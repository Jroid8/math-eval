use std::{cmp::Ordering, str::FromStr};

use crate::{
    FunctionIdentifier as FuncId, VariableIdentifier as VarId,
    number::{
        CommonBuiltinFunc, CommonFuncsTrie, NoStabilityGuard, Number, get_common_method_ptr,
        substitute_common_spec_funcs_eq,
    },
    postfix_tree::PostfixTree,
    syntax::AstNode,
    tokenizer::NumberRecognizer,
    trie::EmptyNameTrie,
};

pub struct StdIntRecognizer(bool);

impl NumberRecognizer for StdIntRecognizer {
    fn new(current: char) -> Option<Self> {
        match current {
            '0'..='9' => Some(Self(false)),
            _ => None,
        }
    }

    fn recognize(&mut self, current: char) -> bool {
        if current == 'e' {
            self.0 = !self.0;
            self.0
        } else {
            current.is_ascii_digit()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntMathError {
    Overflow,
    DivisionByZero,
    LogDomainViolation,
    SqrtDomainViolation,
    FactorialDomainViolation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct PanicFreeInt<N>(pub Result<N, IntMathError>);

macro_rules! pfi_inner {
    ($pfi: expr) => {
        if let Ok(i) = $pfi.0 { i } else { return $pfi }
    };
}

macro_rules! impl_op_for_pfi {
    ($t: ident, $op: ident, $func: ident, $met: ident) => {
        impl std::ops::$op for PanicFreeInt<$t> {
            type Output = Self;

            fn $func(self, rhs: Self) -> Self::Output {
                Self(
                    pfi_inner!(self)
                        .$met(pfi_inner!(rhs))
                        .ok_or(IntMathError::Overflow),
                )
            }
        }
    };
}

macro_rules! impl_all_ops_for_pfi {
    ($($t: ident),+) => {$(
        impl_op_for_pfi!($t, Add, add, checked_add);
        impl_op_for_pfi!($t, Sub, sub, checked_sub);
        impl_op_for_pfi!($t, Mul, mul, checked_mul);

        impl std::ops::Div for PanicFreeInt<$t> {
            type Output = Self;
            fn div(self, rhs: Self) -> Self::Output {
                let rhs = pfi_inner!(rhs);
                if rhs == 0 {
                    return Self(Err(IntMathError::DivisionByZero));
                }
                Self(
                    pfi_inner!(self)
                        .checked_div(rhs)
                        .ok_or(IntMathError::Overflow),
                )
            }
        }

        impl std::ops::Neg for PanicFreeInt<$t> {
            type Output = Self;
            fn neg(self) -> Self::Output {
                Self(pfi_inner!(self).checked_neg().ok_or(IntMathError::Overflow))
            }
        }
        )+};
}

impl_all_ops_for_pfi!(i8, i16, i32, i64, i128, isize);

fn can_overflow<T>(digits: &[u8]) -> bool {
    let threshold = const {
        use std::f64::consts::LOG10_2;
        let bits = 8 * size_of::<T>() - 1;
        (bits as f64 * LOG10_2).ceil() as usize
    };
    digits.len() >= threshold
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum ParseIntError {
    Empty,
    InvalidDigit,
    Overflow,
    Fractional,
}

fn parse_u32(src: &str) -> Result<u32, ParseIntError> {
    let src = src.as_bytes();
    let mut result: u32 = 0;
    if src.is_empty() {
        return Err(ParseIntError::Empty);
    }
    let mut digits = match src {
        [b'+'] => {
            return Err(ParseIntError::InvalidDigit);
        }
        [b'+', rest @ ..] => rest,
        _ => src,
    };
    if can_overflow::<u32>(digits) {
        while let [c, rest @ ..] = digits {
            result = result.checked_mul(10).ok_or(ParseIntError::Overflow)?;
            result = result
                .checked_add(
                    (*c as char)
                        .to_digit(10)
                        .ok_or(ParseIntError::InvalidDigit)?,
                )
                .ok_or(ParseIntError::Overflow)?;
            digits = rest;
        }
    } else {
        while let [c, rest @ ..] = digits {
            result *= 10;
            result += (*c as char)
                .to_digit(10)
                .ok_or(ParseIntError::InvalidDigit)?;
            digits = rest;
        }
    }
    Ok(result)
}

macro_rules! impl_from_str_for_pfi {
    ($($t: ident),+) => {$(
        impl FromStr for PanicFreeInt<$t> {
            type Err = ParseIntError;

            fn from_str(s: &str) -> Result<Self, Self::Err> {
                let mut src = s.as_bytes();
                let mut result: $t = 0;
                if src.is_empty() {
                    return Err(ParseIntError::Empty);
                }
                let mut exp = if let Some(pos) = s.find('e').or_else(|| s.find('E')) {
                    src = &src[..pos];
                    parse_u32(&s[pos + 1..])?
                } else {
                    0
                };
                let (is_positive, mut digits) = match src {
                    [b'+' | b'-'] => {
                        return Err(ParseIntError::InvalidDigit);
                    }
                    [b'+', rest @ ..] => (true, rest),
                    [b'-', rest @ ..] => (false, rest),
                    _ => (true, src),
                };
                if can_overflow::<$t>(digits) {
                    let mut decimal = false;
                    macro_rules! run_checked_loop {
                        ($additive_func: ident) => {
                            while let [c, rest @ ..] = digits {
                                if *c == b'.' {
                                    decimal = true;
                                    digits = &digits[1..];
                                    break;
                                }
                                result = result.checked_mul(10).ok_or(ParseIntError::Overflow)?;
                                let x = (*c as char)
                                    .to_digit(10)
                                    .ok_or(ParseIntError::InvalidDigit)? as $t;
                                result = result.$additive_func(x).ok_or(ParseIntError::Overflow)?;
                                digits = rest;
                            }
                            if decimal {
                                while let [c, rest @ ..] = digits {
                                    exp = exp.checked_sub(1).ok_or(ParseIntError::Fractional)?;
                                    result = result.checked_mul(10).ok_or(ParseIntError::Overflow)?;
                                    let x = (*c as char)
                                        .to_digit(10)
                                        .ok_or(ParseIntError::InvalidDigit)? as $t;
                                    result = result.$additive_func(x).ok_or(ParseIntError::Overflow)?;
                                    digits = rest;
                                }
                            }
                        };
                    }
                    if is_positive {
                        run_checked_loop!(checked_add);
                    } else {
                        run_checked_loop!(checked_sub);
                    }
                } else {
                    let mut decimal = false;
                    macro_rules! run_unchecked_loop {
                        ($additive_op: tt) => {
                            while let [c, rest @ ..] = digits {
                                if *c == b'.' {
                                    decimal = true;
                                    digits = &digits[1..];
                                    break;
                                }
                                result *= 10;
                                let x = (*c as char)
                                    .to_digit(10)
                                    .ok_or(ParseIntError::InvalidDigit)? as $t;
                                result $additive_op x;
                                digits = rest;
                            }
                            if decimal {
                                while let [c, rest @ ..] = digits {
                                    exp = exp.checked_sub(1).ok_or(ParseIntError::Fractional)?;
                                    result *= 10;
                                    let x = (*c as char)
                                        .to_digit(10)
                                        .ok_or(ParseIntError::InvalidDigit)? as $t;
                                    result $additive_op x;
                                    digits = rest;
                                }
                            }
                        };
                    }
                    if is_positive {
                        run_unchecked_loop!(+=);
                    } else {
                        run_unchecked_loop!(-=);
                    }
                }
                result = (10 as $t).checked_pow(exp).and_then(|r| result.checked_mul(r)).ok_or(ParseIntError::Overflow)?;
                Ok(PanicFreeInt(Ok(result)))
            }
        }
    )+};
}

impl_from_str_for_pfi!(i8, i16, i32, i64, i128, isize);

impl<N: Ord> PartialOrd for PanicFreeInt<N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.0.as_ref().ok()?.cmp(other.0.as_ref().ok()?))
    }
}

macro_rules! impl_number_for_pfi {
    ($($t: ident),+) => {$(
        impl Number for PanicFreeInt<$t> {
            type AsArg<'a> = Self;
            type Recognizer = StdIntRecognizer;
            type ConstsTrieType = EmptyNameTrie;
            type BuiltinFuncId = CommonBuiltinFunc;
            type BuiltinFuncsTrieType = CommonFuncsTrie;
            type ImmEvalStabilityGuard = NoStabilityGuard<Self>;

            const CONSTS_TRIE: Self::ConstsTrieType = EmptyNameTrie;
            const BUILTIN_FUNCS_TRIE: Self::BuiltinFuncsTrieType = CommonFuncsTrie;

            fn get_method_ptr(id: Self::BuiltinFuncId) -> super::BfPointer<Self> {
                get_common_method_ptr(id)
            }

            fn substitute_spec_funcs_equivalents<V: VarId, F: FuncId>(
                tree: &mut PostfixTree<AstNode<Self, V, F>>,
            ) {
                substitute_common_spec_funcs_eq(tree);
            }

            fn asarg(&self) -> Self::AsArg<'_> {
                *self
            }

            fn from_i8(value: i8) -> Self {
                Self(Ok(value.into()))
            }

            fn as_i8(&self) -> Option<i8> {
                self.0.ok()?.try_into().ok()
            }

            fn pow(self, rhs: Self) -> Self {
                let rhs = pfi_inner!(rhs);
                if rhs < 0 {
                    Self(Ok(0))
                } else if let Ok(rhs) = rhs.try_into() {
                    Self(
                        pfi_inner!(self)
                            .checked_pow(rhs)
                            .ok_or(IntMathError::Overflow),
                    )
                } else {
                    Self(Err(IntMathError::Overflow))
                }
            }

            fn exp2(self) -> Self {
                let exp = pfi_inner!(self);
                if exp < 0 {
                    Self(Ok(0))
                } else if let Ok(exp) = exp.try_into() {
                    Self((2 as $t).checked_pow(exp).ok_or(IntMathError::Overflow))
                } else {
                    Self(Err(IntMathError::Overflow))
                }
            }

            fn exp10(self) -> Self {
                let exp = pfi_inner!(self);
                if exp < 0 {
                    Self(Ok(0))
                } else if let Ok(exp) = exp.try_into() {
                    Self((10 as $t).checked_pow(exp).ok_or(IntMathError::Overflow))
                } else {
                    Self(Err(IntMathError::Overflow))
                }
            }

            fn log(self, base: Self) -> Self {
                Self(
                    pfi_inner!(self)
                        .checked_ilog(pfi_inner!(base))
                        .ok_or(IntMathError::LogDomainViolation)
                        .and_then(|v| v.try_into().map_err(|_| IntMathError::Overflow)),
                )
            }

            fn log2(self) -> Self {
                Self(
                    pfi_inner!(self)
                        .checked_ilog2()
                        .ok_or(IntMathError::LogDomainViolation)
                        .and_then(|v| v.try_into().map_err(|_| IntMathError::Overflow)),
                )
            }

            fn log10(self) -> Self {
                Self(
                    pfi_inner!(self)
                        .checked_ilog10()
                        .ok_or(IntMathError::LogDomainViolation)
                        .and_then(|v| v.try_into().map_err(|_| IntMathError::Overflow)),
                )
            }

            fn sqrt(self) -> Self {
                Self(
                    pfi_inner!(self)
                        .checked_isqrt()
                        .ok_or(IntMathError::SqrtDomainViolation),
                )
            }

            fn modulo(self, rhs: Self) -> Self {
                let rhs = pfi_inner!(rhs);
                if rhs == 0 {
                    Self(Err(IntMathError::DivisionByZero))
                } else {
                    Self(
                        pfi_inner!(self)
                            .checked_rem_euclid(rhs)
                            .ok_or(IntMathError::Overflow),
                    )
                }
            }

            fn abs(self) -> Self {
                Self(
                    pfi_inner!(self)
                        .checked_abs()
                        .ok_or(IntMathError::SqrtDomainViolation),
                )
            }

            fn sign(self) -> Self {
                Self(self.0.map($t::signum))
            }

            fn min(values: &[Self]) -> Self {
                Self(
                    values
                        .iter()
                        .try_fold($t::MAX, |acc, x| x.0.map(|x| x.min(acc))),
                )
            }

            fn max(values: &[Self]) -> Self {
                Self(
                    values
                        .iter()
                        .try_fold($t::MIN, |acc, x| x.0.map(|x| x.max(acc))),
                )
            }

            fn factorial(self) -> Self {
                const OVERFLOW_N: $t = {
                    let mut n = 3;
                    let mut p: $t = 2;
                    while let Some(res) = p.checked_mul(n) {
                        p = res;
                        n += 1;
                    }
                    n
                };
                let n = pfi_inner!(self);
                if n < 0 {
                    Self(Err(IntMathError::FactorialDomainViolation))
                } else if n >= OVERFLOW_N {
                    Self(Err(IntMathError::Overflow))
                } else {
                    Self(Ok((2..=n).product()))
                }
            }

            fn double_factorial(self) -> Self {
                const OVERFLOW_N: $t = {
                    let mut n = 4;
                    let mut even: $t = 2;
                    let mut odd: $t = 3;
                    loop {
                        if n & 1 == 0 {
                            if let Some(res) = even.checked_mul(n) {
                                even = res;
                            } else {
                                break;
                            }
                        } else {
                            if let Some(res) = odd.checked_mul(n) {
                                odd = res;
                            } else {
                                break;
                            }
                        }
                        n += 1;
                    }
                    n
                };
                let n = pfi_inner!(self);
                if n < 0 {
                    Self(Err(IntMathError::FactorialDomainViolation))
                } else if n >= OVERFLOW_N {
                    Self(Err(IntMathError::Overflow))
                } else {
                    Self(Ok((2..=n).rev().step_by(2).product()))
                }
            }
        }
    )+};
}

impl_number_for_pfi!(i8, i16, i32, i64, i128, isize);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pfi_parse() {
        let parse = |s: &str| -> Result<i32, ParseIntError> {
            s.parse::<PanicFreeInt<i32>>().map(|pfi| pfi.0.unwrap())
        };
        assert_eq!(parse("123"), Ok(123));
        assert_eq!(parse("123e1"), Ok(1230));
        assert_eq!(parse("1e6"), Ok(1000000));
        assert_eq!(parse("3.141e3"), Ok(3141));
        assert_eq!(parse(".3141e4"), Ok(3141));
        assert_eq!(parse("2147483647"), Ok(2147483647));
        assert_eq!(parse("2.14748367e8"), Ok(214748367));
        assert_eq!(parse("1.14748367e9"), Ok(1147483670));
        assert_eq!(parse("1.5"), Err(ParseIntError::Fractional));
        assert_eq!(parse("2.718281828459045e5"), Err(ParseIntError::Fractional));
        assert_eq!(parse("187f"), Err(ParseIntError::InvalidDigit));
        assert_eq!(parse("4294967296"), Err(ParseIntError::Overflow));
        assert_eq!(parse("1e10"), Err(ParseIntError::Overflow));
        assert_eq!(parse("1.1111111111e10"), Err(ParseIntError::Overflow));
    }
}
