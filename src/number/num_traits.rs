use std::{
    fmt::Debug,
    marker::PhantomData,
    ops::{self},
    str::FromStr,
};

use num_traits::{FloatConst, One, ToPrimitive, Zero, real::Real};

use crate::{
    FunctionIdentifier as FuncId, VariableIdentifier as VarId,
    number::{
        BfPointer, Number,
        std_float::{
            STD_FLOAT_CONSTS_TRIE_NODES, StdFloatFunc, StdFloatFuncsTrie, StdFloatLike,
            StdFloatPrecisionGuard, StdFloatRecognizer, substitute_std_float_spec_funcs_eq,
        },
    },
    postfix_tree::PostfixTree,
    syntax::AstNode,
    trie::{NameTrie, TrieNode},
};

#[derive(Debug, PartialEq, Clone, Copy, PartialOrd)]
#[repr(transparent)]
pub struct NumReal<N: Real>(pub N);

impl<N: Real> ops::Add for NumReal<N> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        NumReal(self.0 + rhs.0)
    }
}

impl<N: Real> ops::Sub for NumReal<N> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        NumReal(self.0 - rhs.0)
    }
}

impl<N: Real> ops::Mul for NumReal<N> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        NumReal(self.0 * rhs.0)
    }
}

impl<N: Real> ops::Div for NumReal<N> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        NumReal(self.0 / rhs.0)
    }
}

impl<N: Real> ops::Neg for NumReal<N> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        NumReal(-self.0)
    }
}

impl<N: Real + FromStr> FromStr for NumReal<N> {
    type Err = <N as FromStr>::Err;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        N::from_str(s).map(NumReal)
    }
}

pub struct NumRealConstsNameTrie<F: Real + FloatConst>(PhantomData<F>);

impl<F: Real + FloatConst> NameTrie<NumReal<F>> for NumRealConstsNameTrie<F> {
    fn nodes(&self) -> &[TrieNode] {
        &STD_FLOAT_CONSTS_TRIE_NODES
    }

    fn leaf_to_value(&self, leaf: u32) -> NumReal<F> {
        match leaf {
            0 => NumReal(F::PI()),
            1 => NumReal(F::E()),
            2 => NumReal(F::TAU()),
            _ => unreachable!(),
        }
    }
}

impl<N> Number for NumReal<N>
where
    N: Real + FloatConst + Debug + FromStr + 'static,
{
    type AsArg<'a> = Self;
    type Recognizer = StdFloatRecognizer;
    type ConstsTrieType = NumRealConstsNameTrie<N>;
    type ExtraFuncId = StdFloatFunc;
    type BuiltinFuncsTrieType = StdFloatFuncsTrie;
    type ImmEvalStabilityGuard = StdFloatPrecisionGuard<Self>;

    const CONSTS_TRIE: NumRealConstsNameTrie<N> = NumRealConstsNameTrie(PhantomData);
    const BUILTIN_FUNCS_TRIE: Self::BuiltinFuncsTrieType = StdFloatFuncsTrie;
    const DO_DISPLACING_SIMPLIFICATION: bool = true;

    fn get_method_ptr(id: StdFloatFunc) -> super::BfPointer<Self> {
        match id {
            StdFloatFunc::Sin => BfPointer::Single(Self::sin),
            StdFloatFunc::Cos => BfPointer::Single(Self::cos),
            StdFloatFunc::Tan => BfPointer::Single(Self::tan),
            StdFloatFunc::Cot => BfPointer::Single(Self::cot),
            StdFloatFunc::Sinh => BfPointer::Single(Self::sinh),
            StdFloatFunc::Cosh => BfPointer::Single(Self::cosh),
            StdFloatFunc::Tanh => BfPointer::Single(Self::tanh),
            StdFloatFunc::Coth => BfPointer::Single(Self::coth),
            StdFloatFunc::Asin => BfPointer::Single(Self::asin),
            StdFloatFunc::Acos => BfPointer::Single(Self::acos),
            StdFloatFunc::Atan => BfPointer::Single(Self::atan),
            StdFloatFunc::Acot => BfPointer::Single(Self::acot),
            StdFloatFunc::Atan2 => BfPointer::<Self>::Dual(Self::atan2),
            StdFloatFunc::Asinh => BfPointer::Single(Self::asinh),
            StdFloatFunc::Acosh => BfPointer::Single(Self::acosh),
            StdFloatFunc::Atanh => BfPointer::Single(Self::atanh),
            StdFloatFunc::Acoth => BfPointer::Single(Self::acot),
            StdFloatFunc::Ln => BfPointer::Single(Self::ln),
            StdFloatFunc::Ln1p => BfPointer::Single(Self::ln1p),
            StdFloatFunc::Exp => BfPointer::Single(Self::exp),
            StdFloatFunc::Expm1 => BfPointer::Single(Self::expm1),
            StdFloatFunc::Floor => BfPointer::Single(Self::floor),
            StdFloatFunc::Ceil => BfPointer::Single(Self::ceil),
            StdFloatFunc::Round => BfPointer::Single(Self::round),
            StdFloatFunc::Trunc => BfPointer::Single(Self::trunc),
            StdFloatFunc::Frac => BfPointer::Single(Self::frac),
            StdFloatFunc::Cbrt => BfPointer::Single(Self::cqrt),
        }
    }

    fn asarg(&self) -> Self {
        *self
    }

    fn substitute_spec_funcs_equivalents<V: VarId, F: FuncId>(
        tree: &mut PostfixTree<AstNode<Self, V, F>>,
    ) {
        substitute_std_float_spec_funcs_eq(tree)
    }

    fn from_i8(value: i8) -> Self {
        NumReal(N::from(value).unwrap())
    }

    fn as_i8(&self) -> Option<i8> {
        <N as ToPrimitive>::to_i8(&self.0)
    }

    fn log(self, base: Self) -> Self {
        NumReal(self.0.log(base.0))
    }

    fn log2(self) -> Self {
        NumReal(self.0.log2())
    }

    fn log10(self) -> Self {
        NumReal(self.0.log10())
    }

    fn exp2(self) -> Self {
        NumReal(self.0.exp2())
    }

    fn exp10(self) -> Self {
        Self(N::from(10).unwrap().powf(self.0))
    }

    fn modulo(self, rhs: Self) -> Self {
        NumReal(self.0 % rhs.0)
    }

    fn pow(self, rhs: Self) -> Self {
        NumReal(self.0.powf(rhs.0))
    }

    fn abs(self) -> Self {
        NumReal(self.0.abs())
    }

    fn sign(self) -> Self {
        let zero = <N as Zero>::zero();
        let one = <N as One>::one();
        NumReal(if self.0.is_zero() {
            zero
        } else if self.0 > zero {
            one
        } else {
            -one
        })
    }

    fn sqrt(self) -> Self {
        NumReal(self.0.sqrt())
    }

    fn factorial(self) -> Self {
        let zero = <N as Zero>::zero();
        let one = <N as One>::one();
        if self.0 < zero {
            return NumReal(zero);
        }
        let mut result = one;
        let mut k = self.0.floor();
        while k > one {
            result = result * k;
            k = k - one;
        }
        NumReal(result)
    }

    fn double_factorial(self) -> Self {
        let zero = <N as Zero>::zero();
        let one = <N as One>::one();
        if self.0 < zero {
            return NumReal(zero);
        }
        let mut result = one;
        let mut k = self.0.floor();
        while k > one {
            result = result * k;
            k = k - Self::from_i8(2).0;
        }
        NumReal(result)
    }

    fn min(values: &[Self]) -> Self {
        NumReal(
            values
                .iter()
                .copied()
                .map(|nr| nr.0)
                .reduce(|acc, x| <N as Real>::min(acc, x))
                .unwrap(),
        )
    }

    fn max(values: &[Self]) -> Self {
        NumReal(
            values
                .iter()
                .copied()
                .map(|nr| nr.0)
                .reduce(|acc, x| <N as Real>::max(acc, x))
                .unwrap(),
        )
    }
}

impl<N> StdFloatLike for NumReal<N>
where
    N: Real + FloatConst + Debug + FromStr + 'static,
{
    fn is_two(self) -> bool {
        self.0.to_u8() == Some(2)
    }

    fn is_ten(self) -> bool {
        self.0.to_u8() == Some(10)
    }

    fn sin(self) -> Self {
        NumReal(self.0.sin())
    }

    fn cos(self) -> Self {
        NumReal(self.0.cos())
    }

    fn tan(self) -> Self {
        NumReal(self.0.tan())
    }

    fn cot(self) -> Self {
        let (sin, cos) = self.0.sin_cos();
        NumReal(cos / sin)
    }

    fn asin(self) -> Self {
        NumReal(self.0.asin())
    }

    fn acos(self) -> Self {
        NumReal(self.0.acos())
    }

    fn atan(self) -> Self {
        NumReal(self.0.atan())
    }

    fn atan2(self, x: Self) -> Self {
        NumReal(self.0.atan2(x.0))
    }

    fn acot(self) -> Self {
        NumReal((-self.0).atan() + N::FRAC_PI_2())
    }

    fn sinh(self) -> Self {
        NumReal(self.0.sinh())
    }

    fn cosh(self) -> Self {
        NumReal(self.0.cosh())
    }

    fn tanh(self) -> Self {
        NumReal(self.0.tanh())
    }

    fn coth(self) -> Self {
        self.cosh() / self.sinh()
    }

    fn asinh(self) -> Self {
        NumReal(self.0.asinh())
    }

    fn acosh(self) -> Self {
        NumReal(self.0.acosh())
    }

    fn atanh(self) -> Self {
        NumReal(self.0.atanh())
    }

    fn acoth(self) -> Self {
        NumReal(self.0.recip().atanh())
    }

    fn ln(self) -> Self {
        NumReal(self.0.ln())
    }

    fn ln1p(self) -> Self {
        NumReal(self.0.ln_1p())
    }

    fn exp(self) -> Self {
        NumReal(self.0.exp())
    }

    fn expm1(self) -> Self {
        NumReal(self.0.exp_m1())
    }

    fn floor(self) -> Self {
        NumReal(self.0.floor())
    }

    fn ceil(self) -> Self {
        NumReal(self.0.ceil())
    }

    fn round(self) -> Self {
        NumReal(self.0.round())
    }

    fn trunc(self) -> Self {
        NumReal(self.0.trunc())
    }

    fn frac(self) -> Self {
        NumReal(self.0.fract())
    }

    fn cqrt(self) -> Self {
        NumReal(self.0.cbrt())
    }
}
