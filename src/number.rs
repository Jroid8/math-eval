use std::{
    f64::consts::{E, PI, TAU},
    fmt::{Debug, Display},
    marker::PhantomData,
    ops::{Add, Div, Mul, Neg, Sub},
    str::FromStr,
};

use strum::EnumIter;

use crate::{
    FunctionIdentifier,
    quick_expr::{CtxFuncPtr, FunctionSource, MarkedFunc},
};

#[derive(Debug, Clone, Copy, Hash)]
pub enum NFPointer<N: Number> {
    Single(for<'a> fn(N::AsArg<'a>) -> N),
    Dual(for<'a, 'b> fn(N::AsArg<'a>, N::AsArg<'b>) -> N),
    Flexible(fn(&[N]) -> N),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, EnumIter)]
pub enum NativeFunction {
    Sin,
    Cos,
    Tan,
    Cot,
    Asin,
    Acos,
    Atan,
    Acot,
    Log,
    Log2,
    Log10,
    Ln,
    Exp,
    Floor,
    Ceil,
    Round,
    Trunc,
    Frac,
    Abs,
    Sign,
    Sqrt,
    Cbrt,
    Max,
    Min,
}

impl NativeFunction {
    pub fn parse(input: &str) -> Option<NativeFunction> {
        match input {
            "sin" => Some(NativeFunction::Sin),
            "cos" => Some(NativeFunction::Cos),
            "tan" => Some(NativeFunction::Tan),
            "cot" => Some(NativeFunction::Cot),
            "asin" => Some(NativeFunction::Asin),
            "acos" => Some(NativeFunction::Acos),
            "atan" => Some(NativeFunction::Atan),
            "acot" => Some(NativeFunction::Acot),
            "log" => Some(NativeFunction::Log),
            "log2" => Some(NativeFunction::Log2),
            "log10" => Some(NativeFunction::Log10),
            "ln" => Some(NativeFunction::Ln),
            "lg" => Some(NativeFunction::Log10),
            "lb" => Some(NativeFunction::Log2),
            "exp" => Some(NativeFunction::Exp),
            "floor" => Some(NativeFunction::Floor),
            "ceil" => Some(NativeFunction::Ceil),
            "round" => Some(NativeFunction::Round),
            "trunc" => Some(NativeFunction::Trunc),
            "frac" => Some(NativeFunction::Frac),
            "abs" => Some(NativeFunction::Abs),
            "sign" => Some(NativeFunction::Sign),
            "sqrt" => Some(NativeFunction::Sqrt),
            "cbrt" => Some(NativeFunction::Cbrt),
            "max" => Some(NativeFunction::Max),
            "min" => Some(NativeFunction::Min),
            _ => None,
        }
    }
    pub fn as_pointer<N: Number>(self) -> NFPointer<N> {
        match self {
            NativeFunction::Sin => NFPointer::Single(N::sin),
            NativeFunction::Cos => NFPointer::Single(N::cos),
            NativeFunction::Tan => NFPointer::Single(N::tan),
            NativeFunction::Cot => NFPointer::Single(N::cot),
            NativeFunction::Asin => NFPointer::Single(N::asin),
            NativeFunction::Acos => NFPointer::Single(N::acos),
            NativeFunction::Atan => NFPointer::Single(N::atan),
            NativeFunction::Acot => NFPointer::Single(N::acot),
            NativeFunction::Log => NFPointer::Dual(N::log),
            NativeFunction::Log2 => NFPointer::Single(N::log2),
            NativeFunction::Log10 => NFPointer::Single(N::log10),
            NativeFunction::Ln => NFPointer::Single(N::ln),
            NativeFunction::Exp => NFPointer::Single(N::exp),
            NativeFunction::Floor => NFPointer::Single(N::floor),
            NativeFunction::Ceil => NFPointer::Single(N::ceil),
            NativeFunction::Round => NFPointer::Single(N::round),
            NativeFunction::Trunc => NFPointer::Single(N::trunc),
            NativeFunction::Frac => NFPointer::Single(N::frac),
            NativeFunction::Abs => NFPointer::Single(N::abs),
            NativeFunction::Sign => NFPointer::Single(N::sign),
            NativeFunction::Sqrt => NFPointer::Single(N::sqrt),
            NativeFunction::Cbrt => NFPointer::Single(N::cbrt),
            NativeFunction::Max => NFPointer::Flexible(N::max),
            NativeFunction::Min => NFPointer::Flexible(N::min),
        }
    }
    pub fn as_markedfunc<N: Number, F: FunctionIdentifier>(
        self,
        argc: u8,
    ) -> MarkedFunc<'static, N, F> {
        match self.as_pointer::<N>() {
            NFPointer::Single(func) => MarkedFunc {
                func: CtxFuncPtr::Single(func),
                _src: PhantomData,
                src: FunctionSource::NativeFunction(self),
            },
            NFPointer::Dual(func) => MarkedFunc {
                func: CtxFuncPtr::Dual(func),
                _src: PhantomData,
                src: FunctionSource::NativeFunction(self),
            },
            NFPointer::Flexible(func) => MarkedFunc {
                func: CtxFuncPtr::Flexible(func, argc),
                _src: PhantomData,
                src: FunctionSource::NativeFunction(self),
            },
        }
    }
    pub fn is_fixed(self) -> bool {
        !matches!(self, NativeFunction::Min | NativeFunction::Max)
    }
    pub fn min_args(self) -> u8 {
        match self {
            NativeFunction::Log => 2,
            NativeFunction::Max => 2,
            NativeFunction::Min => 2,
            _ => 1,
        }
    }
    pub fn max_args(self) -> Option<u8> {
        match self {
            NativeFunction::Log => Some(2),
            NativeFunction::Max => None,
            NativeFunction::Min => None,
            _ => Some(1),
        }
    }
}

impl Display for NativeFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            NativeFunction::Sin => "sin",
            NativeFunction::Cos => "cos",
            NativeFunction::Tan => "tan",
            NativeFunction::Cot => "cot",
            NativeFunction::Asin => "asin",
            NativeFunction::Acos => "acos",
            NativeFunction::Atan => "atan",
            NativeFunction::Acot => "acot",
            NativeFunction::Log => "log",
            NativeFunction::Log2 => "log2",
            NativeFunction::Log10 => "log10",
            NativeFunction::Ln => "ln",
            NativeFunction::Exp => "exp",
            NativeFunction::Floor => "floor",
            NativeFunction::Ceil => "ceil",
            NativeFunction::Round => "round",
            NativeFunction::Trunc => "trunc",
            NativeFunction::Frac => "frac",
            NativeFunction::Abs => "abs",
            NativeFunction::Sign => "sign",
            NativeFunction::Sqrt => "sqrt",
            NativeFunction::Cbrt => "cbrt",
            NativeFunction::Max => "max",
            NativeFunction::Min => "min",
        })
    }
}

pub trait Reborrow {
    type This<'a>
    where
        Self: 'a;
    fn reborrow(&self) -> Self::This<'_>;
}

impl<T> Reborrow for &'_ T {
    type This<'a>
        = &'a T
    where
        Self: 'a;
    fn reborrow(&self) -> Self::This<'_> {
        self
    }
}

pub trait Number:
    Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + PartialEq
    + From<i32>
    + FromStr
    + Neg<Output = Self>
    + Clone
    + Debug
    + 'static
{
    type AsArg<'a>: ToOwned<Owned = Self>
        + for<'b> Add<Self::AsArg<'b>, Output = Self>
        + for<'b> Sub<Self::AsArg<'b>, Output = Self>
        + for<'b> Mul<Self::AsArg<'b>, Output = Self>
        + for<'b> Div<Self::AsArg<'b>, Output = Self>
        + for<'b> Neg<Output = Self>
        + for<'b> Reborrow<This<'b> = Self::AsArg<'b>>
        + PartialEq
        + Neg<Output = Self>
        + Copy
        + Debug;

    fn parse_constant(input: &str) -> Option<Self>;
    fn modulo(value: Self::AsArg<'_>, rhs: Self::AsArg<'_>) -> Self;
    fn pow(value: Self::AsArg<'_>, rhs: Self::AsArg<'_>) -> Self;
    fn sin(value: Self::AsArg<'_>) -> Self;
    fn cos(value: Self::AsArg<'_>) -> Self;
    fn tan(value: Self::AsArg<'_>) -> Self;
    fn cot(value: Self::AsArg<'_>) -> Self;
    fn asin(value: Self::AsArg<'_>) -> Self;
    fn acos(value: Self::AsArg<'_>) -> Self;
    fn atan(value: Self::AsArg<'_>) -> Self;
    fn acot(value: Self::AsArg<'_>) -> Self;
    fn log(value: Self::AsArg<'_>, base: Self::AsArg<'_>) -> Self;
    fn log2(value: Self::AsArg<'_>) -> Self;
    fn log10(value: Self::AsArg<'_>) -> Self;
    fn ln(value: Self::AsArg<'_>) -> Self;
    fn exp(value: Self::AsArg<'_>) -> Self;
    fn floor(value: Self::AsArg<'_>) -> Self;
    fn ceil(value: Self::AsArg<'_>) -> Self;
    fn round(value: Self::AsArg<'_>) -> Self;
    fn trunc(value: Self::AsArg<'_>) -> Self;
    fn frac(value: Self::AsArg<'_>) -> Self;
    fn abs(value: Self::AsArg<'_>) -> Self;
    fn sign(value: Self::AsArg<'_>) -> Self;
    fn sqrt(value: Self::AsArg<'_>) -> Self;
    fn cbrt(value: Self::AsArg<'_>) -> Self;
    fn max(value: &[Self]) -> Self;
    fn min(value: &[Self]) -> Self;
    fn factorial(value: Self::AsArg<'_>) -> Self;
    fn double_factorial(value: Self::AsArg<'_>) -> Self;
    fn asarg(&self) -> Self::AsArg<'_>;

    fn negexp(lhs: Self::AsArg<'_>, rhs: Self::AsArg<'_>) -> Self {
        Self::pow(lhs, (-rhs).asarg())
    }
}

impl Number for f64 {
    type AsArg<'a> = f64;

    fn pow(value: Self, rhs: Self) -> Self {
        value.powf(rhs)
    }

    fn parse_constant(input: &str) -> Option<Self> {
        match input {
            "pi" => Some(PI),
            "e" => Some(E),
            "tau" => Some(TAU),
            _ => None,
        }
    }

    fn modulo(value: Self, rhs: Self) -> Self {
        value.rem_euclid(rhs)
    }

    fn sin(value: Self) -> Self {
        value.sin()
    }

    fn cos(value: Self) -> Self {
        value.cos()
    }

    fn tan(value: Self) -> Self {
        value.tan()
    }

    fn cot(value: Self) -> Self {
        // FIX: replace with a more accurate version
        let (sin, cos) = value.sin_cos();
        cos / sin
    }

    fn asin(value: Self) -> Self {
        value.asin()
    }

    fn acos(value: Self) -> Self {
        value.acos()
    }

    fn atan(value: Self) -> Self {
        value.atan()
    }

    fn acot(value: Self) -> Self {
        (-value).atan() + std::f64::consts::FRAC_PI_2
    }

    fn log(value: Self, base: Self) -> Self {
        value.log(base)
    }

    fn log2(value: Self) -> Self {
        value.log2()
    }

    fn log10(value: Self) -> Self {
        value.log10()
    }

    fn ln(value: Self) -> Self {
        value.ln()
    }

    fn exp(value: Self) -> Self {
        value.exp()
    }

    fn floor(value: Self) -> Self {
        value.floor()
    }

    fn ceil(value: Self) -> Self {
        value.ceil()
    }

    fn round(value: Self) -> Self {
        value.round()
    }

    fn trunc(value: Self) -> Self {
        value.trunc()
    }

    fn frac(value: Self) -> Self {
        value.fract()
    }

    fn abs(value: Self) -> Self {
        value.abs()
    }

    fn sign(value: Self) -> Self {
        match value.partial_cmp(&0.0) {
            Some(cmp) => match cmp {
                std::cmp::Ordering::Less => -1.0,
                std::cmp::Ordering::Equal => 0.0,
                std::cmp::Ordering::Greater => 1.0,
            },
            None => value,
        }
    }

    fn sqrt(value: Self) -> Self {
        value.sqrt()
    }

    fn cbrt(value: Self) -> Self {
        value.cbrt()
    }

    fn max(value: &[Self]) -> Self {
        value.iter().copied().fold(f64::MIN, f64::max)
    }

    fn min(value: &[Self]) -> Self {
        value.iter().copied().fold(f64::MAX, f64::min)
    }

    fn factorial(value: Self) -> Self {
        // FIX: replace with gamma function
        if value.is_infinite() || value < 0.0 || value >= u32::MAX as f64 || value.is_nan() {
            return f64::NAN;
        }
        let mut result = 1.0;
        let mut k = value as u32;
        while k > 1 {
            result *= k as f64;
            k -= 1;
        }
        result
    }

    fn double_factorial(value: Self) -> Self {
        if value.is_infinite() || value < 0.0 || value >= u32::MAX as f64 || value.is_nan() {
            return f64::NAN;
        }
        let mut result = 1.0;
        let mut k = value as u32;
        while k > 1 {
            result *= k as f64;
            k -= 2;
        }
        result
    }

    fn asarg(&self) -> Self::AsArg<'_> {
        *self
    }
}

impl Reborrow for f64 {
    type This<'a> = f64;

    fn reborrow(&self) -> Self::This<'_> {
        *self
    }
}
