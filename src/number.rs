use std::{
    fmt::{Debug, Display},
    ops::{Add, Div, Mul, Neg, Sub},
    str::FromStr,
};

#[derive(Clone, Copy)]
pub enum NFPointer<N: MathEvalNumber> {
    Single(fn(N) -> N),
    Dual(fn(N, N) -> Result<N, N::Error>),
    SingleWithError(fn(N) -> Result<N, N::Error>),
    Flexible(fn(&[N]) -> N),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
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
            "ln" => Some(NativeFunction::Ln),
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
    pub fn to_pointer<N: MathEvalNumber>(&self) -> NFPointer<N> {
        match self {
            NativeFunction::Sin => NFPointer::Single(N::sin),
            NativeFunction::Cos => NFPointer::Single(N::cos),
            NativeFunction::Tan => NFPointer::Single(N::tan),
            NativeFunction::Cot => NFPointer::Single(N::cot),
            NativeFunction::Asin => NFPointer::SingleWithError(N::asin),
            NativeFunction::Acos => NFPointer::SingleWithError(N::acos),
            NativeFunction::Atan => NFPointer::SingleWithError(N::atan),
            NativeFunction::Acot => NFPointer::SingleWithError(N::acot),
            NativeFunction::Log => NFPointer::Dual(N::log),
            NativeFunction::Ln => NFPointer::SingleWithError(N::ln),
            NativeFunction::Exp => NFPointer::Single(N::exp),
            NativeFunction::Floor => NFPointer::Single(N::floor),
            NativeFunction::Ceil => NFPointer::Single(N::ceil),
            NativeFunction::Round => NFPointer::Single(N::round),
            NativeFunction::Trunc => NFPointer::Single(N::trunc),
            NativeFunction::Frac => NFPointer::Single(N::frac),
            NativeFunction::Abs => NFPointer::Single(N::abs),
            NativeFunction::Sign => NFPointer::Single(N::sign),
            NativeFunction::Sqrt => NFPointer::SingleWithError(N::sqrt),
            NativeFunction::Cbrt => NFPointer::Single(N::cbrt),
            NativeFunction::Max => NFPointer::Flexible(N::max),
            NativeFunction::Min => NFPointer::Flexible(N::min),
        }
    }
    pub fn is_fixed(&self) -> bool {
        matches!(self, Self::Min | Self::Max)
    }
}

impl Display for NativeFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                NativeFunction::Sin => "sin",
                NativeFunction::Cos => "cos",
                NativeFunction::Tan => "tan",
                NativeFunction::Cot => "cot",
                NativeFunction::Asin => "asin",
                NativeFunction::Acos => "acos",
                NativeFunction::Atan => "atan",
                NativeFunction::Acot => "acot",
                NativeFunction::Log => "log",
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
            }
        )
    }
}

// don't forget to #[inline]
pub trait MathEvalNumber:
    Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + PartialEq
    + From<f64>
    + FromStr
    + Neg<Output = Self>
    + Copy
{
    type Error;

    fn modulo(self, rhs: Self) -> Self;
    fn pow(self, rhs: Self) -> Self;
    fn parse_constant(input: &str) -> Option<Self>;
    fn sin(argument: Self) -> Self;
    fn cos(argument: Self) -> Self;
    fn tan(argument: Self) -> Self;
    fn cot(argument: Self) -> Self;
    fn asin(argument: Self) -> Result<Self, Self::Error>;
    fn acos(argument: Self) -> Result<Self, Self::Error>;
    fn atan(argument: Self) -> Result<Self, Self::Error>;
    fn acot(argument: Self) -> Result<Self, Self::Error>;
    fn log(argument: Self, base: Self) -> Result<Self, Self::Error>;
    fn ln(argument: Self) -> Result<Self, Self::Error>;
    fn exp(argument: Self) -> Self;
    fn floor(argument: Self) -> Self;
    fn ceil(argument: Self) -> Self;
    fn round(argument: Self) -> Self;
    fn trunc(argument: Self) -> Self;
    fn frac(argument: Self) -> Self;
    fn abs(argument: Self) -> Self;
    fn sign(argument: Self) -> Self;
    fn sqrt(argument: Self) -> Result<Self, Self::Error>;
    fn cbrt(argument: Self) -> Self;
    fn max(argument: &[Self]) -> Self;
    fn min(argument: &[Self]) -> Self;
    fn factorial(self) -> Result<Self, Self::Error>;
}

impl MathEvalNumber for f64 {
    type Error = ();

    fn pow(self, rhs: Self) -> Self {
        self.powf(rhs)
    }

    fn parse_constant(input: &str) -> Option<Self> {
        match input {
            "pi" => Some(std::f64::consts::PI),
            "e" => Some(std::f64::consts::E),
            _ => None,
        }
    }

    fn modulo(self, rhs: Self) -> Self {
        self.rem_euclid(rhs)
    }

    fn sin(argument: Self) -> Self {
        argument.sin()
    }

    fn cos(argument: Self) -> Self {
        argument.cos()
    }

    fn tan(argument: Self) -> Self {
        argument.tan()
    }

    fn cot(argument: Self) -> Self {
        let (sin, cos) = argument.sin_cos();
        cos / sin
    }

    fn asin(argument: Self) -> Result<Self, Self::Error> {
        Ok(argument.asin())
    }

    fn acos(argument: Self) -> Result<Self, Self::Error> {
        Ok(argument.acos())
    }

    fn atan(argument: Self) -> Result<Self, Self::Error> {
        Ok(argument.atan())
    }

    fn acot(argument: Self) -> Result<Self, Self::Error> {
        Ok((-argument).atan() + std::f64::consts::FRAC_PI_2)
    }

    fn log(argument: Self, base: Self) -> Result<Self, Self::Error> {
        // log2 and log10 are more accurate than log(2) and log(10) as mentioned by the docs: https://doc.rust-lang.org/std/primitive.f64.html#method.log
        Ok(if base == 10.0 {
            argument.log10()
        } else if base == 2.0 {
            argument.log2()
        } else {
            argument.log(base)
        })
    }

    fn ln(argument: Self) -> Result<Self, Self::Error> {
        Ok(argument.ln())
    }

    fn exp(argument: Self) -> Self {
        argument.exp()
    }

    fn floor(argument: Self) -> Self {
        argument.floor()
    }

    fn ceil(argument: Self) -> Self {
        argument.ceil()
    }

    fn round(argument: Self) -> Self {
        argument.round()
    }

    fn trunc(argument: Self) -> Self {
        argument.trunc()
    }

    fn frac(argument: Self) -> Self {
        argument.fract()
    }

    fn abs(argument: Self) -> Self {
        argument.abs()
    }

    fn sign(argument: Self) -> Self {
        match argument.partial_cmp(&0.0) {
            Some(cmp) => match cmp {
                std::cmp::Ordering::Less => -1.0,
                std::cmp::Ordering::Equal => 0.0,
                std::cmp::Ordering::Greater => 1.0,
            },
            None => argument,
        }
    }

    fn sqrt(argument: Self) -> Result<Self, Self::Error> {
        Ok(argument.sqrt())
    }

    fn cbrt(argument: Self) -> Self {
        argument.cbrt()
    }

    fn max(argument: &[Self]) -> Self {
        argument.iter().copied().fold(f64::MIN, f64::max)
    }

    fn min(argument: &[Self]) -> Self {
        argument.iter().copied().fold(f64::MAX, f64::min)
    }

    fn factorial(self) -> Result<Self, Self::Error> {
        let mut result = 1.0;
        for v in 2..=(self as u32) {
            result *= v as f64;
        }
        Ok(result)
    }
}
