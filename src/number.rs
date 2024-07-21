use std::{
    fmt::{Debug, Display},
    ops::{Add, Div, Mul, Neg, Sub},
    str::FromStr,
};

#[derive(Debug, Clone, Copy, Hash)]
pub enum NFPointer<N: MathEvalNumber> {
    Single(fn(N) -> N),
    Dual(fn(N, N) -> N),
    Flexible(fn(&[N]) -> N),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
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
    pub fn is_fixed(&self) -> bool {
        matches!(self, Self::Min | Self::Max)
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

// don't forget to #[inline]
pub trait MathEvalNumber:
    Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + PartialEq
    + From<i16>
    + FromStr
    + Neg<Output = Self>
    + Copy
    + Debug
    + 'static
{
    fn modulo(self, rhs: Self) -> Self;
    fn pow(self, rhs: Self) -> Self;
    fn parse_constant(input: &str) -> Option<Self>;
    fn sin(argument: Self) -> Self;
    fn cos(argument: Self) -> Self;
    fn tan(argument: Self) -> Self;
    fn cot(argument: Self) -> Self;
    fn asin(argument: Self) -> Self;
    fn acos(argument: Self) -> Self;
    fn atan(argument: Self) -> Self;
    fn acot(argument: Self) -> Self;
    fn log(argument: Self, base: Self) -> Self;
    fn log2(argument: Self) -> Self;
    fn log10(argument: Self) -> Self;
    fn ln(argument: Self) -> Self;
    fn exp(argument: Self) -> Self;
    fn floor(argument: Self) -> Self;
    fn ceil(argument: Self) -> Self;
    fn round(argument: Self) -> Self;
    fn trunc(argument: Self) -> Self;
    fn frac(argument: Self) -> Self;
    fn abs(argument: Self) -> Self;
    fn sign(argument: Self) -> Self;
    fn sqrt(argument: Self) -> Self;
    fn cbrt(argument: Self) -> Self;
    fn max(argument: &[Self]) -> Self;
    fn min(argument: &[Self]) -> Self;
    fn factorial(self) -> Self;
}

macro_rules! impl_float {
    ($ft:ident) => {
        impl MathEvalNumber for $ft {
            fn pow(self, rhs: Self) -> Self {
                self.powf(rhs)
            }

            fn parse_constant(input: &str) -> Option<Self> {
                match input {
                    "pi" => Some(std::$ft::consts::PI),
                    "e" => Some(std::$ft::consts::E),
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

            fn asin(argument: Self) -> Self {
                argument.asin()
            }

            fn acos(argument: Self) -> Self {
                argument.acos()
            }

            fn atan(argument: Self) -> Self {
                argument.atan()
            }

            fn acot(argument: Self) -> Self {
                (-argument).atan() + std::$ft::consts::FRAC_PI_2
            }

            fn log(argument: Self, base: Self) -> Self {
                argument.log(base)
            }

            fn log2(argument: Self) -> Self {
                argument.log2()
            }

            fn log10(argument: Self) -> Self {
                argument.log2()
            }

            fn ln(argument: Self) -> Self {
                argument.ln()
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

            fn sqrt(argument: Self) -> Self {
                argument.sqrt()
            }

            fn cbrt(argument: Self) -> Self {
                argument.cbrt()
            }

            fn max(argument: &[Self]) -> Self {
                argument.iter().copied().fold(Self::MIN, Self::max)
            }

            fn min(argument: &[Self]) -> Self {
                argument.iter().copied().fold(Self::MAX, Self::min)
            }

            fn factorial(self) -> Self {
                let mut result = 1.0;
                for v in 2..=(self as u32) {
                    result *= v as $ft;
                }
                result
            }
        }
    };
}

impl_float!(f64);
impl_float!(f32);
