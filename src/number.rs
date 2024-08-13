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
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn cot(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;
    fn acot(self) -> Self;
    fn log(self, base: Self) -> Self;
    fn log2(self) -> Self;
    fn log10(self) -> Self;
    fn ln(self) -> Self;
    fn exp(self) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn round(self) -> Self;
    fn trunc(self) -> Self;
    fn frac(self) -> Self;
    fn abs(self) -> Self;
    fn sign(self) -> Self;
    fn sqrt(self) -> Self;
    fn cbrt(self) -> Self;
    fn max(values: &[Self]) -> Self;
    fn min(values: &[Self]) -> Self;
    fn factorial(self) -> Self;
}

#[cfg(not(feature = "num-traits"))]
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

            fn sin(self) -> Self {
                self.sin()
            }

            fn cos(self) -> Self {
                self.cos()
            }

            fn tan(self) -> Self {
                self.tan()
            }

            fn cot(self) -> Self {
                let (sin, cos) = self.sin_cos();
                cos / sin
            }

            fn asin(self) -> Self {
                self.asin()
            }

            fn acos(self) -> Self {
                self.acos()
            }

            fn atan(self) -> Self {
                self.atan()
            }

            fn acot(self) -> Self {
                (-self).atan() + std::$ft::consts::FRAC_PI_2
            }

            fn log(self, base: Self) -> Self {
                self.log(base)
            }

            fn log2(self) -> Self {
                self.log2()
            }

            fn log10(self) -> Self {
                self.log2()
            }

            fn ln(self) -> Self {
                self.ln()
            }

            fn exp(self) -> Self {
                self.exp()
            }

            fn floor(self) -> Self {
                self.floor()
            }

            fn ceil(self) -> Self {
                self.ceil()
            }

            fn round(self) -> Self {
                self.round()
            }

            fn trunc(self) -> Self {
                self.trunc()
            }

            fn frac(self) -> Self {
                self.fract()
            }

            fn abs(self) -> Self {
                self.abs()
            }

            fn sign(self) -> Self {
                match self.partial_cmp(&0.0) {
                    Some(cmp) => match cmp {
                        std::cmp::Ordering::Less => -1.0,
                        std::cmp::Ordering::Equal => 0.0,
                        std::cmp::Ordering::Greater => 1.0,
                    },
                    None => self,
                }
            }

            fn sqrt(self) -> Self {
                self.sqrt()
            }

            fn cbrt(self) -> Self {
                self.cbrt()
            }

            fn max(values: &[Self]) -> Self {
                values.iter().copied().fold(Self::MIN, Self::max)
            }

            fn min(values: &[Self]) -> Self {
                values.iter().copied().fold(Self::MAX, Self::min)
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

#[cfg(not(feature = "num-traits"))]
impl_float!(f64);
#[cfg(not(feature = "num-traits"))]
impl_float!(f32);

#[cfg(feature = "num-traits")]
impl<T> MathEvalNumber for T
where
    T: num_traits::Float + From<i16> + FromStr + Debug + 'static + num_traits::FloatConst,
{
    fn modulo(self, rhs: Self) -> Self {
        let r = self % rhs;
        if r.is_sign_negative() {
            r.abs()
        } else {
            r
        }
    }

    fn pow(self, rhs: Self) -> Self {
        self.powf(rhs)
    }

    fn parse_constant(input: &str) -> Option<Self> {
        match input {
            "pi" => Some(Self::PI()),
            "e" => Some(Self::E()),
            "tau" => Some(Self::TAU()),
            _ => None,
        }
    }

    fn sin(self) -> Self {
        self.sin()
    }

    fn cos(self) -> Self {
        self.cos()
    }

    fn tan(self) -> Self {
        self.tan()
    }

    fn cot(self) -> Self {
        let (sin, cos) = self.sin_cos();
        sin / cos
    }

    fn asin(self) -> Self {
        self.asin()
    }

    fn acos(self) -> Self {
        self.acos()
    }

    fn atan(self) -> Self {
        self.atan()
    }

    fn acot(self) -> Self {
        (-self).atan() + Self::FRAC_PI_2()
    }

    fn log(self, base: Self) -> Self {
        self.log(base)
    }

    fn log2(self) -> Self {
        self.log2()
    }

    fn log10(self) -> Self {
        self.log10()
    }

    fn ln(self) -> Self {
        self.ln()
    }

    fn exp(self) -> Self {
        self.exp()
    }

    fn floor(self) -> Self {
        self.floor()
    }

    fn ceil(self) -> Self {
        self.ceil()
    }

    fn round(self) -> Self {
        self.round()
    }

    fn trunc(self) -> Self {
        self.trunc()
    }

    fn frac(self) -> Self {
        self.fract()
    }

    fn abs(self) -> Self {
        self.abs()
    }

    fn sign(self) -> Self {
        self.signum()
    }

    fn sqrt(self) -> Self {
        self.sqrt()
    }

    fn cbrt(self) -> Self {
        self.cbrt()
    }

    fn max(values: &[Self]) -> Self {
        values.iter().copied().reduce(Self::max).unwrap()
    }

    fn min(values: &[Self]) -> Self {
        values.iter().copied().reduce(Self::min).unwrap()
    }

    fn factorial(self) -> Self {
        todo!()
    }
}
