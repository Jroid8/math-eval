use num_traits::{Float, FloatConst};
use std::{
    fmt::{Debug, Display},
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
            NativeFunction::Frac => NFPointer::Single(N::fract),
            NativeFunction::Abs => NFPointer::Single(N::abs),
            NativeFunction::Sign => NFPointer::Single(N::signum),
            NativeFunction::Sqrt => NFPointer::Single(N::sqrt),
            NativeFunction::Cbrt => NFPointer::Single(N::cbrt),
            NativeFunction::Max => NFPointer::Flexible(N::maximum),
            NativeFunction::Min => NFPointer::Flexible(N::minimum),
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

pub trait MathEvalNumber: Float + FromStr + FloatConst + Debug + 'static {
    fn parse_constant(input: &str) -> Option<Self> {
        match input {
            "pi" => Some(Self::PI()),
            "e" => Some(Self::E()),
            "tau" => Some(Self::TAU()),
            _ => None,
        }
    }

    fn cot(self) -> Self {
        (Self::FRAC_PI_2() - self).tan()
    }

    fn acot(self) -> Self {
        (-self).atan() + Self::FRAC_PI_2()
    }

    fn maximum(args: &[Self]) -> Self {
        args.iter().copied().reduce(|acc, c| acc.max(c)).unwrap()
    }

    fn minimum(args: &[Self]) -> Self {
        args.iter().copied().reduce(|acc, c| acc.min(c)).unwrap()
    }

    fn factorial(self) -> Self {
        todo!()
    }
}

impl<T: Float + FromStr + FloatConst + Debug + 'static> MathEvalNumber for T {}
