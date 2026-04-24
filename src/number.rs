use std::{
    f64::consts::{E, PI, TAU},
    fmt::{Debug, Display},
    marker::PhantomData,
    num::NonZeroU8,
    ops::{Add, Div, Mul, Neg, Sub},
    str::FromStr,
};

use strum::{EnumIter, FromRepr};

use crate::{
    FunctionIdentifier as FuncId, nz,
    quick_expr::{CtxFuncPtr, MarkedFunc},
    tokenizer::{NumberRecognizer, StandardFloatRecognizer},
    trie::{NameTrie, TrieNode},
};

#[cfg(debug_assertions)]
use crate::quick_expr::FunctionSource;

#[derive(Debug, Clone, Copy, Hash)]
pub enum BFPointer<N: Number> {
    Single(for<'a> fn(N) -> N),
    Dual(for<'a> fn(N, N::AsArg<'a>) -> N),
    Flexible(fn(&[N]) -> N),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, EnumIter, FromRepr)]
#[repr(u8)]
pub enum BuiltinFunction {
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

impl BuiltinFunction {
    pub fn as_pointer<N: Number>(self) -> BFPointer<N> {
        match self {
            BuiltinFunction::Sin => BFPointer::Single(N::sin),
            BuiltinFunction::Cos => BFPointer::Single(N::cos),
            BuiltinFunction::Tan => BFPointer::Single(N::tan),
            BuiltinFunction::Cot => BFPointer::Single(N::cot),
            BuiltinFunction::Asin => BFPointer::Single(N::asin),
            BuiltinFunction::Acos => BFPointer::Single(N::acos),
            BuiltinFunction::Atan => BFPointer::Single(N::atan),
            BuiltinFunction::Acot => BFPointer::Single(N::acot),
            BuiltinFunction::Log => BFPointer::Dual(N::log),
            BuiltinFunction::Log2 => BFPointer::Single(N::log2),
            BuiltinFunction::Log10 => BFPointer::Single(N::log10),
            BuiltinFunction::Ln => BFPointer::Single(N::ln),
            BuiltinFunction::Exp => BFPointer::Single(N::exp),
            BuiltinFunction::Floor => BFPointer::Single(N::floor),
            BuiltinFunction::Ceil => BFPointer::Single(N::ceil),
            BuiltinFunction::Round => BFPointer::Single(N::round),
            BuiltinFunction::Trunc => BFPointer::Single(N::trunc),
            BuiltinFunction::Frac => BFPointer::Single(N::frac),
            BuiltinFunction::Abs => BFPointer::Single(N::abs),
            BuiltinFunction::Sign => BFPointer::Single(N::sign),
            BuiltinFunction::Sqrt => BFPointer::Single(N::sqrt),
            BuiltinFunction::Cbrt => BFPointer::Single(N::cbrt),
            BuiltinFunction::Max => BFPointer::Flexible(N::max),
            BuiltinFunction::Min => BFPointer::Flexible(N::min),
        }
    }
    pub fn as_markedfunc<N: Number, F: FuncId>(self, argc: NonZeroU8) -> MarkedFunc<'static, N, F> {
        match self.as_pointer::<N>() {
            BFPointer::Single(func) => MarkedFunc {
                func: CtxFuncPtr::Single(func),
                _src: PhantomData,
                #[cfg(debug_assertions)]
                src: FunctionSource::BuiltinFunction(self),
            },
            BFPointer::Dual(func) => MarkedFunc {
                func: CtxFuncPtr::Dual(func),
                _src: PhantomData,
                #[cfg(debug_assertions)]
                src: FunctionSource::BuiltinFunction(self),
            },
            BFPointer::Flexible(func) => MarkedFunc {
                func: CtxFuncPtr::Flexible(func, argc),
                _src: PhantomData,
                #[cfg(debug_assertions)]
                src: FunctionSource::BuiltinFunction(self),
            },
        }
    }
    pub const fn is_flex(self) -> bool {
        matches!(self, BuiltinFunction::Min | BuiltinFunction::Max)
    }
    pub const fn min_args(self) -> NonZeroU8 {
        match self {
            BuiltinFunction::Log | BuiltinFunction::Max | BuiltinFunction::Min => nz!(2),
            _ => nz!(1),
        }
    }
    pub const fn max_args(self) -> Option<NonZeroU8> {
        match self {
            BuiltinFunction::Log => Some(nz!(2)),
            BuiltinFunction::Max | BuiltinFunction::Min => None,
            _ => Some(nz!(1)),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            BuiltinFunction::Sin => "sin",
            BuiltinFunction::Cos => "cos",
            BuiltinFunction::Tan => "tan",
            BuiltinFunction::Cot => "cot",
            BuiltinFunction::Asin => "asin",
            BuiltinFunction::Acos => "acos",
            BuiltinFunction::Atan => "atan",
            BuiltinFunction::Acot => "acot",
            BuiltinFunction::Log => "log",
            BuiltinFunction::Log2 => "log2",
            BuiltinFunction::Log10 => "log10",
            BuiltinFunction::Ln => "ln",
            BuiltinFunction::Exp => "exp",
            BuiltinFunction::Floor => "floor",
            BuiltinFunction::Ceil => "ceil",
            BuiltinFunction::Round => "round",
            BuiltinFunction::Trunc => "trunc",
            BuiltinFunction::Frac => "frac",
            BuiltinFunction::Abs => "abs",
            BuiltinFunction::Sign => "sign",
            BuiltinFunction::Sqrt => "sqrt",
            BuiltinFunction::Cbrt => "cbrt",
            BuiltinFunction::Max => "max",
            BuiltinFunction::Min => "min",
        }
    }
}

const BUILTIN_FUNCS_TRIE_NODES: [TrieNode; 94] = [
    TrieNode::Branch('a', 17),
    TrieNode::Branch('b', 2),
    TrieNode::Branch('s', 1),
    TrieNode::Leaf(18),
    TrieNode::Branch('c', 5),
    TrieNode::Branch('o', 4),
    TrieNode::Branch('s', 1),
    TrieNode::Leaf(5),
    TrieNode::Branch('t', 1),
    TrieNode::Leaf(7),
    TrieNode::Branch('s', 3),
    TrieNode::Branch('i', 2),
    TrieNode::Branch('n', 1),
    TrieNode::Leaf(4),
    TrieNode::Branch('t', 3),
    TrieNode::Branch('a', 2),
    TrieNode::Branch('n', 1),
    TrieNode::Leaf(6),
    TrieNode::Branch('c', 13),
    TrieNode::Branch('b', 3),
    TrieNode::Branch('r', 2),
    TrieNode::Branch('t', 1),
    TrieNode::Leaf(21),
    TrieNode::Branch('e', 3),
    TrieNode::Branch('i', 2),
    TrieNode::Branch('l', 1),
    TrieNode::Leaf(14),
    TrieNode::Branch('o', 4),
    TrieNode::Branch('s', 1),
    TrieNode::Leaf(1),
    TrieNode::Branch('t', 1),
    TrieNode::Leaf(3),
    TrieNode::Branch('e', 3),
    TrieNode::Branch('x', 2),
    TrieNode::Branch('p', 1),
    TrieNode::Leaf(12),
    TrieNode::Branch('f', 9),
    TrieNode::Branch('l', 4),
    TrieNode::Branch('o', 3),
    TrieNode::Branch('o', 2),
    TrieNode::Branch('r', 1),
    TrieNode::Leaf(13),
    TrieNode::Branch('r', 3),
    TrieNode::Branch('a', 2),
    TrieNode::Branch('c', 1),
    TrieNode::Leaf(17),
    TrieNode::Branch('l', 14),
    TrieNode::Branch('b', 1),
    TrieNode::Leaf(9),
    TrieNode::Branch('g', 1),
    TrieNode::Leaf(10),
    TrieNode::Branch('n', 1),
    TrieNode::Leaf(11),
    TrieNode::Branch('o', 7),
    TrieNode::Branch('g', 6),
    TrieNode::Leaf(8),
    TrieNode::Branch('1', 2),
    TrieNode::Branch('0', 1),
    TrieNode::Leaf(10),
    TrieNode::Branch('2', 1),
    TrieNode::Leaf(9),
    TrieNode::Branch('m', 6),
    TrieNode::Branch('a', 2),
    TrieNode::Branch('x', 1),
    TrieNode::Leaf(22),
    TrieNode::Branch('i', 2),
    TrieNode::Branch('n', 1),
    TrieNode::Leaf(23),
    TrieNode::Branch('r', 5),
    TrieNode::Branch('o', 4),
    TrieNode::Branch('u', 3),
    TrieNode::Branch('n', 2),
    TrieNode::Branch('d', 1),
    TrieNode::Leaf(15),
    TrieNode::Branch('s', 10),
    TrieNode::Branch('i', 5),
    TrieNode::Branch('g', 2),
    TrieNode::Branch('n', 1),
    TrieNode::Leaf(19),
    TrieNode::Branch('n', 1),
    TrieNode::Leaf(0),
    TrieNode::Branch('q', 3),
    TrieNode::Branch('r', 2),
    TrieNode::Branch('t', 1),
    TrieNode::Leaf(20),
    TrieNode::Branch('t', 8),
    TrieNode::Branch('a', 2),
    TrieNode::Branch('n', 1),
    TrieNode::Leaf(2),
    TrieNode::Branch('r', 4),
    TrieNode::Branch('u', 3),
    TrieNode::Branch('n', 2),
    TrieNode::Branch('c', 1),
    TrieNode::Leaf(16),
];

pub struct BuiltinFuncsNameTrie;

impl NameTrie<BuiltinFunction> for BuiltinFuncsNameTrie {
    fn nodes(&self) -> &[TrieNode] {
        &BUILTIN_FUNCS_TRIE_NODES
    }
    fn leaf_to_value(&self, leaf: u32) -> BuiltinFunction {
        BuiltinFunction::from_repr(leaf as u8).unwrap()
    }
}

impl Display for BuiltinFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

pub trait Number:
    for<'a> Add<Self::AsArg<'a>, Output = Self>
    + for<'a> Sub<Self::AsArg<'a>, Output = Self>
    + for<'a> Mul<Self::AsArg<'a>, Output = Self>
    + for<'a> Div<Self::AsArg<'a>, Output = Self>
    + Neg<Output = Self>
    + PartialEq
    + From<i32>
    + FromStr
    + Clone
    + Debug
    + 'static
{
    type AsArg<'a>: ToOwned<Owned = Self> + Neg<Output = Self> + PartialEq + Copy + Debug;
    type Recognizer: NumberRecognizer;
    type ConstsNameTrieType: NameTrie<&'static Self>;

    const CONSTS_NAME_TRIE: Self::ConstsNameTrieType;

    fn is_two(value: Self::AsArg<'_>) -> bool;
    fn is_ten(value: Self::AsArg<'_>) -> bool;
    fn asarg(&self) -> Self::AsArg<'_>;

    fn modulo(value: Self, rhs: Self::AsArg<'_>) -> Self;
    fn pow(value: Self, rhs: Self::AsArg<'_>) -> Self;
    fn sin(value: Self) -> Self;
    fn cos(value: Self) -> Self;
    fn tan(value: Self) -> Self;
    fn cot(value: Self) -> Self;
    fn asin(value: Self) -> Self;
    fn acos(value: Self) -> Self;
    fn atan(value: Self) -> Self;
    fn acot(value: Self) -> Self;
    fn log(value: Self, base: Self::AsArg<'_>) -> Self;
    fn log2(value: Self) -> Self;
    fn log10(value: Self) -> Self;
    fn ln(value: Self) -> Self;
    fn exp(value: Self) -> Self;
    fn floor(value: Self) -> Self;
    fn ceil(value: Self) -> Self;
    fn round(value: Self) -> Self;
    fn trunc(value: Self) -> Self;
    fn frac(value: Self) -> Self;
    fn abs(value: Self) -> Self;
    fn sign(value: Self) -> Self;
    fn sqrt(value: Self) -> Self;
    fn cbrt(value: Self) -> Self;
    fn max(value: &[Self]) -> Self;
    fn min(value: &[Self]) -> Self;
    fn factorial(value: Self) -> Self;
    fn double_factorial(value: Self) -> Self;
}

const STD_FLOAT_CONSTS_TRIE_NODES: [TrieNode; 9] = [
    TrieNode::Branch('p', 2),
    TrieNode::Branch('i', 1),
    TrieNode::Leaf(0),
    TrieNode::Branch('e', 1),
    TrieNode::Leaf(1),
    TrieNode::Branch('t', 3),
    TrieNode::Branch('a', 2),
    TrieNode::Branch('u', 1),
    TrieNode::Leaf(2),
];

pub struct StdFloatConstsNameTrie<F: 'static> {
    pi: &'static F,
    e: &'static F,
    tau: &'static F,
}

impl<F> NameTrie<&'static F> for StdFloatConstsNameTrie<F> {
    fn nodes(&self) -> &[TrieNode] {
        &STD_FLOAT_CONSTS_TRIE_NODES
    }

    fn leaf_to_value(&self, leaf: u32) -> &'static F {
        match leaf {
            0 => self.pi,
            1 => self.e,
            2 => self.tau,
            _ => unreachable!(),
        }
    }
}

impl Number for f64 {
    type AsArg<'a> = f64;
    type Recognizer = StandardFloatRecognizer;
    type ConstsNameTrieType = StdFloatConstsNameTrie<f64>;

    const CONSTS_NAME_TRIE: StdFloatConstsNameTrie<f64> = StdFloatConstsNameTrie {
        pi: &PI,
        e: &E,
        tau: &TAU,
    };

    fn asarg(&self) -> Self::AsArg<'_> {
        *self
    }

    fn is_two(value: Self::AsArg<'_>) -> bool {
        value == 2.0
    }

    fn is_ten(value: Self::AsArg<'_>) -> bool {
        value == 10.0
    }

    fn pow(value: Self, rhs: Self) -> Self {
        value.powf(rhs)
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
        if value.is_infinite() || value < 0.0 || value.is_nan() {
            return f64::NAN;
        }
        if value >= 171.0 {
            return f64::INFINITY;
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
        if value.is_infinite() || value < 0.0 || value.is_nan() {
            return f64::NAN;
        }
        if value >= 301.0 {
            return f64::INFINITY;
        }
        let mut result = 1.0;
        let mut k = value as u32;
        while k > 1 {
            result *= k as f64;
            k -= 2;
        }
        result
    }
}
