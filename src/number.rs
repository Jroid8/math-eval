use std::{
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
    tokenizer::NumberRecognizer,
    trie::{NameTrie, TrieNode},
};

mod primitives;

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
    Sinh,
    Cosh,
    Tanh,
    Coth,
    Asin,
    Acos,
    Atan,
    Acot,
    Atan2,
    Asinh,
    Acosh,
    Atanh,
    Acoth,
    Erf,
    Erfc,
    Log,
    Log2,
    Log10,
    Ln,
    Ln1p,
    Exp,
    Exp2,
    Exp10,
    Expm1,
    Floor,
    Ceil,
    Round,
    Trunc,
    Frac,
    Abs,
    Sign,
    Sqrt,
    Cbrt,
    Gamma,
    Lgamma,
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
            BuiltinFunction::Sinh => BFPointer::Single(N::sinh),
            BuiltinFunction::Cosh => BFPointer::Single(N::cosh),
            BuiltinFunction::Tanh => BFPointer::Single(N::tanh),
            BuiltinFunction::Coth => BFPointer::Single(N::coth),
            BuiltinFunction::Asin => BFPointer::Single(N::asin),
            BuiltinFunction::Acos => BFPointer::Single(N::acos),
            BuiltinFunction::Atan => BFPointer::Single(N::atan),
            BuiltinFunction::Acot => BFPointer::Single(N::acot),
            BuiltinFunction::Atan2 => BFPointer::Dual(N::atan2),
            BuiltinFunction::Asinh => BFPointer::Single(N::asinh),
            BuiltinFunction::Acosh => BFPointer::Single(N::acosh),
            BuiltinFunction::Atanh => BFPointer::Single(N::atanh),
            BuiltinFunction::Acoth => BFPointer::Single(N::acoth),
            BuiltinFunction::Erf => BFPointer::Single(N::erf),
            BuiltinFunction::Erfc => BFPointer::Single(N::erfc),
            BuiltinFunction::Log => BFPointer::Dual(N::log),
            BuiltinFunction::Log2 => BFPointer::Single(N::log2),
            BuiltinFunction::Log10 => BFPointer::Single(N::log10),
            BuiltinFunction::Ln => BFPointer::Single(N::ln),
            BuiltinFunction::Ln1p => BFPointer::Single(N::ln_1p),
            BuiltinFunction::Exp => BFPointer::Single(N::exp),
            BuiltinFunction::Exp2 => BFPointer::Single(N::exp2),
            BuiltinFunction::Exp10 => BFPointer::Single(N::exp10),
            BuiltinFunction::Expm1 => BFPointer::Single(N::exp_m1),
            BuiltinFunction::Floor => BFPointer::Single(N::floor),
            BuiltinFunction::Ceil => BFPointer::Single(N::ceil),
            BuiltinFunction::Round => BFPointer::Single(N::round),
            BuiltinFunction::Trunc => BFPointer::Single(N::trunc),
            BuiltinFunction::Frac => BFPointer::Single(N::frac),
            BuiltinFunction::Abs => BFPointer::Single(N::abs),
            BuiltinFunction::Sign => BFPointer::Single(N::sign),
            BuiltinFunction::Sqrt => BFPointer::Single(N::sqrt),
            BuiltinFunction::Cbrt => BFPointer::Single(N::cbrt),
            BuiltinFunction::Gamma => BFPointer::Single(N::gamma),
            BuiltinFunction::Lgamma => BFPointer::Single(N::lgamma),
            BuiltinFunction::Max => BFPointer::Flexible(N::max),
            BuiltinFunction::Min => BFPointer::Flexible(N::min),
        }
    }
    #[allow(dead_code)]
    pub(crate) fn as_markedfunc<N: Number, F: FuncId>(
        self,
        argc: NonZeroU8,
    ) -> MarkedFunc<'static, N, F> {
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
            BuiltinFunction::Sinh => "sinh",
            BuiltinFunction::Cosh => "cosh",
            BuiltinFunction::Tanh => "tanh",
            BuiltinFunction::Coth => "coth",
            BuiltinFunction::Asin => "asin",
            BuiltinFunction::Acos => "acos",
            BuiltinFunction::Atan => "atan",
            BuiltinFunction::Acot => "acot",
            BuiltinFunction::Atan2 => "atan2",
            BuiltinFunction::Asinh => "asinh",
            BuiltinFunction::Acosh => "acosh",
            BuiltinFunction::Atanh => "atanh",
            BuiltinFunction::Acoth => "acoth",
            BuiltinFunction::Erf => "erf",
            BuiltinFunction::Erfc => "erfc",
            BuiltinFunction::Log => "log",
            BuiltinFunction::Log2 => "log2",
            BuiltinFunction::Log10 => "log10",
            BuiltinFunction::Ln => "ln",
            BuiltinFunction::Ln1p => "ln1p",
            BuiltinFunction::Exp => "exp",
            BuiltinFunction::Exp2 => "exp2",
            BuiltinFunction::Exp10 => "exp10",
            BuiltinFunction::Expm1 => "expm1",
            BuiltinFunction::Floor => "floor",
            BuiltinFunction::Ceil => "ceil",
            BuiltinFunction::Round => "round",
            BuiltinFunction::Trunc => "trunc",
            BuiltinFunction::Frac => "frac",
            BuiltinFunction::Abs => "abs",
            BuiltinFunction::Sign => "sign",
            BuiltinFunction::Sqrt => "sqrt",
            BuiltinFunction::Cbrt => "cbrt",
            BuiltinFunction::Gamma => "gamma",
            BuiltinFunction::Lgamma => "lgamma",
            BuiltinFunction::Max => "max",
            BuiltinFunction::Min => "min",
        }
    }
}

static BUILTIN_FUNCS_TRIE_NODES: [TrieNode; 131] = [
    TrieNode::Branch('a', 19),
    TrieNode::Branch('b', 2),
    TrieNode::Branch('s', 1),
    TrieNode::Leaf(BuiltinFunction::Abs as u32),
    TrieNode::Branch('c', 5),
    TrieNode::Branch('o', 4),
    TrieNode::Branch('s', 1),
    TrieNode::Leaf(BuiltinFunction::Acos as u32),
    TrieNode::Branch('t', 1),
    TrieNode::Leaf(BuiltinFunction::Acot as u32),
    TrieNode::Branch('s', 3),
    TrieNode::Branch('i', 2),
    TrieNode::Branch('n', 1),
    TrieNode::Leaf(BuiltinFunction::Asin as u32),
    TrieNode::Branch('t', 5),
    TrieNode::Branch('a', 4),
    TrieNode::Branch('n', 3),
    TrieNode::Leaf(BuiltinFunction::Atan as u32),
    TrieNode::Branch('2', 1),
    TrieNode::Leaf(BuiltinFunction::Atan2 as u32),
    TrieNode::Branch('c', 17),
    TrieNode::Branch('b', 3),
    TrieNode::Branch('r', 2),
    TrieNode::Branch('t', 1),
    TrieNode::Leaf(BuiltinFunction::Cbrt as u32),
    TrieNode::Branch('e', 3),
    TrieNode::Branch('i', 2),
    TrieNode::Branch('l', 1),
    TrieNode::Leaf(BuiltinFunction::Ceil as u32),
    TrieNode::Branch('o', 8),
    TrieNode::Branch('s', 3),
    TrieNode::Leaf(BuiltinFunction::Cos as u32),
    TrieNode::Branch('h', 1),
    TrieNode::Leaf(BuiltinFunction::Cosh as u32),
    TrieNode::Branch('t', 3),
    TrieNode::Leaf(BuiltinFunction::Cot as u32),
    TrieNode::Branch('h', 1),
    TrieNode::Leaf(BuiltinFunction::Coth as u32),
    TrieNode::Branch('e', 16),
    TrieNode::Branch('r', 4),
    TrieNode::Branch('f', 3),
    TrieNode::Leaf(BuiltinFunction::Erf as u32),
    TrieNode::Branch('c', 1),
    TrieNode::Leaf(BuiltinFunction::Erfc as u32),
    TrieNode::Branch('x', 10),
    TrieNode::Branch('p', 9),
    TrieNode::Leaf(BuiltinFunction::Exp as u32),
    TrieNode::Branch('1', 2),
    TrieNode::Branch('0', 1),
    TrieNode::Leaf(BuiltinFunction::Exp10 as u32),
    TrieNode::Branch('2', 1),
    TrieNode::Leaf(BuiltinFunction::Exp2 as u32),
    TrieNode::Branch('m', 2),
    TrieNode::Branch('1', 1),
    TrieNode::Leaf(BuiltinFunction::Expm1 as u32),
    TrieNode::Branch('f', 9),
    TrieNode::Branch('l', 4),
    TrieNode::Branch('o', 3),
    TrieNode::Branch('o', 2),
    TrieNode::Branch('r', 1),
    TrieNode::Leaf(BuiltinFunction::Floor as u32),
    TrieNode::Branch('r', 3),
    TrieNode::Branch('a', 2),
    TrieNode::Branch('c', 1),
    TrieNode::Leaf(BuiltinFunction::Frac as u32),
    TrieNode::Branch('g', 5),
    TrieNode::Branch('a', 4),
    TrieNode::Branch('m', 3),
    TrieNode::Branch('m', 2),
    TrieNode::Branch('a', 1),
    TrieNode::Leaf(BuiltinFunction::Gamma as u32),
    TrieNode::Branch('l', 22),
    TrieNode::Branch('b', 1),
    TrieNode::Leaf(BuiltinFunction::Log2 as u32),
    TrieNode::Branch('g', 6),
    TrieNode::Leaf(BuiltinFunction::Log10 as u32),
    TrieNode::Branch('a', 4),
    TrieNode::Branch('m', 3),
    TrieNode::Branch('m', 2),
    TrieNode::Branch('a', 1),
    TrieNode::Leaf(BuiltinFunction::Lgamma as u32),
    TrieNode::Branch('n', 4),
    TrieNode::Leaf(BuiltinFunction::Ln as u32),
    TrieNode::Branch('p', 2),
    TrieNode::Branch('1', 1),
    TrieNode::Leaf(BuiltinFunction::Ln1p as u32),
    TrieNode::Branch('o', 7),
    TrieNode::Branch('g', 6),
    TrieNode::Leaf(BuiltinFunction::Log as u32),
    TrieNode::Branch('1', 2),
    TrieNode::Branch('0', 1),
    TrieNode::Leaf(BuiltinFunction::Log10 as u32),
    TrieNode::Branch('2', 1),
    TrieNode::Leaf(BuiltinFunction::Log2 as u32),
    TrieNode::Branch('m', 6),
    TrieNode::Branch('a', 2),
    TrieNode::Branch('x', 1),
    TrieNode::Leaf(BuiltinFunction::Max as u32),
    TrieNode::Branch('i', 2),
    TrieNode::Branch('n', 1),
    TrieNode::Leaf(BuiltinFunction::Min as u32),
    TrieNode::Branch('r', 5),
    TrieNode::Branch('o', 4),
    TrieNode::Branch('u', 3),
    TrieNode::Branch('n', 2),
    TrieNode::Branch('d', 1),
    TrieNode::Leaf(BuiltinFunction::Round as u32),
    TrieNode::Branch('s', 12),
    TrieNode::Branch('i', 7),
    TrieNode::Branch('g', 2),
    TrieNode::Branch('n', 1),
    TrieNode::Leaf(BuiltinFunction::Sign as u32),
    TrieNode::Branch('n', 3),
    TrieNode::Leaf(BuiltinFunction::Sin as u32),
    TrieNode::Branch('h', 1),
    TrieNode::Leaf(BuiltinFunction::Sinh as u32),
    TrieNode::Branch('q', 3),
    TrieNode::Branch('r', 2),
    TrieNode::Branch('t', 1),
    TrieNode::Leaf(BuiltinFunction::Sqrt as u32),
    TrieNode::Branch('t', 10),
    TrieNode::Branch('a', 4),
    TrieNode::Branch('n', 3),
    TrieNode::Leaf(BuiltinFunction::Tan as u32),
    TrieNode::Branch('h', 1),
    TrieNode::Leaf(BuiltinFunction::Tanh as u32),
    TrieNode::Branch('r', 4),
    TrieNode::Branch('u', 3),
    TrieNode::Branch('n', 2),
    TrieNode::Branch('c', 1),
    TrieNode::Leaf(BuiltinFunction::Trunc as u32),
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
    + FromStr
    + Clone
    + Debug
    + 'static
{
    type AsArg<'a>: ToOwned<Owned = Self> + Neg<Output = Self> + PartialEq + Copy + Debug;
    type Recognizer: NumberRecognizer;
    type ConstsNameTrieType: NameTrie<Self>;

    const CONSTS_NAME_TRIE: Self::ConstsNameTrieType;

    fn one() -> Self;
    fn zero() -> Self;
    fn is_one(value: Self::AsArg<'_>) -> bool;
    fn is_two(value: Self::AsArg<'_>) -> bool;
    fn is_ten(value: Self::AsArg<'_>) -> bool;
    fn asarg(&self) -> Self::AsArg<'_>;

    fn modulo(self, rhs: Self::AsArg<'_>) -> Self;
    fn pow(self, rhs: Self::AsArg<'_>) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn cot(self) -> Self;
    fn sinh(self) -> Self;
    fn cosh(self) -> Self;
    fn tanh(self) -> Self;
    fn coth(self) -> Self;
    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;
    fn acot(self) -> Self;
    fn atan2(self, x: Self::AsArg<'_>) -> Self;
    fn asinh(self) -> Self;
    fn acosh(self) -> Self;
    fn atanh(self) -> Self;
    fn acoth(self) -> Self;
    fn erf(self) -> Self;
    fn erfc(self) -> Self;
    fn log(self, base: Self::AsArg<'_>) -> Self;
    fn log2(self) -> Self;
    fn log10(self) -> Self;
    fn ln(self) -> Self;
    fn ln_1p(self) -> Self;
    fn exp(self) -> Self;
    fn exp2(self) -> Self;
    fn exp10(self) -> Self;
    fn exp_m1(self) -> Self;
    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn round(self) -> Self;
    fn trunc(self) -> Self;
    fn frac(self) -> Self;
    fn abs(self) -> Self;
    fn sign(self) -> Self;
    fn sqrt(self) -> Self;
    fn cbrt(self) -> Self;
    fn lgamma(self) -> Self;
    fn gamma(self) -> Self;
    fn factorial(self) -> Self;
    fn double_factorial(self) -> Self;
    fn max(values: &[Self]) -> Self;
    fn min(values: &[Self]) -> Self;
}

static STD_FLOAT_CONSTS_TRIE_NODES: [TrieNode; 9] = [
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

pub struct StdFloatConstsNameTrie<F: Clone> {
    pi: F,
    e: F,
    tau: F,
}

impl<F: Clone> NameTrie<F> for StdFloatConstsNameTrie<F> {
    fn nodes(&self) -> &[TrieNode] {
        &STD_FLOAT_CONSTS_TRIE_NODES
    }

    fn leaf_to_value(&self, leaf: u32) -> F {
        match leaf {
            0 => self.pi.clone(),
            1 => self.e.clone(),
            2 => self.tau.clone(),
            _ => unreachable!(),
        }
    }
}
