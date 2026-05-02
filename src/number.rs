use std::{
    fmt::Debug,
    num::NonZeroU8,
    ops::{Add, Div, Mul, Neg, Sub},
    str::FromStr,
};

use crate::{
    BinaryOp, FunctionIdentifier as FuncId, UnaryOp, VariableIdentifier as VarId, nz,
    postfix_tree::PostfixTree, syntax::AstNode, tokenizer::NumberRecognizer, trie::NameTrie,
};

pub mod std_float;

#[derive(Debug, Clone, Copy, Hash)]
pub enum BfPointer<N: Number> {
    Single(for<'a> fn(N) -> N),
    Dual(for<'a> fn(N, N::AsArg<'a>) -> N),
    // Triple(for<'a, 'b> fn(N, N::AsArg<'a>, N::AsArg<'b>) -> N),
    Flexible(fn(&[N]) -> N),
}

impl<N: Number> BfPointer<N> {
    pub fn is_flex(self) -> bool {
        matches!(self, BfPointer::Flexible(_))
    }
}

pub trait BuiltinFuncId: FuncId {
    fn from_common(id: CommonBuiltinFunc) -> Self;
    fn min_args(self) -> NonZeroU8;
    fn max_args(self) -> Option<NonZeroU8>;
    fn is_flex(self) -> bool;
    fn specialize_per_argc(&mut self, argc: NonZeroU8);
}

pub trait ImmEvalStabilityGuard<N: Number>: Sized + Debug {
    fn from_number(num: N) -> Self;
    fn eval(self) -> N;
    fn apply_unary_op(self, opr: UnaryOp) -> Self;
    fn apply_binary_op(self, rhs: Self, opr: BinaryOp) -> Self;
    fn apply_func_single(self, id: N::BuiltinFuncId, func: fn(N) -> N) -> Self;
    fn apply_func_dual(
        self,
        arg2: Self,
        id: N::BuiltinFuncId,
        func: for<'a> fn(N, N::AsArg<'a>) -> N,
    ) -> Self;
    fn apply_func_flex(
        args: std::vec::Drain<'_, Self>,
        id: N::BuiltinFuncId,
        func: fn(&[N]) -> N,
        arg_space: &mut Vec<N>,
    ) -> Self;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommonBuiltinFunc {
    Abs,
    Sign,
    Min,
    Max,
}

impl BuiltinFuncId for CommonBuiltinFunc {
    fn from_common(id: CommonBuiltinFunc) -> Self {
        id
    }

    fn min_args(self) -> NonZeroU8 {
        nz!(1)
    }

    fn max_args(self) -> Option<NonZeroU8> {
        match self {
            Self::Min | Self::Max => None,
            _ => Some(nz!(1)),
        }
    }

    fn is_flex(self) -> bool {
        matches!(self, Self::Min | Self::Max)
    }

    fn specialize_per_argc(&mut self, _argc: NonZeroU8) {}
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
    type ConstsTrieType: NameTrie<Self>;
    type BuiltinFuncId: BuiltinFuncId;
    type BuiltinFuncsTrieType: NameTrie<Self::BuiltinFuncId>;
    type ImmEvalStabilityGuard: ImmEvalStabilityGuard<Self>;

    const CONSTS_TRIE: Self::ConstsTrieType;
    const BUILTIN_FUNCS_TRIE: Self::BuiltinFuncsTrieType;

    fn get_method_ptr(id: Self::BuiltinFuncId) -> BfPointer<Self>;
    fn asarg(&self) -> Self::AsArg<'_>;
    fn substitute_spec_funcs_equivalents<V: VarId, F: FuncId>(
        tree: &mut PostfixTree<AstNode<Self, V, F>>,
    );

    fn one() -> Self;
    fn zero() -> Self;

    fn modulo(self, rhs: Self::AsArg<'_>) -> Self;
    fn pow(self, rhs: Self::AsArg<'_>) -> Self;
    fn abs(self) -> Self;
    fn sign(self) -> Self;
    fn factorial(self) -> Self;
    fn double_factorial(self) -> Self;
    fn max(values: &[Self]) -> Self;
    fn min(values: &[Self]) -> Self;
}
