use std::{
    fmt::Debug,
    num::NonZeroU8,
    ops::{Add, Div, Mul, Neg, Sub},
    str::FromStr,
};

use strum::FromRepr;

use crate::{
    BinaryOp, FunctionIdentifier as FuncId, UnaryOp, VariableIdentifier as VarId, nz,
    postfix_tree::{PostfixTree, subtree_collection::SubtreeCollection},
    syntax::{AstNode, FunctionType},
    tokenizer::NumberRecognizer,
    trie::NameTrie,
};

pub mod std_float;
#[cfg(feature = "libm")]
pub mod libm_ext;

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
    fn into_common(self) -> Option<CommonBuiltinFunc>;
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, FromRepr)]
#[repr(u8)]
pub enum CommonBuiltinFunc {
    Exp2,
    Exp10,
    Log,
    Log2,
    Log10,
    Sqrt,
    Abs,
    Sign,
    Min,
    Max,
}

impl BuiltinFuncId for CommonBuiltinFunc {
    fn from_common(id: CommonBuiltinFunc) -> Self {
        id
    }

    fn into_common(self) -> Option<CommonBuiltinFunc> {
        Some(self)
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

    fn from_i8(value: i8) -> Self;
    fn as_i8(&self) -> Option<i8>;

    fn pow(self, rhs: Self::AsArg<'_>) -> Self;
    fn exp2(self) -> Self;
    fn exp10(self) -> Self;
    fn log(self, base: Self::AsArg<'_>) -> Self;
    fn log2(self) -> Self;
    fn log10(self) -> Self;

    fn sqrt(self) -> Self;
    fn modulo(self, rhs: Self::AsArg<'_>) -> Self;
    fn abs(self) -> Self;
    fn sign(self) -> Self;
    fn factorial(self) -> Self;
    fn double_factorial(self) -> Self;
    fn max(values: &[Self]) -> Self;
    fn min(values: &[Self]) -> Self;
}

#[derive(Debug)]
pub struct NoStabilityGuard<N: Number>(N);

impl<N: Number> ImmEvalStabilityGuard<N> for NoStabilityGuard<N> {
    fn from_number(num: N) -> Self {
        NoStabilityGuard(num)
    }

    fn eval(self) -> N {
        self.0
    }

    fn apply_unary_op(self, opr: UnaryOp) -> Self {
        NoStabilityGuard(opr.eval(self.0))
    }

    fn apply_binary_op(self, rhs: Self, opr: BinaryOp) -> Self {
        NoStabilityGuard(opr.eval(self.0, rhs.0.asarg()))
    }

    fn apply_func_single(self, _id: N::BuiltinFuncId, func: fn(N) -> N) -> Self {
        NoStabilityGuard(func(self.0))
    }

    fn apply_func_dual(
        self,
        arg2: Self,
        _id: N::BuiltinFuncId,
        func: for<'a> fn(N, <N as Number>::AsArg<'a>) -> N,
    ) -> Self {
        NoStabilityGuard(func(self.0, arg2.0.asarg()))
    }

    fn apply_func_flex(
        args: std::vec::Drain<'_, Self>,
        _id: N::BuiltinFuncId,
        func: fn(&[N]) -> N,
        arg_space: &mut Vec<N>,
    ) -> Self {
        arg_space.extend(args.map(|a| a.0));
        NoStabilityGuard(func(&arg_space))
    }
}

pub fn substitute_exp_eq<N: Number, V: VarId, F: FuncId>(
    tree: &mut PostfixTree<AstNode<N, V, F>>,
    symbol_space: &mut SubtreeCollection<AstNode<N, V, F>>,
    target: usize,
) -> bool {
    if matches!(tree[target], AstNode::BinaryOp(BinaryOp::Pow)) {
        let mut children = tree.children_iter(target);
        let param = children.next().unwrap().1;
        let func = match children.next().unwrap() {
            (AstNode::Number(base), _) if base.as_i8() == Some(2) => CommonBuiltinFunc::Exp2,
            (AstNode::Number(base), _) if base.as_i8() == Some(10) => CommonBuiltinFunc::Exp10,
            _ => return false,
        };
        symbol_space.extend_from_tree(&tree, param);
        symbol_space
            .push(AstNode::Function(
                FunctionType::Builtin(N::BuiltinFuncId::from_common(func)),
                nz!(1),
            ))
            .unwrap();
        let sc_head = symbol_space.len() - 1;
        tree.replace_from_sc_move(target, symbol_space, sc_head);
        true
    } else {
        false
    }
}

pub fn substitute_log_eq<N: Number, V: VarId, F: FuncId>(
    tree: &mut PostfixTree<AstNode<N, V, F>>,
    symbol_space: &mut SubtreeCollection<AstNode<N, V, F>>,
    target: usize,
) -> bool {
    if matches!(
        tree[target],
        AstNode::Function(FunctionType::Builtin(func), _)
            if func.into_common() == Some(CommonBuiltinFunc::Log)
    ) {
        let mut children = tree.children_iter(target);
        match children.next().unwrap() {
            (AstNode::Number(base), _) => {
                let func = if base.as_i8() == Some(2) {
                    CommonBuiltinFunc::Log2
                } else if base.as_i8() == Some(10) {
                    CommonBuiltinFunc::Log10
                } else {
                    return false;
                };
                let param = children.next().unwrap().1;
                symbol_space.extend_from_tree(&tree, param);
                symbol_space
                    .push(AstNode::Function(
                        FunctionType::Builtin(N::BuiltinFuncId::from_common(func)),
                        nz!(1),
                    ))
                    .unwrap();
                let sc_head = symbol_space.len() - 1;
                tree.replace_from_sc_move(target, symbol_space, sc_head);
                true
            }
            _ => false,
        }
    } else {
        false
    }
}

pub fn substitute_common_spec_funcs_eq<N: Number, V: VarId, F: FuncId>(
    tree: &mut PostfixTree<AstNode<N, V, F>>,
) {
    let mut symbol_space: SubtreeCollection<AstNode<N, V, F>> =
        SubtreeCollection::from_alloc(Vec::with_capacity(0));
    let mut idx = 2;
    while idx < tree.len() {
        for subs in [substitute_log_eq, substitute_exp_eq] {
            if subs(tree, &mut symbol_space, idx) {
                break;
            }
        }
        idx += 1;
    }
}

pub struct CommonFuncsTrie;

impl NameTrie<CommonBuiltinFunc> for CommonFuncsTrie {
    fn nodes(&self) -> &[crate::trie::TrieNode] {
        &[]
    }

    fn leaf_to_value(&self, leaf: u32) -> CommonBuiltinFunc {
        CommonBuiltinFunc::from_repr(leaf as u8).unwrap()
    }
}

pub fn get_common_method_ptr<N>(id: CommonBuiltinFunc) -> BfPointer<N>
where
    N: Number<BuiltinFuncId = CommonBuiltinFunc>,
{
    match id {
        CommonBuiltinFunc::Exp2 => BfPointer::Single(N::exp2),
        CommonBuiltinFunc::Exp10 => BfPointer::Single(N::exp10),
        CommonBuiltinFunc::Log => BfPointer::Dual(N::log),
        CommonBuiltinFunc::Log2 => BfPointer::Single(N::log2),
        CommonBuiltinFunc::Log10 => BfPointer::Single(N::log10),
        CommonBuiltinFunc::Sqrt => BfPointer::Single(N::sqrt),
        CommonBuiltinFunc::Abs => BfPointer::Single(N::abs),
        CommonBuiltinFunc::Sign => BfPointer::Single(N::sign),
        CommonBuiltinFunc::Min => BfPointer::Flexible(N::min),
        CommonBuiltinFunc::Max => BfPointer::Flexible(N::max),
    }
}
