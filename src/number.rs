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

static COMMON_FUNCS_TRIE_NODES: [TrieNode; 37] = [
    TrieNode::Branch('a', 3),
    TrieNode::Branch('b', 2),
    TrieNode::Branch('s', 1),
    TrieNode::Leaf(CommonBuiltinFunc::Abs as u32),
    TrieNode::Branch('e', 7),
    TrieNode::Branch('x', 6),
    TrieNode::Branch('p', 5),
    TrieNode::Branch('1', 2),
    TrieNode::Branch('0', 1),
    TrieNode::Leaf(CommonBuiltinFunc::Exp10 as u32),
    TrieNode::Branch('2', 1),
    TrieNode::Leaf(CommonBuiltinFunc::Exp2 as u32),
    TrieNode::Branch('l', 8),
    TrieNode::Branch('o', 7),
    TrieNode::Branch('g', 6),
    TrieNode::Leaf(CommonBuiltinFunc::Log as u32),
    TrieNode::Branch('1', 2),
    TrieNode::Branch('0', 1),
    TrieNode::Leaf(CommonBuiltinFunc::Log10 as u32),
    TrieNode::Branch('2', 1),
    TrieNode::Leaf(CommonBuiltinFunc::Log2 as u32),
    TrieNode::Branch('m', 6),
    TrieNode::Branch('a', 2),
    TrieNode::Branch('x', 1),
    TrieNode::Leaf(CommonBuiltinFunc::Max as u32),
    TrieNode::Branch('i', 2),
    TrieNode::Branch('n', 1),
    TrieNode::Leaf(CommonBuiltinFunc::Min as u32),
    TrieNode::Branch('s', 8),
    TrieNode::Branch('i', 3),
    TrieNode::Branch('g', 2),
    TrieNode::Branch('n', 1),
    TrieNode::Leaf(CommonBuiltinFunc::Sign as u32),
    TrieNode::Branch('q', 3),
    TrieNode::Branch('r', 2),
    TrieNode::Branch('t', 1),
    TrieNode::Leaf(CommonBuiltinFunc::Sqrt as u32),
];

pub struct CommonFuncsTrie;

impl NameTrie<CommonBuiltinFunc> for CommonFuncsTrie {
    fn nodes(&self) -> &[crate::trie::TrieNode] {
        &COMMON_FUNCS_TRIE_NODES
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{number::std_float::StdFloatFunc, syntax::MathAst};

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct X;

    #[test]
    fn substitute_log_equivalent() {
        let mut symbol_space = SubtreeCollection::new();
        let mut sle = move |nodes: &[AstNode<f64, X, ()>]| {
            let mut ast = MathAst::from_nodes(nodes.iter().copied()).into_tree();
            let i = 2;
            while i < ast.len() {
                substitute_log_eq(&mut ast, &mut symbol_space, i);
            }
            ast.postorder_iter().cloned().collect::<Vec<_>>()
        };
        assert_eq!(
            sle(&[
                AstNode::Variable(X),
                AstNode::Number(2.0),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Log.into()), nz!(2)),
            ]),
            vec![
                AstNode::Variable(X),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Log2.into()), nz!(1)),
            ]
        );
        assert_eq!(
            sle(&[
                AstNode::Variable(X),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Sqrt.into()), nz!(1)),
                AstNode::Number(2.0),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Log.into()), nz!(2)),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Abs.into()), nz!(1)),
            ]),
            vec![
                AstNode::Variable(X),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Sqrt.into()), nz!(1)),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Log2.into()), nz!(1)),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Abs.into()), nz!(1)),
            ]
        );
    }

    #[test]
    fn substitute_exp_equivalent() {
        let mut symbol_space = SubtreeCollection::new();
        let mut see = move |nodes: &[AstNode<f64, X, ()>]| {
            let mut ast = MathAst::from_nodes(nodes.iter().copied()).into_tree();
            let i = 2;
            while i < ast.len() {
                substitute_exp_eq(&mut ast, &mut symbol_space, i);
            }
            ast.postorder_iter().cloned().collect::<Vec<_>>()
        };
        assert_eq!(
            see(&[
                AstNode::Number(2.0),
                AstNode::Variable(X),
                AstNode::BinaryOp(BinaryOp::Pow),
            ]),
            vec![
                AstNode::Variable(X),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Exp2.into()), nz!(1)),
            ]
        );
        assert_eq!(
            see(&[
                AstNode::Number(10.0),
                AstNode::Variable(X),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Abs.into()), nz!(1)),
                AstNode::BinaryOp(BinaryOp::Pow),
            ]),
            vec![
                AstNode::Variable(X),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Abs.into()), nz!(1)),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Exp10.into()), nz!(1)),
            ]
        );
    }
}
