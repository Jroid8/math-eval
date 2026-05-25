use std::{
    fmt::{Debug, Display},
    num::NonZeroU8,
    ops::{Add, Div, Mul, Neg, Sub},
    str::FromStr,
};

use strum::{FromRepr, VariantArray};

use crate::{
    BinaryOp, FunctionIdentifier as FuncId, UnaryOp, VariableIdentifier as VarId, nz,
    postfix_tree::{PostfixTree, subtree_collection::SubtreeCollection},
    syntax::{AstNode, FunctionType},
    tokenizer::NumberRecognizer,
    trie::{NameTrie, TrieNode},
};

#[cfg(feature = "libm")]
pub mod libm_ext;
#[cfg(feature = "num-traits")]
pub mod num_traits;
pub mod std_float;
pub mod std_int;

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

pub trait ImmEvalStabilityGuard<N: Number>: Sized + Debug {
    fn from_number(num: N) -> Self;
    fn eval(self) -> N;
    fn apply_unary_op(self, opr: UnaryOp) -> Self;
    fn apply_binary_op(self, rhs: Self, opr: BinaryOp) -> Self;
    fn apply_func_single(self, id: BuiltinFunc<N::ExtraFuncId>, func: fn(N) -> N) -> Self;
    fn apply_func_dual(
        self,
        arg2: Self,
        id: BuiltinFunc<N::ExtraFuncId>,
        func: for<'a> fn(N, N::AsArg<'a>) -> N,
    ) -> Self;
    fn apply_func_flex(
        args: std::vec::Drain<'_, Self>,
        id: BuiltinFunc<N::ExtraFuncId>,
        func: fn(&[N]) -> N,
        arg_space: &mut Vec<N>,
    ) -> Self;
}

pub trait ExtraFuncId: FuncId {
    fn min_args(self) -> NonZeroU8;
    fn max_args(self) -> Option<NonZeroU8>;
    fn specialize_per_argc(func: &mut BuiltinFunc<Self>, argc: NonZeroU8);
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuiltinFunc<E: ExtraFuncId> {
    Basic(BasicFunc),
    Ext(E),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, FromRepr, VariantArray)]
#[repr(u8)]
pub enum BasicFunc {
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

impl BasicFunc {
    pub const fn get_method_ptr<N: Number>(self) -> BfPointer<N> {
        match self {
            BasicFunc::Exp2 => BfPointer::Single(N::exp2),
            BasicFunc::Exp10 => BfPointer::Single(N::exp10),
            BasicFunc::Log => BfPointer::Dual(N::log),
            BasicFunc::Log2 => BfPointer::Single(N::log2),
            BasicFunc::Log10 => BfPointer::Single(N::log10),
            BasicFunc::Sqrt => BfPointer::Single(N::sqrt),
            BasicFunc::Abs => BfPointer::Single(N::abs),
            BasicFunc::Sign => BfPointer::Single(N::sign),
            BasicFunc::Min => BfPointer::Flexible(N::min),
            BasicFunc::Max => BfPointer::Flexible(N::max),
        }
    }

    pub const fn min_args(self) -> NonZeroU8 {
        match self {
            BasicFunc::Exp2 => nz!(1),
            BasicFunc::Exp10 => nz!(1),
            BasicFunc::Log => nz!(2),
            BasicFunc::Log2 => nz!(1),
            BasicFunc::Log10 => nz!(1),
            BasicFunc::Sqrt => nz!(1),
            BasicFunc::Abs => nz!(1),
            BasicFunc::Sign => nz!(1),
            BasicFunc::Min => nz!(2),
            BasicFunc::Max => nz!(2),
        }
    }

    pub const fn max_args(self) -> Option<NonZeroU8> {
        match self {
            BasicFunc::Min | BasicFunc::Max => None,
            _ => Some(self.min_args()),
        }
    }

    pub fn is_flex(self) -> bool {
        self.max_args().is_none_or(|m| self.min_args() != m)
    }

    pub const fn name(self) -> &'static str {
        match self {
            BasicFunc::Exp2 => "exp2",
            BasicFunc::Exp10 => "exp10",
            BasicFunc::Log => "log",
            BasicFunc::Log2 => "log2",
            BasicFunc::Log10 => "log10",
            BasicFunc::Sqrt => "sqrt",
            BasicFunc::Abs => "abs",
            BasicFunc::Sign => "sign",
            BasicFunc::Min => "min",
            BasicFunc::Max => "max",
        }
    }

    pub const fn ext_id_offset(repr: u32) -> u32 {
        BasicFunc::VARIANTS.len() as u32 + repr
    }
}

impl Display for BasicFunc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

impl<E: ExtraFuncId> BuiltinFunc<E> {
    pub fn get_method_ptr<N>(self) -> BfPointer<N>
    where
        N: Number<ExtraFuncId = E>,
    {
        match self {
            Self::Basic(id) => id.get_method_ptr(),
            Self::Ext(id) => N::get_method_ptr(id),
        }
    }

    pub fn min_args(self) -> NonZeroU8 {
        match self {
            Self::Basic(id) => id.min_args(),
            Self::Ext(id) => id.min_args(),
        }
    }

    pub fn max_args(self) -> Option<NonZeroU8> {
        match self {
            Self::Basic(id) => id.max_args(),
            Self::Ext(id) => id.max_args(),
        }
    }

    pub fn is_flex(self) -> bool {
        self.max_args().is_none_or(|m| self.min_args() != m)
    }

    pub fn specialize_per_argc(&mut self, argc: NonZeroU8) {
        E::specialize_per_argc(self, argc)
    }
}

impl<E: ExtraFuncId> From<BasicFunc> for BuiltinFunc<E> {
    fn from(value: BasicFunc) -> Self {
        BuiltinFunc::Basic(value)
    }
}

impl<E: ExtraFuncId, C: FuncId> From<BasicFunc> for FunctionType<E, C> {
    fn from(value: BasicFunc) -> Self {
        FunctionType::Builtin(BuiltinFunc::Basic(value))
    }
}

impl<E: ExtraFuncId + Display> Display for BuiltinFunc<E> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BuiltinFunc::Basic(id) => f.write_str(id.name()),
            BuiltinFunc::Ext(id) => <E as Display>::fmt(id, f),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NoExtFunc {}

impl ExtraFuncId for NoExtFunc {
    fn min_args(self) -> NonZeroU8 {
        match self {}
    }

    fn max_args(self) -> Option<NonZeroU8> {
        match self {}
    }

    fn specialize_per_argc(_func: &mut BuiltinFunc<Self>, _argc: NonZeroU8) {
        ()
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
    type ConstsTrieType: NameTrie<Self>;
    type ExtraFuncId: ExtraFuncId;
    type BuiltinFuncsTrieType: NameTrie<BuiltinFunc<Self::ExtraFuncId>>;
    type ImmEvalStabilityGuard: ImmEvalStabilityGuard<Self>;

    const CONSTS_TRIE: Self::ConstsTrieType;
    const BUILTIN_FUNCS_TRIE: Self::BuiltinFuncsTrieType;
    const DO_DISPLACING_SIMPLIFICATION: bool;

    fn get_method_ptr(id: Self::ExtraFuncId) -> BfPointer<Self>;
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

    fn apply_func_single(self, _id: BuiltinFunc<N::ExtraFuncId>, func: fn(N) -> N) -> Self {
        NoStabilityGuard(func(self.0))
    }

    fn apply_func_dual(
        self,
        arg2: Self,
        _id: BuiltinFunc<N::ExtraFuncId>,
        func: for<'a> fn(N, <N as Number>::AsArg<'a>) -> N,
    ) -> Self {
        NoStabilityGuard(func(self.0, arg2.0.asarg()))
    }

    fn apply_func_flex(
        args: std::vec::Drain<'_, Self>,
        _id: BuiltinFunc<N::ExtraFuncId>,
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
            (AstNode::Number(base), _) if base.as_i8() == Some(2) => BasicFunc::Exp2,
            (AstNode::Number(base), _) if base.as_i8() == Some(10) => BasicFunc::Exp10,
            _ => return false,
        };
        symbol_space.extend_from_tree(&tree, param);
        symbol_space
            .push(AstNode::Function(
                FunctionType::Builtin(func.into()),
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
        AstNode::Function(FunctionType::Builtin(BuiltinFunc::Basic(BasicFunc::Log)), _)
    ) {
        let mut children = tree.children_iter(target);
        match children.next().unwrap() {
            (AstNode::Number(base), _) => {
                let func = if base.as_i8() == Some(2) {
                    BasicFunc::Log2
                } else if base.as_i8() == Some(10) {
                    BasicFunc::Log10
                } else {
                    return false;
                };
                let param = children.next().unwrap().1;
                symbol_space.extend_from_tree(&tree, param);
                symbol_space
                    .push(AstNode::Function(
                        FunctionType::Builtin(func.into()),
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

pub fn substitute_basic_funcs_eq<N: Number, V: VarId, F: FuncId>(
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

static BASIC_FUNCS_TRIE_NODES: [TrieNode; 37] = [
    TrieNode::Branch('a', 3),
    TrieNode::Branch('b', 2),
    TrieNode::Branch('s', 1),
    TrieNode::Leaf(BasicFunc::Abs as u32),
    TrieNode::Branch('e', 7),
    TrieNode::Branch('x', 6),
    TrieNode::Branch('p', 5),
    TrieNode::Branch('1', 2),
    TrieNode::Branch('0', 1),
    TrieNode::Leaf(BasicFunc::Exp10 as u32),
    TrieNode::Branch('2', 1),
    TrieNode::Leaf(BasicFunc::Exp2 as u32),
    TrieNode::Branch('l', 8),
    TrieNode::Branch('o', 7),
    TrieNode::Branch('g', 6),
    TrieNode::Leaf(BasicFunc::Log as u32),
    TrieNode::Branch('1', 2),
    TrieNode::Branch('0', 1),
    TrieNode::Leaf(BasicFunc::Log10 as u32),
    TrieNode::Branch('2', 1),
    TrieNode::Leaf(BasicFunc::Log2 as u32),
    TrieNode::Branch('m', 6),
    TrieNode::Branch('a', 2),
    TrieNode::Branch('x', 1),
    TrieNode::Leaf(BasicFunc::Max as u32),
    TrieNode::Branch('i', 2),
    TrieNode::Branch('n', 1),
    TrieNode::Leaf(BasicFunc::Min as u32),
    TrieNode::Branch('s', 8),
    TrieNode::Branch('i', 3),
    TrieNode::Branch('g', 2),
    TrieNode::Branch('n', 1),
    TrieNode::Leaf(BasicFunc::Sign as u32),
    TrieNode::Branch('q', 3),
    TrieNode::Branch('r', 2),
    TrieNode::Branch('t', 1),
    TrieNode::Leaf(BasicFunc::Sqrt as u32),
];

pub struct BasicFuncsTrie;

impl NameTrie<BuiltinFunc<NoExtFunc>> for BasicFuncsTrie {
    fn nodes(&self) -> &[crate::trie::TrieNode] {
        &BASIC_FUNCS_TRIE_NODES
    }

    fn leaf_to_value(&self, leaf: u32) -> BuiltinFunc<NoExtFunc> {
        BuiltinFunc::Basic(BasicFunc::from_repr(leaf as u8).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::syntax::MathAst;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct X;

    #[test]
    fn substitute_log_equivalent() {
        let mut symbol_space = SubtreeCollection::new();
        let mut sle = move |nodes: &[AstNode<f64, X, ()>]| {
            let mut ast = MathAst::from_nodes(nodes.iter().copied()).into_tree();
            let mut i = 2;
            while i < ast.len() {
                substitute_log_eq(&mut ast, &mut symbol_space, i);
                i += 1;
            }
            ast.postorder_iter().cloned().collect::<Vec<_>>()
        };
        assert_eq!(
            sle(&[
                AstNode::Variable(X),
                AstNode::Number(2.0),
                AstNode::Function(BasicFunc::Log.into(), nz!(2)),
            ]),
            vec![
                AstNode::Variable(X),
                AstNode::Function(BasicFunc::Log2.into(), nz!(1)),
            ]
        );
        assert_eq!(
            sle(&[
                AstNode::Variable(X),
                AstNode::Function(BasicFunc::Sqrt.into(), nz!(1)),
                AstNode::Number(2.0),
                AstNode::Function(BasicFunc::Log.into(), nz!(2)),
                AstNode::Function(BasicFunc::Abs.into(), nz!(1)),
            ]),
            vec![
                AstNode::Variable(X),
                AstNode::Function(BasicFunc::Sqrt.into(), nz!(1)),
                AstNode::Function(BasicFunc::Log2.into(), nz!(1)),
                AstNode::Function(BasicFunc::Abs.into(), nz!(1)),
            ]
        );
    }

    #[test]
    fn substitute_exp_equivalent() {
        let mut symbol_space = SubtreeCollection::new();
        let mut see = move |nodes: &[AstNode<f64, X, ()>]| {
            let mut ast = MathAst::from_nodes(nodes.iter().copied()).into_tree();
            let mut i = 2;
            while i < ast.len() {
                substitute_exp_eq(&mut ast, &mut symbol_space, i);
                i += 1;
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
                AstNode::Function(BasicFunc::Exp2.into(), nz!(1)),
            ]
        );
        assert_eq!(
            see(&[
                AstNode::Number(10.0),
                AstNode::Variable(X),
                AstNode::Function(BasicFunc::Abs.into(), nz!(1)),
                AstNode::BinaryOp(BinaryOp::Pow),
            ]),
            vec![
                AstNode::Variable(X),
                AstNode::Function(BasicFunc::Abs.into(), nz!(1)),
                AstNode::Function(BasicFunc::Exp10.into(), nz!(1)),
            ]
        );
    }
}
