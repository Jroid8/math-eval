use std::{fmt::Display, num::NonZeroU8};

use libm::Libm;
use strum::{EnumIter, FromRepr, VariantArray};

use crate::{
    BinaryOp, FunctionIdentifier as FuncId, UnaryOp, VariableIdentifier as VarId,
    number::{
        BasicFunc, BfPointer, BuiltinFunc, ExtraFuncId, ImmEvalStabilityGuard, Number,
        std_float::{
            StdFloatConstsNameTrie, StdFloatFunc, StdFloatFuncsSuperset, StdFloatLike,
            StdFloatRecognizer, substitute_expm1_eq, substitute_ln1p_eq,
        },
        substitute_exp_eq, substitute_log_eq,
    },
    nz,
    postfix_tree::{PostfixTree, subtree_collection::SubtreeCollection},
    syntax::{AstNode, FunctionType},
    trie::{NameTrie, TrieNode},
};

pub trait LibmExtended: StdFloatLike {
    fn ln10() -> Self;
    fn ln2() -> Self;

    fn erf(self) -> Self;
    fn erfc(self) -> Self;

    fn gamma(self) -> Self;
    fn lgamma(self) -> Self;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, FromRepr, EnumIter)]
#[repr(u8)]
pub enum LibmFunc {
    Erf,
    Erfc,
    Gamma,
    Lgamma,
}

impl LibmFunc {
    pub const fn name(self) -> &'static str {
        match self {
            LibmFunc::Erf => "erf",
            LibmFunc::Erfc => "erfc",
            LibmFunc::Gamma => "gamma",
            LibmFunc::Lgamma => "lgamma",
        }
    }
}

impl Display for LibmFunc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum StdLibmFunc {
    Std(StdFloatFunc),
    Libm(LibmFunc),
}

impl From<StdFloatFunc> for StdLibmFunc {
    fn from(value: StdFloatFunc) -> Self {
        Self::Std(value)
    }
}

impl From<StdFloatFunc> for BuiltinFunc<StdLibmFunc> {
    fn from(value: StdFloatFunc) -> Self {
        BuiltinFunc::Ext(StdLibmFunc::Std(value))
    }
}

impl From<BuiltinFunc<StdFloatFunc>> for BuiltinFunc<StdLibmFunc> {
    fn from(value: BuiltinFunc<StdFloatFunc>) -> Self {
        match value {
            BuiltinFunc::Basic(id) => BuiltinFunc::Basic(id),
            BuiltinFunc::Ext(id) => BuiltinFunc::Ext(StdLibmFunc::Std(id)),
        }
    }
}

impl<C: FuncId> From<StdFloatFunc> for FunctionType<StdLibmFunc, C> {
    fn from(value: StdFloatFunc) -> Self {
        FunctionType::Builtin(BuiltinFunc::Ext(StdLibmFunc::Std(value)))
    }
}

impl From<LibmFunc> for StdLibmFunc {
    fn from(value: LibmFunc) -> Self {
        Self::Libm(value)
    }
}

impl From<LibmFunc> for BuiltinFunc<StdLibmFunc> {
    fn from(value: LibmFunc) -> Self {
        BuiltinFunc::Ext(StdLibmFunc::Libm(value))
    }
}

impl<C: FuncId> From<LibmFunc> for FunctionType<StdLibmFunc, C> {
    fn from(value: LibmFunc) -> Self {
        FunctionType::Builtin(BuiltinFunc::Ext(StdLibmFunc::Libm(value)))
    }
}

impl Display for StdLibmFunc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            StdLibmFunc::Std(func) => func.name(),
            StdLibmFunc::Libm(func) => func.name(),
        })
    }
}

impl ExtraFuncId for StdLibmFunc {
    fn min_args(self) -> NonZeroU8 {
        match self {
            StdLibmFunc::Std(id) => id.min_args(),
            StdLibmFunc::Libm(_) => nz!(1),
        }
    }

    fn max_args(self) -> Option<NonZeroU8> {
        match self {
            StdLibmFunc::Std(id) => id.max_args(),
            StdLibmFunc::Libm(_) => Some(nz!(1)),
        }
    }

    fn specialize_per_argc(func: &mut super::BuiltinFunc<Self>, argc: NonZeroU8) {
        if *func == BasicFunc::Log.into() && argc.get() == 1 {
            *func = StdFloatFunc::Ln.into()
        }
    }
}

impl StdFloatFuncsSuperset for StdLibmFunc {
    fn into_std(self) -> Option<StdFloatFunc> {
        match self {
            StdLibmFunc::Std(func) => Some(func),
            StdLibmFunc::Libm(_) => None,
        }
    }
    fn from_std(id: StdFloatFunc) -> Self {
        Self::Std(id)
    }
}

pub trait LibmFuncsSuperset: StdFloatFuncsSuperset {
    fn into_libm(self) -> Option<LibmFunc>;
    fn from_libm(id: LibmFunc) -> Self;
}

impl LibmFuncsSuperset for StdLibmFunc {
    fn into_libm(self) -> Option<LibmFunc> {
        match self {
            Self::Std(_) => None,
            Self::Libm(func) => Some(func),
        }
    }
    fn from_libm(id: LibmFunc) -> Self {
        Self::Libm(id)
    }
}

pub static STD_LIBM_FUNCS_TRIE_NODES: [TrieNode; 167] = [
    TrieNode::Branch('a', 51),
    TrieNode::Branch('b', 2),
    TrieNode::Branch('s', 1),
    TrieNode::Leaf(BasicFunc::Abs as u32),
    TrieNode::Branch('c', 9),
    TrieNode::Branch('o', 8),
    TrieNode::Branch('s', 3),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Acos as u32)),
    TrieNode::Branch('h', 1),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Acosh as u32)),
    TrieNode::Branch('t', 3),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Acot as u32)),
    TrieNode::Branch('h', 1),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Acoth as u32)),
    TrieNode::Branch('r', 25),
    TrieNode::Branch('c', 24),
    TrieNode::Branch('c', 9),
    TrieNode::Branch('o', 8),
    TrieNode::Branch('s', 3),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Acos as u32)),
    TrieNode::Branch('h', 1),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Acosh as u32)),
    TrieNode::Branch('t', 3),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Acot as u32)),
    TrieNode::Branch('h', 1),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Acoth as u32)),
    TrieNode::Branch('s', 5),
    TrieNode::Branch('i', 4),
    TrieNode::Branch('n', 3),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Asin as u32)),
    TrieNode::Branch('h', 1),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Asinh as u32)),
    TrieNode::Branch('t', 7),
    TrieNode::Branch('a', 6),
    TrieNode::Branch('n', 5),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Atan as u32)),
    TrieNode::Branch('2', 1),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Atan2 as u32)),
    TrieNode::Branch('h', 1),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Atanh as u32)),
    TrieNode::Branch('s', 5),
    TrieNode::Branch('i', 4),
    TrieNode::Branch('n', 3),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Asin as u32)),
    TrieNode::Branch('h', 1),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Asinh as u32)),
    TrieNode::Branch('t', 7),
    TrieNode::Branch('a', 6),
    TrieNode::Branch('n', 5),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Atan as u32)),
    TrieNode::Branch('2', 1),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Atan2 as u32)),
    TrieNode::Branch('h', 1),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Atanh as u32)),
    TrieNode::Branch('c', 17),
    TrieNode::Branch('b', 3),
    TrieNode::Branch('r', 2),
    TrieNode::Branch('t', 1),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Cbrt as u32)),
    TrieNode::Branch('e', 3),
    TrieNode::Branch('i', 2),
    TrieNode::Branch('l', 1),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Ceil as u32)),
    TrieNode::Branch('o', 8),
    TrieNode::Branch('s', 3),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Cos as u32)),
    TrieNode::Branch('h', 1),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Cosh as u32)),
    TrieNode::Branch('t', 3),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Cot as u32)),
    TrieNode::Branch('h', 1),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Coth as u32)),
    TrieNode::Branch('e', 16),
    TrieNode::Branch('r', 4),
    TrieNode::Branch('f', 3),
    TrieNode::Leaf(StdFloatFunc::ext_id_offset(LibmFunc::Erf as u32)),
    TrieNode::Branch('c', 1),
    TrieNode::Leaf(StdFloatFunc::ext_id_offset(LibmFunc::Erfc as u32)),
    TrieNode::Branch('x', 10),
    TrieNode::Branch('p', 9),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Exp as u32)),
    TrieNode::Branch('1', 2),
    TrieNode::Branch('0', 1),
    TrieNode::Leaf(BasicFunc::Exp10 as u32),
    TrieNode::Branch('2', 1),
    TrieNode::Leaf(BasicFunc::Exp2 as u32),
    TrieNode::Branch('m', 2),
    TrieNode::Branch('1', 1),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Expm1 as u32)),
    TrieNode::Branch('f', 11),
    TrieNode::Branch('l', 4),
    TrieNode::Branch('o', 3),
    TrieNode::Branch('o', 2),
    TrieNode::Branch('r', 1),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Floor as u32)),
    TrieNode::Branch('r', 5),
    TrieNode::Branch('a', 4),
    TrieNode::Branch('c', 3),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Frac as u32)),
    TrieNode::Branch('t', 1),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Frac as u32)),
    TrieNode::Branch('g', 5),
    TrieNode::Branch('a', 4),
    TrieNode::Branch('m', 3),
    TrieNode::Branch('m', 2),
    TrieNode::Branch('a', 1),
    TrieNode::Leaf(StdFloatFunc::ext_id_offset(LibmFunc::Gamma as u32)),
    TrieNode::Branch('l', 22),
    TrieNode::Branch('b', 1),
    TrieNode::Leaf(BasicFunc::Log2 as u32),
    TrieNode::Branch('g', 6),
    TrieNode::Leaf(BasicFunc::Log2 as u32),
    TrieNode::Branch('a', 4),
    TrieNode::Branch('m', 3),
    TrieNode::Branch('m', 2),
    TrieNode::Branch('a', 1),
    TrieNode::Leaf(StdFloatFunc::ext_id_offset(LibmFunc::Lgamma as u32)),
    TrieNode::Branch('n', 4),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Ln as u32)),
    TrieNode::Branch('1', 2),
    TrieNode::Branch('p', 1),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Ln1p as u32)),
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
    TrieNode::Branch('r', 5),
    TrieNode::Branch('o', 4),
    TrieNode::Branch('u', 3),
    TrieNode::Branch('n', 2),
    TrieNode::Branch('d', 1),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Round as u32)),
    TrieNode::Branch('s', 12),
    TrieNode::Branch('i', 7),
    TrieNode::Branch('g', 2),
    TrieNode::Branch('n', 1),
    TrieNode::Leaf(BasicFunc::Sign as u32),
    TrieNode::Branch('n', 3),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Sin as u32)),
    TrieNode::Branch('h', 1),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Sinh as u32)),
    TrieNode::Branch('q', 3),
    TrieNode::Branch('r', 2),
    TrieNode::Branch('t', 1),
    TrieNode::Leaf(BasicFunc::Sqrt as u32),
    TrieNode::Branch('t', 10),
    TrieNode::Branch('a', 4),
    TrieNode::Branch('n', 3),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Tan as u32)),
    TrieNode::Branch('h', 1),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Tanh as u32)),
    TrieNode::Branch('r', 4),
    TrieNode::Branch('u', 3),
    TrieNode::Branch('n', 2),
    TrieNode::Branch('c', 1),
    TrieNode::Leaf(BasicFunc::ext_id_offset(StdFloatFunc::Trunc as u32)),
];

pub struct StdLibmFuncsTrie;

impl NameTrie<BuiltinFunc<StdLibmFunc>> for StdLibmFuncsTrie {
    fn nodes(&self) -> &[TrieNode] {
        &STD_LIBM_FUNCS_TRIE_NODES
    }

    fn leaf_to_value(&self, leaf: u32) -> BuiltinFunc<StdLibmFunc> {
        let mut leaf = leaf as u8;
        if let Some(id) = BasicFunc::from_repr(leaf) {
            BuiltinFunc::Basic(id)
        } else {
            leaf -= BasicFunc::VARIANTS.len() as u8;
            BuiltinFunc::Ext(if let Some(id) = StdFloatFunc::from_repr(leaf) {
                StdLibmFunc::Std(id)
            } else {
                leaf -= StdFloatFunc::VARIANTS.len() as u8;
                StdLibmFunc::Libm(LibmFunc::from_repr(leaf).unwrap())
            })
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StdLibmStabilityGuard<N> {
    Number(N),
    Exp(N),
    OnePlus(N),
    Gamma(N),
    Erf(N),
}

impl<N> ImmEvalStabilityGuard<N> for StdLibmStabilityGuard<N>
where
    N: LibmExtended<ExtraFuncId = StdLibmFunc>,
{
    fn from_number(num: N) -> Self {
        Self::Number(num)
    }

    fn eval(self) -> N {
        match self {
            Self::Number(num) => num,
            Self::Exp(num) => num.exp(),
            Self::OnePlus(num) => num + N::from_i8(1),
            Self::Gamma(num) => num.gamma(),
            Self::Erf(num) => num.erf(),
        }
    }

    fn apply_unary_op(self, opr: UnaryOp) -> Self {
        if opr == UnaryOp::Fac {
            return Self::Gamma(self.eval() + N::from_i8(1));
        }
        Self::Number(opr.eval(self.eval()))
    }

    fn apply_binary_op(self, rhs: Self, opr: BinaryOp) -> Self {
        match (self, rhs) {
            (Self::Exp(x), Self::Number(y)) if opr == BinaryOp::Sub && y == N::from_i8(1) => {
                Self::Number(x.expm1())
            }
            (Self::Number(x), y) | (y, Self::Number(x))
                if opr == BinaryOp::Add && x == N::from_i8(1) =>
            {
                Self::OnePlus(y.eval())
            }
            (Self::Number(base), exp) if opr == BinaryOp::Pow => {
                let exp = exp.eval();
                if base.is_ten() {
                    return Self::Number(exp.exp10());
                }
                Self::Number(if base.is_two() {
                    exp.exp2()
                } else {
                    base.pow(exp.asarg())
                })
            }
            (Self::Number(x), Self::Erf(y)) if opr == BinaryOp::Sub && x == N::from_i8(1) => {
                Self::Number(y.erfc())
            }
            (lhs, rhs) => Self::Number(opr.eval(lhs.eval(), rhs.eval())),
        }
    }

    fn apply_func_single(self, id: BuiltinFunc<StdLibmFunc>, func: fn(N) -> N) -> Self {
        match (id, self) {
            (BuiltinFunc::Ext(StdLibmFunc::Std(StdFloatFunc::Exp)), _) => Self::Exp(self.eval()),
            (BuiltinFunc::Ext(StdLibmFunc::Std(StdFloatFunc::Ln)), Self::OnePlus(num)) => {
                Self::Number(num.ln1p())
            }
            (BuiltinFunc::Ext(StdLibmFunc::Std(StdFloatFunc::Ln)), Self::Gamma(num)) => {
                Self::Number(num.lgamma())
            }
            (BuiltinFunc::Ext(StdLibmFunc::Std(StdFloatFunc::Ln)), Self::Exp(num)) => {
                Self::Number(num)
            }
            (BuiltinFunc::Basic(BasicFunc::Log10), Self::Gamma(num)) => {
                Self::Number(num.lgamma() / N::ln10())
            }
            (BuiltinFunc::Basic(BasicFunc::Log2), Self::Gamma(num)) => {
                Self::Number(num.lgamma() / N::ln2())
            }
            (BuiltinFunc::Ext(StdLibmFunc::Libm(LibmFunc::Gamma)), _) => Self::Gamma(self.eval()),
            (BuiltinFunc::Ext(StdLibmFunc::Libm(LibmFunc::Erf)), _) => Self::Erf(self.eval()),
            _ => Self::Number(func(self.eval())),
        }
    }

    fn apply_func_dual(
        self,
        arg2: Self,
        id: BuiltinFunc<StdLibmFunc>,
        func: fn(N, N) -> N,
    ) -> Self {
        Self::Number(if id == BasicFunc::Log.into() {
            let base = arg2.eval();
            if let Self::Gamma(num) = self {
                return Self::Number(num.lgamma() / base.ln());
            }
            let num = self.eval();
            if base.is_ten() {
                num.log10()
            } else if base.is_two() {
                num.log2()
            } else {
                num.log(base)
            }
        } else {
            func(self.eval(), arg2.eval())
        })
    }

    fn apply_func_flex(
        args: std::vec::Drain<'_, Self>,
        id: BuiltinFunc<StdLibmFunc>,
        _func: fn(&[N]) -> N,
        arg_space: &mut Vec<N>,
    ) -> Self {
        arg_space.extend(args.map(Self::eval));
        Self::Number(match id {
            BuiltinFunc::Basic(BasicFunc::Min) => N::min(&arg_space),
            BuiltinFunc::Basic(BasicFunc::Max) => N::max(&arg_space),
            _ => unreachable!(),
        })
    }
}

pub fn substitute_lgamma_eq<N, B, V: VarId, F: FuncId>(
    tree: &mut PostfixTree<AstNode<N, V, F>>,
    symbol_space: &mut SubtreeCollection<AstNode<N, V, F>>,
    target: usize,
) -> bool
where
    N: LibmExtended<ExtraFuncId = B>,
    B: LibmFuncsSuperset,
{
    #[derive(Debug, PartialEq, Eq, Clone, Copy)]
    enum LogBase {
        E,
        Ten,
        Two,
        Value(usize),
    }

    let base = if let AstNode::Function(FunctionType::Builtin(func), _) = tree[target] {
        match func {
            BuiltinFunc::Basic(BasicFunc::Log10) => LogBase::Ten,
            BuiltinFunc::Basic(BasicFunc::Log2) => LogBase::Two,
            BuiltinFunc::Basic(BasicFunc::Log) => {
                LogBase::Value(tree.nth_child(target, 1).unwrap())
            }
            BuiltinFunc::Ext(id) if id.into_std() == Some(StdFloatFunc::Ln) => LogBase::E,
            _ => return false,
        }
    } else {
        return false;
    };
    let child = tree.nth_child(target, 0).unwrap();
    match tree[child] {
        AstNode::Function(FunctionType::Builtin(BuiltinFunc::Ext(func)), _)
            if func.into_libm() == Some(LibmFunc::Gamma) =>
        {
            symbol_space.extend_from_tree(tree, tree.nth_child(child, 0).unwrap());
        }
        AstNode::UnaryOp(UnaryOp::Fac) => {
            symbol_space.extend_from_tree(tree, tree.nth_child(child, 0).unwrap());
            symbol_space.push(AstNode::Number(N::from_i8(1))).unwrap();
            symbol_space.push(AstNode::BinaryOp(BinaryOp::Add)).unwrap();
        }
        _ => return false,
    };
    symbol_space
        .push(AstNode::Function(
            FunctionType::Builtin(BuiltinFunc::Ext(B::from_libm(LibmFunc::Lgamma))),
            nz!(1),
        ))
        .unwrap();
    match base {
        LogBase::E => (),
        LogBase::Ten => symbol_space.push(AstNode::Number(N::ln10())).unwrap(),
        LogBase::Two => symbol_space.push(AstNode::Number(N::ln2())).unwrap(),
        LogBase::Value(idx) => {
            symbol_space.extend_from_tree(tree, idx);
            symbol_space
                .push(AstNode::Function(
                    FunctionType::Builtin(BuiltinFunc::Ext(B::from_std(StdFloatFunc::Ln))),
                    nz!(1),
                ))
                .unwrap();
        }
    }
    if base != LogBase::E {
        symbol_space.push(AstNode::BinaryOp(BinaryOp::Div)).unwrap();
    }
    let sc_head = symbol_space.len() - 1;
    tree.replace_from_sc_move(target, symbol_space, sc_head);
    true
}

pub fn substitute_erfc_eq<N, B, V: VarId, F: FuncId>(
    tree: &mut PostfixTree<AstNode<N, V, F>>,
    symbol_space: &mut SubtreeCollection<AstNode<N, V, F>>,
    target: usize,
) -> bool
where
    N: LibmExtended<ExtraFuncId = B>,
    B: LibmFuncsSuperset,
{
    if matches!(tree[target], AstNode::BinaryOp(BinaryOp::Sub)) {
        let mut children = tree.children_iter(target);
        if let (
            (AstNode::Function(FunctionType::Builtin(BuiltinFunc::Ext(func)), _), idx),
            (AstNode::Number(lhs), _),
        ) = (children.next().unwrap(), children.next().unwrap())
            && func.into_libm() == Some(LibmFunc::Erf)
            && *lhs == N::from_i8(1)
        {
            let param = tree.children_iter(idx).next().unwrap().1;
            symbol_space.extend_from_tree(&tree, param);
            symbol_space
                .push(AstNode::Function(
                    FunctionType::Builtin(BuiltinFunc::Ext(B::from_libm(LibmFunc::Erfc))),
                    nz!(1),
                ))
                .unwrap();
            let sc_head = symbol_space.len() - 1;
            tree.replace_from_sc_move(target, symbol_space, sc_head);
            true
        } else {
            false
        }
    } else {
        false
    }
}

pub fn substitute_libm_ext_spec_funcs_eq<N, B, V: VarId, F: FuncId>(
    tree: &mut PostfixTree<AstNode<N, V, F>>,
) where
    N: LibmExtended<ExtraFuncId = B>,
    B: LibmFuncsSuperset,
{
    let mut symbol_space: SubtreeCollection<AstNode<N, V, F>> =
        SubtreeCollection::from_alloc(Vec::with_capacity(0));
    let mut idx = 2;
    while idx < tree.len() {
        for subs in [
            substitute_expm1_eq,
            substitute_ln1p_eq,
            substitute_log_eq,
            substitute_erfc_eq,
            substitute_lgamma_eq,
            substitute_exp_eq,
        ] {
            if subs(tree, &mut symbol_space, idx) {
                break;
            }
        }
        idx += 1;
    }
}

macro_rules! impl_number_for_std_float {
    ($t: ident) => {
        impl Number for $t {
            type AsArg<'a> = Self;
            type Recognizer = StdFloatRecognizer;
            type ConstsTrieType = StdFloatConstsNameTrie<Self>;
            type ExtraFuncId = StdLibmFunc;
            type BuiltinFuncsTrieType = StdLibmFuncsTrie;
            type ImmEvalStabilityGuard = StdLibmStabilityGuard<Self>;

            const CONSTS_TRIE: StdFloatConstsNameTrie<Self> = StdFloatConstsNameTrie {
                pi: std::$t::consts::PI,
                e: std::$t::consts::E,
                tau: std::$t::consts::TAU,
            };
            const BUILTIN_FUNCS_TRIE: Self::BuiltinFuncsTrieType = StdLibmFuncsTrie;
            const DO_DISPLACING_SIMPLIFICATION: bool = true;

            fn from_i8(value: i8) -> Self {
                value.into()
            }

            fn as_i8(&self) -> Option<i8> {
                if self.is_nan() || self.is_infinite() || *self != self.trunc() {
                    return None;
                }
                if *self < i8::MIN as $t || *self > i8::MAX as $t {
                    return None;
                }
                Some(*self as i8)
            }

            fn get_method_ptr(id: StdLibmFunc) -> BfPointer<$t> {
                match id {
                    StdLibmFunc::Std(StdFloatFunc::Sin) => BfPointer::Single(Self::sin),
                    StdLibmFunc::Std(StdFloatFunc::Cos) => BfPointer::Single(Self::cos),
                    StdLibmFunc::Std(StdFloatFunc::Tan) => BfPointer::Single(Self::tan),
                    StdLibmFunc::Std(StdFloatFunc::Cot) => BfPointer::Single(|x| {
                        let (sin, cos) = x.sin_cos();
                        cos / sin
                    }),
                    StdLibmFunc::Std(StdFloatFunc::Sinh) => BfPointer::Single(Self::sinh),
                    StdLibmFunc::Std(StdFloatFunc::Cosh) => BfPointer::Single(Self::cosh),
                    StdLibmFunc::Std(StdFloatFunc::Tanh) => BfPointer::Single(Self::tanh),
                    StdLibmFunc::Std(StdFloatFunc::Coth) => {
                        BfPointer::Single(|x| (-x).atan() + std::$t::consts::FRAC_PI_2)
                    }
                    StdLibmFunc::Std(StdFloatFunc::Asin) => BfPointer::Single(Self::asin),
                    StdLibmFunc::Std(StdFloatFunc::Acos) => BfPointer::Single(Self::acos),
                    StdLibmFunc::Std(StdFloatFunc::Atan) => BfPointer::Single(Self::atan),
                    StdLibmFunc::Std(StdFloatFunc::Acot) => {
                        BfPointer::Single(|x| x.cosh() / x.sinh())
                    }
                    StdLibmFunc::Std(StdFloatFunc::Atan2) => BfPointer::<Self>::Dual($t::atan2),
                    StdLibmFunc::Std(StdFloatFunc::Asinh) => BfPointer::Single(Self::asinh),
                    StdLibmFunc::Std(StdFloatFunc::Acosh) => BfPointer::Single(Self::acosh),
                    StdLibmFunc::Std(StdFloatFunc::Atanh) => BfPointer::Single(Self::atanh),
                    StdLibmFunc::Std(StdFloatFunc::Acoth) => {
                        BfPointer::Single(|x| x.recip().atanh())
                    }
                    StdLibmFunc::Std(StdFloatFunc::Ln) => BfPointer::Single(Self::ln),
                    StdLibmFunc::Std(StdFloatFunc::Ln1p) => BfPointer::Single(Self::ln_1p),
                    StdLibmFunc::Std(StdFloatFunc::Exp) => BfPointer::Single(Self::exp),
                    StdLibmFunc::Std(StdFloatFunc::Expm1) => BfPointer::Single(Self::exp_m1),
                    StdLibmFunc::Std(StdFloatFunc::Floor) => BfPointer::Single(Self::floor),
                    StdLibmFunc::Std(StdFloatFunc::Ceil) => BfPointer::Single(Self::ceil),
                    StdLibmFunc::Std(StdFloatFunc::Round) => BfPointer::Single(Self::round),
                    StdLibmFunc::Std(StdFloatFunc::Trunc) => BfPointer::Single(Self::trunc),
                    StdLibmFunc::Std(StdFloatFunc::Frac) => BfPointer::Single(Self::fract),
                    StdLibmFunc::Std(StdFloatFunc::Cbrt) => BfPointer::Single(Self::cbrt),
                    StdLibmFunc::Libm(LibmFunc::Erf) => BfPointer::Single(Libm::<Self>::erf),
                    StdLibmFunc::Libm(LibmFunc::Erfc) => BfPointer::Single(Libm::<Self>::erfc),
                    StdLibmFunc::Libm(LibmFunc::Gamma) => BfPointer::Single(Libm::<Self>::tgamma),
                    StdLibmFunc::Libm(LibmFunc::Lgamma) => BfPointer::Single(Libm::<Self>::lgamma),
                }
            }

            fn substitute_spec_funcs_equivalents<V: VarId, F: FuncId>(
                tree: &mut PostfixTree<AstNode<Self, V, F>>,
            ) {
                substitute_libm_ext_spec_funcs_eq(tree);
            }

            fn asarg(&self) -> Self::AsArg<'_> {
                *self
            }

            fn pow(self, rhs: Self) -> Self {
                self.powf(rhs)
            }

            fn exp2(self) -> Self {
                self.exp2()
            }

            fn exp10(self) -> Self {
                Libm::<Self>::exp10(self)
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

            fn sqrt(self) -> Self {
                self.sqrt()
            }

            fn modulo(self, rhs: Self) -> Self {
                self.rem_euclid(rhs)
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

            fn max(values: &[Self]) -> Self {
                values
                    .iter()
                    .copied()
                    .max_by(|x, y| x.total_cmp(y))
                    .unwrap()
            }

            fn min(values: &[Self]) -> Self {
                values
                    .iter()
                    .copied()
                    .min_by(|x, y| x.total_cmp(y))
                    .unwrap()
            }

            fn factorial(self) -> Self {
                libm::Libm::<Self>::tgamma(self + 1.0)
            }

            fn double_factorial(self) -> Self {
                if self.is_infinite() || self < 0.0 || self.is_nan() {
                    return Self::NAN;
                }
                if self >= 301.0 {
                    return Self::INFINITY;
                }
                let mut result = 1.0;
                let mut k = self as u32;
                while k > 1 {
                    result *= k as $t;
                    k -= 2;
                }
                result
            }
        }
    };
}

impl_number_for_std_float!(f64);
impl_number_for_std_float!(f32);

macro_rules! impl_libmext_for_std_float {
    ($t: ident) => {
        impl LibmExtended for $t {
            fn ln10() -> Self {
                std::$t::consts::LN_10
            }

            fn ln2() -> Self {
                std::$t::consts::LN_2
            }

            fn erf(self) -> Self {
                Libm::<Self>::erf(self)
            }

            fn erfc(self) -> Self {
                Libm::<Self>::erfc(self)
            }

            fn gamma(self) -> Self {
                Libm::<Self>::tgamma(self)
            }

            fn lgamma(self) -> Self {
                Libm::<Self>::lgamma(self)
            }
        }
    };
}

impl_libmext_for_std_float!(f64);
impl_libmext_for_std_float!(f32);

#[cfg(test)]
mod tests {
    use strum::{FromRepr, IntoEnumIterator};

    use super::StdLibmStabilityGuard as Slsg;
    use super::*;
    use crate::syntax::MathAst;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, FromRepr)]
    #[repr(u8)]
    enum TestVar {
        X,
        Y,
    }

    #[test]
    fn stability_guard() {
        assert_eq!(
            Slsg::Number(0.1).apply_func_single(LibmFunc::Erf.into(), |_| panic!()),
            Slsg::Erf(0.1),
        );
        assert_eq!(
            Slsg::Number(1.0).apply_binary_op(Slsg::Erf(0.1), BinaryOp::Sub),
            Slsg::Number(libm::erfc(0.1)),
        );
        assert_eq!(
            Slsg::Number(2.0).apply_binary_op(Slsg::Erf(0.1), BinaryOp::Sub),
            Slsg::Number(2.0 - libm::erf(0.1)),
        );
        assert_eq!(
            Slsg::Number(3.0).apply_func_single(StdFloatFunc::Exp.into(), |_| panic!()),
            Slsg::Exp(3.0),
        );
        assert_eq!(
            Slsg::Exp(3.0).apply_binary_op(Slsg::Number(1.0), BinaryOp::Sub),
            Slsg::Number(3f64.exp_m1()),
        );
        assert_eq!(
            Slsg::Exp(3.0).apply_binary_op(Slsg::Number(5.0), BinaryOp::Sub),
            Slsg::Number(3f64.exp() - 5.0),
        );
        assert_eq!(
            Slsg::Number(1.0).apply_unary_op(UnaryOp::Fac),
            Slsg::Gamma(2.0)
        );
        assert_eq!(
            Slsg::Number(1.0).apply_func_single(LibmFunc::Gamma.into(), |_| panic!()),
            Slsg::Gamma(1.0)
        );
        assert_eq!(
            Slsg::Gamma(1000.0).apply_func_single(StdFloatFunc::Ln.into(), |_| panic!()),
            Slsg::Number(libm::lgamma(1000.0))
        );
        assert_eq!(
            Slsg::Number(8.0).apply_func_single(StdFloatFunc::Ln.into(), f64::ln),
            Slsg::Number(8f64.ln())
        );
        assert_eq!(
            Slsg::Gamma(1000.0).apply_func_single(BasicFunc::Log2.into(), |_| panic!()),
            Slsg::Number(libm::lgamma(1000.0) / 2f64.ln())
        );
        assert_eq!(
            Slsg::Gamma(1000.0).apply_func_single(BasicFunc::Log10.into(), |_| panic!()),
            Slsg::Number(libm::lgamma(1000.0) / 10f64.ln())
        );
        assert_eq!(
            Slsg::Gamma(1000f64).apply_func_dual(
                Slsg::Number(3f64),
                BasicFunc::Log.into(),
                |_, _| panic!()
            ),
            Slsg::Number(libm::lgamma(1000.0) / 3f64.ln())
        );
        assert_eq!(
            Slsg::Gamma(10.0).apply_func_single(StdFloatFunc::Sin.into(), f64::sin),
            Slsg::Number(libm::tgamma(10.0).sin())
        );
        assert_eq!(
            Slsg::Number(0.0).apply_binary_op(Slsg::Number(1.0), BinaryOp::Add),
            Slsg::OnePlus(0.0)
        );
        assert_eq!(
            Slsg::OnePlus(8.0).apply_func_single(StdFloatFunc::Ln.into(), |_| panic!()),
            Slsg::Number(8f64.ln_1p())
        );
        assert_eq!(
            Slsg::OnePlus(0.0).apply_binary_op(Slsg::Number(1.8), BinaryOp::Add),
            Slsg::Number(2.8)
        );
        assert_eq!(
            Slsg::OnePlus(8.11).apply_binary_op(Slsg::Number(1.0), BinaryOp::Add),
            Slsg::OnePlus(9.11)
        );
        assert_eq!(
            Slsg::Number(1.0).apply_binary_op(Slsg::Number(9.3), BinaryOp::Add),
            Slsg::OnePlus(9.3)
        );
        assert_eq!(
            Slsg::Number(4.2).apply_binary_op(Slsg::Number(1.0), BinaryOp::Add),
            Slsg::OnePlus(4.2)
        );
        assert_eq!(
            Slsg::Number(10.0).apply_binary_op(Slsg::Number(23.0), BinaryOp::Pow),
            Slsg::Number(1e23)
        );
        assert_eq!(
            Slsg::Number(2.0).apply_binary_op(Slsg::Number(100.0), BinaryOp::Pow),
            Slsg::Number(100f64.exp2())
        );
        assert_eq!(
            Slsg::Number(977f64).apply_func_dual(
                Slsg::Number(10f64),
                BasicFunc::Log.into(),
                |_, _| panic!()
            ),
            Slsg::Number(977f64.log10())
        );
        assert_eq!(
            Slsg::Number(888f64).apply_func_dual(
                Slsg::Number(2f64),
                BasicFunc::Log.into(),
                |_, _| panic!()
            ),
            Slsg::Number(888f64.log2())
        );
        assert_eq!(
            Slsg::Number(1.0).apply_unary_op(UnaryOp::Neg),
            Slsg::Number(-1.0)
        );
        assert_eq!(
            Slsg::Number(2.0).apply_binary_op(Slsg::Number(9.3), BinaryOp::Add),
            Slsg::Number(11.3)
        );
    }

    #[test]
    fn substitute_spec_funcs() {
        let usf = |nodes: &[AstNode<f64, TestVar, ()>]| {
            let mut ast = MathAst::from_nodes(nodes.iter().copied());
            ast.substitute_spec_funcs_equivalents();
            ast.into_tree()
                .postorder_iter()
                .cloned()
                .collect::<Vec<_>>()
        };
        assert_eq!(
            usf(&[
                AstNode::Number(1.0),
                AstNode::Variable(TestVar::X),
                AstNode::Function(FunctionType::Builtin(LibmFunc::Erf.into()), nz!(1)),
                AstNode::BinaryOp(BinaryOp::Sub),
            ]),
            vec![
                AstNode::Variable(TestVar::X),
                AstNode::Function(FunctionType::Builtin(LibmFunc::Erfc.into()), nz!(1)),
            ]
        );
        assert_eq!(
            usf(&[
                AstNode::Variable(TestVar::X),
                AstNode::Function(FunctionType::Builtin(LibmFunc::Gamma.into()), nz!(1)),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Ln.into()), nz!(1)),
                AstNode::Variable(TestVar::Y),
                AstNode::Function(FunctionType::Builtin(LibmFunc::Gamma.into()), nz!(1)),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Ln.into()), nz!(1)),
                AstNode::BinaryOp(BinaryOp::Sub),
                AstNode::Variable(TestVar::X),
                AstNode::Variable(TestVar::Y),
                AstNode::BinaryOp(BinaryOp::Sub),
                AstNode::Function(FunctionType::Builtin(LibmFunc::Gamma.into()), nz!(1)),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Ln.into()), nz!(1)),
                AstNode::BinaryOp(BinaryOp::Sub),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Exp.into()), nz!(1))
            ]),
            vec![
                AstNode::Variable(TestVar::X),
                AstNode::Function(FunctionType::Builtin(LibmFunc::Lgamma.into()), nz!(1)),
                AstNode::Variable(TestVar::Y),
                AstNode::Function(FunctionType::Builtin(LibmFunc::Lgamma.into()), nz!(1)),
                AstNode::BinaryOp(BinaryOp::Sub),
                AstNode::Variable(TestVar::X),
                AstNode::Variable(TestVar::Y),
                AstNode::BinaryOp(BinaryOp::Sub),
                AstNode::Function(FunctionType::Builtin(LibmFunc::Lgamma.into()), nz!(1)),
                AstNode::BinaryOp(BinaryOp::Sub),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Exp.into()), nz!(1))
            ]
        );
        assert_eq!(
            usf(&[
                AstNode::Variable(TestVar::X),
                AstNode::UnaryOp(UnaryOp::Fac),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Ln.into()), nz!(1)),
            ]),
            vec![
                AstNode::Variable(TestVar::X),
                AstNode::Number(1.0),
                AstNode::BinaryOp(BinaryOp::Add),
                AstNode::Function(FunctionType::Builtin(LibmFunc::Lgamma.into()), nz!(1)),
            ]
        );
        assert_eq!(
            usf(&[
                AstNode::Variable(TestVar::X),
                AstNode::Number(2.0),
                AstNode::Function(FunctionType::Builtin(BasicFunc::Log.into()), nz!(2)),
            ]),
            vec![
                AstNode::Variable(TestVar::X),
                AstNode::Function(FunctionType::Builtin(BasicFunc::Log2.into()), nz!(1)),
            ]
        );
        assert_eq!(
            usf(&[
                AstNode::Variable(TestVar::X),
                AstNode::Function(FunctionType::Builtin(BasicFunc::Sqrt.into()), nz!(1)),
                AstNode::Number(2.0),
                AstNode::Function(FunctionType::Builtin(BasicFunc::Log.into()), nz!(2)),
                AstNode::Function(FunctionType::Builtin(BasicFunc::Abs.into()), nz!(1)),
            ]),
            vec![
                AstNode::Variable(TestVar::X),
                AstNode::Function(FunctionType::Builtin(BasicFunc::Sqrt.into()), nz!(1)),
                AstNode::Function(FunctionType::Builtin(BasicFunc::Log2.into()), nz!(1)),
                AstNode::Function(FunctionType::Builtin(BasicFunc::Abs.into()), nz!(1)),
            ]
        );
        assert_eq!(
            usf(&[
                AstNode::Variable(TestVar::X),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Sin.into()), nz!(1)),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Exp.into()), nz!(1)),
                AstNode::Number(1.0),
                AstNode::BinaryOp(BinaryOp::Sub),
            ]),
            vec![
                AstNode::Variable(TestVar::X),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Sin.into()), nz!(1)),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Expm1.into()), nz!(1)),
            ]
        );
        assert_eq!(
            usf(&[
                AstNode::Variable(TestVar::Y),
                AstNode::Function(FunctionType::Builtin(BasicFunc::Abs.into()), nz!(1)),
                AstNode::Number(1.0),
                AstNode::BinaryOp(BinaryOp::Add),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Ln.into()), nz!(1)),
            ]),
            vec![
                AstNode::Variable(TestVar::Y),
                AstNode::Function(FunctionType::Builtin(BasicFunc::Abs.into()), nz!(1)),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Ln1p.into()), nz!(1)),
            ]
        );
        assert_eq!(
            usf(&[
                AstNode::Number(1.0),
                AstNode::Variable(TestVar::Y),
                AstNode::Function(FunctionType::Builtin(BasicFunc::Abs.into()), nz!(1)),
                AstNode::BinaryOp(BinaryOp::Add),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Ln.into()), nz!(1)),
            ]),
            vec![
                AstNode::Variable(TestVar::Y),
                AstNode::Function(FunctionType::Builtin(BasicFunc::Abs.into()), nz!(1)),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Ln1p.into()), nz!(1)),
            ]
        );
    }

    #[test]
    fn func_parsing() {
        for &id in BasicFunc::VARIANTS {
            assert_eq!(
                StdLibmFuncsTrie.exact_match(id.name()),
                Some(BuiltinFunc::Basic(id)),
                "name: {}",
                id.name()
            );
        }
        for &id in StdFloatFunc::VARIANTS {
            assert_eq!(
                StdLibmFuncsTrie.exact_match(id.name()),
                Some(BuiltinFunc::Ext(StdLibmFunc::Std(id))),
                "name: {}",
                id.name()
            );
        }
        for id in LibmFunc::iter() {
            assert_eq!(
                StdLibmFuncsTrie.exact_match(id.name()),
                Some(BuiltinFunc::Ext(StdLibmFunc::Libm(id))),
                "name: {}",
                id.name()
            );
        }
    }
}
