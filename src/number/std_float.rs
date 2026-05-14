use std::{fmt::Display, num::NonZeroU8};

use strum::{FromRepr, VariantArray};

use crate::{
    BinaryOp, FunctionIdentifier as FuncId, UnaryOp, VariableIdentifier as VarId,
    number::{BuiltinFuncId, CommonBuiltinFunc, ImmEvalStabilityGuard, Number},
    nz,
    postfix_tree::{PostfixTree, subtree_collection::SubtreeCollection},
    syntax::{AstNode, FunctionType},
    tokenizer::NumberRecognizer,
    trie::{NameTrie, TrieNode},
};

pub trait StdFloatLike: for<'a> Number<AsArg<'a> = Self> + Copy {
    fn is_two(self) -> bool;
    fn is_ten(self) -> bool;

    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn tan(self) -> Self;
    fn cot(self) -> Self;

    fn asin(self) -> Self;
    fn acos(self) -> Self;
    fn atan(self) -> Self;
    fn atan2(self, x: Self) -> Self;
    fn acot(self) -> Self;

    fn sinh(self) -> Self;
    fn cosh(self) -> Self;
    fn tanh(self) -> Self;
    fn coth(self) -> Self;

    fn asinh(self) -> Self;
    fn acosh(self) -> Self;
    fn atanh(self) -> Self;
    fn acoth(self) -> Self;

    fn ln(self) -> Self;
    fn ln1p(self) -> Self;

    fn exp(self) -> Self;
    fn expm1(self) -> Self;

    fn floor(self) -> Self;
    fn ceil(self) -> Self;
    fn round(self) -> Self;
    fn trunc(self) -> Self;
    fn frac(self) -> Self;

    fn cqrt(self) -> Self;
}

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, FromRepr, VariantArray)]
#[repr(u8)]
pub enum StdFloatFunc {
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
    Max,
    Min,
}

impl StdFloatFunc {
    pub const fn name(self) -> &'static str {
        match self {
            StdFloatFunc::Sin => "sin",
            StdFloatFunc::Cos => "cos",
            StdFloatFunc::Tan => "tan",
            StdFloatFunc::Cot => "cot",
            StdFloatFunc::Sinh => "sinh",
            StdFloatFunc::Cosh => "cosh",
            StdFloatFunc::Tanh => "tanh",
            StdFloatFunc::Coth => "coth",
            StdFloatFunc::Asin => "asin",
            StdFloatFunc::Acos => "acos",
            StdFloatFunc::Atan => "atan",
            StdFloatFunc::Acot => "acot",
            StdFloatFunc::Atan2 => "atan2",
            StdFloatFunc::Asinh => "asinh",
            StdFloatFunc::Acosh => "acosh",
            StdFloatFunc::Atanh => "atanh",
            StdFloatFunc::Acoth => "acoth",
            StdFloatFunc::Log => "log",
            StdFloatFunc::Log2 => "log2",
            StdFloatFunc::Log10 => "log10",
            StdFloatFunc::Ln => "ln",
            StdFloatFunc::Ln1p => "ln1p",
            StdFloatFunc::Exp => "exp",
            StdFloatFunc::Exp2 => "exp2",
            StdFloatFunc::Exp10 => "exp10",
            StdFloatFunc::Expm1 => "expm1",
            StdFloatFunc::Floor => "floor",
            StdFloatFunc::Ceil => "ceil",
            StdFloatFunc::Round => "round",
            StdFloatFunc::Trunc => "trunc",
            StdFloatFunc::Frac => "frac",
            StdFloatFunc::Abs => "abs",
            StdFloatFunc::Sign => "sign",
            StdFloatFunc::Sqrt => "sqrt",
            StdFloatFunc::Cbrt => "cbrt",
            StdFloatFunc::Max => "max",
            StdFloatFunc::Min => "min",
        }
    }
}

impl BuiltinFuncId for StdFloatFunc {
    fn from_common(id: CommonBuiltinFunc) -> Self {
        match id {
            CommonBuiltinFunc::Exp2 => Self::Exp2,
            CommonBuiltinFunc::Exp10 => Self::Exp10,
            CommonBuiltinFunc::Log => Self::Log,
            CommonBuiltinFunc::Log2 => Self::Log2,
            CommonBuiltinFunc::Log10 => Self::Log10,
            CommonBuiltinFunc::Sqrt => Self::Sqrt,
            CommonBuiltinFunc::Abs => Self::Abs,
            CommonBuiltinFunc::Sign => Self::Sign,
            CommonBuiltinFunc::Min => Self::Min,
            CommonBuiltinFunc::Max => Self::Max,
        }
    }

    fn into_common(self) -> Option<CommonBuiltinFunc> {
        match self {
            Self::Log => Some(CommonBuiltinFunc::Log),
            Self::Log2 => Some(CommonBuiltinFunc::Log2),
            Self::Log10 => Some(CommonBuiltinFunc::Log10),
            Self::Abs => Some(CommonBuiltinFunc::Abs),
            Self::Sign => Some(CommonBuiltinFunc::Sign),
            Self::Min => Some(CommonBuiltinFunc::Min),
            Self::Max => Some(CommonBuiltinFunc::Max),
            _ => None,
        }
    }

    fn is_flex(self) -> bool {
        matches!(self, StdFloatFunc::Min | StdFloatFunc::Max)
    }

    fn min_args(self) -> NonZeroU8 {
        match self {
            StdFloatFunc::Log | StdFloatFunc::Max | StdFloatFunc::Min => {
                nz!(2)
            }
            _ => nz!(1),
        }
    }

    fn max_args(self) -> Option<NonZeroU8> {
        match self {
            StdFloatFunc::Log => Some(nz!(2)),
            StdFloatFunc::Max | StdFloatFunc::Min => None,
            _ => Some(nz!(1)),
        }
    }

    fn specialize_per_argc(&mut self, argc: NonZeroU8) {
        if *self == StdFloatFunc::Log && argc.get() == 1 {
            *self = StdFloatFunc::Ln;
        }
    }
}

impl Display for StdFloatFunc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

pub trait StdFloatFuncsSuperset: BuiltinFuncId {
    fn into_std(self) -> Option<StdFloatFunc>;
    fn from_std(id: StdFloatFunc) -> Self;
}

impl StdFloatFuncsSuperset for StdFloatFunc {
    fn into_std(self) -> Option<StdFloatFunc> {
        Some(self)
    }
    fn from_std(id: StdFloatFunc) -> Self {
        id
    }
}

pub static STD_FLOAT_FUNCS_TRIE_NODES: [TrieNode; 146] = [
    TrieNode::Branch('a', 51),
    TrieNode::Branch('b', 2),
    TrieNode::Branch('s', 1),
    TrieNode::Leaf(StdFloatFunc::Abs as u32),
    TrieNode::Branch('c', 9),
    TrieNode::Branch('o', 8),
    TrieNode::Branch('s', 3),
    TrieNode::Leaf(StdFloatFunc::Acos as u32),
    TrieNode::Branch('h', 1),
    TrieNode::Leaf(StdFloatFunc::Acosh as u32),
    TrieNode::Branch('t', 3),
    TrieNode::Leaf(StdFloatFunc::Acot as u32),
    TrieNode::Branch('h', 1),
    TrieNode::Leaf(StdFloatFunc::Acoth as u32),
    TrieNode::Branch('r', 25),
    TrieNode::Branch('c', 24),
    TrieNode::Branch('c', 9),
    TrieNode::Branch('o', 8),
    TrieNode::Branch('s', 3),
    TrieNode::Leaf(StdFloatFunc::Acos as u32),
    TrieNode::Branch('h', 1),
    TrieNode::Leaf(StdFloatFunc::Acosh as u32),
    TrieNode::Branch('t', 3),
    TrieNode::Leaf(StdFloatFunc::Acot as u32),
    TrieNode::Branch('h', 1),
    TrieNode::Leaf(StdFloatFunc::Acoth as u32),
    TrieNode::Branch('s', 5),
    TrieNode::Branch('i', 4),
    TrieNode::Branch('n', 3),
    TrieNode::Leaf(StdFloatFunc::Asin as u32),
    TrieNode::Branch('h', 1),
    TrieNode::Leaf(StdFloatFunc::Asinh as u32),
    TrieNode::Branch('t', 7),
    TrieNode::Branch('a', 6),
    TrieNode::Branch('n', 5),
    TrieNode::Leaf(StdFloatFunc::Atan as u32),
    TrieNode::Branch('2', 1),
    TrieNode::Leaf(StdFloatFunc::Atan2 as u32),
    TrieNode::Branch('h', 1),
    TrieNode::Leaf(StdFloatFunc::Atanh as u32),
    TrieNode::Branch('s', 5),
    TrieNode::Branch('i', 4),
    TrieNode::Branch('n', 3),
    TrieNode::Leaf(StdFloatFunc::Asin as u32),
    TrieNode::Branch('h', 1),
    TrieNode::Leaf(StdFloatFunc::Asinh as u32),
    TrieNode::Branch('t', 5),
    TrieNode::Branch('a', 4),
    TrieNode::Branch('n', 3),
    TrieNode::Leaf(StdFloatFunc::Atan as u32),
    TrieNode::Branch('h', 1),
    TrieNode::Leaf(StdFloatFunc::Atanh as u32),
    TrieNode::Branch('c', 17),
    TrieNode::Branch('b', 3),
    TrieNode::Branch('r', 2),
    TrieNode::Branch('t', 1),
    TrieNode::Leaf(StdFloatFunc::Cbrt as u32),
    TrieNode::Branch('e', 3),
    TrieNode::Branch('i', 2),
    TrieNode::Branch('l', 1),
    TrieNode::Leaf(StdFloatFunc::Ceil as u32),
    TrieNode::Branch('o', 8),
    TrieNode::Branch('s', 3),
    TrieNode::Leaf(StdFloatFunc::Cos as u32),
    TrieNode::Branch('h', 1),
    TrieNode::Leaf(StdFloatFunc::Cosh as u32),
    TrieNode::Branch('t', 3),
    TrieNode::Leaf(StdFloatFunc::Cot as u32),
    TrieNode::Branch('h', 1),
    TrieNode::Leaf(StdFloatFunc::Coth as u32),
    TrieNode::Branch('e', 8),
    TrieNode::Branch('x', 7),
    TrieNode::Branch('p', 6),
    TrieNode::Leaf(StdFloatFunc::Exp as u32),
    TrieNode::Branch('2', 1),
    TrieNode::Leaf(StdFloatFunc::Exp2 as u32),
    TrieNode::Branch('m', 2),
    TrieNode::Branch('1', 1),
    TrieNode::Leaf(StdFloatFunc::Expm1 as u32),
    TrieNode::Branch('f', 11),
    TrieNode::Branch('l', 4),
    TrieNode::Branch('o', 3),
    TrieNode::Branch('o', 2),
    TrieNode::Branch('r', 1),
    TrieNode::Leaf(StdFloatFunc::Floor as u32),
    TrieNode::Branch('r', 5),
    TrieNode::Branch('a', 4),
    TrieNode::Branch('c', 3),
    TrieNode::Leaf(StdFloatFunc::Frac as u32),
    TrieNode::Branch('t', 1),
    TrieNode::Leaf(StdFloatFunc::Frac as u32),
    TrieNode::Branch('l', 17),
    TrieNode::Branch('b', 1),
    TrieNode::Leaf(StdFloatFunc::Log2 as u32),
    TrieNode::Branch('g', 1),
    TrieNode::Leaf(StdFloatFunc::Log2 as u32),
    TrieNode::Branch('n', 4),
    TrieNode::Leaf(StdFloatFunc::Ln as u32),
    TrieNode::Branch('1', 2),
    TrieNode::Branch('p', 1),
    TrieNode::Leaf(StdFloatFunc::Ln1p as u32),
    TrieNode::Branch('o', 7),
    TrieNode::Branch('g', 6),
    TrieNode::Leaf(StdFloatFunc::Log as u32),
    TrieNode::Branch('1', 2),
    TrieNode::Branch('0', 1),
    TrieNode::Leaf(StdFloatFunc::Log10 as u32),
    TrieNode::Branch('2', 1),
    TrieNode::Leaf(StdFloatFunc::Log2 as u32),
    TrieNode::Branch('m', 6),
    TrieNode::Branch('a', 2),
    TrieNode::Branch('x', 1),
    TrieNode::Leaf(StdFloatFunc::Max as u32),
    TrieNode::Branch('i', 2),
    TrieNode::Branch('n', 1),
    TrieNode::Leaf(StdFloatFunc::Min as u32),
    TrieNode::Branch('r', 5),
    TrieNode::Branch('o', 4),
    TrieNode::Branch('u', 3),
    TrieNode::Branch('n', 2),
    TrieNode::Branch('d', 1),
    TrieNode::Leaf(StdFloatFunc::Round as u32),
    TrieNode::Branch('s', 12),
    TrieNode::Branch('i', 7),
    TrieNode::Branch('g', 2),
    TrieNode::Branch('n', 1),
    TrieNode::Leaf(StdFloatFunc::Sign as u32),
    TrieNode::Branch('n', 3),
    TrieNode::Leaf(StdFloatFunc::Sin as u32),
    TrieNode::Branch('h', 1),
    TrieNode::Leaf(StdFloatFunc::Sinh as u32),
    TrieNode::Branch('q', 3),
    TrieNode::Branch('r', 2),
    TrieNode::Branch('t', 1),
    TrieNode::Leaf(StdFloatFunc::Sqrt as u32),
    TrieNode::Branch('t', 10),
    TrieNode::Branch('a', 4),
    TrieNode::Branch('n', 3),
    TrieNode::Leaf(StdFloatFunc::Tan as u32),
    TrieNode::Branch('h', 1),
    TrieNode::Leaf(StdFloatFunc::Tanh as u32),
    TrieNode::Branch('r', 4),
    TrieNode::Branch('u', 3),
    TrieNode::Branch('n', 2),
    TrieNode::Branch('c', 1),
    TrieNode::Leaf(StdFloatFunc::Trunc as u32),
];

pub struct StdFloatFuncsTrie;

impl NameTrie<StdFloatFunc> for StdFloatFuncsTrie {
    fn nodes(&self) -> &[TrieNode] {
        &STD_FLOAT_FUNCS_TRIE_NODES
    }
    fn leaf_to_value(&self, leaf: u32) -> StdFloatFunc {
        StdFloatFunc::from_repr(leaf as u8).unwrap()
    }
}

pub static STD_FLOAT_CONSTS_TRIE_NODES: [TrieNode; 9] = [
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

pub struct StdFloatConstsNameTrie<F: Copy> {
    pub pi: F,
    pub e: F,
    pub tau: F,
}

impl<F: Copy> NameTrie<F> for StdFloatConstsNameTrie<F> {
    fn nodes(&self) -> &[TrieNode] {
        &STD_FLOAT_CONSTS_TRIE_NODES
    }

    fn leaf_to_value(&self, leaf: u32) -> F {
        match leaf {
            0 => self.pi,
            1 => self.e,
            2 => self.tau,
            _ => unreachable!(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StdFloatPrecisionGuard<N> {
    Number(N),
    Exp(N),
    OnePlus(N),
}

impl<N> ImmEvalStabilityGuard<N> for StdFloatPrecisionGuard<N>
where
    N: StdFloatLike<BuiltinFuncId = StdFloatFunc>,
{
    fn from_number(num: N) -> Self {
        Self::Number(num)
    }

    fn eval(self) -> N {
        match self {
            Self::Number(num) => num,
            Self::Exp(num) => num.exp(),
            Self::OnePlus(num) => num + N::from_i8(1),
        }
    }

    fn apply_unary_op(self, opr: UnaryOp) -> Self {
        Self::Number(opr.eval(self.eval()))
    }

    fn apply_binary_op(self, rhs: Self, opr: BinaryOp) -> Self {
        match (self, rhs) {
            (Self::Exp(x), Self::Number(y)) if opr == BinaryOp::Sub && y.as_i8() == Some(1) => {
                Self::Number(x.expm1())
            }
            (Self::Number(x), y) | (y, Self::Number(x))
                if opr == BinaryOp::Add && x.as_i8() == Some(1) =>
            {
                Self::OnePlus(y.eval())
            }
            (Self::Number(base), exp) if opr == BinaryOp::Pow => {
                let exp = exp.eval();
                Self::Number(if base.as_i8() == Some(2) {
                    exp.exp2()
                } else {
                    base.pow(exp.asarg())
                })
            }
            (lhs, rhs) => Self::Number(opr.eval(lhs.eval(), rhs.eval())),
        }
    }

    fn apply_func_single(self, id: StdFloatFunc, func: fn(N) -> N) -> Self {
        match id {
            StdFloatFunc::Exp => Self::Exp(self.eval()),
            StdFloatFunc::Ln => Self::Number(match self {
                Self::Exp(num) => num,
                Self::OnePlus(x) => x.ln1p(),
                _ => self.eval().ln(),
            }),
            _ => Self::Number(func(self.eval())),
        }
    }

    fn apply_func_dual(self, arg2: Self, id: StdFloatFunc, func: for<'a> fn(N, N) -> N) -> Self {
        Self::Number(if id == StdFloatFunc::Log {
            let base = arg2.eval();
            let num = self.eval();
            if base.as_i8() == Some(10) {
                num.log10()
            } else if base.as_i8() == Some(2) {
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
        id: StdFloatFunc,
        _func: fn(&[N]) -> N,
        arg_space: &mut Vec<N>,
    ) -> Self {
        arg_space.extend(args.map(Self::eval));
        Self::Number(match id {
            StdFloatFunc::Min => N::min(&arg_space),
            StdFloatFunc::Max => N::max(&arg_space),
            _ => unreachable!(),
        })
    }
}

pub fn substitute_ln1p_eq<N, B, V: VarId, F: FuncId>(
    tree: &mut PostfixTree<AstNode<N, V, F>>,
    symbol_space: &mut SubtreeCollection<AstNode<N, V, F>>,
    target: usize,
) -> bool
where
    N: StdFloatLike<BuiltinFuncId = B>,
    B: StdFloatFuncsSuperset,
{
    if matches!(
        tree[target],
        AstNode::Function(FunctionType::Builtin(func), _)
            if func.into_std() == Some(StdFloatFunc::Ln)
    ) {
        let child = tree.nth_child(target, 0).unwrap();
        if matches!(tree[child], AstNode::BinaryOp(BinaryOp::Add)) {
            let mut children = tree.children_iter(child);
            match (children.next().unwrap(), children.next().unwrap()) {
                ((AstNode::Number(num), _), (_, param))
                | ((_, param), (AstNode::Number(num), _))
                    if num.as_i8() == Some(1) =>
                {
                    symbol_space.extend_from_tree(&tree, param);
                    symbol_space
                        .push(AstNode::Function(
                            FunctionType::Builtin(B::from_std(StdFloatFunc::Ln1p)),
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
    } else {
        false
    }
}

pub fn substitute_expm1_eq<N, B, V: VarId, F: FuncId>(
    tree: &mut PostfixTree<AstNode<N, V, F>>,
    symbol_space: &mut SubtreeCollection<AstNode<N, V, F>>,
    target: usize,
) -> bool
where
    N: StdFloatLike<BuiltinFuncId = B>,
    B: StdFloatFuncsSuperset,
{
    if matches!(tree[target], AstNode::BinaryOp(BinaryOp::Sub)) {
        let mut children = tree.children_iter(target);
        match (children.next().unwrap(), children.next().unwrap()) {
            (
                (AstNode::Number(rhs), _),
                (AstNode::Function(FunctionType::Builtin(func), _), idx),
            ) if func.into_std() == Some(StdFloatFunc::Exp) && rhs.as_i8() == Some(1) => {
                let param = tree.nth_child(idx, 0).unwrap();
                symbol_space.extend_from_tree(&tree, param);
                symbol_space
                    .push(AstNode::Function(
                        FunctionType::Builtin(B::from_std(StdFloatFunc::Expm1)),
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

#[cfg(not(feature = "libm"))]
pub fn substitute_std_float_spec_funcs_eq<N, B, V: VarId, F: FuncId>(
    tree: &mut PostfixTree<AstNode<N, V, F>>,
) where
    N: StdFloatLike<BuiltinFuncId = B>,
    B: StdFloatFuncsSuperset,
{
    let mut symbol_space: SubtreeCollection<AstNode<N, V, F>> =
        SubtreeCollection::from_alloc(Vec::with_capacity(0));
    let mut idx = 2;
    while idx < tree.len() {
        for subs in [
            substitute_expm1_eq,
            substitute_ln1p_eq,
            super::substitute_log_eq,
            super::substitute_exp_eq,
        ] {
            if subs(tree, &mut symbol_space, idx) {
                break;
            }
        }
        idx += 1;
    }
}

pub struct StdFloatRecognizer(bool);

impl NumberRecognizer for StdFloatRecognizer {
    fn new(current: char) -> Option<Self> {
        match current {
            '0'..='9' => Some(Self(false)),
            '.' => Some(Self(true)),
            _ => None,
        }
    }

    fn recognize(&mut self, current: char) -> bool {
        if (current == 'e' || current == '.') && !self.0 {
            self.0 = true;
            true
        } else {
            current.is_ascii_digit()
        }
    }
}

macro_rules! impl_number_for_std_float {
    ($t: ident) => {
        #[cfg(not(feature = "libm"))]
        impl Number for $t {
            type AsArg<'a> = Self;
            type Recognizer = StdFloatRecognizer;
            type ConstsTrieType = StdFloatConstsNameTrie<Self>;
            type BuiltinFuncId = StdFloatFunc;
            type BuiltinFuncsTrieType = StdFloatFuncsTrie;
            type ImmEvalStabilityGuard = StdFloatPrecisionGuard<Self>;

            const CONSTS_TRIE: StdFloatConstsNameTrie<Self> = StdFloatConstsNameTrie {
                pi: std::$t::consts::PI,
                e: std::$t::consts::E,
                tau: std::$t::consts::TAU,
            };
            const BUILTIN_FUNCS_TRIE: Self::BuiltinFuncsTrieType = StdFloatFuncsTrie;

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

            fn get_method_ptr(id: StdFloatFunc) -> super::BfPointer<Self> {
                use crate::number::BfPointer;
                match id {
                    StdFloatFunc::Sin => BfPointer::Single(Self::sin),
                    StdFloatFunc::Cos => BfPointer::Single(Self::cos),
                    StdFloatFunc::Tan => BfPointer::Single(Self::tan),
                    StdFloatFunc::Cot => BfPointer::Single(|x| {
                        let (sin, cos) = x.sin_cos();
                        cos / sin
                    }),
                    StdFloatFunc::Sinh => BfPointer::Single(Self::sinh),
                    StdFloatFunc::Cosh => BfPointer::Single(Self::cosh),
                    StdFloatFunc::Tanh => BfPointer::Single(Self::tanh),
                    StdFloatFunc::Coth => {
                        BfPointer::Single(|x| (-x).atan() + std::$t::consts::FRAC_PI_2)
                    }
                    StdFloatFunc::Asin => BfPointer::Single(Self::asin),
                    StdFloatFunc::Acos => BfPointer::Single(Self::acos),
                    StdFloatFunc::Atan => BfPointer::Single(Self::atan),
                    StdFloatFunc::Acot => BfPointer::Single(|x| x.cosh() / x.sinh()),
                    StdFloatFunc::Atan2 => BfPointer::<Self>::Dual(Self::atan2),
                    StdFloatFunc::Asinh => BfPointer::Single(Self::asinh),
                    StdFloatFunc::Acosh => BfPointer::Single(Self::acosh),
                    StdFloatFunc::Atanh => BfPointer::Single(Self::atanh),
                    StdFloatFunc::Acoth => BfPointer::Single(|x| x.recip().atanh()),
                    StdFloatFunc::Log => BfPointer::<Self>::Dual(Self::log),
                    StdFloatFunc::Log2 => BfPointer::Single(Self::log2),
                    StdFloatFunc::Log10 => BfPointer::Single(Self::log10),
                    StdFloatFunc::Ln => BfPointer::Single(Self::ln),
                    StdFloatFunc::Ln1p => BfPointer::Single(Self::ln_1p),
                    StdFloatFunc::Exp => BfPointer::Single(Self::exp),
                    StdFloatFunc::Exp2 => BfPointer::Single(Self::exp2),
                    StdFloatFunc::Exp10 => BfPointer::Single(Self::exp10),
                    StdFloatFunc::Expm1 => BfPointer::Single(Self::exp_m1),
                    StdFloatFunc::Floor => BfPointer::Single(Self::floor),
                    StdFloatFunc::Ceil => BfPointer::Single(Self::ceil),
                    StdFloatFunc::Round => BfPointer::Single(Self::round),
                    StdFloatFunc::Trunc => BfPointer::Single(Self::trunc),
                    StdFloatFunc::Frac => BfPointer::Single(Self::fract),
                    StdFloatFunc::Abs => BfPointer::Single(Self::abs),
                    StdFloatFunc::Sign => BfPointer::Single(Self::sign),
                    StdFloatFunc::Sqrt => BfPointer::Single(Self::sqrt),
                    StdFloatFunc::Cbrt => BfPointer::Single(Self::cbrt),
                    StdFloatFunc::Max => BfPointer::Flexible(<Self as Number>::max),
                    StdFloatFunc::Min => BfPointer::Flexible(<Self as Number>::min),
                }
            }

            fn substitute_spec_funcs_equivalents<V: VarId, F: FuncId>(
                tree: &mut PostfixTree<AstNode<Self, V, F>>,
            ) {
                substitute_std_float_spec_funcs_eq(tree);
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
                (10 as $t).powf(self)
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

            fn sqrt(self) -> Self {
                self.sqrt()
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
                if self.is_infinite() || self < 0.0 || self.is_nan() {
                    return Self::NAN;
                }
                if self >= 171.0 {
                    return Self::INFINITY;
                }
                let mut result = 1.0;
                let mut k = self as u32;
                while k > 1 {
                    result *= k as $t;
                    k -= 1;
                }
                result
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

macro_rules! impl_sfl_for_std_float {
    ($t: ident) => {
        impl StdFloatLike for $t {
            fn is_two(self) -> bool {
                self == 2.0
            }

            fn is_ten(self) -> bool {
                self == 10.0
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

            fn atan2(self, x: Self) -> Self {
                self.atan2(x)
            }

            fn acot(self) -> Self {
                std::$t::consts::FRAC_PI_2 - self.atan()
            }

            fn sinh(self) -> Self {
                self.sinh()
            }

            fn cosh(self) -> Self {
                self.cosh()
            }

            fn tanh(self) -> Self {
                self.tanh()
            }

            fn coth(self) -> Self {
                self.cosh() / self.sinh()
            }

            fn asinh(self) -> Self {
                self.asinh()
            }

            fn acosh(self) -> Self {
                self.acosh()
            }

            fn atanh(self) -> Self {
                self.atanh()
            }

            fn acoth(self) -> Self {
                self.recip().atanh()
            }

            fn ln(self) -> Self {
                self.ln()
            }

            fn ln1p(self) -> Self {
                self.ln_1p()
            }

            fn exp(self) -> Self {
                self.exp()
            }

            fn expm1(self) -> Self {
                self.exp_m1()
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

            fn cqrt(self) -> Self {
                self.cbrt()
            }
        }
    };
}

impl_sfl_for_std_float!(f64);
impl_sfl_for_std_float!(f32);

#[cfg(test)]
mod tests {
    use strum::VariantArray;
    use super::*;

    #[test]
    #[cfg(not(feature = "libm"))]
    fn precision_guard() {
        use super::{StdFloatFunc, StdFloatPrecisionGuard as Sfsg};
        use crate::{BinaryOp, UnaryOp, number::ImmEvalStabilityGuard};

        assert_eq!(
            Sfsg::Number(3.0).apply_func_single(StdFloatFunc::Exp.into(), |_| panic!()),
            Sfsg::Exp(3.0),
        );
        assert_eq!(
            Sfsg::Exp(3.0).apply_binary_op(Sfsg::Number(1.0), BinaryOp::Sub),
            Sfsg::Number(3f64.exp_m1()),
        );
        assert_eq!(
            Sfsg::Exp(3.0).apply_binary_op(Sfsg::Number(5.0), BinaryOp::Sub),
            Sfsg::Number(3f64.exp() - 5.0),
        );
        assert_eq!(
            Sfsg::Number(0.0).apply_binary_op(Sfsg::Number(1.0), BinaryOp::Add),
            Sfsg::OnePlus(0.0)
        );
        assert_eq!(
            Sfsg::OnePlus(8.0).apply_func_single(StdFloatFunc::Ln.into(), |_| panic!()),
            Sfsg::Number(8f64.ln_1p())
        );
        assert_eq!(
            Sfsg::OnePlus(0.0).apply_binary_op(Sfsg::Number(1.8), BinaryOp::Add),
            Sfsg::Number(2.8)
        );
        assert_eq!(
            Sfsg::OnePlus(8.11).apply_binary_op(Sfsg::Number(1.0), BinaryOp::Add),
            Sfsg::OnePlus(9.11)
        );
        assert_eq!(
            Sfsg::Number(1.0).apply_binary_op(Sfsg::Number(9.3), BinaryOp::Add),
            Sfsg::OnePlus(9.3)
        );
        assert_eq!(
            Sfsg::Number(4.2).apply_binary_op(Sfsg::Number(1.0), BinaryOp::Add),
            Sfsg::OnePlus(4.2)
        );
        assert_eq!(
            Sfsg::Number(2.0).apply_binary_op(Sfsg::Number(100.0), BinaryOp::Pow),
            Sfsg::Number(100f64.exp2())
        );
        assert_eq!(
            Sfsg::Number(977f64).apply_func_dual(
                Sfsg::Number(10.0),
                StdFloatFunc::Log,
                |_, _| panic!()
            ),
            Sfsg::Number(977f64.log10())
        );
        assert_eq!(
            Sfsg::Number(888f64).apply_func_dual(
                Sfsg::Number(2.0),
                StdFloatFunc::Log,
                |_, _| panic!()
            ),
            Sfsg::Number(888f64.log2())
        );
        assert_eq!(
            Sfsg::Number(1.0).apply_unary_op(UnaryOp::Neg),
            Sfsg::Number(-1.0)
        );
        assert_eq!(
            Sfsg::Number(2.0).apply_binary_op(Sfsg::Number(9.3), BinaryOp::Add),
            Sfsg::Number(11.3)
        );
    }

    #[test]
    fn substitute_spec_funcs() {
        use super::StdFloatFunc;
        use crate::{
            BinaryOp, nz,
            syntax::{AstNode, FunctionType, MathAst},
        };

        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        enum TestVar {
            X,
            Y,
        }

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
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Abs.into()), nz!(1)),
                AstNode::Number(1.0),
                AstNode::BinaryOp(BinaryOp::Add),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Ln.into()), nz!(1)),
            ]),
            vec![
                AstNode::Variable(TestVar::Y),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Abs.into()), nz!(1)),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Ln1p.into()), nz!(1)),
            ]
        );
        assert_eq!(
            usf(&[
                AstNode::Number(1.0),
                AstNode::Variable(TestVar::Y),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Abs.into()), nz!(1)),
                AstNode::BinaryOp(BinaryOp::Add),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Ln.into()), nz!(1)),
            ]),
            vec![
                AstNode::Variable(TestVar::Y),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Abs.into()), nz!(1)),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Ln1p.into()), nz!(1)),
            ]
        );
    }
}
