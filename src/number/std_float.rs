use std::{fmt::Display, num::NonZeroU8};

use libm::Libm;
use strum::{EnumIter, FromRepr};

use crate::{
    BinaryOp, UnaryOp,
    number::{BfPointer, BuiltinFuncId, CommonBuiltinFunc, ImmEvalStabilityGuard, Number},
    nz,
    postfix_tree::{PostfixTree, subtree_collection::SubtreeCollection},
    syntax::{AstNode, FunctionType},
    tokenizer::StandardFloatRecognizer,
    trie::{NameTrie, TrieNode},
};

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, EnumIter, FromRepr)]
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
            StdFloatFunc::Erf => "erf",
            StdFloatFunc::Erfc => "erfc",
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
            StdFloatFunc::Gamma => "gamma",
            StdFloatFunc::Lgamma => "lgamma",
            StdFloatFunc::Max => "max",
            StdFloatFunc::Min => "min",
        }
    }
}

impl BuiltinFuncId for StdFloatFunc {
    fn from_common(id: CommonBuiltinFunc) -> Self {
        match id {
            CommonBuiltinFunc::Abs => Self::Abs,
            CommonBuiltinFunc::Sign => Self::Sign,
            CommonBuiltinFunc::Min => Self::Min,
            CommonBuiltinFunc::Max => Self::Max,
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

static BUILTIN_FUNCS_TRIE_NODES: [TrieNode; 131] = [
    TrieNode::Branch('a', 19),
    TrieNode::Branch('b', 2),
    TrieNode::Branch('s', 1),
    TrieNode::Leaf(StdFloatFunc::Abs as u32),
    TrieNode::Branch('c', 5),
    TrieNode::Branch('o', 4),
    TrieNode::Branch('s', 1),
    TrieNode::Leaf(StdFloatFunc::Acos as u32),
    TrieNode::Branch('t', 1),
    TrieNode::Leaf(StdFloatFunc::Acot as u32),
    TrieNode::Branch('s', 3),
    TrieNode::Branch('i', 2),
    TrieNode::Branch('n', 1),
    TrieNode::Leaf(StdFloatFunc::Asin as u32),
    TrieNode::Branch('t', 5),
    TrieNode::Branch('a', 4),
    TrieNode::Branch('n', 3),
    TrieNode::Leaf(StdFloatFunc::Atan as u32),
    TrieNode::Branch('2', 1),
    TrieNode::Leaf(StdFloatFunc::Atan2 as u32),
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
    TrieNode::Branch('e', 16),
    TrieNode::Branch('r', 4),
    TrieNode::Branch('f', 3),
    TrieNode::Leaf(StdFloatFunc::Erf as u32),
    TrieNode::Branch('c', 1),
    TrieNode::Leaf(StdFloatFunc::Erfc as u32),
    TrieNode::Branch('x', 10),
    TrieNode::Branch('p', 9),
    TrieNode::Leaf(StdFloatFunc::Exp as u32),
    TrieNode::Branch('1', 2),
    TrieNode::Branch('0', 1),
    TrieNode::Leaf(StdFloatFunc::Exp10 as u32),
    TrieNode::Branch('2', 1),
    TrieNode::Leaf(StdFloatFunc::Exp2 as u32),
    TrieNode::Branch('m', 2),
    TrieNode::Branch('1', 1),
    TrieNode::Leaf(StdFloatFunc::Expm1 as u32),
    TrieNode::Branch('f', 9),
    TrieNode::Branch('l', 4),
    TrieNode::Branch('o', 3),
    TrieNode::Branch('o', 2),
    TrieNode::Branch('r', 1),
    TrieNode::Leaf(StdFloatFunc::Floor as u32),
    TrieNode::Branch('r', 3),
    TrieNode::Branch('a', 2),
    TrieNode::Branch('c', 1),
    TrieNode::Leaf(StdFloatFunc::Frac as u32),
    TrieNode::Branch('g', 5),
    TrieNode::Branch('a', 4),
    TrieNode::Branch('m', 3),
    TrieNode::Branch('m', 2),
    TrieNode::Branch('a', 1),
    TrieNode::Leaf(StdFloatFunc::Gamma as u32),
    TrieNode::Branch('l', 22),
    TrieNode::Branch('b', 1),
    TrieNode::Leaf(StdFloatFunc::Log2 as u32),
    TrieNode::Branch('g', 6),
    TrieNode::Leaf(StdFloatFunc::Log10 as u32),
    TrieNode::Branch('a', 4),
    TrieNode::Branch('m', 3),
    TrieNode::Branch('m', 2),
    TrieNode::Branch('a', 1),
    TrieNode::Leaf(StdFloatFunc::Lgamma as u32),
    TrieNode::Branch('n', 4),
    TrieNode::Leaf(StdFloatFunc::Ln as u32),
    TrieNode::Branch('p', 2),
    TrieNode::Branch('1', 1),
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
        &BUILTIN_FUNCS_TRIE_NODES
    }
    fn leaf_to_value(&self, leaf: u32) -> StdFloatFunc {
        StdFloatFunc::from_repr(leaf as u8).unwrap()
    }
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

pub struct StdFloatConstsNameTrie<F: Copy> {
    pi: F,
    e: F,
    tau: F,
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
pub enum StdFloatStabilityGuard<N> {
    Number(N),
    Erf(N),
    Exp(N),
    Gamma(N),
    OnePlus(N),
}

impl ImmEvalStabilityGuard<f64> for StdFloatStabilityGuard<f64> {
    fn from_number(num: f64) -> Self {
        Self::Number(num)
    }

    fn eval(self) -> f64 {
        match self {
            Self::Number(num) => num,
            Self::Erf(num) => Libm::<f64>::erf(num),
            Self::Exp(num) => num.exp(),
            Self::Gamma(num) => Libm::<f64>::tgamma(num),
            Self::OnePlus(num) => num + 1.0,
        }
    }

    fn apply_unary_op(self, opr: UnaryOp) -> Self {
        match self {
            Self::Number(num) if opr == UnaryOp::Fac => Self::Gamma(num + 1.0),
            _ => Self::Number(opr.eval(self.eval())),
        }
    }

    fn apply_binary_op(self, rhs: Self, opr: BinaryOp) -> Self {
        match (self, rhs) {
            (Self::Number(x), Self::Erf(y)) if opr == BinaryOp::Sub && x == 1.0 => {
                Self::Number(libm::erfc(y))
            }
            (Self::Exp(x), Self::Number(y)) if opr == BinaryOp::Sub && y == 1.0 => {
                Self::Number(x.exp_m1())
            }
            (Self::Number(x), y) | (y, Self::Number(x)) if opr == BinaryOp::Add && x == 1.0 => {
                Self::OnePlus(y.eval())
            }
            (Self::Number(base), exp) if opr == BinaryOp::Pow => {
                let exp = exp.eval();
                Self::Number(if base == 10.0 {
                    Libm::<f64>::exp10(exp)
                } else if base == 2.0 {
                    exp.exp2()
                } else {
                    base.pow(exp.asarg())
                })
            }
            (lhs, rhs) => Self::Number(opr.eval(lhs.eval(), rhs.eval())),
        }
    }

    fn apply_func_single(self, id: StdFloatFunc, func: fn(f64) -> f64) -> Self {
        match id {
            StdFloatFunc::Erf => Self::Erf(self.eval()),
            StdFloatFunc::Exp => Self::Exp(self.eval()),
            StdFloatFunc::Gamma => Self::Gamma(self.eval()),
            StdFloatFunc::Ln => Self::Number(match self {
                Self::Gamma(num) => libm::lgamma(num),
                Self::Exp(num) => num,
                Self::OnePlus(x) => x.ln_1p(),
                _ => self.eval().ln(),
            }),
            StdFloatFunc::Log10 => {
                // 171.6243769563027 IS FOR f64
                // use 35.040096 for f32
                Self::Number(
                    if let Self::Gamma(num) = self
                        && num > 171.6243769563027
                    {
                        libm::lgamma(num) / 10f64.ln()
                    } else {
                        self.eval().log10()
                    },
                )
            }
            StdFloatFunc::Log2 => Self::Number(
                if let Self::Gamma(num) = self
                    && num > 171.6243769563027
                {
                    libm::lgamma(num) / 2f64.ln()
                } else {
                    self.eval().log2()
                },
            ),
            _ => Self::Number(func(self.eval())),
        }
    }

    fn apply_func_dual(self, arg2: Self, id: StdFloatFunc, func: fn(f64, f64) -> f64) -> Self {
        Self::Number(if id == StdFloatFunc::Log {
            let base = arg2.eval();
            if let Self::Gamma(num) = self
                && num > 171.6243769563027
            {
                libm::lgamma(num) / base.ln()
            } else {
                let num = self.eval();
                if base == 10.0 {
                    num.log10()
                } else if base == 2.0 {
                    num.log2()
                } else {
                    num.log(base)
                }
            }
        } else {
            func(self.eval(), arg2.eval())
        })
    }

    fn apply_func_flex(
        args: std::vec::Drain<'_, Self>,
        id: StdFloatFunc,
        _func: fn(&[f64]) -> f64,
        _arg_space: &mut Vec<f64>,
    ) -> Self {
        Self::Number(match id {
            StdFloatFunc::Min => args.map(Self::eval).min_by(|x, y| x.total_cmp(y)).unwrap(),
            StdFloatFunc::Max => args.map(Self::eval).max_by(|x, y| x.total_cmp(y)).unwrap(),
            _ => unreachable!(),
        })
    }
}

impl Number for f64 {
    type AsArg<'a> = Self;
    type Recognizer = StandardFloatRecognizer;
    type ConstsTrieType = StdFloatConstsNameTrie<Self>;
    type BuiltinFuncId = StdFloatFunc;
    type BuiltinFuncsTrieType = StdFloatFuncsTrie;
    type ImmEvalStabilityGuard = StdFloatStabilityGuard<Self>;

    const CONSTS_TRIE: StdFloatConstsNameTrie<Self> = StdFloatConstsNameTrie {
        pi: std::f64::consts::PI,
        e: std::f64::consts::E,
        tau: std::f64::consts::TAU,
    };
    const BUILTIN_FUNCS_TRIE: Self::BuiltinFuncsTrieType = StdFloatFuncsTrie;

    fn one() -> Self {
        1.0
    }

    fn zero() -> Self {
        0.0
    }

    fn get_method_ptr(id: StdFloatFunc) -> BfPointer<f64> {
        match id {
            StdFloatFunc::Sin => BfPointer::Single(f64::sin),
            StdFloatFunc::Cos => BfPointer::Single(f64::cos),
            StdFloatFunc::Tan => BfPointer::Single(f64::tan),
            StdFloatFunc::Cot => BfPointer::Single(|x| {
                let (sin, cos) = x.sin_cos();
                cos / sin
            }),
            StdFloatFunc::Sinh => BfPointer::Single(f64::sinh),
            StdFloatFunc::Cosh => BfPointer::Single(f64::cosh),
            StdFloatFunc::Tanh => BfPointer::Single(f64::tanh),
            StdFloatFunc::Coth => BfPointer::Single(|x| (-x).atan() + std::f64::consts::FRAC_PI_2),
            StdFloatFunc::Asin => BfPointer::Single(f64::asin),
            StdFloatFunc::Acos => BfPointer::Single(f64::acos),
            StdFloatFunc::Atan => BfPointer::Single(f64::atan),
            StdFloatFunc::Acot => BfPointer::Single(|x| x.cosh() / x.sinh()),
            StdFloatFunc::Atan2 => BfPointer::<f64>::Dual(f64::atan2),
            StdFloatFunc::Asinh => BfPointer::Single(f64::asinh),
            StdFloatFunc::Acosh => BfPointer::Single(f64::acosh),
            StdFloatFunc::Atanh => BfPointer::Single(f64::atanh),
            StdFloatFunc::Acoth => BfPointer::Single(|x| x.recip().atanh()),
            StdFloatFunc::Erf => BfPointer::Single(libm::erf),
            StdFloatFunc::Erfc => BfPointer::Single(libm::erfc),
            StdFloatFunc::Log => BfPointer::<f64>::Dual(f64::log),
            StdFloatFunc::Log2 => BfPointer::Single(f64::log2),
            StdFloatFunc::Log10 => BfPointer::Single(f64::log10),
            StdFloatFunc::Ln => BfPointer::Single(f64::ln),
            StdFloatFunc::Ln1p => BfPointer::Single(f64::ln_1p),
            StdFloatFunc::Exp => BfPointer::Single(f64::exp),
            StdFloatFunc::Exp2 => BfPointer::Single(f64::exp2),
            StdFloatFunc::Exp10 => BfPointer::Single(libm::exp10),
            StdFloatFunc::Expm1 => BfPointer::Single(f64::exp_m1),
            StdFloatFunc::Floor => BfPointer::Single(f64::floor),
            StdFloatFunc::Ceil => BfPointer::Single(f64::ceil),
            StdFloatFunc::Round => BfPointer::Single(f64::round),
            StdFloatFunc::Trunc => BfPointer::Single(f64::trunc),
            StdFloatFunc::Frac => BfPointer::Single(f64::fract),
            StdFloatFunc::Abs => BfPointer::Single(f64::abs),
            StdFloatFunc::Sign => BfPointer::Single(f64::sign),
            StdFloatFunc::Sqrt => BfPointer::Single(f64::sqrt),
            StdFloatFunc::Cbrt => BfPointer::Single(f64::cbrt),
            StdFloatFunc::Gamma => BfPointer::Single(libm::tgamma),
            StdFloatFunc::Lgamma => BfPointer::Single(libm::lgamma),
            StdFloatFunc::Max => BfPointer::Flexible(<f64 as Number>::max),
            StdFloatFunc::Min => BfPointer::Flexible(<f64 as Number>::min),
        }
    }

    fn substitute_spec_funcs_equivalents<
        V: crate::VariableIdentifier,
        F: crate::FunctionIdentifier,
    >(
        tree: &mut crate::postfix_tree::PostfixTree<crate::syntax::AstNode<Self, V, F>>,
    ) {
        macro_rules! log_detector {
            ($base: expr, $node: expr) => {
                (
                    &|target, tree| {
                        if matches!(
                            tree[target],
                            AstNode::Function(FunctionType::Builtin(StdFloatFunc::Log), _)
                        ) {
                            let mut children = tree.children_iter(target);
                            match children.next().unwrap() {
                                (AstNode::Number(base), _) if *base == $base => {
                                    Some(children.next().unwrap().1)
                                }
                                _ => None,
                            }
                        } else {
                            None
                        }
                    },
                    &[AstNode::Function(FunctionType::Builtin($node), nz!(1))],
                )
            };
        }
        macro_rules! exp_detector {
            ($base: expr, $node: expr) => {
                (
                    &|target, tree| {
                        if matches!(tree[target], AstNode::BinaryOp(BinaryOp::Pow)) {
                            let mut children = tree.children_iter(target);
                            let idx = children.next().unwrap().1;
                            match children.next().unwrap() {
                                (AstNode::Number(base), _) if *base == $base => Some(idx),
                                _ => None,
                            }
                        } else {
                            None
                        }
                    },
                    &[AstNode::Function(FunctionType::Builtin($node), nz!(1))],
                )
            };
        }
        let detrep_pairs: [(
            &dyn Fn(usize, &PostfixTree<AstNode<Self, V, F>>) -> Option<usize>,
            &'static [AstNode<Self, V, F>],
        ); _] = [
            (
                &|target, tree: &PostfixTree<AstNode<Self, V, F>>| {
                    if matches!(tree[target], AstNode::BinaryOp(BinaryOp::Sub)) {
                        let mut children = tree.children_iter(target);
                        match (children.next().unwrap(), children.next().unwrap()) {
                            (
                                (
                                    AstNode::Function(FunctionType::Builtin(StdFloatFunc::Erf), _),
                                    idx,
                                ),
                                (AstNode::Number(lhs), _),
                            ) if *lhs == 1.0 => Some(tree.children_iter(idx).next().unwrap().1),
                            _ => None,
                        }
                    } else {
                        None
                    }
                },
                &[AstNode::Function(
                    FunctionType::Builtin(StdFloatFunc::Erfc),
                    nz!(1),
                )],
            ),
            log_detector!(10.0, StdFloatFunc::Log10),
            log_detector!(2.0, StdFloatFunc::Log2),
            exp_detector!(10.0, StdFloatFunc::Exp10),
            exp_detector!(2.0, StdFloatFunc::Exp2),
            (
                &|target, tree| {
                    if matches!(tree[target], AstNode::BinaryOp(BinaryOp::Sub)) {
                        let mut children = tree.children_iter(target);
                        match (children.next().unwrap(), children.next().unwrap()) {
                            (
                                (AstNode::Number(rhs), _),
                                (
                                    AstNode::Function(FunctionType::Builtin(StdFloatFunc::Exp), _),
                                    idx,
                                ),
                            ) if *rhs == 1.0 => Some(tree.children_iter(idx).next().unwrap().1),
                            _ => None,
                        }
                    } else {
                        None
                    }
                },
                &[AstNode::Function(
                    FunctionType::Builtin(StdFloatFunc::Expm1),
                    nz!(1),
                )],
            ),
            (
                &|target, tree| {
                    if matches!(
                        tree[target],
                        AstNode::Function(FunctionType::Builtin(StdFloatFunc::Ln), _)
                    ) {
                        let (child_node, child_idx) = tree.children_iter(target).next().unwrap();
                        if matches!(child_node, AstNode::BinaryOp(BinaryOp::Add)) {
                            let mut children = tree.children_iter(child_idx);
                            match (children.next().unwrap(), children.next().unwrap()) {
                                ((AstNode::Number(num), _), (_, idx))
                                | ((_, idx), (AstNode::Number(num), _))
                                    if *num == 1.0 =>
                                {
                                    Some(idx)
                                }
                                _ => None,
                            }
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                },
                &[AstNode::Function(
                    FunctionType::Builtin(StdFloatFunc::Ln1p),
                    nz!(1),
                )],
            ),
            (
                &|target, tree| {
                    if matches!(
                        tree[target],
                        AstNode::Function(FunctionType::Builtin(StdFloatFunc::Ln), _)
                    ) {
                        let (child_node, child_idx) = tree.children_iter(target).next().unwrap();
                        if matches!(
                            child_node,
                            AstNode::Function(FunctionType::Builtin(StdFloatFunc::Gamma), _)
                        ) {
                            Some(tree.children_iter(child_idx).next().unwrap().1)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                },
                &[AstNode::Function(
                    FunctionType::Builtin(StdFloatFunc::Lgamma),
                    nz!(1),
                )],
            ),
            (
                &|target, tree| {
                    if matches!(
                        tree[target],
                        AstNode::Function(FunctionType::Builtin(StdFloatFunc::Ln), _)
                    ) {
                        let (child_node, child_idx) = tree.children_iter(target).next().unwrap();
                        if matches!(
                            child_node,
                            AstNode::UnaryOp(UnaryOp::Fac)
                        ) {
                            Some(tree.children_iter(child_idx).next().unwrap().1)
                        } else {
                            None
                        }
                    } else {
                        None
                    }
                },
                &[
                    AstNode::Number(1.0),
                    AstNode::BinaryOp(BinaryOp::Add),
                    AstNode::Function(FunctionType::Builtin(StdFloatFunc::Lgamma), nz!(1)),
                ],
            ),
        ];
        let mut symbol_space: SubtreeCollection<AstNode<Self, V, F>> =
            SubtreeCollection::from_alloc(Vec::with_capacity(0));
        let mut idx = 2;
        while idx < tree.len() {
            for &(detector, replacement) in &detrep_pairs {
                if let Some(param) = detector(idx, &tree) {
                    symbol_space.extend_from_tree(&tree, param);
                    for node in replacement {
                        symbol_space.push(node.clone()).unwrap();
                    }
                    let sc_head = symbol_space.len() - 1;
                    tree.replace_from_sc_move(idx, &mut symbol_space, sc_head);
                    break;
                };
            }
            idx += 1;
        }
    }

    fn asarg(&self) -> Self::AsArg<'_> {
        *self
    }

    fn pow(self, rhs: Self) -> Self {
        self.powf(rhs)
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
            result *= k as f64;
            k -= 2;
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::StdFloatStabilityGuard as Sfsg;
    use super::*;
    use crate::syntax::MathAst;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, FromRepr)]
    #[repr(u8)]
    enum TestVar {
        X,
        Y,
    }

    #[test]
    fn std_float_stability_guard() {
        assert_eq!(
            Sfsg::Number(0.1).apply_func_single(StdFloatFunc::Erf, |_| panic!()),
            Sfsg::Erf(0.1),
        );
        assert_eq!(
            Sfsg::Number(1.0).apply_binary_op(Sfsg::Erf(0.1), BinaryOp::Sub),
            Sfsg::Number(libm::erfc(0.1)),
        );
        assert_eq!(
            Sfsg::Number(2.0).apply_binary_op(Sfsg::Erf(0.1), BinaryOp::Sub),
            Sfsg::Number(2.0 - libm::erf(0.1)),
        );
        assert_eq!(
            Sfsg::Number(3.0).apply_func_single(StdFloatFunc::Exp, |_| panic!()),
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
            Sfsg::Number(1.0).apply_unary_op(UnaryOp::Fac),
            Sfsg::Gamma(2.0)
        );
        assert_eq!(
            Sfsg::Number(1.0).apply_func_single(StdFloatFunc::Gamma, |_| panic!()),
            Sfsg::Gamma(1.0)
        );
        assert_eq!(
            Sfsg::Gamma(1000.0).apply_func_single(StdFloatFunc::Ln, |_| panic!()),
            Sfsg::Number(libm::lgamma(1000.0))
        );
        assert_eq!(
            Sfsg::Number(8.0).apply_func_single(StdFloatFunc::Ln, |_| panic!()),
            Sfsg::Number(8f64.ln())
        );
        assert_eq!(
            Sfsg::Gamma(1000.0).apply_func_single(StdFloatFunc::Log2, |_| panic!()),
            Sfsg::Number(libm::lgamma(1000.0) / 2f64.ln())
        );
        assert_eq!(
            Sfsg::Gamma(1000.0).apply_func_single(StdFloatFunc::Log10, |_| panic!()),
            Sfsg::Number(libm::lgamma(1000.0) / 10f64.ln())
        );
        assert_eq!(
            Sfsg::Gamma(1000.0).apply_func_dual(
                Sfsg::Number(3.0),
                StdFloatFunc::Log,
                |_, _| panic!()
            ),
            Sfsg::Number(libm::lgamma(1000.0) / 3f64.ln())
        );
        assert_eq!(
            Sfsg::Gamma(10.0).apply_func_single(StdFloatFunc::Sin, f64::sin),
            Sfsg::Number(libm::tgamma(10.0).sin())
        );
        assert_eq!(
            Sfsg::Number(0.0).apply_binary_op(Sfsg::Number(1.0), BinaryOp::Add),
            Sfsg::OnePlus(0.0)
        );
        assert_eq!(
            Sfsg::OnePlus(8.0).apply_func_single(StdFloatFunc::Ln, |_| panic!()),
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
            Sfsg::Number(10.0).apply_binary_op(Sfsg::Number(23.0), BinaryOp::Pow),
            Sfsg::Number(1e23)
        );
        assert_eq!(
            Sfsg::Number(2.0).apply_binary_op(Sfsg::Number(100.0), BinaryOp::Pow),
            Sfsg::Number(100f64.exp2())
        );
        assert_eq!(
            Sfsg::Number(977.0).apply_func_dual(
                Sfsg::Number(10.0),
                StdFloatFunc::Log,
                |_, _| panic!()
            ),
            Sfsg::Number(977f64.log10())
        );
        assert_eq!(
            Sfsg::Number(888.0).apply_func_dual(
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
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Erf), nz!(1)),
                AstNode::BinaryOp(BinaryOp::Sub),
            ]),
            vec![
                AstNode::Variable(TestVar::X),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Erfc), nz!(1)),
            ]
        );
        assert_eq!(
            usf(&[
                AstNode::Variable(TestVar::X),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Gamma), nz!(1)),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Ln), nz!(1)),
                AstNode::Variable(TestVar::Y),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Gamma), nz!(1)),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Ln), nz!(1)),
                AstNode::BinaryOp(BinaryOp::Sub),
                AstNode::Variable(TestVar::X),
                AstNode::Variable(TestVar::Y),
                AstNode::BinaryOp(BinaryOp::Sub),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Gamma), nz!(1)),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Ln), nz!(1)),
                AstNode::BinaryOp(BinaryOp::Sub),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Exp), nz!(1))
            ]),
            vec![
                AstNode::Variable(TestVar::X),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Lgamma), nz!(1)),
                AstNode::Variable(TestVar::Y),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Lgamma), nz!(1)),
                AstNode::BinaryOp(BinaryOp::Sub),
                AstNode::Variable(TestVar::X),
                AstNode::Variable(TestVar::Y),
                AstNode::BinaryOp(BinaryOp::Sub),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Lgamma), nz!(1)),
                AstNode::BinaryOp(BinaryOp::Sub),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Exp), nz!(1))
            ]
        );
        assert_eq!(
            usf(&[
                AstNode::Variable(TestVar::X),
                AstNode::UnaryOp(UnaryOp::Fac),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Ln), nz!(1)),
            ]),
            vec![
                AstNode::Variable(TestVar::X),
                AstNode::Number(1.0),
                AstNode::BinaryOp(BinaryOp::Add),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Lgamma), nz!(1)),
            ]
        );
        assert_eq!(
            usf(&[
                AstNode::Variable(TestVar::X),
                AstNode::Number(2.0),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Log), nz!(2)),
            ]),
            vec![
                AstNode::Variable(TestVar::X),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Log2), nz!(1)),
            ]
        );
        assert_eq!(
            usf(&[
                AstNode::Variable(TestVar::X),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Sqrt), nz!(1)),
                AstNode::Number(2.0),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Log), nz!(2)),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Abs), nz!(1)),
            ]),
            vec![
                AstNode::Variable(TestVar::X),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Sqrt), nz!(1)),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Log2), nz!(1)),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Abs), nz!(1)),
            ]
        );
        assert_eq!(
            usf(&[
                AstNode::Number(2.0),
                AstNode::Variable(TestVar::X),
                AstNode::BinaryOp(BinaryOp::Pow),
            ]),
            vec![
                AstNode::Variable(TestVar::X),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Exp2), nz!(1)),
            ]
        );
        assert_eq!(
            usf(&[
                AstNode::Number(10.0),
                AstNode::Variable(TestVar::X),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Abs), nz!(1)),
                AstNode::BinaryOp(BinaryOp::Pow),
            ]),
            vec![
                AstNode::Variable(TestVar::X),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Abs), nz!(1)),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Exp10), nz!(1)),
            ]
        );
        assert_eq!(
            usf(&[
                AstNode::Variable(TestVar::X),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Sin), nz!(1)),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Exp), nz!(1)),
                AstNode::Number(1.0),
                AstNode::BinaryOp(BinaryOp::Sub),
            ]),
            vec![
                AstNode::Variable(TestVar::X),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Sin), nz!(1)),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Expm1), nz!(1)),
            ]
        );
        assert_eq!(
            usf(&[
                AstNode::Variable(TestVar::Y),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Abs), nz!(1)),
                AstNode::Number(1.0),
                AstNode::BinaryOp(BinaryOp::Add),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Ln), nz!(1)),
            ]),
            vec![
                AstNode::Variable(TestVar::Y),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Abs), nz!(1)),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Ln1p), nz!(1)),
            ]
        );
        assert_eq!(
            usf(&[
                AstNode::Number(1.0),
                AstNode::Variable(TestVar::Y),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Abs), nz!(1)),
                AstNode::BinaryOp(BinaryOp::Add),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Ln), nz!(1)),
            ]),
            vec![
                AstNode::Variable(TestVar::Y),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Abs), nz!(1)),
                AstNode::Function(FunctionType::Builtin(StdFloatFunc::Ln1p), nz!(1)),
            ]
        );
    }
}
