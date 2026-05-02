#![allow(dead_code)]

use math_eval::{
    BinaryOp, FunctionIdentifier, UnaryOp, VariableIdentifier,
    number::std_float::StdFloatFunc,
    postfix_tree::Node,
    syntax::{AstNode, FunctionType},
};

pub fn rand_f64() -> f64 {
    f64::from_bits(fastrand::u64(..))
}

fn rand_bf_1p() -> StdFloatFunc {
    use StdFloatFunc::*;
    *weighted_choice(&[
        (Sin, 4),
        (Cos, 4),
        (Tan, 3),
        (Cot, 1),
        (Asin, 2),
        (Acos, 2),
        (Atan, 3),
        (Acot, 2),
        (Log2, 2),
        (Log10, 2),
        (Ln, 3),
        (Exp, 2),
        (Floor, 2),
        (Ceil, 2),
        (Round, 2),
        (Trunc, 1),
        (Frac, 1),
        (Abs, 4),
        (Sign, 2),
        (Sqrt, 4),
        (Cbrt, 1),
    ])
    .unwrap()
}

fn rand_unaryop() -> UnaryOp {
    match fastrand::u8(0..20) {
        0..16 => UnaryOp::Neg,
        16..19 => UnaryOp::Fac,
        19 => UnaryOp::DoubleFac,
        _ => unreachable!(),
    }
}

fn rand_bf_2p() -> StdFloatFunc {
    match fastrand::u8(0..11) {
        0..5 => StdFloatFunc::Min,
        5..10 => StdFloatFunc::Max,
        10 => StdFloatFunc::Log,
        _ => unreachable!(),
    }
}

fn rand_binaryop() -> BinaryOp {
    use BinaryOp::*;
    *weighted_choice(&[(Add, 3), (Sub, 2), (Mul, 3), (Div, 2), (Pow, 2), (Mod, 1)]).unwrap()
}

#[inline]
fn weighted_choice<'a, T>(items: &'a [(T, u8)]) -> Option<&'a T> {
    let Some(last) = items.last() else {
        return None;
    };
    let weight_sum = items.iter().map(|(_, w)| w).sum();
    let rn = fastrand::u8(0..weight_sum);
    let mut acu = 0;
    for (e, w) in &items[..items.len() - 1] {
        if rn < w + acu {
            return Some(e);
        } else {
            acu += w;
        }
    }
    Some(&last.0)
}

#[macro_export]
macro_rules! nz {
    ($v: literal) => {
        const { std::num::NonZero::new($v).unwrap() }
    };
}

const CHILD_COUNT_WEIGHTS: [(u8, u8); 4] = [(1, 5), (2, 8), (3, 2), (4, 1)];

#[derive(Debug, Clone)]
pub(crate) struct AstGen<'a, V: VariableIdentifier, F: FunctionIdentifier> {
    rem_nodes: usize,
    orphans: usize,
    variables: &'a [V],
    functions_1p: &'a [F],
    functions_2p: &'a [F],
    functions_3p: &'a [F],
    functions_4p: &'a [F],
}

impl<'a, V: VariableIdentifier, F: FunctionIdentifier> AstGen<'a, V, F> {
    pub(crate) fn new(
        target: usize,
        variables: &'a [V],
        functions_1p: &'a [F],
        functions_2p: &'a [F],
        functions_3p: &'a [F],
        functions_4p: &'a [F],
    ) -> AstGen<'a, V, F> {
        AstGen {
            rem_nodes: target,
            orphans: 0,
            variables,
            functions_1p,
            functions_2p,
            functions_3p,
            functions_4p,
        }
    }

    fn rand_branch(&self, exclude_uniary: bool) -> AstNode<f64, V, F> {
        let cap = self.orphans.min(CHILD_COUNT_WEIGHTS.len()) as usize;
        let selection = if exclude_uniary {
            &CHILD_COUNT_WEIGHTS[1..cap]
        } else {
            &CHILD_COUNT_WEIGHTS[..cap]
        };
        match *weighted_choice(selection).unwrap() {
            1 => match fastrand::u8(0..10) {
                0..6 => AstNode::Function(FunctionType::Builtin(rand_bf_1p()), nz!(1)),
                6..9 => AstNode::UnaryOp(rand_unaryop()),
                9 => AstNode::Function(
                    FunctionType::Custom(*fastrand::choice(self.functions_1p).unwrap()),
                    nz!(1),
                ),
                _ => unreachable!(),
            },
            2 => match fastrand::u8(0..20) {
                0..16 => AstNode::BinaryOp(rand_binaryop()),
                16..19 => AstNode::Function(FunctionType::Builtin(rand_bf_2p()), nz!(2)),
                19 => AstNode::Function(
                    FunctionType::Custom(*fastrand::choice(self.functions_2p).unwrap()),
                    nz!(2),
                ),
                _ => unreachable!(),
            },
            3 => AstNode::Function(
                match fastrand::u8(0..7) {
                    0 => FunctionType::Builtin(StdFloatFunc::Min),
                    1 => FunctionType::Builtin(StdFloatFunc::Max),
                    2..7 => FunctionType::Custom(*fastrand::choice(self.functions_3p).unwrap()),
                    _ => unreachable!(),
                },
                nz!(3),
            ),
            4 => AstNode::Function(
                match fastrand::u8(0..7) {
                    0 => FunctionType::Builtin(StdFloatFunc::Min),
                    1 => FunctionType::Builtin(StdFloatFunc::Max),
                    2..7 => FunctionType::Custom(*fastrand::choice(self.functions_4p).unwrap()),
                    _ => unreachable!(),
                },
                nz!(4),
            ),
            _ => unreachable!(),
        }
    }

    fn rand_leaf(&self) -> AstNode<f64, V, F> {
        match fastrand::u8(0..=7) {
            0..2 => AstNode::Number(fastrand::i8(-10..10) as f64),
            2 => AstNode::Number(fastrand::i16(-1000..1000) as f64 / 100.0),
            3..6 => AstNode::Variable(*fastrand::choice(self.variables).unwrap()),
            6 => AstNode::Number(fastrand::i32(..) as f64),
            7 => AstNode::Number(rand_f64()),
            _ => unreachable!(),
        }
    }

    fn branch_chance(&self) -> f32 {
        1.0 - 1.0 / (1.0 + (self.orphans as f32) / 4.0)
    }
}

impl<V, F> Iterator for AstGen<'_, V, F>
where
    V: VariableIdentifier,
    F: FunctionIdentifier,
{
    type Item = AstNode<f64, V, F>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.rem_nodes > 0 {
            self.rem_nodes -= 1;
            if fastrand::f32() < self.branch_chance() {
                let node = self.rand_branch(false);
                self.orphans -= node.children() - 1;
                Some(node)
            } else {
                self.orphans += 1;
                Some(self.rand_leaf())
            }
        } else {
            if self.orphans > 1 {
                let node = self.rand_branch(true);
                self.orphans -= node.children() - 1;
                Some(node)
            } else {
                None
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.rem_nodes, Some(self.rem_nodes + self.orphans - 1))
    }
}
