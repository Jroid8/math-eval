#![allow(dead_code)]

use math_eval::{
    BinaryOp, FunctionIdentifier, UnaryOp, VariableIdentifier,
    number::{Number, std_float::StdFloatFunc},
    postfix_tree::Node,
    syntax::{AstNode, FunctionType},
};

pub fn rand_f64() -> f64 {
    f64::from_bits(fastrand::u64(..))
}

fn rand_bf_1p() -> <f64 as Number>::BuiltinFuncId {
    use StdFloatFunc::*;
    #[cfg(feature = "libm")]
    use math_eval::number::libm_ext::LibmFunc::*;
    fastrand::choice([
        Sin.into(),
        Cos.into(),
        Tan.into(),
        Cot.into(),
        Asin.into(),
        Acos.into(),
        Atan.into(),
        Acot.into(),
        Sinh.into(),
        Cosh.into(),
        Tanh.into(),
        Coth.into(),
        Asinh.into(),
        Acosh.into(),
        Atanh.into(),
        Acoth.into(),
        Log2.into(),
        Log10.into(),
        Ln1p.into(),
        Ln.into(),
        Exp.into(),
        Expm1.into(),
        Floor.into(),
        Ceil.into(),
        Round.into(),
        Trunc.into(),
        Frac.into(),
        Abs.into(),
        Sign.into(),
        Sqrt.into(),
        Cbrt.into(),
        #[cfg(feature = "libm")]
        Erf.into(),
        #[cfg(feature = "libm")]
        Erfc.into(),
        #[cfg(feature = "libm")]
        Exp10.into(),
        #[cfg(feature = "libm")]
        Gamma.into(),
        #[cfg(feature = "libm")]
        Lgamma.into(),
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
    fastrand::choice([StdFloatFunc::Min, StdFloatFunc::Max, StdFloatFunc::Log]).unwrap()
}

fn rand_binaryop() -> BinaryOp {
    use BinaryOp::*;
    fastrand::choice([Add, Sub, Mul, Div, Pow, Mod]).unwrap()
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
                16..19 => AstNode::Function(FunctionType::Builtin(rand_bf_2p().into()), nz!(2)),
                19 => AstNode::Function(
                    FunctionType::Custom(*fastrand::choice(self.functions_2p).unwrap()),
                    nz!(2),
                ),
                _ => unreachable!(),
            },
            3 => AstNode::Function(
                match fastrand::u8(0..7) {
                    0 => FunctionType::Builtin(StdFloatFunc::Min.into()),
                    1 => FunctionType::Builtin(StdFloatFunc::Max.into()),
                    2..7 => FunctionType::Custom(*fastrand::choice(self.functions_3p).unwrap()),
                    _ => unreachable!(),
                },
                nz!(3),
            ),
            4 => AstNode::Function(
                match fastrand::u8(0..7) {
                    0 => FunctionType::Builtin(StdFloatFunc::Min.into()),
                    1 => FunctionType::Builtin(StdFloatFunc::Max.into()),
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
