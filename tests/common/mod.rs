use std::fmt::Display;

use math_eval::{
    BinaryOp, FunctionPointer, UnaryOp, VariableStore,
    number::NativeFunction,
    postfix_tree::Node,
    syntax::{AstNode, FunctionType, MathAst},
};
use strum::{EnumIter, IntoEnumIterator};

#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumIter)]
pub(crate) enum MyVar {
    X,
    Y,
    Z,
    T,
}

impl MyVar {
    pub(crate) fn parse(input: &str) -> Option<Self> {
        match input {
            "x" => Some(Self::X),
            "y" => Some(Self::Y),
            "z" => Some(Self::Z),
            "t" => Some(Self::T),
            _ => None,
        }
    }
}

impl Display for MyVar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MyVar::X => f.write_str("x"),
            MyVar::Y => f.write_str("y"),
            MyVar::Z => f.write_str("z"),
            MyVar::T => f.write_str("t"),
        }
    }
}

pub(crate) struct MyStore(pub(crate) [f64; 4]);

impl VariableStore<f64, MyVar> for MyStore {
    fn get(&self, var: MyVar) -> f64 {
        match var {
            MyVar::X => self.0[0],
            MyVar::Y => self.0[1],
            MyVar::Z => self.0[2],
            MyVar::T => self.0[3],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumIter)]
pub(crate) enum MyFunc {
    Deg2Rad,
    Clamp,
    Digits,
}

impl MyFunc {
    pub(crate) fn parse(input: &str) -> Option<(Self, u8, Option<u8>)> {
        match input {
            "deg2rad" => Some((MyFunc::Deg2Rad, 1, Some(1))),
            "clamp" => Some((MyFunc::Clamp, 3, Some(3))),
            "digits" => Some((MyFunc::Digits, 1, None)),
            _ => None,
        }
    }

    pub(crate) fn as_pointer(self) -> FunctionPointer<'static, f64> {
        match self {
            MyFunc::Deg2Rad => FunctionPointer::Single(|x: f64| x.to_radians()),
            MyFunc::Clamp => {
                FunctionPointer::Triple(|x: f64, min: f64, max: f64| x.min(max).max(min))
            }
            MyFunc::Digits => FunctionPointer::Flexible(|values: &[f64]| {
                values
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| 10f64.powi(i as i32) * v)
                    .sum()
            }),
        }
    }
}

impl Display for MyFunc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MyFunc::Deg2Rad => f.write_str("deg2rad"),
            MyFunc::Clamp => f.write_str("clamp"),
            MyFunc::Digits => f.write_str("digits"),
        }
    }
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

fn rand_nf_1p() -> NativeFunction {
    use NativeFunction::*;
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
    match fastrand::u8(0..10) {
        0..6 => UnaryOp::Neg,
        6..9 => UnaryOp::Fac,
        9 => UnaryOp::DoubleFac,
        _ => unreachable!(),
    }
}

fn rand_nf_2p() -> NativeFunction {
    match fastrand::u8(0..11) {
        0..5 => NativeFunction::Min,
        5..10 => NativeFunction::Max,
        10 => NativeFunction::Log,
        _ => unreachable!(),
    }
}

fn rand_binaryop() -> BinaryOp {
    use BinaryOp::*;
    *weighted_choice(&[(Add, 3), (Sub, 2), (Mul, 3), (Div, 2), (Pow, 2), (Mod, 1)]).unwrap()
}

#[derive(Debug, Clone)]
struct AstGen {
    rem_nodes: usize,
    orphans: usize,
}

const CHILD_COUNT_WEIGHTS: [(u8, u8); 4] = [(1, 4), (2, 6), (3, 2), (4, 1)];

impl AstGen {
    fn new(target: usize) -> AstGen {
        AstGen {
            rem_nodes: target,
            orphans: 0,
        }
    }

    fn rand_branch(&self, exclude_uniary: bool) -> AstNode<f64, MyVar, MyFunc> {
        let cap = self.orphans.min(CHILD_COUNT_WEIGHTS.len()) as usize;
        let selection = if exclude_uniary {
            &CHILD_COUNT_WEIGHTS[1..cap]
        } else {
            &CHILD_COUNT_WEIGHTS[..cap]
        };
        match *weighted_choice(selection).unwrap() {
            1 => match fastrand::u8(0..10) {
                0..6 => AstNode::Function(FunctionType::Native(rand_nf_1p()), 1),
                6..9 => AstNode::UnaryOp(rand_unaryop()),
                9 => AstNode::Function(FunctionType::Custom(MyFunc::Deg2Rad), 1),
                _ => unreachable!(),
            },
            2 => {
                if fastrand::u8(0..20) == 19 {
                    AstNode::Function(FunctionType::Native(rand_nf_2p()), 2)
                } else {
                    AstNode::BinaryOp(rand_binaryop())
                }
            }
            3 => AstNode::Function(
                fastrand::choice([
                    NativeFunction::Min.into(),
                    NativeFunction::Max.into(),
                    FunctionType::Custom(MyFunc::Digits),
                    FunctionType::Custom(MyFunc::Clamp),
                ])
                .unwrap(),
                3,
            ),
            4 => AstNode::Function(
                fastrand::choice([
                    NativeFunction::Min.into(),
                    NativeFunction::Max.into(),
                    FunctionType::Custom(MyFunc::Digits),
                ])
                .unwrap(),
                4,
            ),
            _ => unreachable!(),
        }
    }

    fn rand_leaf() -> AstNode<f64, MyVar, MyFunc> {
        match fastrand::u8(0..3) {
            0..2 => AstNode::Number(fastrand::u8(0..10) as f64 / 10.0),
            2 => AstNode::Variable(fastrand::choice(MyVar::iter()).unwrap()),
            _ => unreachable!(),
        }
    }

    fn branch_chance(&self) -> f32 {
        1.0 - 1.0 / (1.0 + (self.orphans as f32) / 4.0)
    }
}

impl Iterator for AstGen {
    type Item = AstNode<f64, MyVar, MyFunc>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.rem_nodes > 0 {
            self.rem_nodes -= 1;
            if fastrand::f32() < self.branch_chance() {
                let node = self.rand_branch(false);
                self.orphans -= node.children() - 1;
                Some(node)
            } else {
                self.orphans += 1;
                Some(Self::rand_leaf())
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

pub(crate) fn rand_ast(size: usize) -> MathAst<f64, MyVar, MyFunc> {
    MathAst::from_nodes(AstGen::new(size))
}
