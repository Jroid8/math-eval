use std::{f64::consts::PI, fmt::Display, ops::RangeInclusive};

use fastrand_contrib::f64_range;
use math_eval::{
    FunctionPointer, VariableStore,
    quick_expr::QuickExpr,
    syntax::{CfInfo, MathAst},
    tokenizer::{StandardFloatRecognizer as Sfr, TokenStream},
    trie::{NameTrie, TrieNode},
};
use strum::FromRepr;

use crate::common::{AstGen, rand_f64};

mod common;

#[test]
fn correctness_baseline() {
    for case in BASELINE_CASES {
        case.perform();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, FromRepr)]
#[repr(u8)]
enum MyVar {
    X,
    Y,
    Sigma,
}

impl Display for MyVar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            MyVar::X => "x",
            MyVar::Y => "y",
            MyVar::Sigma => "σ",
        })
    }
}

struct MyVarsNameTrie;

impl NameTrie<MyVar> for MyVarsNameTrie {
    fn nodes(&self) -> &[TrieNode] {
        &[
            TrieNode::Branch('x', 1),
            TrieNode::Leaf(MyVar::X as u32),
            TrieNode::Branch('y', 1),
            TrieNode::Leaf(MyVar::Y as u32),
            TrieNode::Branch('σ', 1),
            TrieNode::Leaf(MyVar::Sigma as u32),
        ]
    }
    fn leaf_to_value(&self, leaf: u32) -> MyVar {
        MyVar::from_repr(leaf as u8).unwrap()
    }
}

fn detrand_u64(mut x: u64, y: u64) -> u64 {
    x = x.wrapping_add(y);
    let y = y as u32;
    x ^= x.wrapping_shr(y);
    x = (x as u128 * 0xff51afd7ed558ccdu128) as u64;
    x ^= x.wrapping_shr(y >> 6);
    x = (x as u128 * 0xc4ceb9fe1a85ec53u128) as u64;
    x ^= x.wrapping_shr(y >> 12);
    x = (x as u128 * y as u128) as u64;
    x ^ x.wrapping_shr(y >> 18)
}

fn detrand_f64(x: f64, y: u64) -> f64 {
    const EXP_VALUE: u64 = 0b1111111111 << 52;
    const MANTISSA_MASK: u64 = (1 << 52) - 1;
    f64::from_bits(EXP_VALUE | detrand_u64(x.to_bits(), y) & MANTISSA_MASK) - 1.0
}

fn terra(x: f64, y: f64, coords: &[(f64, f64, f64)]) -> f64 {
    coords
        .iter()
        .map(|(x0, y0, z)| z / (1.0 + (x - x0).powi(2) + (y - y0).powi(2)))
        .sum()
}

fn average(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, FromRepr)]
#[repr(u8)]
enum MyFunc {
    Rad2Deg,
    Average,
    Random,
    Terra,
}

impl MyFunc {
    fn with_info(self) -> CfInfo<MyFunc> {
        match self {
            MyFunc::Rad2Deg => CfInfo::new(MyFunc::Rad2Deg, nz!(1), Some(nz!(1))),
            MyFunc::Average => CfInfo::new(MyFunc::Average, nz!(1), None),
            MyFunc::Random => CfInfo::new(MyFunc::Random, nz!(1), Some(nz!(1))),
            MyFunc::Terra => CfInfo::new(MyFunc::Terra, nz!(2), Some(nz!(2))),
        }
    }

    fn to_pointer<'a>(self, ctx: &'a CtxfulMyFuncs) -> FunctionPointer<'a, f64> {
        match self {
            MyFunc::Rad2Deg => FunctionPointer::<f64>::Single(f64::to_degrees),
            MyFunc::Average => FunctionPointer::Flexible(average),
            MyFunc::Random => FunctionPointer::<f64>::DynSingle(&ctx.rand),
            MyFunc::Terra => FunctionPointer::<f64>::DynDual(&ctx.terra),
        }
    }
}

impl Display for MyFunc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            MyFunc::Rad2Deg => "rad2deg",
            MyFunc::Average => "avg",
            MyFunc::Random => "rand",
            MyFunc::Terra => "terra",
        })
    }
}

struct MyFuncsNameTrie;

impl NameTrie<CfInfo<MyFunc>> for MyFuncsNameTrie {
    fn nodes(&self) -> &[TrieNode] {
        &[
            TrieNode::Branch('a', 9),
            TrieNode::Branch('v', 8),
            TrieNode::Branch('e', 5),
            TrieNode::Branch('r', 4),
            TrieNode::Branch('a', 3),
            TrieNode::Branch('g', 2),
            TrieNode::Branch('e', 1),
            TrieNode::Leaf(MyFunc::Average as u32),
            TrieNode::Branch('g', 1),
            TrieNode::Leaf(MyFunc::Average as u32),
            TrieNode::Branch('r', 13),
            TrieNode::Branch('a', 12),
            TrieNode::Branch('d', 5),
            TrieNode::Branch('2', 4),
            TrieNode::Branch('d', 3),
            TrieNode::Branch('e', 2),
            TrieNode::Branch('g', 1),
            TrieNode::Leaf(MyFunc::Rad2Deg as u32),
            TrieNode::Branch('n', 5),
            TrieNode::Branch('d', 4),
            TrieNode::Leaf(MyFunc::Random as u32),
            TrieNode::Branch('o', 2),
            TrieNode::Branch('m', 1),
            TrieNode::Leaf(MyFunc::Random as u32),
            TrieNode::Branch('t', 5),
            TrieNode::Branch('e', 4),
            TrieNode::Branch('r', 3),
            TrieNode::Branch('r', 2),
            TrieNode::Branch('a', 1),
            TrieNode::Leaf(MyFunc::Terra as u32),
        ]
    }
    fn leaf_to_value(&self, leaf: u32) -> CfInfo<MyFunc> {
        MyFunc::from_repr(leaf as u8).unwrap().with_info()
    }
}

struct CtxfulMyFuncs {
    rand: Box<dyn Fn(f64) -> f64>,
    terra: Box<dyn Fn(f64, f64) -> f64>,
}

impl CtxfulMyFuncs {
    fn new() -> Self {
        let seed = fastrand::u64(..);
        let coords = (0..fastrand::u8(1..8))
            .map(|_| {
                (
                    f64_range(-32.0..=32.0),
                    f64_range(-32.0..=32.0),
                    f64_range(-5.0..=5.0),
                )
            })
            .collect::<Vec<_>>();
        Self {
            rand: Box::new(move |x| detrand_f64(x, seed)),
            terra: Box::new(move |x, y| terra(x, y, &coords)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct MyStore {
    x: f64,
    y: f64,
    sigma: f64,
}

impl VariableStore<f64, MyVar> for MyStore {
    fn get<'a>(&'a self, var: MyVar) -> f64 {
        match var {
            MyVar::X => self.x,
            MyVar::Y => self.y,
            MyVar::Sigma => self.sigma,
        }
    }
}

const G: f64 = 6.6743e-11;
const C: f64 = 299792458.0;
const W: f64 = 520.0;

struct MyConstsNameTrie;

impl NameTrie<&'static f64> for MyConstsNameTrie {
    fn nodes(&self) -> &[TrieNode] {
        &[
            TrieNode::Branch('g', 1),
            TrieNode::Leaf(0),
            TrieNode::Branch('G', 1),
            TrieNode::Leaf(0),
            TrieNode::Branch('c', 1),
            TrieNode::Leaf(1),
            TrieNode::Branch('w', 1),
            TrieNode::Leaf(2),
        ]
    }

    fn leaf_to_value(&self, leaf: u32) -> &'static f64 {
        match leaf {
            0 => &G,
            1 => &C,
            2 => &W,
            _ => unreachable!(),
        }
    }
}

struct BaselineTestCase {
    rust_func: fn(f64, f64, f64, &CtxfulMyFuncs) -> f64,
    str_expr: &'static str,
    x_range: RangeInclusive<f64>,
    y_range: RangeInclusive<f64>,
    sigma_range: RangeInclusive<f64>,
}

const TOLERANCE: f64 = f64::EPSILON * 4.0;

impl BaselineTestCase {
    fn perform(&self) {
        let ast = MathAst::new(
            &TokenStream::new::<Sfr>(self.str_expr).unwrap(),
            &MyConstsNameTrie,
            &MyFuncsNameTrie,
            &MyVarsNameTrie,
        )
        .unwrap();
        let ctxful_funcs = CtxfulMyFuncs::new();
        let mf2p = |f| MyFunc::to_pointer(f, &ctxful_funcs);
        let quick = QuickExpr::new(ast.clone(), mf2p);
        let mut stack = Vec::with_capacity(quick.stack_req_capacity());
        for _ in 0..200 {
            let x = f64_range(self.x_range.clone());
            let y = f64_range(self.y_range.clone());
            let sigma = f64_range(self.sigma_range.clone());
            let comp_res = (self.rust_func)(x, y, sigma, &ctxful_funcs);
            let quick_res = quick.eval(&MyStore { x, y, sigma }, &mut stack);
            let ast_res = ast.eval(mf2p, &MyStore { x, y, sigma });
            let syno_res = MathAst::parse_and_eval(
                &TokenStream::new::<Sfr>(self.str_expr).unwrap(),
                &MyConstsNameTrie,
                &MyFuncsNameTrie,
                &MyVarsNameTrie,
                &MyStore { x, y, sigma },
                mf2p,
            )
            .unwrap();
            if (comp_res - ast_res).abs() > TOLERANCE
                || (comp_res - syno_res).abs() > TOLERANCE
                || (comp_res - quick_res).abs() > TOLERANCE
            {
                panic!(
                    "evaluation of \"{}\" produced incorrect result for values x={x}, y={y}, σ={sigma}.\ntrue:\t{comp_res}\nast:\t{ast_res}\nsyno:\t{syno_res}\nquick:\t{quick_res}",
                    self.str_expr
                )
            }
        }
    }
}

const BASELINE_CASES: [BaselineTestCase; 10] = [
    BaselineTestCase {
        rust_func: expr1,
        str_expr: "cos(x*pi/10*(1.3+sin(σ/10))+sin(y*pi*sin(σ/17)+16*sin(σ))+2σ)+0.05",
        x_range: -1e10..=1e10,
        y_range: -1e10..=1e10,
        sigma_range: -1e10..=1e10,
    },
    BaselineTestCase {
        rust_func: expr2,
        str_expr: "rad2deg(σ - atan(y / x))",
        x_range: 0.0..=1e10,
        y_range: 1e-10..=1e10,
        sigma_range: 0.1..=PI,
    },
    BaselineTestCase {
        rust_func: expr3,
        str_expr: "ceil(x * log10(ceil(σ)))",
        x_range: 0.0..=1e10,
        y_range: 0.0..=0.0,
        sigma_range: 1.1..=1e3,
    },
    BaselineTestCase {
        rust_func: expr4,
        str_expr: "exp(ln(x)+ln(x+1)+ln(x+2)+ln(x+3)+ln(x+4)-(ln(y)+ln(y+1)+ln(y+2)+ln(y+3)+ln(y+4)))",
        x_range: -1e10..=1e10,
        y_range: -1e10..=1e10,
        sigma_range: 0.0..=0.0,
    },
    BaselineTestCase {
        rust_func: expr5,
        str_expr: "2Gx/c^2",
        x_range: 1.0..=1e20,
        y_range: 0.0..=0.0,
        sigma_range: 0.0..=0.0,
    },
    BaselineTestCase {
        rust_func: expr6,
        str_expr: "max(rand(x+wy)/10,terra(2x,2y))",
        x_range: -22.0..=22.0,
        y_range: -22.0..=22.0,
        sigma_range: 0.0..=0.0,
    },
    BaselineTestCase {
        rust_func: expr7,
        x_range: -100.0..=100.0,
        str_expr: "5avg(x,y,σ,100randσ)",
        y_range: -100.0..=100.0,
        sigma_range: -100.0..=100.0,
    },
    BaselineTestCase {
        rust_func: expr8,
        str_expr: "ceil(x)!/(ceil(y)!(ceil(x) - ceil(y))!)",
        x_range: 50.0..=100.0,
        y_range: 1.0..=39.9,
        sigma_range: 0.0..=0.0,
    },
    BaselineTestCase {
        rust_func: expr9,
        str_expr: "|6x + 8y - σ|/10",
        x_range: -100.0..=100.0,
        y_range: -100.0..=100.0,
        sigma_range: -100.0..=100.0,
    },
    BaselineTestCase {
        rust_func: expr10,
        str_expr: "|x-y|rand(σ)+min(x,y)",
        x_range: -1e3..=1e3,
        y_range: -1e3..=1e3,
        sigma_range: 0.0..=1.0,
    },
];

fn expr1(x: f64, y: f64, s: f64, _ctx: &CtxfulMyFuncs) -> f64 {
    (x * PI / 10.0 * (1.3 + (s / 10.0).sin())
        + (y * PI * (s / 17.0).sin() + 16.0 * s.sin()).sin()
        + 2.0 * s)
        .cos()
        + 0.05
}

fn expr2(x: f64, y: f64, s: f64, _ctx: &CtxfulMyFuncs) -> f64 {
    (s - (y / x).atan()).to_degrees()
}

fn expr3(x: f64, _y: f64, s: f64, _ctx: &CtxfulMyFuncs) -> f64 {
    (x * s.ceil().log10()).ceil()
}

fn expr4(x: f64, y: f64, _s: f64, _ctx: &CtxfulMyFuncs) -> f64 {
    let lnx: f64 = (0..5).map(|i| (x + i as f64).ln()).sum();
    let lny: f64 = (0..5).map(|i| (y + i as f64).ln()).sum();
    (lnx - lny).exp()
}

fn expr5(x: f64, _y: f64, _s: f64, _ctx: &CtxfulMyFuncs) -> f64 {
    2.0 * G * x / C.powi(2)
}

fn expr6(x: f64, y: f64, _s: f64, ctx: &CtxfulMyFuncs) -> f64 {
    (ctx.terra)(2.0 * x, 2.0 * y).max((ctx.rand)(x + W * y) / 10.0)
}

fn expr7(x: f64, y: f64, s: f64, ctx: &CtxfulMyFuncs) -> f64 {
    5.0 * average(&[x, y, s, 100.0 * (ctx.rand)(s)])
}

fn expr8(x: f64, y: f64, _s: f64, _ctx: &CtxfulMyFuncs) -> f64 {
    let fact = |x| <f64 as math_eval::number::Number>::factorial(x);
    fact(x.ceil()) / (fact(y.ceil()) * fact(x.ceil() - y.ceil()))
}

fn expr9(x: f64, y: f64, s: f64, _ctx: &CtxfulMyFuncs) -> f64 {
    (6.0 * x + 8.0 * y - s).abs() / 10.0
}

fn expr10(x: f64, y: f64, s: f64, ctx: &CtxfulMyFuncs) -> f64 {
    (x - y).abs() * (ctx.rand)(s) + x.min(y)
}

#[test]
fn correctness_alignment() {
    for i in 4..120 {
        let ast = MathAst::from_nodes(AstGen::new(
            i / 2,
            &[MyVar::X, MyVar::Y, MyVar::Sigma],
            &[MyFunc::Rad2Deg, MyFunc::Random],
            &[MyFunc::Average, MyFunc::Terra],
            &[MyFunc::Average],
            &[MyFunc::Average],
        ));
        let ctxful_funcs = CtxfulMyFuncs::new();
        let mf2p = |f| MyFunc::to_pointer(f, &ctxful_funcs);
        let vars = MyStore {
            x: rand_f64(),
            y: rand_f64(),
            sigma: rand_f64(),
        };
        let ast_res = ast.eval(mf2p, &vars);
        let quick = QuickExpr::new(ast.clone(), mf2p);
        let mut stack = Vec::with_capacity(quick.stack_req_capacity());
        let quick_res = quick.eval(&vars, &mut stack);
        if (ast_res - quick_res).abs() > TOLERANCE {
            panic!(
                "evaluation of \"{}\" produced inconsistent results for values x={}, y={}, σ={}.\nast:\t{ast_res}\nquick:\t{quick_res}\nast (debug): {:?}",
                ast,
                vars.x,
                vars.y,
                vars.sigma,
                ast.as_tree().postorder_iter().collect::<Vec<_>>()
            )
        }
    }
}
