use std::{f64::consts::PI, ops::RangeInclusive};

use fastrand_contrib::f64_range;
use math_eval::{
    FunctionPointer, VariableStore, quick_expr::QuickExpr, syntax::MathAst, tokenizer::TokenStream,
};

#[test]
fn correctness() {
    for case in CASES {
        case.perform();
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MyVar {
    X,
    Y,
    Sigma,
}

impl MyVar {
    fn parse(input: &str) -> Option<MyVar> {
        match input {
            "x" => Some(MyVar::X),
            "y" => Some(MyVar::Y),
            "σ" => Some(MyVar::Sigma),
            _ => None,
        }
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MyFunc {
    Rad2Deg,
    Average,
    Random,
    Terra,
}

impl MyFunc {
    fn parse(input: &str) -> Option<(MyFunc, u8, Option<u8>)> {
        match input {
            "rad2deg" => Some((MyFunc::Rad2Deg, 1, Some(1))),
            "avg" | "average" => Some((MyFunc::Average, 1, None)),
            "rand" | "random" => Some((MyFunc::Random, 1, Some(1))),
            "terra" => Some((MyFunc::Terra, 2, Some(2))),
            _ => None,
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

fn parse_consts(input: &str) -> Option<f64> {
    fastrand::u64(..);
    match input {
        "g" | "G" => Some(G),
        "c" => Some(C),
        "w" => Some(W),
        _ => None,
    }
}

struct TestCase {
    rust_func: fn(f64, f64, f64, &CtxfulMyFuncs) -> f64,
    str_expr: &'static str,
    x_range: RangeInclusive<f64>,
    y_range: RangeInclusive<f64>,
    sigma_range: RangeInclusive<f64>,
}

const TOLERANCE: f64 = f64::EPSILON * 4.0;

impl TestCase {
    fn perform(&self) {
        let ast = MathAst::new(
            &TokenStream::new(self.str_expr).unwrap(),
            parse_consts,
            MyFunc::parse,
            MyVar::parse,
        )
        .unwrap();
        let ctxful_funcs = CtxfulMyFuncs::new();
        let mf2p = |f| MyFunc::to_pointer(f, &ctxful_funcs);
        let quick = QuickExpr::new(ast.clone(), mf2p);
        let mut stack = Vec::with_capacity(quick.stack_req_capacity().unwrap());
        for _ in 0..200 {
            let x = f64_range(self.x_range.clone());
            let y = f64_range(self.y_range.clone());
            let sigma = f64_range(self.sigma_range.clone());
            let comp_res = (self.rust_func)(x, y, sigma, &ctxful_funcs);
            let quick_res = quick.eval(&MyStore { x, y, sigma }, &mut stack).unwrap();
            let ast_res = ast.eval(mf2p, &MyStore { x, y, sigma });
            let syno_res = MathAst::parse_and_eval(
                &TokenStream::new(self.str_expr).unwrap(),
                parse_consts,
                MyFunc::parse,
                MyVar::parse,
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

const CASES: [TestCase; 7] = [
    TestCase {
        rust_func: expr1,
        str_expr: "cos(x*pi/10*(1.3+sin(σ/10))+sin(y*pi*sin(σ/17)+16*sin(σ))+2σ)+0.05",
        x_range: -1e10..=1e10,
        y_range: -1e10..=1e10,
        sigma_range: -1e10..=1e10,
    },
    TestCase {
        rust_func: expr2,
        str_expr: "rad2deg(σ - atan(y / x))",
        x_range: 0.0..=1e10,
        y_range: 1e-10..=1e10,
        sigma_range: 0.1..=PI,
    },
    TestCase {
        rust_func: expr3,
        str_expr: "ceil(x * log10(ceil(σ)))",
        x_range: 0.0..=1e10,
        y_range: 0.0..=0.0,
        sigma_range: 1.1..=1e3,
    },
    TestCase {
        rust_func: expr4,
        str_expr: "exp(ln(x)+ln(x+1)+ln(x+2)+ln(x+3)+ln(x+4)-(ln(y)+ln(y+1)+ln(y+2)+ln(y+3)+ln(y+4)))",
        x_range: -1e10..=1e10,
        y_range: -1e10..=1e10,
        sigma_range: 0.0..=0.0,
    },
    TestCase {
        rust_func: expr5,
        str_expr: "2Gx/c^2",
        x_range: 1.0..=1e20,
        y_range: 0.0..=0.0,
        sigma_range: 0.0..=0.0,
    },
    TestCase {
        rust_func: expr6,
        str_expr: "max(rand(x+wy)/10,terra(2x,2y))",
        x_range: -22.0..=22.0,
        y_range: -22.0..=22.0,
        sigma_range: 0.0..=0.0,
    },
    TestCase {
        rust_func: expr7,
        str_expr: "5avg(x,y,σ,100rand(x))",
        x_range: -100.0..=100.0,
        y_range: -100.0..=100.0,
        sigma_range: -100.0..=100.0,
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
    5.0 * average(&[x, y, s, 100.0 * (ctx.rand)(x)])
}
