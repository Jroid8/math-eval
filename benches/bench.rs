use std::time::Duration;

use criterion::{
    Bencher, BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};
use fastrand::Rng;
use math_eval::{
    FunctionPointer, VariableStore,
    quick_expr::QuickExpr,
    syntax::{CfInfo, MathAst},
    tokenizer::{StandardFloatRecognizer as Sfr, TokenStream},
    trie::{EmptyNameTrie, NameTrie, TrieNode},
};
use meval::{Context, Expr};
use strum::FromRepr;

fn dist(x: f64, y: f64) -> f64 {
    (x * x + y * y).sqrt()
}

fn average(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
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

fn meval_calc_bencher(b: &mut Bencher<'_>, input: &str) {
    let mut context = Context::new();
    let seed = fastrand::u64(..);
    context.func2("dist", dist);
    context.funcn("avg", average, 1..);
    context.func("log", f64::log10);
    context.func("rand", move |x| detrand_f64(x, seed));
    let expr = input
        .parse::<Expr>()
        .unwrap()
        .bind3_with_context(context, "x", "y", "t")
        .unwrap();
    let mut rng = fastrand::Rng::new();
    b.iter_batched(
        || MyStore::rand(&mut rng),
        |vars| black_box(expr(vars.x, vars.y, vars.t)),
        criterion::BatchSize::SmallInput,
    );
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, FromRepr)]
#[repr(u8)]
enum MyVar {
    X,
    Y,
    T,
}

struct MyVarsNameTrie;

impl NameTrie<MyVar> for MyVarsNameTrie {
    fn nodes(&self) -> &[TrieNode] {
        &[
            TrieNode::Branch('x', 1),
            TrieNode::Leaf(MyVar::X as u32),
            TrieNode::Branch('y', 1),
            TrieNode::Leaf(MyVar::Y as u32),
            TrieNode::Branch('t', 1),
            TrieNode::Leaf(MyVar::T as u32),
        ]
    }
    fn leaf_to_value(&self, leaf: u32) -> MyVar {
        MyVar::from_repr(leaf as u8).unwrap()
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, FromRepr)]
#[repr(u8)]
enum MyFunc {
    Dist,
    Average,
    Random,
}

macro_rules! nz {
    ($v: literal) => {
        const { std::num::NonZero::new($v).unwrap() }
    };
}

impl MyFunc {
    fn with_info(self) -> CfInfo<Self> {
        match self {
            MyFunc::Dist => CfInfo::new(MyFunc::Dist, nz!(2), Some(nz!(2))),
            MyFunc::Random => CfInfo::new(MyFunc::Random, nz!(1), Some(nz!(1))),
            MyFunc::Average => CfInfo::new(MyFunc::Average, nz!(1), None),
        }
    }

    fn to_pointer<'a>(self, rand_fn: &'a impl Fn(f64) -> f64) -> FunctionPointer<'a, f64> {
        match self {
            MyFunc::Dist => FunctionPointer::<f64>::Dual(dist),
            MyFunc::Average => FunctionPointer::Flexible(average),
            MyFunc::Random => FunctionPointer::<f64>::DynSingle(rand_fn),
        }
    }
}

struct MyFuncsNameTrie;

impl NameTrie<CfInfo<MyFunc>> for MyFuncsNameTrie {
    fn nodes(&self) -> &[TrieNode] {
        &[
            TrieNode::Branch('d', 4),
            TrieNode::Branch('i', 3),
            TrieNode::Branch('s', 2),
            TrieNode::Branch('t', 1),
            TrieNode::Leaf(MyFunc::Dist as u32),
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
            TrieNode::Branch('r', 7),
            TrieNode::Branch('a', 6),
            TrieNode::Branch('n', 5),
            TrieNode::Branch('d', 4),
            TrieNode::Leaf(MyFunc::Random as u32),
            TrieNode::Branch('o', 2),
            TrieNode::Branch('m', 1),
            TrieNode::Leaf(MyFunc::Random as u32),
        ]
    }

    fn leaf_to_value(&self, leaf: u32) -> CfInfo<MyFunc> {
        MyFunc::from_repr(leaf as u8).unwrap().with_info()
    }
}

#[derive(Debug, Clone, Copy)]
struct MyStore {
    x: f64,
    y: f64,
    t: f64,
}

impl MyStore {
    fn rand(rng: &mut Rng) -> Self {
        Self {
            x: (rng.f64() - 0.5) * 1e7,
            y: (rng.f64() - 0.5) * 1e7,
            t: rng.f64() * 1e7,
        }
    }
}

impl VariableStore<f64, MyVar> for MyStore {
    fn get<'a>(&'a self, var: MyVar) -> f64 {
        match var {
            MyVar::X => self.x,
            MyVar::Y => self.y,
            MyVar::T => self.t,
        }
    }
}

fn matheval_calc_bencher(b: &mut Bencher<'_>, qexpr: &QuickExpr<'_, f64, MyVar, MyFunc>) {
    let mut stack = Vec::with_capacity(qexpr.stack_req_capacity());
    let mut rng = fastrand::Rng::new();
    b.iter_batched(
        || MyStore::rand(&mut rng),
        |store| black_box(qexpr.eval(&store, &mut stack)),
        criterion::BatchSize::SmallInput,
    )
}

fn calculation(crit: &mut Criterion) {
    let exprs = [
        ("x+y", "simple addition"),
        ("sin(x)", "simple builtin function"),
        ("dist(x, y)", "simple custom function"),
        ("abs(x-y)*rand(t)+min(x,y)", "random value between x and y"),
        (
            "avg(8*x,y+56,(t+43)^2,100*rand(t))",
            "custom function with four arguments",
        ),
        ("sin(x*17/5)+cos(y+729166/7933)", "with simplification"),
        (
            "sin(x+cos(y^(1/6)))*log(895731)",
            "with ahead of time evaluation",
        ),
        (
            "sin(x*pi/10*(1.3+sin(t/10))+t*2+sin(y*pi*sin(t/17)+16*sin(t)))+0.05",
            "long trigonometric formula",
        ),
        (
            "exp(ln(x)+ln(x+1)+ln(x+2)+ln(x+3)+ln(x+4)-ln(y)-ln(y+1)-ln(y+2)-ln(y+3)-ln(y+4))",
            "long logarithmic formula",
        ),
    ];
    let mut group = crit.benchmark_group("Calculation");
    group.measurement_time(Duration::from_secs(10));
    for (input, desc) in exprs {
        let mut ast = MathAst::new(
            &TokenStream::new::<Sfr>(input).unwrap(),
            &EmptyNameTrie,
            &MyFuncsNameTrie,
            &MyVarsNameTrie,
        )
        .unwrap();
        let seed = fastrand::u64(..);
        let rand_fn = move |x| detrand_f64(x, seed);
        ast.aot_evaluation(|id| id.to_pointer(&rand_fn));
        ast.displacing_simplification();
        group.throughput(Throughput::Elements(ast.as_tree().len() as u64));
        let qexpr = QuickExpr::new(ast, |id| id.to_pointer(&rand_fn));
        group.bench_with_input(
            BenchmarkId::new("math-eval", desc),
            &qexpr,
            matheval_calc_bencher,
        );

        group.bench_with_input(BenchmarkId::new("meval", desc), input, meval_calc_bencher);
    }
    group.finish();
}

fn matheval_parse_bencher(b: &mut Bencher<'_>, input: &str) {
    let seed = fastrand::u64(..);
    let rand_fn = move |x| detrand_f64(x, seed);
    let vars = MyStore::rand(&mut Rng::new());
    b.iter(|| {
        black_box(math_eval::evaluate(
            input,
            &EmptyNameTrie,
            &MyFuncsNameTrie,
            &MyVarsNameTrie,
            |id| id.to_pointer(&rand_fn),
            &vars,
        ))
    });
}

fn meval_parse_bencher(b: &mut Bencher<'_>, input: &str) {
    let mut context = Context::new();
    let seed = fastrand::u64(..);
    context.func2("dist", dist);
    context.funcn("avg", average, 1..);
    context.func("log", f64::log10);
    context.func("rand", move |x| detrand_f64(x, seed));
    let vars = MyStore::rand(&mut Rng::new());
    b.iter(|| {
        black_box(input
            .parse::<Expr>()
            .unwrap()
            .bind3_with_context(context.clone(), "x", "y", "t")
            .unwrap()(vars.x, vars.y, vars.t));
    });
}

fn parsing(crit: &mut Criterion) {
    let exprs = [
        ("x+y", "simple addition", true),
        (
            "max(2, x, 8*y, t^2, x*y+1)",
            "builtin function with 5 arguments",
            true,
        ),
        (
            "|x-y|rand(t)+min(x,y)",
            "random value between x and y",
            false,
        ),
        ("|x||y|-sin(t)", "ambiguous pipe symbols", false),
        ("lnsinx+3y(t-5)", "misleading tokens", false),
        (
            "sin(x*pi/10*(1.3+sin(t/10))+t*2+sin(y*pi*sin(t/17)+16*sin(t)))+0.05",
            "long trigonometric formula",
            true,
        ),
        (
            "exp(ln(x)+ln(x+1)+ln(x+2)+ln(x+3)+ln(x+4)-ln(y)-ln(y+1)-ln(y+2)-ln(y+3)-ln(y+4))",
            "long logarithmic formula",
            true,
        ),
    ];
    let mut group = crit.benchmark_group("Parsing");
    group.measurement_time(Duration::from_secs(10));
    for (input, desc, bench_meval) in exprs {
        group.throughput(Throughput::Bytes(input.len() as u64));
        group.bench_with_input(
            BenchmarkId::new("math-eval", desc),
            input,
            matheval_parse_bencher,
        );
        if bench_meval {
            group.bench_with_input(BenchmarkId::new("meval", desc), input, meval_parse_bencher);
        }
    }
}

criterion_group!(benches, calculation, parsing);
criterion_main!(benches);
