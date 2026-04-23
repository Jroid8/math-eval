use criterion::{
    Bencher, BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};
use math_eval::{
    FunctionPointer, VariableStore,
    quick_expr::QuickExpr,
    syntax::MathAst,
    tokenizer::{StandardFloatRecognizer as Sfr, TokenStream},
    trie::{EmptyNameTrie, NameTrie, TrieNode},
};
use meval::{Context, Expr};
use strum::FromRepr;

fn dist(x: f64, y: f64) -> f64 {
    (x * x + y * y).sqrt()
}

fn slope(argv: &[f64]) -> f64 {
    (argv[3] - argv[2]) / (argv[1] - argv[0])
}

fn meval_calc_bencher(b: &mut Bencher<'_>, input: &str) {
    let mut context = Context::new();
    context.func2("dist", dist);
    context.funcn("slope", slope, 4);
    context.func("log", |v: f64| v.log10());
    let expr = input
        .parse::<Expr>()
        .unwrap()
        .bind3_with_context(context, "x", "y", "t")
        .unwrap();
    let (mut x, mut y, mut t) = (0.0, 0.0, 0.0);
    b.iter(|| {
        black_box(expr(x, y, t));
        x += 1.0;
        y += 1.0;
        t += 0.0625;
    })
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
    Slope,
}

impl MyFunc {
    fn with_mmargs(self) -> (Self, u8, Option<u8>) {
        match self {
            MyFunc::Dist => (MyFunc::Dist, 2, Some(2)),
            MyFunc::Slope => (MyFunc::Slope, 4, Some(4)),
        }
    }

    fn to_pointer(self) -> FunctionPointer<'static, f64> {
        match self {
            MyFunc::Dist => FunctionPointer::<f64>::Dual(dist),
            MyFunc::Slope => FunctionPointer::<f64>::Flexible(slope),
        }
    }
}

struct MyFuncsNameTrie;

impl NameTrie<(MyFunc, u8, Option<u8>)> for MyFuncsNameTrie {
    fn nodes(&self) -> &[TrieNode] {
        &[
            TrieNode::Branch('d', 4),
            TrieNode::Branch('i', 3),
            TrieNode::Branch('s', 2),
            TrieNode::Branch('t', 1),
            TrieNode::Leaf(MyFunc::Dist as u32),
            TrieNode::Branch('s', 5),
            TrieNode::Branch('l', 4),
            TrieNode::Branch('o', 3),
            TrieNode::Branch('p', 2),
            TrieNode::Branch('e', 1),
            TrieNode::Leaf(MyFunc::Slope as u32),
        ]
    }

    fn leaf_to_value(&self, leaf: u32) -> (MyFunc, u8, Option<u8>) {
        MyFunc::from_repr(leaf as u8).unwrap().with_mmargs()
    }
}

#[derive(Debug)]
struct MyStore {
    x: f64,
    y: f64,
    t: f64,
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

fn matheval_calc_bencher(b: &mut Bencher<'_>, ast: &MathAst<f64, MyVar, MyFunc>) {
    let qexpr = QuickExpr::new(ast.clone(), MyFunc::to_pointer);
    let mut stack = Vec::with_capacity(qexpr.stack_req_capacity().unwrap());
    let (mut x, mut y, mut t) = (0.0, 0.0, 0.0);
    b.iter(|| {
        black_box(qexpr.eval(&MyStore { x, y, t }, &mut stack).unwrap());
        x += 1.0;
        y -= 1.0;
        t += 0.0625;
    })
}

fn calculation(crit: &mut Criterion) {
    let exprs = [
        ("x+y", "one addition"),
        ("sin(x)", "one builtin function single input"),
        ("dist(x, y)", "one custom function two inputs"),
        ("slope(x,y,x+17,t)", "one custom function four inputs"),
        ("10*sin(t)", "2 instructions"),
        ("3*x^2 + 2*x - 5", "quadradic"),
        ("sin(x*17/5)+cos(y+729166/7933)", "simplification"),
        (
            "sin(x+cos(y^(1/6)))*log(895731)",
            "ahead of time evaluation",
        ),
        (
            "sin(x*pi/10*(1.3+sin(t/10))+t*2+sin(y*pi*sin(t/17)+16*sin(t)))+0.05",
            "long",
        ),
    ];
    let mut group = crit.benchmark_group("Calculation");
    for (input, desc) in exprs {
        let ast = MathAst::new(
            &TokenStream::new::<Sfr>(input).unwrap(),
            &EmptyNameTrie,
            &MyFuncsNameTrie,
            &MyVarsNameTrie,
        )
        .unwrap();
        group.throughput(Throughput::Elements(ast.as_tree().len() as u64));
        group.bench_with_input(
            BenchmarkId::new("math-eval", desc),
            &ast,
            matheval_calc_bencher,
        );
        group.bench_with_input(BenchmarkId::new("meval", desc), input, meval_calc_bencher);
    }
    group.finish();
}

criterion_group!(benches, calculation);
criterion_main!(benches);
