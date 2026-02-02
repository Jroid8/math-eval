use criterion::{
    Bencher, BenchmarkId, Criterion, Throughput, black_box, criterion_group, criterion_main,
};
use math_eval::{
    FunctionPointer, VariableStore, quick_expr::QuickExpr, syntax::MathAst, tokenizer::TokenStream,
};
use meval::{Context, Expr};

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

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum MyVar {
    X,
    Y,
    T,
}

impl MyVar {
    fn parse(input: &str) -> Option<Self> {
        match input {
            "x" => Some(MyVar::X),
            "y" => Some(MyVar::Y),
            "t" => Some(MyVar::T),
            _ => None,
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
enum MyFunc {
    Dist,
    Slope,
}

impl MyFunc {
    fn parse(input: &str) -> Option<(Self, u8, Option<u8>)> {
        match input {
            "dist" => Some((MyFunc::Dist, 2, Some(2))),
            "slope" => Some((MyFunc::Slope, 4, Some(4))),
            _ => None,
        }
    }

    fn to_pointer(self) -> FunctionPointer<'static, f64> {
        match self {
            MyFunc::Dist => FunctionPointer::<f64>::Dual(dist),
            MyFunc::Slope => FunctionPointer::<f64>::Flexible(slope),
        }
    }
}

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
        black_box(qexpr.eval(MyStore { x, y, t }, &mut stack).unwrap());
        x += 1.0;
        y -= 1.0;
        t += 0.0625;
    })
}

fn calculation(crit: &mut Criterion) {
    let exprs = [
        ("x+y", "one addition"),
        ("sin(x)", "one native function single input"),
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
            TokenStream::new(input).unwrap(),
            |_| None,
            MyFunc::parse,
            MyVar::parse,
        ).unwrap();
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
