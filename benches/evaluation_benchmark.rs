use criterion::{
    black_box, criterion_group, criterion_main, Bencher, BenchmarkId, Criterion, Throughput,
};
use math_eval::{
    optimizations::MathAssembly,
    syntax::{FunctionIdentifier, SyntaxTree, VariableIdentifier},
    tokenizer::{token_stream::TokenStream, token_tree::TokenTree},
};
use meval::{Context, Expr};
use std::{cmp::min, time::Duration};

#[derive(Clone, Debug, Copy)]
enum MyVar {
    X,
    Y,
    T,
}

impl VariableIdentifier for MyVar {
    fn parse(input: &str) -> Option<Self> {
        match input {
            "x" => Some(MyVar::X),
            "y" => Some(MyVar::Y),
            "t" => Some(MyVar::T),
            _ => None,
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Debug)]
enum MyFunc {
    Dist,
    Slope,
}

impl FunctionIdentifier for MyFunc {
    type Error = ();

    fn parse(input: &str) -> Option<Self> {
        match input {
            "dist" => Some(MyFunc::Dist),
            "slope" => Some(MyFunc::Slope),
            _ => None,
        }
    }

    fn minimum_arg_count(&self) -> u8 {
        match self {
            MyFunc::Dist => 2,
            MyFunc::Slope => 4,
        }
    }

    fn maximum_arg_count(&self) -> Option<u8> {
        Some(self.minimum_arg_count())
    }
}

fn dist(input: &[f64]) -> f64 {
    (input[0] * input[0] + input[1] * input[1]).sqrt()
}

fn slope(input: &[f64]) -> f64 {
    (input[3] - input[1]) / (input[2] - input[0])
}

fn to_expr(input: &str) -> MathAssembly<'_, f64, MyVar, ()> {
    SyntaxTree::<f64, MyVar, MyFunc>::new(
        &TokenTree::new(&TokenStream::new(input).unwrap()).unwrap(),
        |_| None,
    )
    .unwrap()
    .to_asm(|fi: &MyFunc| match fi {
        MyFunc::Dist => &|input: &[f64]| Ok(dist(input)),
        MyFunc::Slope => &|input: &[f64]| Ok(slope(input)),
    })
}

fn meval_bencher(b: &mut Bencher, input: &str) {
    let mut context = Context::new();
    context.funcn("dist", dist, 2);
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

fn matheval_bencher(b: &mut Bencher, input: &str) {
    let mut expr = to_expr(input);
    let (mut x, mut y, mut t) = (0.0, 0.0, 0.0);
    b.iter(|| {
        black_box(
            expr.eval(|var| match var {
                MyVar::X => x,
                MyVar::Y => y,
                MyVar::T => t,
            })
            .unwrap(),
        );
        x += 1.0;
        y -= 1.0;
        t += 0.0625;
    })
}

const MATH_EXPRESSIONS: [(&str, &str); 8] = [
    ("x+y", "one addition"),
    ("sin(x)", "one native function"),
    ("slope(x,y,x+17,t)", "one custom function"),
    ("10*sin(t)", "3 instructions"),
    ("sin(x*17/5)+cos(y+729166/7933)", "simplification"),
    ("sin(x+cos(y^(1/6)))*log(895731)", "ahead of time evaluation"),
    ("sin(x*pi/10*(1.3+sin(t/10))+t*2+sin(y*pi*sin(t/17)+16*sin(t)))+0.05", "practical"),
    ("sin(atan(y/(x+1))+sqrt(x^2+y^2)-t*10)*sin(atan(y/(x+1))+sqrt(x^2+y^2)/1.5-t*11+sqrt(x^2+y^2)*sin(t*0.025))*(1+sin(sqrt(x^2+y^2)/5+atan(y/(x+1))*2+t))", "duplicates"),
];

fn get_throughput(input: &str) -> u64 {
    TokenStream::new(input).unwrap().0.len() as u64
}

fn hardcoded(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("Hardcoded");
    group.measurement_time(Duration::from_secs(15));
    for input in MATH_EXPRESSIONS {
        group.bench_with_input(
            BenchmarkId::new("math-eval", input.1),
            input.0,
            matheval_bencher,
        );
        group.bench_with_input(BenchmarkId::new("meval", input.1), input.0, meval_bencher);
    }
    group.finish();
}

fn random(crit: &mut Criterion) {
    let mut group = crit.benchmark_group("Random");
    fastrand::seed(2606283414);
    group.measurement_time(Duration::from_secs(20));
    for size in (2usize..=12).map(|i| i.pow(2)) {
        let input = generate(size);
        let name = format!("{size}:{}", &input[..min(input.len(), 20)]);
        group.throughput(Throughput::Elements(get_throughput(&input)));
        group.bench_with_input(
            BenchmarkId::new("math-eval", &name),
            input.as_str(),
            matheval_bencher,
        );
        group.bench_with_input(
            BenchmarkId::new("meval", &name),
            input.as_str(),
            meval_bencher,
        );
    }
    group.finish();
}

fn generate(target_size: usize) -> String {
    let branch_nodes: &[&str] = &[
        "{}+{}",
        "{}-{}",
        "{}*{}",
        "{}/(({})^2+1)",
        "{}/max({},1)",
        "sin({})",
        "cos({})",
        "tan({})",
        "log(max({},1))",
        "log(({})^2+1)",
        "abs({})",
        "sqrt(abs({}))",
        "max({},{},{})",
        "min({},{},{})",
        "max({},{},{},{})",
        "min({},{},{},{})",
        "max({},{},{},{},{})",
        "min({},{},{},{},{})",
        "sqrt(({})^2 + ({})^2)",
        "dist({},{})",
        "slope({},{},{}+1,{})",
    ];
    let mut result = String::from("{}");
    for _ in 0..target_size {
        let rand_slice = fastrand::usize(0..result.len());
        let index = result[rand_slice..]
            .find("{}")
            .map(|i| i + rand_slice)
            .or(result.find("{}"))
            .unwrap();
        let selected = fastrand::choice(branch_nodes).unwrap();
        result.replace_range(index..index + 2, selected);
    }
    let wpin = |val: String| -> String {
        if val.starts_with('-') {
            format!("({})", val)
        } else {
            val
        }
    };
    while let Some(index) = result.find("{}") {
        result.replace_range(
            index..index + 2,
            match fastrand::u8(0..30) {
                0..=4 => wpin(fastrand::i32(i32::MIN..i32::MAX).to_string()),
                5..=8 => {
                    wpin((fastrand::f64() * fastrand::i32(i32::MIN..i32::MAX) as f64).to_string())
                }
                9 => "pi".to_string(),
                _ => fastrand::choice(&["x", "y", "t"]).unwrap().to_string(),
            }
            .as_str(),
        );
    }
    result
}

criterion_group!(benches, random, hardcoded);
criterion_main!(benches);
