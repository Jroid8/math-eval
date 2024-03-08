use indicatif::{ProgressBar, ProgressStyle};
use math_eval::{
    syntax::{FunctionIdentifier, SyntaxTree, VariableIdentifier},
    tokenizer::{token_stream::TokenStream, token_tree::TokenTree},
};

#[derive(Clone, Copy, Debug)]
enum MyVar {
    X,
    Y,
    Z,
    T,
}

impl VariableIdentifier for MyVar {
    fn parse(input: &str) -> Option<Self> {
        match input {
            "x" => Some(MyVar::X),
            "y" => Some(MyVar::Y),
            "z" => Some(MyVar::Z),
            "t" => Some(MyVar::T),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MyFunc {
    Dist,
    Dot,
}

#[derive(Debug)]
struct IllegalValue;

impl FunctionIdentifier for MyFunc {
    fn parse(input: &str) -> Option<Self> {
        match input {
            "dist" => Some(Self::Dist),
            "dot" => Some(Self::Dot),
            _ => None,
        }
    }

    fn minimum_arg_count(&self) -> u8 {
        2
    }

    fn maximum_arg_count(&self) -> Option<u8> {
        Some(2)
    }
}

fn custom_functions<'a>(fi: &MyFunc) -> &'a dyn Fn(&[f64]) -> f64 {
    match fi {
        MyFunc::Dist => &|input: &[f64]| input[0] * input[0] + input[1] * input[1],
        MyFunc::Dot => &|input: &[f64]| input[0] * input[1] + input[1] * input[1],
    }
}

#[test]
fn test_random() {
    let gen_var = || -> f64 {
        fastrand::f64() * 10f64.powi(fastrand::i32(f64::MIN_10_EXP..=f64::MAX_10_EXP / 5))
    };
    let progress_bar =
        ProgressBar::new(10).with_style(ProgressStyle::default_bar().progress_chars("#>-"));
    for s in 1..=10 {
        for _ in 0..400 {
            let (x, y, z, t) = (gen_var(), gen_var(), gen_var(), gen_var());
            let input = generate(2usize.pow(s));
            let tokenstream = TokenStream::new(&input).unwrap();
            let tokentree = TokenTree::new(&tokenstream).unwrap();
            let mut syntree = SyntaxTree::<f64, MyVar, MyFunc>::new(&tokentree, |_| None).unwrap();
            syntree.aot_evaluation(custom_functions);
            syntree.displacing_simplification();
            syntree.aot_evaluation(custom_functions);
            let mut expr = syntree.to_asm(custom_functions);
            expr.eval(|var| match var {
                MyVar::X => x,
                MyVar::Y => y,
                MyVar::Z => z,
                MyVar::T => t,
            });
        }
        progress_bar.inc(1);
    }
}

fn generate(target_size: usize) -> String {
    let branch_nodes: &[&str] = &[
        "{}+{}",
        "{}-{}",
        "{}*{}",
        "{}/(({})^2+0.1)",
        "{}/max({},0.1)",
        "sin({})",
        "cos({})",
        "tan({})",
        "log(max({},0.1))",
        "log(({})^2+0.1)",
        "abs({})",
        "sqrt(abs({}))",
        "sqrt(({})^2)",
        "max({},{})",
        "min({},{})",
        "max({},{},{})",
        "min({},{},{})",
        "max({},{},{},{})",
        "min({},{},{},{})",
        "max({},{},{},{},{})",
        "min({},{},{},{},{})",
        "sqrt(({})^2 + ({})^2)",
        "dist({},{})",
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
                _ => fastrand::choice(&["x", "y", "z", "t"]).unwrap().to_string(),
            }
            .as_str(),
        );
    }
    result
}
