use std::panic;

fn gen_random_f64() -> f64 {
    fastrand::f64()
        * 10f64.powi(fastrand::i32(0..=f64::MANTISSA_DIGITS as i32))
        * (if fastrand::bool() { 1.0 } else { -1.0 })
}

fn generate_infallible_expr(target_size: usize) -> String {
    let branch_nodes: &[&str] = &[
        "$+$",
        "$-$",
        "$*$",
        "$/(($)^2+0.1)",
        "$/max($,0.1)",
        "sin($)",
        "cos($)",
        "tan($)",
        "log(max($,0.1))",
        "log(($)^2+0.1)",
        "log10(max($,0.1))",
        "log10(($)^2+0.1)",
        "log2(max($,0.1))",
        "log2(($)^2+0.1)",
        "abs($)",
        "sqrt(abs($))",
        "max($,$)",
        "min($,$)",
        "max($,$,$)",
        "min($,$,$)",
        "max($,$,$,$)",
        "min($,$,$,$)",
        "max($,$,$,$,$)",
        "min($,$,$,$,$)",
        "sqrt(($)^2 + ($)^2)",
        "dist($,$)",
        "mean($,$)",
        "mean($,$,$)",
        "mean($,$,$,$)",
        "mean($,$,$,$,$)",
        "mean($,$,$,$,$,$)",
        "min(max($,0), 10)!",
    ];
    let mut result = String::from('$');
    for _ in 0..target_size {
        let rand_slice = fastrand::usize(0..result.len());
        let index = result[rand_slice..]
            .find('$')
            .map(|i| i + rand_slice)
            .or(result.find('$'))
            .unwrap();
        let selected = fastrand::choice(branch_nodes).unwrap();
        result.replace_range(index..=index, selected);
    }
    while let Some(index) = result.find('$') {
        result.replace_range(
            index..=index,
            match fastrand::u8(0..30) {
                0..=4 => fastrand::i32(i32::MIN..i32::MAX).to_string(),
                5..=8 => (gen_random_f64()).to_string(),
                9 => "pi".to_string(),
                10 => "e".to_string(),
                11 => "c".to_string(),
                _ => fastrand::choice(&["x", "y", "z", "t"]).unwrap().to_string(),
            }
            .as_str(),
        );
    }
    result
}

#[test]
fn test_operator() {
    for _ in 1..1000 {
        let opr = fastrand::choice(['+', '-', '*', '/']).unwrap();
        let x = gen_random_f64();
        let y = match gen_random_f64() {
            0.0 => {
                if opr == '/' {
                    1.0
                } else {
                    0.0
                }
            }
            x => x,
        };
        let expr_str = format!("{x}{opr}{y}");
        let mut expr = math_eval::parse(
            &expr_str,
            |_| None,
            |_| None::<((), _, _)>,
            |_| None::<()>,
            |_| &|_| 0.0,
        )
        .unwrap();
        let result = match opr {
            '+' => x + y,
            '-' => x - y,
            '*' => x * y,
            '/' => x / y,
            _ => unreachable!(),
        };
        let eval_result = expr.eval(|_| 0.0);
        if expr.eval(|_| 0.0) != result {
            panic!("the result of \"{expr_str}\" didn't match the calculated result\n{eval_result} != {result}")
        }
    }
}

#[test]
fn test_all_valid() {
    let parser = math_eval::EvalBuilder::new()
        .add_function("mean", 2, None, &|inputs: &[f64]| {
            inputs.iter().sum::<f64>() / inputs.len() as f64
        })
        .add_function("dist", 2, Some(2), &|inputs: &[f64]| {
            (inputs[0].powi(2) + inputs[1].powi(2)).sqrt()
        })
        .add_variable("x")
        .add_variable("y")
        .add_variable("z")
        .add_variable("t")
        .add_constant("c", 299792458.0)
        .build_as_parser();
    for n in 1..=10 {
        for _ in 1..30 {
            let expr = generate_infallible_expr(n * n);
            let result = parser(
                &expr,
                gen_random_f64(),
                gen_random_f64(),
                gen_random_f64(),
                gen_random_f64(),
            );
            match result {
                Ok(value) => {
                    if value.is_nan() {
                        panic!("NaN result from: {expr}");
                    }
                }
                Err(error) => {
                    error.print_colored(&expr);
                    panic!("Error returned when parsing {error}")
                }
            }
        }
    }
}

#[derive(Clone)]
enum MyVars {
    X,
    Y,
    A,
}
#[derive(Clone)]
enum MyFuncs {
    Mean,
    Dist,
}

#[test]
fn test_fuzz_symbols() {
    let mut symbols: Vec<String> = "1234567890+-*/^!qwertyuiopasdfghjklzxcvbnm()[]{},"
        .chars()
        .map(String::from)
        .collect();
    symbols.extend(
        [
            "sin", "cos", "tan", "cot", "asin", "acos", "atan", "acot", "log", "log2", "log10",
            "ln", "exp", "floor", "ceil", "round", "trunc", "frac", "abs", "sign", "sqrt", "cbrt",
            "max", "min", "mean", "dist", "ke",
        ]
        .into_iter()
        .map(String::from),
    );
    for _ in 1..200 {
        let expr: String = (1..fastrand::u8(30..80))
            .map(|_| fastrand::choice(&symbols).unwrap().as_str())
            .collect();
        if let Err(err) = std::panic::catch_unwind(|| {
            math_eval::parse(
                &expr,
                |inp| if inp == "ke" { Some(8.99e9) } else { None },
                |inp| match inp {
                    "mean" => Some((MyFuncs::Mean, 2, None)),
                    "dist" => Some((MyFuncs::Dist, 2, Some(2))),
                    _ => None,
                },
                |inp| match inp {
                    "x" => Some(MyVars::X),
                    "y" => Some(MyVars::Y),
                    "a" => Some(MyVars::A),
                    _ => None,
                },
                |func| match func {
                    MyFuncs::Mean => &|inp| inp.iter().sum::<f64>() / inp.len() as f64,
                    MyFuncs::Dist => &|inp| (inp[0] * inp[0] + inp[1] * inp[1]).sqrt(),
                },
            )
        }) {
            println!("{expr}");
            panic::resume_unwind(err);
        }
    }
}

#[test]
fn test_fuzz_all() {
    for _ in 1..100 {
        let noise: String = (0..100).map(|_| fastrand::char('\x00'..char::MAX)).collect();
        if let Err(err) = std::panic::catch_unwind(|| {
            math_eval::parse(
                &noise,
                |inp| if inp == "ke" { Some(8.99e9) } else { None },
                |inp| match inp {
                    "mean" => Some((MyFuncs::Mean, 2, None)),
                    "dist" => Some((MyFuncs::Dist, 2, Some(2))),
                    _ => None,
                },
                |inp| match inp {
                    "x" => Some(MyVars::X),
                    "y" => Some(MyVars::Y),
                    "a" => Some(MyVars::A),
                    _ => None,
                },
                |func| match func {
                    MyFuncs::Mean => &|inp| inp.iter().sum::<f64>() / inp.len() as f64,
                    MyFuncs::Dist => &|inp| (inp[0] * inp[0] + inp[1] * inp[1]).sqrt(),
                },
            )
        }) {
            println!("{noise:?}");
            panic::resume_unwind(err);
        }
    }
}
