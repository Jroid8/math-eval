fn gen_random_f64() -> f64 {
    fastrand::f64() * 10f64.powi(fastrand::i32(0..=f64::MANTISSA_DIGITS as i32))
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
    let wpin = |val: String| -> String {
        if val.starts_with('-') {
            format!("({})", val)
        } else {
            val
        }
    };
    while let Some(index) = result.find('$') {
        result.replace_range(
            index..=index,
            match fastrand::u8(0..30) {
                0..=4 => wpin(fastrand::i32(i32::MIN..i32::MAX).to_string()),
                5..=8 => wpin((gen_random_f64()).to_string()),
                9 => "pi".to_string(),
                10 => "e".to_string(),
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
fn test_all() {
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
            ).unwrap();
            if result.is_nan() {
                panic!("nan result from: {}", expr)
            }
        }
    }
}
