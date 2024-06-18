fn gen_random_f64() -> f64 {
    fastrand::f64() * 10f64.powi(fastrand::i32(0..=f64::MANTISSA_DIGITS as i32))
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
