use std::{fmt::Display, panic};

use math_eval::{
    FunctionPointer, VariableStore,
    number::NativeFunction,
    syntax::MathAst,
    tokenizer::{Token, TokenStream},
};

fn gen_random_f64() -> f64 {
    (fastrand::f64() - 0.5)
        * 10f64.powi(fastrand::i32(0..=f64::MANTISSA_DIGITS as i32))
}

#[test]
fn tokenizer() {
    for _ in 1..100 {
        let noise: String = (0..fastrand::u8(8..100))
            .map(|_| fastrand::char('\x00'..char::MAX))
            .collect();
        if let Err(err) = std::panic::catch_unwind(|| {
            let _ = TokenStream::new(&noise);
        }) {
            println!("input: {noise:?}");
            panic::resume_unwind(err);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumIter)]
enum MyVar {
    X,
    Y,
    Z,
    Theta,
}

impl MyVar {
    fn parse(input: &str) -> Option<Self> {
        match input {
            "x" => Some(Self::X),
            "y" => Some(Self::Y),
            "z" => Some(Self::Z),
            "theta" | "Θ" => Some(Self::Theta),
            _ => None,
        }
    }
}

impl Display for MyVar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MyVar::X => f.write_str("x"),
            MyVar::Y => f.write_str("y"),
            MyVar::Z => f.write_str("z"),
            MyVar::Theta => f.write_str("Θ"),
        }
    }
}

struct MyStore([f64; 4]);

impl VariableStore<f64, MyVar> for MyStore {
    fn get(&self, var: MyVar) -> f64 {
        match var {
            MyVar::X => self.0[0],
            MyVar::Y => self.0[1],
            MyVar::Z => self.0[2],
            MyVar::Theta => self.0[3],
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumIter)]
enum MyFunc {
    Deg2Rad,
    Clamp,
    Digits,
}

impl MyFunc {
    fn parse(input: &str) -> Option<(Self, u8, Option<u8>)> {
        match input {
            "deg2rad" => Some((MyFunc::Deg2Rad, 1, Some(1))),
            "clamp" => Some((MyFunc::Clamp, 3, Some(3))),
            "digits" => Some((MyFunc::Digits, 1, None)),
            _ => None,
        }
    }
    fn as_pointer(self) -> FunctionPointer<'static, f64> {
        match self {
            MyFunc::Deg2Rad => FunctionPointer::Single(|x: f64| x.to_radians()),
            MyFunc::Clamp => {
                FunctionPointer::Triple(|x: f64, min: f64, max: f64| x.min(max).max(min))
            }
            MyFunc::Digits => FunctionPointer::Flexible(|values: &[f64]| {
                values
                    .iter()
                    .enumerate()
                    .map(|(i, &v)| 10f64.powi(i as i32) * v)
                    .sum()
            }),
        }
    }
}

impl Display for MyFunc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MyFunc::Deg2Rad => f.write_str("deg2rad"),
            MyFunc::Clamp => f.write_str("clamp"),
            MyFunc::Digits => f.write_str("digits"),
        }
    }
}

#[test]
fn parser() {
    const OPERATORS: [char; 7] = ['+', '-', '*', '/', '^', '%', '!'];
    let rand_token = || match fastrand::u8(0..8) {
        0 => Token::Number(gen_random_f64().to_string()),
        1 => Token::Operator(fastrand::choice(OPERATORS).unwrap()),
        2 => Token::Variable(fastrand::choice(MyVar::iter()).unwrap().to_string()),
        3 => Token::Function(if fastrand::u8(0..100) < 80 {
            fastrand::choice(NativeFunction::iter()).unwrap().to_string()
        } else {
            fastrand::choice(MyFunc::iter()).unwrap().to_string()
        }),
        4 => Token::OpenParen,
        5 => Token::CloseParen,
        6 => Token::Comma,
        7 => Token::Pipe,
        _ => unreachable!(),
    };
    for _ in 0..1000 {
        let input: Vec<Token<_>> = (0..fastrand::u8(1..50)).map(|_| rand_token()).collect();
        if let Err(err) = std::panic::catch_unwind(|| {
            let _ = MathAst::new(&input, |_| None::<f64>, MyFunc::parse, MyVar::parse);
            let _ = MathAst::parse_and_eval(
                &input,
                |_| None::<f64>,
                MyFunc::parse,
                MyVar::parse,
                &MyStore([3., 5., 1., 0.8]),
                MyFunc::as_pointer,
            );
        }) {
            for tk in input {
                eprint!("{tk}");
            }
            eprintln!();
            panic::resume_unwind(err);
        }
    }
}
