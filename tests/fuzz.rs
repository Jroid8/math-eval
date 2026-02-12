use std::{any::Any, panic};

use math_eval::{
    number::NativeFunction,
    syntax::MathAst,
    tokenizer::{OprToken, Token, TokenStream},
};
use strum::IntoEnumIterator;

use crate::common::{MyFunc, MyStore, MyVar, rand_ast};

mod common;

fn rand_f64() -> f64 {
    (fastrand::f64() - 0.5) * 10f64.powi(fastrand::i32(0..=f64::MANTISSA_DIGITS as i32))
}

#[test]
fn fuzz_tokenizer() {
    for _ in 1..1000 {
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

#[test]
fn fuzz_parser() {
    fn rand_token() -> Token<String> {
        match fastrand::u8(0..8) {
            0 => Token::Number(rand_f64().to_string()),
            1 => Token::Operator(fastrand::choice(OprToken::iter()).unwrap()),
            2 => Token::Variable(fastrand::choice(MyVar::iter()).unwrap().to_string()),
            3 => Token::Function(if fastrand::u8(0..100) < 80 {
                fastrand::choice(NativeFunction::iter())
                    .unwrap()
                    .to_string()
            } else {
                fastrand::choice(MyFunc::iter()).unwrap().to_string()
            }),
            4 => Token::OpenParen,
            5 => Token::CloseParen,
            6 => Token::Comma,
            7 => Token::Pipe,
            _ => unreachable!(),
        }
    }
    fn tryinput(input: &[Token<String>]) -> Result<(), Box<dyn Any + Send + 'static>> {
        std::panic::catch_unwind(|| {
            let _ = MathAst::new(&input, |_| None::<f64>, MyFunc::parse, MyVar::parse);
            let _ = MathAst::parse_and_eval(
                &input,
                |_| None::<f64>,
                MyFunc::parse,
                MyVar::parse,
                &MyStore([3., 5., 1., 0.8]),
                MyFunc::as_pointer,
            );
        })
    }
    for s in [1, 30, 100] {
        for _ in 0..10000/s {
            let input: Vec<Token<_>> = (0..fastrand::u16(s..s + 10)).map(|_| rand_token()).collect();
            if let Err(pan) = tryinput(&input) {
                eprintln!("input: {input:?}");
                panic::resume_unwind(pan);
            }
        }
    }
}
