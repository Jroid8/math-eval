use std::{any::Any, fmt::Display, iter::repeat_with, panic};

use math_eval::{
    FunctionPointer, VariableStore,
    number::NativeFunction,
    quick_expr::QuickExpr,
    syntax::MathAst,
    tokenizer::{OprToken, StandardFloatRecognizer as Sfr, Token, TokenStream},
    trie::{EmptyNameTrie, NameTrie, TrieNode, VecNameTrie},
};
use strum::{EnumIter, FromRepr, IntoEnumIterator};

use crate::common::{AstGen, rand_f64};

mod common;

#[test]
fn fuzz_tokenizer() {
    for _ in 1..1000 {
        let noise: String = (0..fastrand::u8(8..100))
            .map(|_| fastrand::char('\x00'..char::MAX))
            .collect();
        if let Err(err) = panic::catch_unwind(|| {
            let _ = TokenStream::new::<Sfr>(&noise);
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
        panic::catch_unwind(|| {
            let _ =
                MathAst::<f64, _, _>::new(&input, &EmptyNameTrie, &MyFuncNameTrie, &MyVarNameTrie);
            let _ = MathAst::parse_and_eval(
                &input,
                &EmptyNameTrie,
                &MyFuncNameTrie,
                &MyVarNameTrie,
                &MyStore::randomize(),
                MyFunc::as_pointer,
            );
        })
    }
    for s in [1, 30, 100] {
        for _ in 0..1000 {
            let input: Vec<Token<_>> = (0..fastrand::u16(s..s + 10))
                .map(|_| rand_token())
                .collect();
            if let Err(pan) = tryinput(&input) {
                eprintln!("input: {input:?}");
                panic::resume_unwind(pan);
            }
        }
    }
}

fn report_ast_panic(input: MathAst<f64, MyVar, MyFunc>, pan: Box<dyn Any + Send + 'static>) -> ! {
    eprintln!("input: {}", input);
    eprintln!(
        "input (debug format): {:?}",
        input.into_tree().postorder_iter().collect::<Vec<_>>()
    );
    panic::resume_unwind(pan);
}

#[test]
fn fuzz_quickexpr() {
    for size in 1..=3usize {
        for _ in 0..700 {
            let expr = rand_ast(4usize.pow(size as u32));
            if let Err(pan) = std::panic::catch_unwind(|| {
                let expr = QuickExpr::new(expr.clone(), MyFunc::as_pointer);
                let Ok(stack_cap) = expr.stack_req_capacity() else {
                    return;
                };
                let _ = expr.eval(&MyStore::randomize(), &mut Vec::with_capacity(stack_cap));
            }) {
                report_ast_panic(expr, pan);
            }
        }
    }
}

#[test]
fn fuzz_displacing_simplification() {
    for size in 1..=3usize {
        for _ in 0..1000 {
            let expr = rand_ast(4usize.pow(size as u32));
            if let Err(pan) = std::panic::catch_unwind(|| {
                expr.clone().displacing_simplification();
            }) {
                report_ast_panic(expr, pan);
            }
        }
    }
}

#[test]
fn fuzz_aot_evaluation() {
    for size in 1..=3usize {
        for _ in 0..1000 {
            let expr = rand_ast(4usize.pow(size as u32));
            if let Err(pan) = std::panic::catch_unwind(|| {
                expr.clone().aot_evaluation(MyFunc::as_pointer);
            }) {
                report_ast_panic(expr, pan);
            }
        }
    }
}

#[test]
fn fuzz_ast_eval() {
    for size in 1..=3usize {
        for _ in 0..1000 {
            let expr = rand_ast(4usize.pow(size as u32));
            if let Err(pan) = std::panic::catch_unwind(|| {
                expr.eval(MyFunc::as_pointer, &MyStore::randomize());
            }) {
                report_ast_panic(expr, pan);
            }
        }
    }
}

#[test]
fn fuzz_name_trie_new() {
    for _ in 0..1000 {
        let names: Vec<String> = repeat_with(|| {
            repeat_with(|| fastrand::char(..))
                .take(fastrand::usize(1..30))
                .collect::<String>()
        })
        .take(fastrand::u8(3..100).into())
        .collect();
        if let Err(err) = panic::catch_unwind(|| {
            let mut pairs: Vec<(&str, u8)> = names
                .iter()
                .enumerate()
                .map(|(i, s)| (s.as_str(), i as u8))
                .collect();
            let _ = VecNameTrie::new(&mut pairs);
        }) {
            println!("names: {names:?}");
            panic::resume_unwind(err);
        }
    }
}

#[test]
fn fuzz_name_trie_get() {
    for _ in 0..1000 {
        let names: Vec<String> = repeat_with(|| {
            repeat_with(|| fastrand::char(..))
                .take(fastrand::usize(1..30))
                .collect::<String>()
        })
        .take(fastrand::u8(3..100).into())
        .collect();
        let mut pairs: Vec<(&str, u8)> = names
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_str(), i as u8))
            .collect();
        let trie = VecNameTrie::new(&mut pairs);
        for _ in 0..10 {
            let input = repeat_with(|| fastrand::char(..))
                .take(fastrand::usize(1..30))
                .collect::<String>();
            if let Err(err) = panic::catch_unwind(|| {
                let _ = trie.exact_match(&input);
            }) {
                println!("names: {names:?}\ninput: {input:?}");
                panic::resume_unwind(err);
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumIter, FromRepr)]
#[repr(u8)]
enum MyVar {
    X,
    Y,
    Sigma,
}

struct MyVarNameTrie;

impl NameTrie<MyVar> for MyVarNameTrie {
    fn nodes(&self) -> &[TrieNode] {
        &[
            TrieNode::Branch('x', 1),
            TrieNode::Leaf(1),
            TrieNode::Branch('y', 1),
            TrieNode::Leaf(1),
            TrieNode::Branch('σ', 1),
            TrieNode::Leaf(1),
        ]
    }
    fn leaf_to_value(&self, leaf: u32) -> MyVar {
        MyVar::from_repr(leaf as u8).unwrap()
    }
}

impl Display for MyVar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MyVar::X => f.write_str("x"),
            MyVar::Y => f.write_str("y"),
            MyVar::Sigma => f.write_str("σ"),
        }
    }
}

#[derive(Debug)]
struct MyStore {
    x: f64,
    y: f64,
    sigma: f64,
}

impl MyStore {
    fn randomize() -> MyStore {
        MyStore {
            x: rand_f64(),
            y: rand_f64(),
            sigma: rand_f64(),
        }
    }
}

impl VariableStore<f64, MyVar> for MyStore {
    fn get(&self, var: MyVar) -> f64 {
        match var {
            MyVar::X => self.x,
            MyVar::Y => self.y,
            MyVar::Sigma => self.sigma,
        }
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn clamp(value: f64, min: f64, max: f64) -> f64 {
    value.min(max).max(min)
}

fn digits(values: &[f64]) -> f64 {
    values
        .iter()
        .enumerate()
        .map(|(i, &v)| 10f64.powi(i as i32) * v)
        .sum()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, EnumIter, FromRepr)]
#[repr(u8)]
enum MyFunc {
    Sigmoid,
    Clamp,
    Digits,
}

impl MyFunc {
    fn with_mmargs(self) -> (Self, u8, Option<u8>) {
        match self {
            MyFunc::Sigmoid => (MyFunc::Sigmoid, 1, Some(1)),
            MyFunc::Clamp => (MyFunc::Clamp, 3, Some(3)),
            MyFunc::Digits => (MyFunc::Digits, 1, None),
        }
    }

    fn as_pointer(self) -> FunctionPointer<'static, f64> {
        match self {
            MyFunc::Sigmoid => FunctionPointer::<f64>::Single(sigmoid),
            MyFunc::Clamp => FunctionPointer::<f64>::Triple(clamp),
            MyFunc::Digits => FunctionPointer::Flexible(digits),
        }
    }
}

struct MyFuncNameTrie;

impl NameTrie<(MyFunc, u8, Option<u8>)> for MyFuncNameTrie {
    fn nodes(&self) -> &[TrieNode] {
        &[
            TrieNode::Branch('c', 5),
            TrieNode::Branch('l', 4),
            TrieNode::Branch('a', 3),
            TrieNode::Branch('m', 2),
            TrieNode::Branch('p', 1),
            TrieNode::Leaf(MyFunc::Clamp as u32),
            TrieNode::Branch('d', 6),
            TrieNode::Branch('i', 5),
            TrieNode::Branch('g', 4),
            TrieNode::Branch('i', 3),
            TrieNode::Branch('t', 2),
            TrieNode::Branch('s', 1),
            TrieNode::Leaf(MyFunc::Digits as u32),
            TrieNode::Branch('s', 7),
            TrieNode::Branch('i', 6),
            TrieNode::Branch('g', 5),
            TrieNode::Branch('m', 4),
            TrieNode::Branch('o', 3),
            TrieNode::Branch('i', 2),
            TrieNode::Branch('d', 1),
            TrieNode::Leaf(MyFunc::Sigmoid as u32),
        ]
    }
    fn leaf_to_value(&self, leaf: u32) -> (MyFunc, u8, Option<u8>) {
        MyFunc::from_repr(leaf as u8).unwrap().with_mmargs()
    }
}

impl Display for MyFunc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MyFunc::Sigmoid => f.write_str("sigmoid"),
            MyFunc::Clamp => f.write_str("clamp"),
            MyFunc::Digits => f.write_str("digits"),
        }
    }
}

fn rand_ast(size: usize) -> MathAst<f64, MyVar, MyFunc> {
    MathAst::from_nodes(AstGen::new(
        size,
        &[MyVar::X, MyVar::Y, MyVar::Sigma],
        &[MyFunc::Sigmoid],
        &[MyFunc::Digits],
        &[MyFunc::Clamp, MyFunc::Digits],
        &[MyFunc::Digits],
    ))
}
