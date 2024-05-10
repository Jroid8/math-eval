use std::ops::RangeInclusive;

use asm::MathAssembly;
use number::MathEvalNumber;
use syntax::SyntaxTree;
use tokenizer::{
    token_stream::TokenStream,
    token_tree::TokenTree,
};

pub mod asm;
pub mod number;
pub mod syntax;
pub mod tokenizer;
pub mod tree_utils;

pub struct ParsingError {
    kind: ParsingErrorKind,
    at: RangeInclusive<usize>,
}

impl ParsingError {
    pub fn kind(&self) -> &ParsingErrorKind {
        &self.kind
    }

    pub fn at(&self) -> &RangeInclusive<usize> {
        &self.at
    }
}

pub enum ParsingErrorKind {
    UnexpectedCharacter,
    CommaOutsideFunction,
    MissingOpenParenthesis,
    MissingCloseParenthesis,
    NumberParsingError,
    MisplacedOperator,
    UnknownVariableOrConstant,
    UnknownFunction,
    NotEnoughArguments,
    TooManyArguments,
}


pub fn parse<'a, N: MathEvalNumber, V: Clone, F: Clone>(
    input: &str,
    custom_constant_parser: impl Fn(&str) -> Option<N>,
    custom_function_parser: impl Fn(&str) -> Option<(F, u8, Option<u8>)>,
    custom_variable_parser: impl Fn(&str) -> Option<V>,
    function_to_pointer: &impl Fn(&F) -> &'a dyn Fn(&[N]) -> N,
) -> Result<MathAssembly<'a, N, V, F>, ParsingError> {
    let token_stream = TokenStream::new(input).map_err(|e| e.to_general())?;
    let token_tree = TokenTree::new(&token_stream).map_err(|e| e.to_general(input, &token_stream))?;
    let mut syntax_tree = SyntaxTree::new(
        &token_tree,
        custom_constant_parser,
        custom_function_parser,
        custom_variable_parser,
    )
    .map_err(|e| e.to_general(input, &token_tree))?;
    syntax_tree.aot_evaluation(function_to_pointer);
    syntax_tree.displacing_simplification();
    Ok(MathAssembly::new(
        &syntax_tree.0.arena,
        syntax_tree.0.root,
        function_to_pointer,
    ))
}

#[test]
fn test_token2index() {
    let input = " sin(pi) +1";
    let ts = TokenStream::new(input).unwrap();
    assert_eq!(token2index(input, &ts, 3), 9);
    assert_eq!(token2index(input, &ts, 4), 10);
    assert_eq!(token2index(input, &ts, 0), 1);
    assert_eq!(token2index(input, &ts, 1), 5);
}

#[test]
fn test_token2range() {
    let input = " max(pi, 1, -4)*3";
    let ts = TokenStream::new(input).unwrap();
    assert_eq!(token2range(input, &ts, 0), 1..=4);
    assert_eq!(token2range(input, &ts, 1), 5..=6);
    assert_eq!(token2range(input, &ts, 2), 7..=7);
    assert_eq!(token2range(input, &ts, 3), 9..=9);
}

#[test]
fn test_tokennode2range() {
    let input = " max(1, -18) * sin(pi)";
    let ts = TokenStream::new(input).unwrap();
    let tt = TokenTree::new(&ts).unwrap();
    println!(
        "{:?}",
        tt.0.arena[tt.0.root.descendants(&tt.0.arena).nth(10).unwrap()].get()
    );
    assert_eq!(
        tokennode2range(
            input,
            &tt,
            tt.0.root.descendants(&tt.0.arena).nth(3).unwrap()
        ),
        5..=5
    );
    assert_eq!(
        tokennode2range(
            input,
            &tt,
            tt.0.root.descendants(&tt.0.arena).nth(6).unwrap()
        ),
        9..=10
    );
    assert_eq!(
        tokennode2range(input, &tt, tt.0.root.children(&tt.0.arena).nth(1).unwrap()),
        13..=13
    );
    assert_eq!(
        tokennode2range(
            input,
            &tt,
            tt.0.root.descendants(&tt.0.arena).nth(10).unwrap()
        ),
        19..=20
    );
}
