use std::ops::{Range, RangeInclusive};

use number::MathEvalNumber;
use syntax::{FunctionIdentifier, SyntaxTree, VariableIdentifier, SyntaxError};
use tokenizer::{
    token_stream::{Token, TokenStream},
    token_tree::{TokenTree, TokenTreeError},
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
}

fn token2index(input: &str, token_stream: &TokenStream, token_index: usize) -> usize {
    let mut index = 0;
    while input.chars().nth(index).unwrap().is_whitespace() {
        index += 1
    }
    for token in &token_stream.0[..token_index] {
        index += match token {
            Token::Function(s) => s.len() + 1, // this token counts as both the function name and the opening parentheses
            Token::Number(s) | Token::Variable(s) => s.len(),
            Token::Operation(_) | Token::OpenParen | Token::CloseParen | Token::Comma => 1,
        };
        while input.chars().nth(index).unwrap().is_whitespace() {
            index += 1
        }
    }
    index
}

fn token2range(
    input: &str,
    token_stream: &TokenStream,
    token_index: usize,
) -> RangeInclusive<usize> {
    let mut index = token2index(input, token_stream, token_index + 1) - 1;
    while input.chars().nth(index).unwrap().is_whitespace() {
        index -= 1;
    }
    token2index(input, token_stream, token_index)..=index
}

pub fn parse<N: MathEvalNumber, V: VariableIdentifier, F: FunctionIdentifier>(
    input: &str,
    custom_constant_parser: impl Fn(&str) -> Option<N>,
) -> Result<SyntaxTree<N, V, F>, ParsingError> {
    let token_stream = TokenStream::new(input).map_err(|i| ParsingError {
        kind: ParsingErrorKind::UnexpectedCharacter,
        at: i..=i,
    })?;
    let token_tree = TokenTree::new(&token_stream).map_err(|e| match e {
        TokenTreeError::CommaOutsideFunction(i) => ParsingError {
            kind: ParsingErrorKind::CommaOutsideFunction,
            at: token2range(input, &token_stream, i),
        },
        TokenTreeError::MissingOpenParenthesis(i) => ParsingError {
            kind: ParsingErrorKind::MissingOpenParenthesis,
            at: token2range(input, &token_stream, i),
        },
        TokenTreeError::MissingCloseParenthesis(i) => ParsingError {
            kind: ParsingErrorKind::MissingCloseParenthesis,
            at: token2range(input, &token_stream, i),
        },
    })?;
    let syntax_tree = SyntaxTree::new(&token_tree, custom_constant_parser).map_err(|(err, node)| match err {
        SyntaxError::NumberParsingError => todo!(),
        SyntaxError::MisplacedOperator => todo!(),
        SyntaxError::UnknownVariableOrConstant => todo!(),
        SyntaxError::UnknownFunction => todo!(),
        SyntaxError::NotEnoughArguments => todo!(),
        SyntaxError::TooManyArguments => todo!(),
    });
    todo!()
}

#[test]
fn test_token2index() {
    let input = " sin(pi) +1";
    let ts = TokenStream::new(input).unwrap();
    assert_eq!(token2index(input, &ts, 4), 9);
    assert_eq!(token2index(input, &ts, 5), 10);
    assert_eq!(token2index(input, &ts, 0), 1);
    assert_eq!(token2index(input, &ts, 1), 4);
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
