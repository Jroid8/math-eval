use std::ops::RangeInclusive;

use asm::MathAssembly;
use indextree::{NodeEdge, NodeId};
use number::MathEvalNumber;
use syntax::{FunctionIdentifier, SyntaxError, SyntaxTree, VariableIdentifier};
use tokenizer::{
    token_stream::{Token, TokenStream},
    token_tree::{TokenNode, TokenTree, TokenTreeError},
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

fn tokennode2range(
    input: &str,
    token_tree: &TokenTree<'_>,
    target: NodeId,
) -> RangeInclusive<usize> {
    let mut index = 0;
    macro_rules! count_space {
        () => {
            while input.chars().nth(index).unwrap().is_whitespace() {
                index += 1
            }
        };
    }
    for node in token_tree.0.root.traverse(&token_tree.0.arena).skip(1) {
        match node {
            NodeEdge::Start(node) => {
                if *token_tree.0.arena[node].get() != TokenNode::Argument {
                    count_space!();
                }
                let old = index;
                index += match token_tree.0.arena[node].get() {
                    TokenNode::Number(s) | TokenNode::Variable(s) => s.len(),
                    TokenNode::Operation(_) => 1,
                    TokenNode::Parentheses => 1,
                    TokenNode::Function(f) => f.len() + 1,
                    TokenNode::Argument => 0,
                };
                if node == target {
                    return old..=index - 1;
                }
            }
            NodeEdge::End(node) => match token_tree.0.arena[node].get() {
                TokenNode::Argument => {
                    if node
                        .following_siblings(&token_tree.0.arena)
                        .nth(1)
                        .is_some()
                    {
                        count_space!();
                        index += 1;
                    }
                }
                TokenNode::Parentheses | TokenNode::Function(_) => {
                    count_space!();
                    index += 1
                }
                _ => (),
            },
        }
    }
    unreachable!()
}

pub fn parse<'a, N: MathEvalNumber, V: VariableIdentifier, F: FunctionIdentifier>(
    input: &str,
    custom_constant_parser: impl Fn(&str) -> Option<N>,
    custom_function_parser: impl Fn(&str) -> Option<(F, u8, Option<u8>)>,
    custom_variable_parser: impl Fn(&str) -> Option<V>,
    function_to_pointer: &impl Fn(&F) -> &'a dyn Fn(&[N]) -> N,
) -> Result<MathAssembly<'a, N, V, F>, ParsingError> {
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
    let mut syntax_tree = SyntaxTree::new(
        &token_tree,
        custom_constant_parser,
        custom_function_parser,
        custom_variable_parser,
    )
    .map_err(|(err, node)| {
        let at = tokennode2range(input, &token_tree, node);
        let kind = match err {
            SyntaxError::NumberParsingError => ParsingErrorKind::NumberParsingError,
            SyntaxError::MisplacedOperator => ParsingErrorKind::MisplacedOperator,
            SyntaxError::UnknownVariableOrConstant => ParsingErrorKind::UnknownVariableOrConstant,
            SyntaxError::UnknownFunction => ParsingErrorKind::UnknownFunction,
            SyntaxError::NotEnoughArguments => ParsingErrorKind::NotEnoughArguments,
            SyntaxError::TooManyArguments => ParsingErrorKind::TooManyArguments,
        };
        ParsingError { at, kind }
    })?;
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
