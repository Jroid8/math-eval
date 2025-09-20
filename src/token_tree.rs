use std::ops::RangeInclusive;

use crate::token_stream::{Token, TokenStream};
use crate::{ParsingError, ParsingErrorKind, subslice_start};

pub(crate) fn token2index(
    input: &str,
    token_stream: &TokenStream<'_>,
    token_index: usize,
) -> usize {
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

pub(crate) fn token2range(
    input: &str,
    token_stream: &TokenStream<'_>,
    token_index: usize,
) -> RangeInclusive<usize> {
    let start = token2index(input, token_stream, token_index);
    let end = start
        + match &token_stream.0[token_index] {
            Token::Number(s) | Token::Variable(s) => s.len(),
            Token::Operation(_) | Token::OpenParen | Token::CloseParen | Token::Comma => 1,
            Token::Function(func) => dbg!(func.len() + 1),
        }
        - 1;
    start..=end
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenNode<'a> {
    Number(&'a str),
    Operation(char),
    Variable(&'a str),
    Function(&'a str, usize),
    Parentheses(usize),
    Comma,
    Close,
}

impl<'a> TokenNode<'a> {
    fn get_address(&self) -> Option<usize> {
        match self {
            Self::Function(_, addr) | Self::Parentheses(addr) => Some(*addr),
            _ => None,
        }
    }
    fn set_address(&mut self, new_addr: usize) {
        match self {
            Self::Function(_, addr) | Self::Parentheses(addr) => *addr = new_addr,
            _ => panic!(),
        }
    }
}

fn paren_inner_length(tokens: &[Token<'_>]) -> Option<usize> {
    let mut nesting = 1;
    for (i, tk) in tokens.iter().enumerate() {
        match tk {
            Token::Function(_) | Token::OpenParen => {
                nesting += 1;
            }
            Token::CloseParen => {
                if nesting == 1 {
                    return Some(i);
                } else {
                    nesting -= 1;
                }
            }
            _ => (),
        }
    }
    None
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenTree<'a>(pub(crate) Vec<TokenNode<'a>>);

impl<'a> TokenTree<'a> {
    pub fn new(tokens: &[Token<'a>]) -> Result<TokenTree<'a>, TokenTreeError> {
        let mut result: Vec<TokenNode<'a>> = Vec::with_capacity(tokens.len() + 1);
        let mut ret_stack: Vec<(usize, &[Token<'a>])> = Vec::new();
        ret_stack.push((usize::MAX, tokens));
        while let Some((nested_from, tks_slice)) = ret_stack.pop() {
            if nested_from != usize::MAX {
                let index = result.len();
                result[nested_from].set_address(index);
            }
            let mut index = 0;
            while let Some(tk) = tks_slice.get(index) {
                index += 1;
                if matches!(*tk, Token::Function(_) | Token::OpenParen) {
                    let true_idx = subslice_start(tokens, tks_slice).unwrap() + index - 1;
                    let inner_len = paren_inner_length(&tks_slice[index..])
                        .ok_or(TokenTreeError::MissingCloseParenthesis(true_idx))?;
                    if inner_len == 0 {
                        return Err(TokenTreeError::EmptyParenthesis(true_idx));
                    }
                    ret_stack.push((result.len(), &tks_slice[index..index + inner_len]));
                    index += inner_len + 1;
                }
                result.push(match *tk {
                    Token::Number(num) => TokenNode::Number(num),
                    Token::Operation(opr) => TokenNode::Operation(opr),
                    Token::Variable(var) => TokenNode::Variable(var),
                    Token::Function(func) => TokenNode::Function(func, 0),
                    Token::OpenParen => TokenNode::Parentheses(0),
                    Token::CloseParen => {
                        return Err(TokenTreeError::ExtraClosingParenthesis(
                            subslice_start(tokens, tks_slice).unwrap() + index - 1,
                        ));
                    }
                    Token::Comma => TokenNode::Comma,
                });
            }
            result.push(TokenNode::Close);
        }
        debug_assert!(
            result
                .iter()
                .all(|tn| tn.get_address().is_none_or(|a| a > 0)),
            "{result:?}"
        );
        debug_assert_eq!(result.len(), tokens.len() + 1);
        Ok(TokenTree(result))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TokenTreeError {
    ExtraClosingParenthesis(usize),
    MissingCloseParenthesis(usize),
    EmptyParenthesis(usize),
}

impl TokenTreeError {
    pub fn to_general(self, input: &str, token_stream: &TokenStream<'_>) -> ParsingError {
        match self {
            TokenTreeError::ExtraClosingParenthesis(i) => ParsingError {
                kind: ParsingErrorKind::MissingOpenParenthesis,
                at: token2range(input, token_stream, i),
            },
            TokenTreeError::MissingCloseParenthesis(i) => ParsingError {
                kind: ParsingErrorKind::MissingCloseParenthesis,
                at: token2range(input, token_stream, i),
            },
            TokenTreeError::EmptyParenthesis(i) => ParsingError {
                kind: ParsingErrorKind::EmptyParenthesis,
                at: token2range(input, token_stream, i),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::TokenNode::Close as TNClose;
    use super::TokenNode::Comma as TNComma;
    use super::TokenNode::Function as TNFunction;
    use super::TokenNode::Number as TNNumber;
    use super::TokenNode::Operation as TNOperation;
    use super::TokenNode::Parentheses as TNParentheses;
    use super::TokenNode::Variable as TNVariable;
    use super::*;
    use crate::token_stream::Token::*;

    #[test]
    fn test_paren_inner_length() {
        assert_eq!(paren_inner_length(&[CloseParen]), Some(0));
        assert_eq!(paren_inner_length(&[Number("1"), CloseParen]), Some(1));
        assert_eq!(
            paren_inner_length(&[Number("1"), CloseParen, Comma, Comma, Comma, Comma]),
            Some(1)
        );
        assert_eq!(
            paren_inner_length(&[
                OpenParen,
                Number("1"),
                CloseParen,
                Number("7"),
                CloseParen,
                CloseParen,
                CloseParen,
                CloseParen,
            ]),
            Some(4)
        );
        assert_eq!(
            paren_inner_length(&[
                Number("1"),
                Comma,
                OpenParen,
                Comma,
                OpenParen,
                Comma,
                CloseParen,
                Comma,
                CloseParen,
                Comma,
                CloseParen,
                Comma,
            ]),
            Some(10)
        );
        assert_eq!(paren_inner_length(&[]), None);
        assert_eq!(paren_inner_length(&[OpenParen]), None);
        assert_eq!(
            paren_inner_length(&[OpenParen, OpenParen, CloseParen]),
            None
        );
        assert_eq!(
            paren_inner_length(&[OpenParen, Number("178"), CloseParen, OpenParen]),
            None
        );
        assert_eq!(
            paren_inner_length(&[
                OpenParen,
                OpenParen,
                Number("178"),
                CloseParen,
                OpenParen,
                Comma,
                CloseParen,
                Comma,
                CloseParen,
                OpenParen
            ]),
            None
        );
        assert_eq!(
            paren_inner_length(&[
                OpenParen,
                OpenParen,
                Number("178"),
                CloseParen,
                OpenParen,
                Comma,
                CloseParen,
                Comma,
                OpenParen,
                Comma,
                CloseParen,
            ]),
            None
        );
    }

    #[test]
    fn test_treefy() {
        fn treefy<'a>(tokens: &[Token<'a>]) -> Result<Vec<TokenNode<'a>>, TokenTreeError> {
            TokenTree::new(tokens).map(|t| t.0)
        }

        assert_eq!(treefy(&[Number("192")]), Ok(vec![TNNumber("192"), TNClose]));
        assert_eq!(
            treefy(&[OpenParen, Number("3.14"), CloseParen]),
            Ok(vec![TNParentheses(2), TNClose, TNNumber("3.14"), TNClose])
        );
        assert_eq!(
            treefy(&[
                Number("7"),
                Operation('+'),
                OpenParen,
                Number("0.1"),
                CloseParen
            ]),
            Ok(vec![
                TNNumber("7"),
                TNOperation('+'),
                TNParentheses(4),
                TNClose,
                TNNumber("0.1"),
                TNClose,
            ])
        );
        assert_eq!(
            treefy(&[Function("sin"), Number("0"), CloseParen]),
            Ok(vec![TNFunction("sin", 2), TNClose, TNNumber("0"), TNClose])
        );
        println!("CASE START");
        assert_eq!(
            treefy(&[
                Function("tan"),
                OpenParen,
                Number("-4.3"),
                CloseParen,
                Operation('/'),
                Number("-99.3"),
                CloseParen
            ]),
            Ok(vec![
                TNFunction("tan", 2),
                TNClose,
                TNParentheses(6),
                TNOperation('/'),
                TNNumber("-99.3"),
                TNClose,
                TNNumber("-4.3"),
                TNClose,
            ])
        );
        // tan(x)+5(-73)
        assert_eq!(
            treefy(&[
                Function("tan"),
                Variable("x"),
                CloseParen,
                Operation('+'),
                Number("5"),
                OpenParen,
                Number("-73"),
                CloseParen
            ]),
            Ok(vec![
                TNFunction("tan", 7),
                TNOperation('+'),
                TNNumber("5"),
                TNParentheses(5),
                TNClose,
                TNNumber("-73"),
                TNClose,
                TNVariable("x"),
                TNClose
            ])
        );
        // max(2*(x+1), 0)*(t+1)
        assert_eq!(
            treefy(&[
                Function("max"),
                Number("2"),
                Operation('*'),
                OpenParen,
                Variable("x"),
                Operation('+'),
                Number("1"),
                CloseParen,
                Comma,
                Number("0"),
                CloseParen,
                Operation('*'),
                OpenParen,
                Variable("t"),
                Operation('+'),
                Number("1"),
                CloseParen,
            ]),
            Ok(vec![
                TNFunction("max", 8),
                TNOperation('*'),
                TNParentheses(4),
                TNClose,
                TNVariable("t"),
                TNOperation('+'),
                TNNumber("1"),
                TNClose,
                TNNumber("2"),
                TNOperation('*'),
                TNParentheses(14),
                TNComma,
                TNNumber("0"),
                TNClose,
                TNVariable("x"),
                TNOperation('+'),
                TNNumber("1"),
                TNClose,
            ])
        );

        assert_eq!(
            treefy(&[OpenParen]),
            Err(TokenTreeError::MissingCloseParenthesis(0))
        );
        assert_eq!(
            treefy(&[OpenParen, Number("725.4")]),
            Err(TokenTreeError::MissingCloseParenthesis(0))
        );
        assert_eq!(
            treefy(&[Operation('+'), Number("-110"), OpenParen]),
            Err(TokenTreeError::MissingCloseParenthesis(2))
        );
        assert_eq!(
            treefy(&[Function("sin"), Number("1452.333")]),
            Err(TokenTreeError::MissingCloseParenthesis(0))
        );
        assert_eq!(
            treefy(&[
                OpenParen,
                OpenParen,
                Number("-1"),
                CloseParen,
                CloseParen,
                Function("sin"),
                Number("1452.333")
            ]),
            Err(TokenTreeError::MissingCloseParenthesis(5))
        );
        assert_eq!(
            treefy(&[
                Function("sin"),
                Number("1452.333"),
                CloseParen,
                OpenParen,
                OpenParen,
                Number("-1"),
                CloseParen,
            ]),
            Err(TokenTreeError::MissingCloseParenthesis(3))
        );
        assert_eq!(
            treefy(&[CloseParen]),
            Err(TokenTreeError::ExtraClosingParenthesis(0))
        );
        assert_eq!(
            treefy(&[OpenParen, Number("455829"), CloseParen, CloseParen]),
            Err(TokenTreeError::ExtraClosingParenthesis(3))
        );
        assert_eq!(
            treefy(&[OpenParen, CloseParen]),
            Err(TokenTreeError::EmptyParenthesis(0))
        );
        assert_eq!(
            treefy(&[OpenParen, OpenParen, CloseParen, CloseParen]),
            Err(TokenTreeError::EmptyParenthesis(1))
        );
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
}
