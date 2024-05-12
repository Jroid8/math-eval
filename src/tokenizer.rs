pub mod token_stream {
    use nom::{
        character::complete::{alpha1, char, digit0, digit1, one_of, space0},
        combinator::{opt, recognize},
        sequence::tuple,
    };

    #[derive(Debug, PartialEq, Eq, Clone, Hash)]
    pub enum Token {
        Number(String),
        Operation(char),
        Variable(String),
        Function(String),
        OpenParen,
        CloseParen,
        Comma,
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
    pub struct TokenStream(pub Vec<Token>);

    impl TokenStream {
        pub fn new(mut input: &str) -> Result<TokenStream, TokenizationError> {
            let mut result = Vec::new();
            let mut index = 0;
            macro_rules! test {
                ($t:expr => $u:expr) => {
                    if let Ok((rest, matched)) = $t(input) {
                        result.push($u(matched));
                        index += input.len() - rest.len();
                        input = rest;
                        continue;
                    }
                };
            }
            while !input.is_empty() {
                input = space0::<_, ()>(input).unwrap_or((input, "")).0;
                test!(recognize(tuple((digit1::<_, ()>, opt(char('.')), digit0)))
              => |matched: &str| Token::Number(matched.to_string()));
                test!(one_of::<_, _, ()>("-+*/^%!")
              => Token::Operation);
                test!(char::<_, ()>(',')
              => |_| Token::Comma);
                test!(tuple((alpha1::<_, ()>, char('(')))
              => |matched: (&str, char)| Token::Function(matched.0.to_string()));
                test!(alpha1::<_, ()> => |matched: &str| Token::Variable(matched.to_string()));
                test!(one_of::<_, _, ()>(")]}")
              => |_| Token::CloseParen);
                test!(one_of::<_, _, ()>("([{")
              => |_| Token::OpenParen);
                return Err(TokenizationError(index));
            }
            Ok(TokenStream(result))
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
    pub struct TokenizationError(pub usize);

    impl TokenizationError {
        pub fn to_general(self) -> crate::ParsingError {
            crate::ParsingError {
                kind: crate::ParsingErrorKind::UnexpectedCharacter,
                at: self.0..=self.0,
            }
        }
    }
}

pub mod token_tree {
    use std::ops::RangeInclusive;

    use indextree::Arena;

    use crate::tokenizer::token_stream::{Token, TokenStream};
    use crate::tree_utils::Tree;
    use crate::{ParsingError, ParsingErrorKind};

    pub(crate) fn token2index(
        input: &str,
        token_stream: &TokenStream,
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
        token_stream: &TokenStream,
        token_index: usize,
    ) -> RangeInclusive<usize> {
        let mut index = token2index(input, token_stream, token_index + 1) - 1;
        while input.chars().nth(index).unwrap().is_whitespace() {
            index -= 1;
        }
        token2index(input, token_stream, token_index)..=index
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub enum TokenNode<'a> {
        Number(&'a str),
        Operation(char),
        Variable(&'a str),
        Parentheses,
        Function(&'a str),
        Argument,
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct TokenTree<'a>(pub Tree<TokenNode<'a>>);

    impl<'a> TokenTree<'a> {
        pub fn new(tokens: &'a TokenStream) -> Result<TokenTree<'a>, TokenTreeError> {
            let mut arena = Arena::new();
            let root = arena.new_node(TokenNode::Parentheses);
            let mut current_node = root;
            for (index, token) in tokens.0.iter().enumerate() {
                match token {
                    Token::Number(num) => {
                        current_node.append_value(TokenNode::Number(num.as_str()), &mut arena);
                    }
                    Token::Operation(opr) => {
                        current_node.append_value(TokenNode::Operation(*opr), &mut arena);
                    }
                    Token::Variable(var) => {
                        current_node.append_value(TokenNode::Variable(var.as_str()), &mut arena);
                    }
                    Token::Function(func) => {
                        current_node = current_node
                            .append_value(TokenNode::Function(func.as_str()), &mut arena)
                            .append_value(TokenNode::Argument, &mut arena);
                    }
                    Token::OpenParen => {
                        current_node =
                            current_node.append_value(TokenNode::Parentheses, &mut arena);
                    }
                    Token::CloseParen => {
                        if let Some(parent) = current_node.ancestors(&arena).nth(1) {
                            current_node = if matches!(arena[parent].get(), TokenNode::Function(_))
                            {
                                parent.ancestors(&arena).nth(1).unwrap()
                            } else {
                                parent
                            }
                        } else {
                            return Err(TokenTreeError::MissingOpenParenthesis(index));
                        }
                    }
                    Token::Comma => {
                        if let Some(func_node) = current_node.ancestors(&arena).nth(1) {
                            if matches!(arena[func_node].get(), TokenNode::Function(_)) {
                                current_node =
                                    func_node.append_value(TokenNode::Argument, &mut arena);
                            } else {
                                return Err(TokenTreeError::CommaOutsideFunction(index));
                            }
                        } else {
                            return Err(TokenTreeError::CommaOutsideFunction(index));
                        }
                    }
                }
            }
            if current_node == root {
                Ok(TokenTree(Tree { arena, root }))
            } else {
                let mut parens = 0;
                let mut unclosed_paren: Option<usize> = None;
                for (index, token) in tokens.0.iter().enumerate().rev() {
                    match *token {
                        Token::CloseParen => {
                            parens += 1;
                        }
                        Token::OpenParen | Token::Function(_) => {
                            if parens == 0 {
                                unclosed_paren = Some(index)
                            } else {
                                parens -= 1;
                            }
                        }
                        _ => (),
                    }
                }
                Err(TokenTreeError::MissingCloseParenthesis(
                    unclosed_paren.unwrap(),
                ))
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub enum TokenTreeError {
        CommaOutsideFunction(usize),
        MissingOpenParenthesis(usize),
        MissingCloseParenthesis(usize),
    }

    impl TokenTreeError {
        pub fn to_general(self, input: &str, token_stream: &TokenStream) -> ParsingError {
            match self {
                TokenTreeError::CommaOutsideFunction(i) => ParsingError {
                    kind: ParsingErrorKind::CommaOutsideFunction,
                    at: token2range(input, token_stream, i),
                },
                TokenTreeError::MissingOpenParenthesis(i) => ParsingError {
                    kind: ParsingErrorKind::MissingOpenParenthesis,
                    at: token2range(input, token_stream, i),
                },
                TokenTreeError::MissingCloseParenthesis(i) => ParsingError {
                    kind: ParsingErrorKind::MissingCloseParenthesis,
                    at: token2range(input, token_stream, i),
                },
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::token_stream::*;
    use super::token_tree::*;

    #[test]
    fn test_tokenizer() {
        use super::token_stream::Token::*;
        assert_eq!(
            TokenStream::new("11").unwrap().0,
            vec![Number("11".to_string())]
        );
        assert_eq!(
            TokenStream::new("(11)").unwrap().0,
            vec![OpenParen, Number("11".to_string()), CloseParen]
        );
        assert_eq!(
            TokenStream::new("[11]").unwrap().0,
            vec![OpenParen, Number("11".to_string()), CloseParen]
        );
        assert_eq!(
            TokenStream::new("{11}").unwrap().0,
            vec![OpenParen, Number("11".to_string()), CloseParen]
        );
        assert_eq!(
            TokenStream::new("10*5").unwrap().0,
            vec![
                Number("10".to_string()),
                Operation('*'),
                Number("5".to_string())
            ]
        );
        assert_eq!(
            TokenStream::new("10   *            5").unwrap().0,
            vec![
                Number("10".to_string()),
                Operation('*'),
                Number("5".to_string())
            ]
        );
        assert_eq!(
            TokenStream::new("10.10+5.18").unwrap().0,
            vec![
                Number("10.10".to_string()),
                Operation('+'),
                Number("5.18".to_string())
            ]
        );
        assert_eq!(
            TokenStream::new("1.10/pi").unwrap().0,
            vec![
                Number("1.10".to_string()),
                Operation('/'),
                Variable("pi".to_string())
            ]
        );
        assert_eq!(
            TokenStream::new("x+y").unwrap().0,
            vec![
                Variable("x".to_string()),
                Operation('+'),
                Variable("y".to_string())
            ]
        );
        assert_eq!(
            TokenStream::new("sin(x)").unwrap().0,
            vec![
                Function("sin".to_string()),
                Variable("x".to_string()),
                CloseParen
            ]
        );
        assert_eq!(TokenStream::new("ت").unwrap_err(), TokenizationError(0));
        assert_eq!(TokenStream::new("10+ت").unwrap_err(), TokenizationError(3));
    }

    macro_rules! branch {
        ($node:expr, $($children:expr),+) => {
            VecTree::Branch($node,vec![$($children),+])
        };
    }

    #[test]
    fn test_treefy() {
        use super::token_stream::Token::*;
        use crate::tree_utils::VecTree::{self, Leaf};

        macro_rules! treefy {
            ($($s:expr),+) => {{
                TokenTree::new(&TokenStream(vec![$($s),*])).map(|tt|
                   tt.0.root.children(&tt.0.arena).map(|n| VecTree::new(&tt.0.arena,n)).collect::<Vec<_>>()
                )
            }};
        }

        assert_eq!(
            treefy!(Number("10".to_string())),
            Ok(vec![Leaf(TokenNode::Number("10"))])
        );
        assert_eq!(
            treefy!(OpenParen, Number("10".to_string()), CloseParen),
            Ok(vec![branch!(
                TokenNode::Parentheses,
                Leaf(TokenNode::Number("10"))
            )])
        );
        assert_eq!(
            treefy!(
                Number("10".to_string()),
                Operation('+'),
                OpenParen,
                Number("10".to_string()),
                CloseParen
            ),
            Ok(vec![
                Leaf(TokenNode::Number("10")),
                Leaf(TokenNode::Operation('+')),
                branch!(TokenNode::Parentheses, Leaf(TokenNode::Number("10")))
            ])
        );
        assert_eq!(
            treefy!(
                Function("sin".to_string()),
                Number("10".to_string()),
                CloseParen
            ),
            Ok(vec![branch!(
                TokenNode::Function("sin"),
                branch!(TokenNode::Argument, Leaf(TokenNode::Number("10")))
            )])
        );
        assert_eq!(
            treefy!(
                Function("sin".to_string()),
                OpenParen,
                Number("10".to_string()),
                CloseParen,
                Operation('/'),
                Number("10".to_string()),
                CloseParen
            ),
            Ok(vec![branch!(
                TokenNode::Function("sin"),
                branch!(
                    TokenNode::Argument,
                    branch!(TokenNode::Parentheses, Leaf(TokenNode::Number("10"))),
                    Leaf(TokenNode::Operation('/')),
                    Leaf(TokenNode::Number("10"))
                )
            )])
        );
        assert_eq!(
            treefy!(
                Function("tan".to_string()),
                Variable("x".to_string()),
                CloseParen,
                Operation('+'),
                Number("1".to_string()),
                OpenParen,
                Number("103".to_string()),
                CloseParen
            ),
            Ok(vec![
                branch!(
                    TokenNode::Function("tan"),
                    branch!(TokenNode::Argument, Leaf(TokenNode::Variable("x")))
                ),
                Leaf(TokenNode::Operation('+')),
                Leaf(TokenNode::Number("1")),
                branch!(TokenNode::Parentheses, Leaf(TokenNode::Number("103")))
            ])
        );

        assert_eq!(
            treefy!(OpenParen),
            Err(TokenTreeError::MissingCloseParenthesis(0))
        );
        assert_eq!(
            treefy!(OpenParen, Number("11".to_string())),
            Err(TokenTreeError::MissingCloseParenthesis(0))
        );
        assert_eq!(
            treefy!(Number("11".to_string()), OpenParen),
            Err(TokenTreeError::MissingCloseParenthesis(1))
        );
        assert_eq!(
            treefy!(Function("sin".to_string()), Number("121".to_string())),
            Err(TokenTreeError::MissingCloseParenthesis(0))
        );
        assert_eq!(
            treefy!(Number("121".to_string()), Function("sin".to_string())),
            Err(TokenTreeError::MissingCloseParenthesis(1))
        );
        assert_eq!(
            treefy!(
                Function("sin".to_string()),
                OpenParen,
                Number("121".to_string()),
                CloseParen
            ),
            Err(TokenTreeError::MissingCloseParenthesis(0))
        );
        assert_eq!(
            treefy!(CloseParen),
            Err(TokenTreeError::MissingOpenParenthesis(0))
        );
        assert_eq!(
            treefy!(Number("455829".to_string()), CloseParen),
            Err(TokenTreeError::MissingOpenParenthesis(1))
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
