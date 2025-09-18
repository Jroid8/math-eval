use nom::{
    character::complete::{alpha1, alphanumeric1, char, digit0, digit1, one_of, space0},
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
            test!(tuple((alphanumeric1::<_, ()>, char('(')))
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

#[cfg(test)]
mod tests {
    use super::*;
    use super::Token::*;

    #[test]
    fn test_tokenizer() {
        assert_eq!(
            TokenStream::new("1").unwrap().0,
            vec![Number("1".to_string())]
        );
        assert_eq!(
            TokenStream::new("2291").unwrap().0,
            vec![Number("2291".to_string())]
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
            TokenStream::new("839   *            4").unwrap().0,
            vec![
                Number("839".to_string()),
                Operation('*'),
                Number("4".to_string())
            ]
        );
        assert_eq!(
            TokenStream::new("7.620-90.001").unwrap().0,
            vec![
                Number("7.620".to_string()),
                Operation('-'),
                Number("90.001".to_string())
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
        assert_eq!(TokenStream::new("=").unwrap_err(), TokenizationError(0));
        assert_eq!(TokenStream::new("10+ت").unwrap_err(), TokenizationError(3));
    }
}
