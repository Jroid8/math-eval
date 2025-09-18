#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub enum Token<'a> {
    Number(&'a str),
    Operation(char),
    Variable(&'a str),
    Function(&'a str),
    OpenParen,
    CloseParen,
    Comma,
}

#[derive(Debug, Clone, Copy)]
enum CharNotion {
    Number,
    Operation,
    Negation,
    Alphabet,
    OpenParen,
    CloseParen,
    Comma,
    Dot,
    Space,
}

fn recognize(input: char) -> Option<CharNotion> {
    match input {
        '0'..='9' => Some(CharNotion::Number),
        '.' => Some(CharNotion::Dot),
        '-' => Some(CharNotion::Negation),
        '+' | '*' | '/' | '^' | '%' | '!' => Some(CharNotion::Operation),
        'a'..='z' | 'A'..='Z' => Some(CharNotion::Alphabet),
        '(' | '[' | '{' => Some(CharNotion::OpenParen),
        ')' | ']' | '}' => Some(CharNotion::CloseParen),
        ',' => Some(CharNotion::Comma),
        ' ' | '\x09'..='\x0d' => Some(CharNotion::Space),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct TokenStream<'a>(pub(crate) Vec<Token<'a>>);

impl<'a> TokenStream<'a> {
    pub fn new(input: &'a str) -> Result<TokenStream<'a>, TokenizationError> {
        enum Reading {
            Nothing,
            Number(usize, bool),
            VarFunc(usize),
        }

        let mut result = Vec::new();
        let mut state = Reading::Nothing;
        for (pos, cha) in input.char_indices() {
            let notion = recognize(cha).ok_or(TokenizationError(pos))?;
            loop {
                match state {
                    Reading::Nothing => match notion {
                        CharNotion::Number | CharNotion::Negation | CharNotion::Dot => {
                            state = Reading::Number(pos, cha == '.')
                        }
                        CharNotion::Operation => result.push(Token::Operation(cha)),
                        CharNotion::Alphabet => state = Reading::VarFunc(pos),
                        CharNotion::OpenParen => result.push(Token::OpenParen),
                        CharNotion::CloseParen => result.push(Token::CloseParen),
                        CharNotion::Comma => result.push(Token::CloseParen),
                        CharNotion::Space => (),
                    },
                    Reading::Number(start, dec) => match notion {
                        CharNotion::Number => (),
                        CharNotion::Dot => {
                            if dec {
                                return Err(TokenizationError(pos));
                            } else {
                                state = Reading::Number(start, true);
                            }
                        }
                        CharNotion::Negation
                        | CharNotion::Operation
                        | CharNotion::Alphabet
                        | CharNotion::OpenParen
                        | CharNotion::CloseParen
                        | CharNotion::Comma
                        | CharNotion::Space => {
                            match &input[start..pos] {
                                "-" => result.push(Token::Operation('-')),
                                "." => return Err(TokenizationError(start)),
                                "-." => return Err(TokenizationError(start)),
                                _ => result.push(Token::Number(&input[start..pos])),
                            }
                            state = Reading::Nothing;
                            continue;
                        }
                    },
                    Reading::VarFunc(start) => match notion {
                        CharNotion::Alphabet | CharNotion::Number => (),
                        CharNotion::OpenParen => {
                            result.push(Token::Function(&input[start..pos]));
                            state = Reading::Nothing;
                        }
                        CharNotion::Negation
                        | CharNotion::Operation
                        | CharNotion::CloseParen
                        | CharNotion::Dot
                        | CharNotion::Comma
                        | CharNotion::Space => {
                            result.push(Token::Variable(&input[start..pos]));
                            state = Reading::Nothing;
                            continue;
                        }
                    },
                }
                break;
            }
        }
        match state {
            Reading::Nothing => (),
            Reading::Number(start, _) => match &input[start..] {
                // this shoudln't be allowed, but it's not the right place to deal with it
                "-" => result.push(Token::Operation('-')),
                "." => return Err(TokenizationError(start)),
                "-." => return Err(TokenizationError(start)),
                _ => result.push(Token::Number(&input[start..])),
            },
            Reading::VarFunc(start) => result.push(Token::Variable(&input[start..])),
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
    use super::Token::*;
    use super::*;

    #[test]
    fn test_tokenizer() {
        assert_eq!(TokenStream::new("1"), Ok(TokenStream(vec![Number("1")])));
        assert_eq!(
            TokenStream::new("2291"),
            Ok(TokenStream(vec![Number("2291")]))
        );
        assert_eq!(
            TokenStream::new("-1.0"),
            Ok(TokenStream(vec![Number("-1.0")]))
        );
        assert_eq!(
            TokenStream::new("(11)"),
            Ok(TokenStream(vec![OpenParen, Number("11"), CloseParen]))
        );
        assert_eq!(
            TokenStream::new("[11]"),
            Ok(TokenStream(vec![OpenParen, Number("11"), CloseParen]))
        );
        assert_eq!(
            TokenStream::new("{11}"),
            Ok(TokenStream(vec![OpenParen, Number("11"), CloseParen]))
        );
        assert_eq!(
            TokenStream::new("-(pi)"),
            Ok(TokenStream(vec![
                Operation('-'),
                OpenParen,
                Variable("pi"),
                CloseParen
            ]))
        );
        assert_eq!(
            TokenStream::new("10*5"),
            Ok(TokenStream(vec![Number("10"), Operation('*'), Number("5")]))
        );
        assert_eq!(
            TokenStream::new("839   *            4"),
            Ok(TokenStream(vec![
                Number("839"),
                Operation('*'),
                Number("4")
            ]))
        );
        assert_eq!(
            TokenStream::new("7.620-90.001"),
            Ok(TokenStream(vec![Number("7.620"), Number("-90.001")]))
        );
        assert_eq!(
            TokenStream::new("1.10/pi"),
            Ok(TokenStream(vec![
                Number("1.10"),
                Operation('/'),
                Variable("pi")
            ]))
        );
        assert_eq!(
            TokenStream::new("x+y"),
            Ok(TokenStream(vec![
                Variable("x"),
                Operation('+'),
                Variable("y")
            ]))
        );
        assert_eq!(
            TokenStream::new("sin(x)"),
            Ok(TokenStream(vec![
                Function("sin"),
                Variable("x"),
                CloseParen
            ]))
        );
        assert_eq!(TokenStream::new(""), Ok(TokenStream(vec![])));
        assert_eq!(TokenStream::new("   "), Ok(TokenStream(vec![])));
        assert_eq!(
            TokenStream::new("((([{ 3 }])))"),
            Ok(TokenStream(vec![
                OpenParen,
                OpenParen,
                OpenParen,
                OpenParen,
                OpenParen,
                Number("3"),
                CloseParen,
                CloseParen,
                CloseParen,
                CloseParen,
                CloseParen
            ]))
        );
        assert_eq!(
            TokenStream::new("cos(1+sin(0))"),
            Ok(TokenStream(vec![
                Function("cos"),
                Number("1"),
                Operation('+'),
                Function("sin"),
                Number("0"),
                CloseParen,
                CloseParen
            ]))
        );
        assert_eq!(TokenStream::new(".5"), Ok(TokenStream(vec![Number(".5")])));
        assert_eq!(
            TokenStream::new("-.25"),
            Ok(TokenStream(vec![Number("-.25")]))
        );
        assert_eq!(
            TokenStream::new("theta+phi"),
            Ok(TokenStream(vec![
                Variable("theta"),
                Operation('+'),
                Variable("phi")
            ]))
        );
        assert_eq!(
            TokenStream::new("2^10"),
            Ok(TokenStream(vec![Number("2"), Operation('^'), Number("10")]))
        );
        assert_eq!(TokenStream::new("="), Err(TokenizationError(0)));
        assert_eq!(TokenStream::new("99$"), Err(TokenizationError(2)));
        assert_eq!(TokenStream::new("@3"), Err(TokenizationError(0)));
        assert_eq!(TokenStream::new("10+ت"), Err(TokenizationError(3)));
        assert_eq!(TokenStream::new("."), Err(TokenizationError(0)));
        assert_eq!(TokenStream::new("-."), Err(TokenizationError(0)));
    }
}
