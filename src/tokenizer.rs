use std::fmt::{Debug, Display};

use strum::EnumIter;

#[derive(Debug, PartialEq, Eq, Clone, Copy, Hash, EnumIter)]
pub enum OprToken {
    Plus,
    Minus,
    Multiply,
    Divide,
    Power,
    Factorial,
    DoubleFactorial,
    Modulo,
    NegExp,
    DoubleStar,
}

impl OprToken {
    pub fn length(&self) -> usize {
        match self {
            Self::Plus
            | Self::Minus
            | Self::Multiply
            | Self::Divide
            | Self::Power
            | Self::Modulo
            | Self::Factorial => 1,
            Self::DoubleFactorial | Self::NegExp | Self::DoubleStar => 2,
        }
    }
}

impl Display for OprToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Plus => f.write_str("+"),
            Self::Minus => f.write_str("-"),
            Self::Multiply => f.write_str("*"),
            Self::Divide => f.write_str("/"),
            Self::Power => f.write_str("^"),
            Self::Factorial => f.write_str("!"),
            Self::DoubleFactorial => f.write_str("!!"),
            Self::Modulo => f.write_str("%"),
            Self::DoubleStar => f.write_str("**"),
            Self::NegExp => f.write_str("^-"),
        }
    }
}

#[derive(PartialEq, Eq, Clone, Copy, Hash)]
pub enum Token<S: AsRef<str>> {
    Number(S),
    Operator(OprToken),
    Variable(S),
    Function(S),
    OpenParen,
    CloseParen,
    Comma,
    Pipe,
}

impl<S: AsRef<str>> Token<S> {
    pub fn length(&self) -> usize {
        match self {
            // this token captures both the function name and the opening parentheses
            Token::Function(s) => s.as_ref().len() + 1,
            Token::Number(s) | Token::Variable(s) => s.as_ref().len(),
            Token::Operator(opr) => opr.length(),
            Token::OpenParen | Token::CloseParen | Token::Comma | Token::Pipe => 1,
        }
    }
}

impl<S: AsRef<str>> Debug for Token<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Number(num) => f.debug_tuple("Number").field(&num.as_ref()).finish(),
            Self::Operator(opr) => f.debug_tuple("Operator").field(opr).finish(),
            Self::Variable(var) => f.debug_tuple("Variable").field(&var.as_ref()).finish(),
            Self::Function(func) => f.debug_tuple("Function").field(&func.as_ref()).finish(),
            Self::OpenParen => write!(f, "OpenParen"),
            Self::CloseParen => write!(f, "CloseParen"),
            Self::Comma => write!(f, "Comma"),
            Self::Pipe => write!(f, "Pipe"),
        }
    }
}

impl<S: AsRef<str>> Display for Token<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Token::Number(num) => f.write_str(num.as_ref()),
            Token::Operator(opr) => write!(f, "{opr}"),
            Token::Variable(var) => f.write_str(var.as_ref()),
            Token::Function(func) => f.write_str(func.as_ref()),
            Token::OpenParen => f.write_str("("),
            Token::CloseParen => f.write_str(")"),
            Token::Comma => f.write_str(","),
            Token::Pipe => f.write_str("|"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OprChar {
    Plus,
    Minus,
    Star,
    Slash,
    Caret,
    Percent,
    Exclamation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CharNotion {
    Number,
    Operation(OprChar),
    Pipe,
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
        '+' => Some(CharNotion::Operation(OprChar::Plus)),
        '-' => Some(CharNotion::Operation(OprChar::Minus)),
        '*' => Some(CharNotion::Operation(OprChar::Star)),
        '/' => Some(CharNotion::Operation(OprChar::Slash)),
        '^' => Some(CharNotion::Operation(OprChar::Caret)),
        '%' => Some(CharNotion::Operation(OprChar::Percent)),
        '!' => Some(CharNotion::Operation(OprChar::Exclamation)),
        'a'..='z' | 'A'..='Z' => Some(CharNotion::Alphabet),
        '(' | '[' | '{' => Some(CharNotion::OpenParen),
        ')' | ']' | '}' => Some(CharNotion::CloseParen),
        ',' => Some(CharNotion::Comma),
        ' ' | '\x09'..='\x0d' => Some(CharNotion::Space),
        '|' => Some(CharNotion::Pipe),
        _ => None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct TokenStream<S: AsRef<str>>(pub(crate) Vec<Token<S>>);

impl<'a> TokenStream<&'a str> {
    pub fn new(input: &'a str) -> Result<Self, TokenizationError> {
        enum Reading {
            Nothing,
            Number(usize, bool),
            VarFunc(usize),
            Star,
            Caret,
            Exclamation,
        }

        let mut result = Vec::new();
        let mut state = Reading::Nothing;
        for (pos, cha) in input.char_indices() {
            let notion = recognize(cha).ok_or(TokenizationError(pos))?;
            loop {
                match state {
                    Reading::Nothing => match notion {
                        CharNotion::Number | CharNotion::Dot => {
                            state = Reading::Number(pos, cha == '.')
                        }
                        CharNotion::Operation(oc) => match oc {
                            OprChar::Plus => result.push(Token::Operator(OprToken::Plus)),
                            OprChar::Minus => result.push(Token::Operator(OprToken::Minus)),
                            OprChar::Slash => result.push(Token::Operator(OprToken::Divide)),
                            OprChar::Percent => result.push(Token::Operator(OprToken::Modulo)),
                            OprChar::Star => state = Reading::Star,
                            OprChar::Caret => state = Reading::Caret,
                            OprChar::Exclamation => state = Reading::Exclamation,
                        },
                        CharNotion::Alphabet => state = Reading::VarFunc(pos),
                        CharNotion::OpenParen => result.push(Token::OpenParen),
                        CharNotion::CloseParen => result.push(Token::CloseParen),
                        CharNotion::Comma => result.push(Token::Comma),
                        CharNotion::Pipe => result.push(Token::Pipe),
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
                        CharNotion::Operation(_)
                        | CharNotion::Alphabet
                        | CharNotion::OpenParen
                        | CharNotion::CloseParen
                        | CharNotion::Comma
                        | CharNotion::Pipe
                        | CharNotion::Space => {
                            if &input[start..pos] == "." {
                                return Err(TokenizationError(start));
                            } else {
                                result.push(Token::Number(&input[start..pos]))
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
                        CharNotion::Operation(_)
                        | CharNotion::CloseParen
                        | CharNotion::Dot
                        | CharNotion::Comma
                        | CharNotion::Pipe
                        | CharNotion::Space => {
                            result.push(Token::Variable(&input[start..pos]));
                            state = Reading::Nothing;
                            continue;
                        }
                    },
                    Reading::Star => {
                        if notion == CharNotion::Operation(OprChar::Star) {
                            result.push(Token::Operator(OprToken::DoubleStar));
                            state = Reading::Nothing;
                        } else {
                            result.push(Token::Operator(OprToken::Multiply));
                            state = Reading::Nothing;
                            continue;
                        }
                    }
                    Reading::Caret => {
                        if notion == CharNotion::Operation(OprChar::Minus) {
                            result.push(Token::Operator(OprToken::NegExp));
                            state = Reading::Nothing;
                        } else {
                            result.push(Token::Operator(OprToken::Power));
                            state = Reading::Nothing;
                            continue;
                        }
                    }
                    Reading::Exclamation => {
                        if notion == CharNotion::Operation(OprChar::Exclamation) {
                            result.push(Token::Operator(OprToken::DoubleFactorial));
                            state = Reading::Nothing;
                        } else {
                            result.push(Token::Operator(OprToken::Factorial));
                            state = Reading::Nothing;
                            continue;
                        }
                    }
                }
                break;
            }
        }
        match state {
            Reading::Nothing => (),
            Reading::Number(start, _) => {
                if &input[start..] == "." {
                    return Err(TokenizationError(start));
                } else {
                    result.push(Token::Number(&input[start..]))
                }
            }
            Reading::VarFunc(start) => result.push(Token::Variable(&input[start..])),
            Reading::Star => result.push(Token::Operator(OprToken::Multiply)),
            Reading::Caret => result.push(Token::Operator(OprToken::Power)),
            Reading::Exclamation => result.push(Token::Operator(OprToken::Factorial)),
        }
        Ok(TokenStream(result))
    }
}

impl<S: AsRef<str>> AsRef<[Token<S>]> for TokenStream<S> {
    fn as_ref(&self) -> &[Token<S>] {
        &self.0
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
    fn tokenizer() {
        assert_eq!(TokenStream::new("1"), Ok(TokenStream(vec![Number("1")])));
        assert_eq!(
            TokenStream::new("2291"),
            Ok(TokenStream(vec![Number("2291")]))
        );
        assert_eq!(
            TokenStream::new("-1.0"),
            Ok(TokenStream(vec![Operator(OprToken::Minus), Number("1.0")]))
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
                Operator(OprToken::Minus),
                OpenParen,
                Variable("pi"),
                CloseParen
            ]))
        );
        assert_eq!(
            TokenStream::new("10*5"),
            Ok(TokenStream(vec![
                Number("10"),
                Operator(OprToken::Multiply),
                Number("5")
            ]))
        );
        assert_eq!(
            TokenStream::new("839   *            4"),
            Ok(TokenStream(vec![
                Number("839"),
                Operator(OprToken::Multiply),
                Number("4")
            ]))
        );
        assert_eq!(
            TokenStream::new("7.620-90.001"),
            Ok(TokenStream(vec![
                Number("7.620"),
                Operator(OprToken::Minus),
                Number("90.001")
            ]))
        );
        assert_eq!(
            TokenStream::new("1.10/pi"),
            Ok(TokenStream(vec![
                Number("1.10"),
                Operator(OprToken::Divide),
                Variable("pi")
            ]))
        );
        assert_eq!(
            TokenStream::new("x+y"),
            Ok(TokenStream(vec![
                Variable("x"),
                Operator(OprToken::Plus),
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
        assert_eq!(
            TokenStream::new("log(x, 5)"),
            Ok(TokenStream(vec![
                Function("log"),
                Variable("x"),
                Comma,
                Number("5"),
                CloseParen
            ]))
        );
        assert_eq!(
            TokenStream::new("|x|"),
            Ok(TokenStream(vec![Pipe, Variable("x"), Pipe]))
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
                Operator(OprToken::Plus),
                Function("sin"),
                Number("0"),
                CloseParen,
                CloseParen
            ]))
        );
        assert_eq!(TokenStream::new(".5"), Ok(TokenStream(vec![Number(".5")])));
        assert_eq!(
            TokenStream::new("theta+phi"),
            Ok(TokenStream(vec![
                Variable("theta"),
                Operator(OprToken::Plus),
                Variable("phi")
            ]))
        );
        assert_eq!(
            TokenStream::new("2^10"),
            Ok(TokenStream(vec![
                Number("2"),
                Operator(OprToken::Power),
                Number("10")
            ]))
        );
        assert_eq!(
            TokenStream::new("2^-10"),
            Ok(TokenStream(vec![
                Number("2"),
                Operator(OprToken::NegExp),
                Number("10")
            ]))
        );
        assert_eq!(
            TokenStream::new("2**10"),
            Ok(TokenStream(vec![
                Number("2"),
                Operator(OprToken::DoubleStar),
                Number("10")
            ]))
        );
        assert_eq!(
            TokenStream::new("x!!!"),
            Ok(TokenStream(vec![
                Variable("x"),
                Operator(OprToken::DoubleFactorial),
                Operator(OprToken::Factorial)
            ]))
        );
        assert_eq!(TokenStream::new("="), Err(TokenizationError(0)));
        assert_eq!(TokenStream::new("99$"), Err(TokenizationError(2)));
        assert_eq!(TokenStream::new("@3"), Err(TokenizationError(0)));
        assert_eq!(TokenStream::new("10+ت"), Err(TokenizationError(3)));
        assert_eq!(TokenStream::new("."), Err(TokenizationError(0)));
    }
}
