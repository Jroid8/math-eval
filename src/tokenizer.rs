use std::fmt::{Debug, Display};

use strum::EnumIter;

#[derive(Debug, PartialEq, Eq, Clone, Copy, EnumIter)]
pub enum OprToken {
    Plus,
    Minus,
    Multiply,
    Divide,
    Power,
    Factorial,
    DoubleFactorial,
    Modulo,
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
            Self::DoubleFactorial | Self::DoubleStar => 2,
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
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, EnumIter)]
pub enum DelimEdge {
    Opening,
    Closing,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy, EnumIter)]
pub enum DelimKind {
    Paren,
    Brace,
    Bracket,
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct DelimiterToken(pub DelimKind, pub DelimEdge);

impl Display for DelimiterToken {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            DelimiterToken(DelimKind::Paren, DelimEdge::Opening) => "(",
            DelimiterToken(DelimKind::Brace, DelimEdge::Opening) => "[",
            DelimiterToken(DelimKind::Bracket, DelimEdge::Opening) => "{",
            DelimiterToken(DelimKind::Paren, DelimEdge::Closing) => ")",
            DelimiterToken(DelimKind::Brace, DelimEdge::Closing) => "]",
            DelimiterToken(DelimKind::Bracket, DelimEdge::Closing) => "}",
        })
    }
}

#[derive(PartialEq, Eq, Clone, Copy)]
pub enum Token<S: AsRef<str>> {
    Number(S),
    Operator(OprToken),
    Variable(S),
    Function(S, DelimKind),
    Delimiter(DelimiterToken),
    Comma,
    Pipe,
}

impl<S: AsRef<str>> Token<S> {
    pub fn byte_len(&self) -> usize {
        match self {
            Token::Function(s, _) => s.as_ref().len() + 1,
            Token::Number(s) | Token::Variable(s) => s.as_ref().len(),
            Token::Operator(opr) => opr.length(),
            Token::Delimiter(_) | Token::Comma | Token::Pipe => 1,
        }
    }
}

impl<S: AsRef<str>> Debug for Token<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Number(num) => f.debug_tuple("Number").field(&num.as_ref()).finish(),
            Self::Operator(opr) => f.debug_tuple("Operator").field(opr).finish(),
            Self::Variable(var) => f.debug_tuple("Variable").field(&var.as_ref()).finish(),
            Self::Function(func, kind) => f
                .debug_tuple("Function")
                .field(&func.as_ref())
                .field(&kind)
                .finish(),
            Self::Delimiter(delim) => f.debug_tuple("Delimiter").field(&delim).finish(),
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
            Token::Function(func, kind) => {
                write!(
                    f,
                    "{}{}",
                    func.as_ref(),
                    DelimiterToken(*kind, DelimEdge::Opening)
                )
            }
            Token::Delimiter(delim) => write!(f, "{delim}"),
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
    Power,
    Percent,
    Exclamation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CharNotion {
    Operation(OprChar),
    Pipe,
    Alphabet,
    Number,
    Delimiter(DelimiterToken),
    Comma,
    Space,
}

fn recognize(input: char) -> Option<CharNotion> {
    match input {
        '0'..='9' => Some(CharNotion::Number),
        '+' => Some(CharNotion::Operation(OprChar::Plus)),
        '-' => Some(CharNotion::Operation(OprChar::Minus)),
        '*' => Some(CharNotion::Operation(OprChar::Star)),
        '/' => Some(CharNotion::Operation(OprChar::Slash)),
        '^' => Some(CharNotion::Operation(OprChar::Power)),
        '%' => Some(CharNotion::Operation(OprChar::Percent)),
        '!' => Some(CharNotion::Operation(OprChar::Exclamation)),
        '(' => Some(CharNotion::Delimiter(DelimiterToken(
            DelimKind::Paren,
            DelimEdge::Opening,
        ))),
        ')' => Some(CharNotion::Delimiter(DelimiterToken(
            DelimKind::Paren,
            DelimEdge::Closing,
        ))),
        '[' => Some(CharNotion::Delimiter(DelimiterToken(
            DelimKind::Brace,
            DelimEdge::Opening,
        ))),
        ']' => Some(CharNotion::Delimiter(DelimiterToken(
            DelimKind::Brace,
            DelimEdge::Closing,
        ))),
        '{' => Some(CharNotion::Delimiter(DelimiterToken(
            DelimKind::Bracket,
            DelimEdge::Opening,
        ))),
        '}' => Some(CharNotion::Delimiter(DelimiterToken(
            DelimKind::Bracket,
            DelimEdge::Closing,
        ))),
        ',' => Some(CharNotion::Comma),
        ' ' | '\x09'..='\x0d' => Some(CharNotion::Space),
        '|' => Some(CharNotion::Pipe),
        ch if ch.is_alphabetic() => Some(CharNotion::Alphabet),
        _ => None,
    }
}

pub trait NumberRecognizer: Sized {
    fn new(current: char) -> Option<Self>;
    fn recognize(&mut self, current: char) -> bool;
}

pub struct StandardFloatRecognizer(bool);

impl NumberRecognizer for StandardFloatRecognizer {
    fn new(current: char) -> Option<Self> {
        match current {
            '0'..='9' => Some(Self(false)),
            '.' => Some(Self(true)),
            _ => None,
        }
    }

    fn recognize(&mut self, current: char) -> bool {
        if (current == 'e' || current == '.') && !self.0 {
            self.0 = true;
            true
        } else {
            matches!(current, '0'..='9')
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct TokenStream<S: AsRef<str>>(pub(crate) Vec<Token<S>>);

impl<'a> TokenStream<&'a str> {
    pub fn new<N: NumberRecognizer>(input: &'a str) -> Result<Self, TokenizationError> {
        enum Reading<N: NumberRecognizer> {
            Nothing,
            Number(usize, N),
            VarFunc(usize),
            Star,
            Exclamation,
        }

        let mut result = Vec::new();
        let mut state: Reading<N> = Reading::Nothing;
        for (pos, cha) in input.char_indices() {
            let notion = recognize(cha);
            loop {
                match &mut state {
                    Reading::Nothing => match notion {
                        Some(CharNotion::Operation(oc)) => match oc {
                            OprChar::Plus => result.push(Token::Operator(OprToken::Plus)),
                            OprChar::Minus => result.push(Token::Operator(OprToken::Minus)),
                            OprChar::Slash => result.push(Token::Operator(OprToken::Divide)),
                            OprChar::Percent => result.push(Token::Operator(OprToken::Modulo)),
                            OprChar::Power => result.push(Token::Operator(OprToken::Power)),
                            OprChar::Star => state = Reading::Star,
                            OprChar::Exclamation => state = Reading::Exclamation,
                        },
                        Some(CharNotion::Alphabet) => state = Reading::VarFunc(pos),
                        Some(CharNotion::Comma) => result.push(Token::Comma),
                        Some(CharNotion::Pipe) => result.push(Token::Pipe),
                        Some(CharNotion::Delimiter(delim)) => result.push(Token::Delimiter(delim)),
                        Some(CharNotion::Space) => (),
                        None | Some(CharNotion::Number) => {
                            if let Some(nr) = N::new(cha) {
                                state = Reading::Number(pos, nr);
                            } else {
                                return Err(TokenizationError(pos));
                            }
                        }
                    },
                    Reading::Number(start, nr) => {
                        if !nr.recognize(cha) {
                            result.push(Token::Number(&input[*start..pos]));
                            state = Reading::Nothing;
                            continue;
                        }
                    }
                    Reading::VarFunc(start) => match notion {
                        Some(CharNotion::Alphabet | CharNotion::Number) => (),
                        Some(CharNotion::Delimiter(DelimiterToken(kind, DelimEdge::Opening))) => {
                            result.push(Token::Function(&input[*start..pos], kind));
                            state = Reading::Nothing;
                        }
                        None
                        | Some(
                            CharNotion::Operation(_)
                            | CharNotion::Comma
                            | CharNotion::Pipe
                            | CharNotion::Space
                            | CharNotion::Delimiter(DelimiterToken(_, DelimEdge::Closing)),
                        ) => {
                            result.push(Token::Variable(&input[*start..pos]));
                            state = Reading::Nothing;
                            continue;
                        }
                    },
                    Reading::Star => {
                        if notion == Some(CharNotion::Operation(OprChar::Star)) {
                            result.push(Token::Operator(OprToken::DoubleStar));
                            state = Reading::Nothing;
                        } else {
                            result.push(Token::Operator(OprToken::Multiply));
                            state = Reading::Nothing;
                            continue;
                        }
                    }
                    Reading::Exclamation => {
                        if notion == Some(CharNotion::Operation(OprChar::Exclamation)) {
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
            Reading::Number(start, _) => result.push(Token::Number(&input[start..])),
            Reading::VarFunc(start) => result.push(Token::Variable(&input[start..])),
            Reading::Star => result.push(Token::Operator(OprToken::Multiply)),
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

#[derive(Debug, Clone, PartialEq, Eq, Default)]
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
    use super::DelimEdge::*;
    use super::DelimKind::*;
    use super::Token::*;
    use super::*;

    #[test]
    fn tokenizer() {
        let tokenize = |s| TokenStream::new::<StandardFloatRecognizer>(s);
        assert_eq!(tokenize("1"), Ok(TokenStream(vec![Number("1")])));
        assert_eq!(tokenize("2291"), Ok(TokenStream(vec![Number("2291")])));
        assert_eq!(
            tokenize("-1.0"),
            Ok(TokenStream(vec![Operator(OprToken::Minus), Number("1.0")]))
        );
        assert_eq!(
            tokenize("(11)"),
            Ok(TokenStream(vec![
                Delimiter(DelimiterToken(Paren, Opening)),
                Number("11"),
                Delimiter(DelimiterToken(Paren, Closing))
            ]))
        );
        assert_eq!(
            tokenize("[11]"),
            Ok(TokenStream(vec![
                Delimiter(DelimiterToken(Brace, Opening)),
                Number("11"),
                Delimiter(DelimiterToken(Brace, Closing))
            ]))
        );
        assert_eq!(
            tokenize("{11}"),
            Ok(TokenStream(vec![
                Delimiter(DelimiterToken(Bracket, Opening)),
                Number("11"),
                Delimiter(DelimiterToken(Bracket, Closing))
            ]))
        );
        assert_eq!(
            tokenize("-{pi}"),
            Ok(TokenStream(vec![
                Operator(OprToken::Minus),
                Delimiter(DelimiterToken(Bracket, Opening)),
                Variable("pi"),
                Delimiter(DelimiterToken(Bracket, Closing))
            ]))
        );
        assert_eq!(
            tokenize("10*5"),
            Ok(TokenStream(vec![
                Number("10"),
                Operator(OprToken::Multiply),
                Number("5")
            ]))
        );
        assert_eq!(
            tokenize("839   *            4"),
            Ok(TokenStream(vec![
                Number("839"),
                Operator(OprToken::Multiply),
                Number("4")
            ]))
        );
        assert_eq!(
            tokenize("7.620-90.001"),
            Ok(TokenStream(vec![
                Number("7.620"),
                Operator(OprToken::Minus),
                Number("90.001")
            ]))
        );
        assert_eq!(
            tokenize("1.10/pi"),
            Ok(TokenStream(vec![
                Number("1.10"),
                Operator(OprToken::Divide),
                Variable("pi")
            ]))
        );
        assert_eq!(
            tokenize("x+y"),
            Ok(TokenStream(vec![
                Variable("x"),
                Operator(OprToken::Plus),
                Variable("y")
            ]))
        );
        assert_eq!(
            tokenize("sin(x)"),
            Ok(TokenStream(vec![
                Function("sin", Paren),
                Variable("x"),
                Delimiter(DelimiterToken(Paren, Closing))
            ]))
        );
        assert_eq!(
            tokenize("sin[x]"),
            Ok(TokenStream(vec![
                Function("sin", Brace),
                Variable("x"),
                Delimiter(DelimiterToken(Brace, Closing))
            ]))
        );
        assert_eq!(
            tokenize("log(x, 5)"),
            Ok(TokenStream(vec![
                Function("log", Paren),
                Variable("x"),
                Comma,
                Number("5"),
                Delimiter(DelimiterToken(Paren, Closing))
            ]))
        );
        assert_eq!(
            tokenize("|x|"),
            Ok(TokenStream(vec![Pipe, Variable("x"), Pipe]))
        );
        assert_eq!(tokenize(""), Ok(TokenStream(vec![])));
        assert_eq!(tokenize("   "), Ok(TokenStream(vec![])));
        assert_eq!(
            tokenize("cos{1+sin(0)}"),
            Ok(TokenStream(vec![
                Function("cos", Bracket),
                Number("1"),
                Operator(OprToken::Plus),
                Function("sin", Paren),
                Number("0"),
                Delimiter(DelimiterToken(Paren, Closing)),
                Delimiter(DelimiterToken(Bracket, Closing))
            ]))
        );
        assert_eq!(tokenize(".5"), Ok(TokenStream(vec![Number(".5")])));
        assert_eq!(
            tokenize("θ+τ"),
            Ok(TokenStream(vec![
                Variable("θ"),
                Operator(OprToken::Plus),
                Variable("τ")
            ]))
        );
        assert_eq!(
            tokenize("2^10"),
            Ok(TokenStream(vec![
                Number("2"),
                Operator(OprToken::Power),
                Number("10")
            ]))
        );
        assert_eq!(
            tokenize("2**10"),
            Ok(TokenStream(vec![
                Number("2"),
                Operator(OprToken::DoubleStar),
                Number("10")
            ]))
        );
        assert_eq!(
            tokenize("x!!!"),
            Ok(TokenStream(vec![
                Variable("x"),
                Operator(OprToken::DoubleFactorial),
                Operator(OprToken::Factorial)
            ]))
        );
        assert_eq!(tokenize("="), Err(TokenizationError(0)));
        assert_eq!(tokenize("99$"), Err(TokenizationError(2)));
        assert_eq!(tokenize("@3"), Err(TokenizationError(0)));
    }
}
