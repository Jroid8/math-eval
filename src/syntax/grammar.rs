// E -> PE | ES | EIE | |E| | F(A) | FE | T | (E) | [E] | {E}
// P -> + | -
// S -> ! | !!
// I -> + | - | * | / | ^
// A -> E,A | E
// T -> N | V

use std::fmt::Debug;

use crate::{
    FunctionIdentifier as FuncId, VariableIdentifier as VarId,
    number::Number,
    syntax::{CfInfo, SyntaxError, SyntaxErrorKind},
    tokenizer::{DelimEdge, DelimKind, DelimiterToken, OprToken, Token},
    trie::NameTrie,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Expecting {
    #[default]
    Prefix,
    Suffix,
}

fn suffix_is_fnp<N: Number, V: VarId, F: FuncId>(
    name: &str,
    custom_variables: &impl NameTrie<V>,
    custom_functions: &impl NameTrie<CfInfo<F>>,
) -> bool {
    if custom_variables.exact_match(name).is_some() {
        return false;
    }
    for (i, _) in name.char_indices() {
        let name = &name[i..];
        if N::BUILTIN_FUNCS_TRIE.exact_match(name).is_some()
            || custom_functions.exact_match(name).is_some()
        {
            return true;
        }
    }
    false
}

impl Expecting {
    fn next<S: AsRef<str>, N: Number, V: VarId, F: FuncId>(
        self,
        cur: &Token<S>,
        opening_pipe: bool,
        custom_variables: &impl NameTrie<V>,
        custom_functions: &impl NameTrie<CfInfo<F>>,
    ) -> Result<Self, SyntaxErrorKind> {
        match self {
            Expecting::Prefix => match cur {
                // prefix
                Token::Operator(OprToken::Plus | OprToken::Minus)
                | Token::Delimiter(DelimiterToken(_, DelimEdge::Opening))
                | Token::Function(_, _)
                | Token::Pipe => Ok(self),
                // suffix & infix
                Token::Delimiter(DelimiterToken(_, DelimEdge::Closing)) | Token::Comma => {
                    Err(SyntaxErrorKind::UnexpectedToken)
                }
                Token::Operator(
                    OprToken::Factorial
                    | OprToken::DoubleFactorial
                    | OprToken::Multiply
                    | OprToken::Divide
                    | OprToken::Power
                    | OprToken::Modulo
                    | OprToken::DoubleStar,
                ) => Err(SyntaxErrorKind::MisplacedOperator),
                // terminal
                Token::Number(_) => Ok(Expecting::Suffix),
                Token::Variable(name) => Ok(
                    if !suffix_is_fnp::<N, V, F>(name.as_ref(), custom_variables, custom_functions)
                    {
                        Expecting::Suffix
                    } else {
                        self
                    },
                ),
            },
            Expecting::Suffix => Ok(match cur {
                // suffix
                Token::Operator(OprToken::Factorial | OprToken::DoubleFactorial)
                | Token::Delimiter(DelimiterToken(_, DelimEdge::Closing)) => self,
                // prefix
                Token::Delimiter(DelimiterToken(_, DelimEdge::Opening)) | Token::Function(_, _) => {
                    Expecting::Prefix
                }
                // pipe may be prefix or suffix
                Token::Pipe => {
                    if opening_pipe {
                        Expecting::Prefix
                    } else {
                        self
                    }
                }
                // infix
                Token::Operator(
                    OprToken::Plus
                    | OprToken::Minus
                    | OprToken::Multiply
                    | OprToken::Divide
                    | OprToken::Power
                    | OprToken::Modulo
                    | OprToken::DoubleStar,
                )
                | Token::Comma => Expecting::Prefix,
                // terminal
                Token::Number(_) => self,
                Token::Variable(name) => {
                    if suffix_is_fnp::<N, V, F>(name.as_ref(), custom_variables, custom_functions) {
                        Expecting::Prefix
                    } else {
                        self
                    }
                }
            }),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ResOprToken {
    Positive,
    Negative,
    Multiply,
    Divide,
    Power,
    Factorial,
    DoubleFactorial,
    Modulo,
    Add,
    Subtract,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ResDelimKind {
    Paren,
    Brace,
    Bracket,
    Pipe,
}

impl ResDelimKind {
    fn as_missing_opening_err(self) -> SyntaxErrorKind {
        match self {
            ResDelimKind::Paren => SyntaxErrorKind::MissingOpeningParenthesis,
            ResDelimKind::Pipe => SyntaxErrorKind::MissingOpeningPipe,
            ResDelimKind::Bracket => SyntaxErrorKind::MissingOpeningBrackets,
            ResDelimKind::Brace => SyntaxErrorKind::MissingOpeningBraces,
        }
    }
    fn as_missing_closing_err(self) -> SyntaxErrorKind {
        match self {
            ResDelimKind::Paren => SyntaxErrorKind::MissingClosingParenthesis,
            ResDelimKind::Pipe => SyntaxErrorKind::MissingClosingPipe,
            ResDelimKind::Bracket => SyntaxErrorKind::MissingClosingBrackets,
            ResDelimKind::Brace => SyntaxErrorKind::MissingClosingBraces,
        }
    }
    fn as_empty_err(self) -> SyntaxErrorKind {
        match self {
            ResDelimKind::Paren => SyntaxErrorKind::EmptyParenthesis,
            ResDelimKind::Pipe => SyntaxErrorKind::EmptyPipePair,
            ResDelimKind::Bracket => SyntaxErrorKind::EmptyBrackets,
            ResDelimKind::Brace => SyntaxErrorKind::EmptyBraces,
        }
    }
}

impl From<DelimKind> for ResDelimKind {
    fn from(value: DelimKind) -> Self {
        match value {
            DelimKind::Paren => ResDelimKind::Paren,
            DelimKind::Brace => ResDelimKind::Brace,
            DelimKind::Bracket => ResDelimKind::Bracket,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum ResToken<'a> {
    Number(&'a str),
    Operator(ResOprToken),
    Variable(&'a str),
    Function(&'a str),
    OpenDelim,
    CloseDelim,
    OpenPipe,
    ClosePipe,
    Comma,
}

impl From<DelimiterToken> for ResToken<'_> {
    fn from(value: DelimiterToken) -> Self {
        match value {
            DelimiterToken(_, DelimEdge::Opening) => ResToken::OpenDelim,
            DelimiterToken(_, DelimEdge::Closing) => ResToken::CloseDelim,
        }
    }
}

fn clarify_error<S: AsRef<str>>(
    tokens: &[Token<S>],
    kind: SyntaxErrorKind,
    pos: usize,
) -> SyntaxError {
    if kind == SyntaxErrorKind::UnexpectedToken && pos > 0 {
        match tokens[pos - 1..=pos] {
            [
                Token::Comma,
                Token::Delimiter(DelimiterToken(_, DelimEdge::Closing)),
            ]
            | [
                Token::Delimiter(DelimiterToken(_, DelimEdge::Opening)) | Token::Function(_, _),
                Token::Comma,
            ]
            | [Token::Comma, Token::Comma]
            | [
                Token::Function(_, _),
                Token::Delimiter(DelimiterToken(_, DelimEdge::Closing)),
            ] => SyntaxError(SyntaxErrorKind::EmptyArgument, pos - 1..=pos),
            [
                Token::Delimiter(DelimiterToken(kind, DelimEdge::Opening)),
                Token::Delimiter(DelimiterToken(_, DelimEdge::Closing)),
            ] => SyntaxError(ResDelimKind::from(kind).as_empty_err(), pos - 1..=pos),
            [Token::Pipe, Token::Pipe] => {
                SyntaxError(SyntaxErrorKind::EmptyPipePair, pos - 1..=pos)
            }
            [
                Token::Operator(_),
                Token::Delimiter(DelimiterToken(_, DelimEdge::Closing)) | Token::Comma,
            ] => SyntaxError(SyntaxErrorKind::MisplacedOperator, pos - 1..=pos - 1),
            _ => SyntaxError(kind, pos..=pos),
        }
    } else {
        SyntaxError(kind, pos..=pos)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DgEdge {
    Opening(ResDelimKind),
    Closing(usize),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct DgRecord(usize, DgEdge);

#[derive(Debug, Clone)]
struct DelimiterGuard<'a, S: AsRef<str>> {
    tokens: &'a [Token<S>],
    records: Vec<DgRecord>,
    pos: usize,
}

impl<'a, S: AsRef<str>> DelimiterGuard<'a, S> {
    fn new(tokens: &'a [Token<S>]) -> Self {
        Self {
            tokens,
            records: Vec::with_capacity(
                tokens
                    .iter()
                    .filter(|tk| matches!(tk, Token::Delimiter(_) | Token::Pipe))
                    .count()
                    .div_ceil(2),
            ),
            pos: 0,
        }
    }

    #[inline]
    fn last_open_from(&self, mut idx: usize) -> Option<(ResDelimKind, usize, usize)> {
        loop {
            match self.records[idx] {
                DgRecord(pos, DgEdge::Opening(kind)) => return Some((kind, idx, pos)),
                DgRecord(_, DgEdge::Closing(i)) => idx = i.checked_sub(1)?,
            }
        }
    }

    #[inline]
    fn last_open(&self) -> Option<(ResDelimKind, usize, usize)> {
        self.last_open_from(self.records.len().checked_sub(1)?)
    }

    fn delim_closed(&mut self, closing_kind: ResDelimKind) -> Result<(), SyntaxError> {
        if let Some((last_kind, last_idx, last_pos)) = self.last_open() {
            if last_kind == closing_kind {
                self.records
                    .push(DgRecord(self.pos, DgEdge::Closing(last_idx)));
                Ok(())
            } else {
                if let Some(mut start) = last_idx.checked_sub(1) {
                    let mut prev = (last_kind, last_pos);
                    while let Some((kind, idx, pos)) = self.last_open_from(start) {
                        if kind == closing_kind {
                            return Err(SyntaxError(
                                prev.0.as_missing_closing_err(),
                                prev.1..=prev.1,
                            ));
                        }
                        if let Some(ni) = idx.checked_sub(1) {
                            start = ni;
                        } else {
                            break;
                        }
                        prev = (kind, pos);
                    }
                }
                Err(SyntaxError(
                    closing_kind.as_missing_opening_err(),
                    self.pos..=self.pos,
                ))
            }
        } else {
            Err(SyntaxError(
                closing_kind.as_missing_opening_err(),
                self.pos..=self.pos,
            ))
        }
    }

    fn next(&mut self, exp: Expecting) -> Result<(), SyntaxError> {
        match self.tokens[self.pos] {
            Token::Delimiter(DelimiterToken(kind, DelimEdge::Opening))
            | Token::Function(_, kind) => self
                .records
                .push(DgRecord(self.pos, DgEdge::Opening(kind.into()))),
            Token::Delimiter(DelimiterToken(kind, DelimEdge::Closing)) => {
                self.delim_closed(kind.into())?
            }
            Token::Pipe => {
                if exp == Expecting::Prefix {
                    self.records
                        .push(DgRecord(self.pos, DgEdge::Opening(ResDelimKind::Pipe)))
                } else {
                    self.delim_closed(ResDelimKind::Pipe)?
                }
            }
            _ => (),
        }
        self.pos += 1;
        if self.pos == self.tokens.len()
            && let Some((kind, _, pos)) = self.last_open()
        {
            Err(SyntaxError(kind.as_missing_closing_err(), pos..=pos))
        } else {
            Ok(())
        }
    }

    fn back_to(&mut self, pos: usize) {
        if let Some(start) = self.records.iter().position(|r| pos <= r.0) {
            self.records.truncate(start);
        }
        self.pos = pos;
    }
}

#[derive(Clone)]
pub(super) struct ResolvedTkStream<'a, S: AsRef<str>> {
    tokens: &'a [Token<S>],
    exp_list: Vec<Expecting>,
}

impl<S: AsRef<str>> Debug for ResolvedTkStream<'_, S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolvedTkStream")
            .field("tokens", &self.tokens)
            .field("exp_list", &self.exp_list)
            .finish()
    }
}

impl<'a, S: AsRef<str>> ResolvedTkStream<'a, S> {
    pub(super) fn new<N: Number, V: VarId, F: FuncId>(
        tokens: &'a impl AsRef<[Token<S>]>,
        custom_variables: &impl NameTrie<V>,
        custom_functions: &impl NameTrie<CfInfo<F>>,
    ) -> Result<Self, SyntaxError> {
        let tokens = tokens.as_ref();
        if tokens.is_empty() {
            return Err(SyntaxError(SyntaxErrorKind::EmptyInput, 0..=0));
        }
        let last_exp_guard = |exp: Expecting, pos: usize| -> Result<Expecting, SyntaxError> {
            if pos == tokens.len() - 1 && exp == Expecting::Prefix {
                Err(if matches!(tokens.last().unwrap(), Token::Operator(_)) {
                    SyntaxError(SyntaxErrorKind::MisplacedOperator, pos..=pos)
                } else {
                    SyntaxError(SyntaxErrorKind::UnexpectedToken, pos..=pos)
                })
            } else {
                Ok(exp)
            }
        };
        let mut delimiter_guard = DelimiterGuard::new(tokens);
        let mut farthest_error: Option<(SyntaxError, usize)> = None;
        let mut exp_list: Vec<Expecting> = Vec::with_capacity(tokens.len());
        let mut opening_pipe = true;
        loop {
            let pos = exp_list.len();
            match exp_list
                .last()
                .copied()
                .unwrap_or_default()
                .next::<S, N, V, F>(
                    &tokens[pos],
                    opening_pipe,
                    custom_variables,
                    custom_functions,
                )
                .map_err(|err| clarify_error(tokens, err, pos))
                .and_then(|e| last_exp_guard(e, pos))
                .map_err(|err| (err, None))
                .and_then(|exp| match delimiter_guard.next(exp) {
                    Ok(()) => Ok(exp),
                    Err(err) => Err((err, Some(exp))),
                }) {
                Ok(next) => {
                    exp_list.push(next);
                    opening_pipe = false;
                    if exp_list.len() == tokens.len() {
                        return Ok(ResolvedTkStream { tokens, exp_list });
                    }
                }
                Err((err, tried_exp)) => {
                    if farthest_error.as_ref().is_none_or(|&(_, p)| p < pos) {
                        farthest_error = Some((err, pos));
                    }
                    // backtracking
                    if let Some(tried_exp) = tried_exp
                        && matches!(tokens[pos], Token::Pipe)
                        && tried_exp == Expecting::Suffix
                    {
                        opening_pipe = true;
                    } else {
                        loop {
                            let Some(exp) = exp_list.pop() else {
                                return Err(farthest_error.unwrap().0);
                            };
                            if matches!(tokens[exp_list.len()], Token::Pipe)
                                && exp == Expecting::Suffix
                            {
                                opening_pipe = true;
                                delimiter_guard.back_to(exp_list.len());
                                break;
                            }
                        }
                    }
                }
            }
        }
    }

    #[inline]
    pub(super) fn has_implied_mult(&self, idx: usize) -> bool {
        if idx == 0 {
            return false;
        }
        self.exp_list[idx - 1] == Expecting::Suffix
            && (self.exp_list[idx] == Expecting::Prefix
                && !matches!(self.tokens[idx], Token::Operator(_) | Token::Comma)
                || self.exp_list[idx] == Expecting::Suffix
                    && matches!(self.tokens[idx], Token::Variable(_) | Token::Number(_)))
    }

    #[inline]
    pub(super) const fn len(&self) -> usize {
        self.tokens.len()
    }

    #[inline]
    pub(super) fn get(&self, idx: usize) -> Option<ResToken<'a>> {
        Some(match self.tokens.get(idx)? {
            Token::Number(num) => ResToken::Number(num.as_ref()),
            Token::Variable(var) => ResToken::Variable(var.as_ref()),
            Token::Function(func, _) => ResToken::Function(func.as_ref()),
            Token::Delimiter(delim) => (*delim).into(),
            Token::Comma => ResToken::Comma,
            Token::Operator(OprToken::Multiply) => ResToken::Operator(ResOprToken::Multiply),
            Token::Operator(OprToken::Divide) => ResToken::Operator(ResOprToken::Divide),
            Token::Operator(OprToken::Power | OprToken::DoubleStar) => {
                ResToken::Operator(ResOprToken::Power)
            }
            Token::Operator(OprToken::Factorial) => ResToken::Operator(ResOprToken::Factorial),
            Token::Operator(OprToken::DoubleFactorial) => {
                ResToken::Operator(ResOprToken::DoubleFactorial)
            }
            Token::Operator(OprToken::Modulo) => ResToken::Operator(ResOprToken::Modulo),
            Token::Pipe => match self.exp_list[idx] {
                Expecting::Prefix => ResToken::OpenPipe,
                Expecting::Suffix => ResToken::ClosePipe,
            },
            Token::Operator(OprToken::Plus) => {
                if idx == 0 || self.exp_list[idx - 1] == Expecting::Prefix {
                    ResToken::Operator(ResOprToken::Positive)
                } else {
                    ResToken::Operator(ResOprToken::Add)
                }
            }
            Token::Operator(OprToken::Minus) => {
                if idx == 0 || self.exp_list[idx - 1] == Expecting::Prefix {
                    ResToken::Operator(ResOprToken::Negative)
                } else {
                    ResToken::Operator(ResOprToken::Subtract)
                }
            }
        })
    }

    #[inline]
    pub(super) const fn iter(&'a self) -> ResTkStreamIter<'a, S> {
        ResTkStreamIter::new(self)
    }
}

pub(super) struct ResTkStreamIter<'a, S: AsRef<str>> {
    src: &'a ResolvedTkStream<'a, S>,
    start: usize,
    end: usize,
}

impl<'a, S: AsRef<str>> ResTkStreamIter<'a, S> {
    #[inline]
    const fn new(src: &'a ResolvedTkStream<'a, S>) -> Self {
        ResTkStreamIter {
            src,
            start: 0,
            end: src.len(),
        }
    }
}

impl<'a, S: AsRef<str>> Iterator for ResTkStreamIter<'a, S> {
    type Item = ResToken<'a>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.start < self.end {
            let item = self.src.get(self.start).unwrap();
            self.start += 1;
            Some(item)
        } else {
            None
        }
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let target = self.start + n;
        if target < self.end {
            let item = self.src.get(target).unwrap();
            self.start = target + 1;
            Some(item)
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl<'a, S: AsRef<str>> ExactSizeIterator for ResTkStreamIter<'a, S> {
    #[inline]
    fn len(&self) -> usize {
        self.end - self.start
    }
}

impl<'a, S: AsRef<str>> DoubleEndedIterator for ResTkStreamIter<'a, S> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.start < self.end {
            self.end -= 1;
            Some(self.src.get(self.end).unwrap())
        } else {
            None
        }
    }

    #[inline]
    fn nth_back(&mut self, n: usize) -> Option<Self::Item> {
        self.end = self.end.checked_sub(n + 1)?;
        if self.start <= self.end {
            Some(self.src.get(self.end).unwrap())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Expecting::*;
    use super::*;
    use crate::nz;
    use crate::tokenizer::Token;
    use crate::trie::TrieNode;
    use strum::FromRepr;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, FromRepr)]
    #[repr(u8)]
    enum TestVar {
        X,
        Y,
        T,
    }

    struct TestVarsNameTrie;

    impl NameTrie<TestVar> for TestVarsNameTrie {
        fn nodes(&self) -> &[TrieNode] {
            &[
                TrieNode::Branch('x', 1),
                TrieNode::Leaf(TestVar::X as u32),
                TrieNode::Branch('y', 1),
                TrieNode::Leaf(TestVar::Y as u32),
                TrieNode::Branch('t', 1),
                TrieNode::Leaf(TestVar::T as u32),
            ]
        }
        fn leaf_to_value(&self, leaf: u32) -> TestVar {
            TestVar::from_repr(leaf as u8).unwrap()
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct TestFunc;

    struct TestFuncNameTrie;

    impl NameTrie<CfInfo<TestFunc>> for TestFuncNameTrie {
        fn nodes(&self) -> &[TrieNode] {
            &[
                TrieNode::Branch('f', 4),
                TrieNode::Branch('u', 3),
                TrieNode::Branch('n', 2),
                TrieNode::Branch('c', 1),
                TrieNode::Leaf(0),
            ]
        }
        fn leaf_to_value(&self, _leaf: u32) -> CfInfo<TestFunc> {
            CfInfo::new(TestFunc, nz!(1), None)
        }
    }

    #[test]
    fn disambiguation() {
        let disambiguate = |tokens: &[Token<&'static str>]| {
            ResolvedTkStream::new::<f64, _, _>(&tokens, &TestVarsNameTrie, &TestFuncNameTrie)
                .map(|restks| restks.exp_list)
        };
        assert_eq!(
            disambiguate(&[
                Token::Pipe,
                Token::Number("1"),
                Token::Operator(OprToken::Plus),
                Token::Pipe,
                Token::Variable("x"),
                Token::Pipe,
                Token::Pipe,
            ]),
            Ok(vec![Prefix, Suffix, Prefix, Prefix, Suffix, Suffix, Suffix])
        );
        assert_eq!(
            disambiguate(&[
                Token::Pipe,
                Token::Number("3"),
                Token::Pipe,
                Token::Variable("x"),
                Token::Pipe,
                Token::Pipe,
            ]),
            Ok(vec![Prefix, Suffix, Prefix, Suffix, Suffix, Suffix])
        );
        assert_eq!(
            disambiguate(&[
                Token::Pipe,
                Token::Variable("y"),
                Token::Pipe,
                Token::Pipe,
                Token::Variable("x"),
                Token::Pipe,
            ]),
            Ok(vec![Prefix, Suffix, Suffix, Prefix, Suffix, Suffix])
        );
    }

    #[test]
    fn res_tk_stream_iter() {
        let rtks = ResolvedTkStream::new::<f64, _, _>(
            &[
                Token::Number("1"),
                Token::Operator(OprToken::Plus),
                Token::Function("sin", DelimKind::Paren),
                Token::Number("2"),
                Token::Variable("x"),
                Token::Delimiter(DelimiterToken(DelimKind::Paren, DelimEdge::Closing)),
            ],
            &TestVarsNameTrie,
            &TestFuncNameTrie,
        )
        .unwrap();
        let mut rtksi1 = rtks.iter();
        assert_eq!(rtksi1.next(), Some(ResToken::Number("1")));
        assert_eq!(rtksi1.next(), Some(ResToken::Operator(ResOprToken::Add)));
        assert_eq!(rtksi1.next_back(), Some(ResToken::CloseDelim));
        assert_eq!(rtksi1.next_back(), Some(ResToken::Variable("x")));
        assert_eq!(rtksi1.next(), Some(ResToken::Function("sin")));
        assert_eq!(rtksi1.next_back(), Some(ResToken::Number("2")));
        assert_eq!(rtksi1.next(), None);
        assert_eq!(rtksi1.next_back(), None);

        let mut rtksi2 = rtks.iter();
        assert_eq!(rtksi2.nth(1), Some(ResToken::Operator(ResOprToken::Add)));
        assert_eq!(rtksi2.nth(2), Some(ResToken::Variable("x")));
        let mut rtksi3 = rtks.iter();
        assert_eq!(rtksi3.nth_back(1), Some(ResToken::Variable("x")));
        assert_eq!(
            rtksi3.nth_back(2),
            Some(ResToken::Operator(ResOprToken::Add))
        );
    }
}
