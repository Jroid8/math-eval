use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::ops::RangeInclusive;

use crate::asm::{CFPointer, MathAssembly, Stack};
use crate::number::{MathEvalNumber, NFPointer, NativeFunction};
use crate::tokenizer::{Token, TokenStream};
use crate::tree_utils::Tree;
use crate::{FunctionIdentifier, NAME_LIMIT, ParsingError, ParsingErrorKind, VariableIdentifier};
use indextree::{Arena, NodeEdge, NodeId};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnOperation {
    Fac,
    Neg,
}

impl UnOperation {
    pub fn parse(input: char) -> Option<Self> {
        match input {
            '!' => Some(UnOperation::Fac),
            '-' => Some(UnOperation::Neg),
            _ => None,
        }
    }

    pub fn eval<N: MathEvalNumber>(self, value: N::AsArg<'_>) -> N {
        match self {
            UnOperation::Fac => N::factorial(value),
            UnOperation::Neg => -value,
        }
    }

    pub fn as_char(&self) -> char {
        match self {
            UnOperation::Fac => '!',
            UnOperation::Neg => '-',
        }
    }
}

impl Display for UnOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_char())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BiOperation {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Mod,
}

impl BiOperation {
    pub fn parse(input: char) -> Option<BiOperation> {
        match input {
            '+' => Some(BiOperation::Add),
            '-' => Some(BiOperation::Sub),
            '*' => Some(BiOperation::Mul),
            '/' => Some(BiOperation::Div),
            '^' => Some(BiOperation::Pow),
            '%' => Some(BiOperation::Mod),
            _ => None,
        }
    }
    pub fn eval<N: MathEvalNumber>(self, lhs: N::AsArg<'_>, rhs: N::AsArg<'_>) -> N {
        match self {
            BiOperation::Add => lhs + rhs,
            BiOperation::Sub => lhs - rhs,
            BiOperation::Mul => lhs * rhs,
            BiOperation::Div => lhs / rhs,
            BiOperation::Pow => N::pow(lhs, rhs),
            BiOperation::Mod => N::modulo(lhs, rhs),
        }
    }
    pub fn as_char(self) -> char {
        match self {
            BiOperation::Add => '+',
            BiOperation::Sub => '-',
            BiOperation::Mul => '*',
            BiOperation::Div => '/',
            BiOperation::Pow => '^',
            BiOperation::Mod => '%',
        }
    }
    pub fn is_commutative(self) -> bool {
        matches!(self, BiOperation::Add | BiOperation::Mul)
    }
    pub fn precedence(self) -> u8 {
        match self {
            BiOperation::Add => 0,
            BiOperation::Sub => 0,
            BiOperation::Mul => 1,
            BiOperation::Div => 1,
            BiOperation::Mod => 1,
            BiOperation::Pow => 2,
        }
    }
    pub fn is_left_associative(self) -> bool {
        !matches!(self, BiOperation::Pow)
    }
}

impl Display for BiOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_char())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SyntaxNode<N, V, F>
where
    N: MathEvalNumber,
    V: VariableIdentifier,
    F: FunctionIdentifier,
{
    Number(N),
    Variable(V),
    BiOperation(BiOperation),
    UnOperation(UnOperation),
    NativeFunction(NativeFunction),
    CustomFunction(F),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SyntaxErrorKind {
    NumberParsingError,
    MisplacedOperator,
    UnknownVariableOrConstant,
    UnknownFunction,
    NotEnoughArguments,
    TooManyArguments,
    EmptyParenthesis,
    EmptyArgument,
    EmptyInput,
    MissingOpeningParenthesis,
    MissingClosingParenthesis,
    CommaOutsideFunction,
    PipeAbsNotClosed,
    NameTooLong,
}

pub(crate) fn token_range_to_str_range(
    input: &str,
    token_stream: &TokenStream<'_>,
    token_range: RangeInclusive<usize>,
) -> RangeInclusive<usize> {
    let mut start = 0;
    let mut index = 0;
    for (tk_idx, token) in token_stream.0[..=*token_range.end()].iter().enumerate() {
        while input.chars().nth(index).unwrap().is_whitespace() {
            index += 1
        }
        if tk_idx == *token_range.start() {
            start = index;
        }
        index += token.length();
    }
    start..=index - 1
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SyntaxError(SyntaxErrorKind, RangeInclusive<usize>);

impl SyntaxError {
    pub fn to_general(self, input: &str, token_stream: &TokenStream<'_>) -> ParsingError {
        ParsingError {
            at: if self.0 == SyntaxErrorKind::EmptyInput {
                0..=0
            } else {
                token_range_to_str_range(input, token_stream, self.1)
            },
            kind: match self.0 {
                SyntaxErrorKind::NumberParsingError => ParsingErrorKind::NumberParsingError,
                SyntaxErrorKind::MisplacedOperator => ParsingErrorKind::MisplacedOperator,
                SyntaxErrorKind::UnknownVariableOrConstant => {
                    ParsingErrorKind::UnknownVariableOrConstant
                }
                SyntaxErrorKind::UnknownFunction => ParsingErrorKind::UnknownFunction,
                SyntaxErrorKind::NotEnoughArguments => ParsingErrorKind::NotEnoughArguments,
                SyntaxErrorKind::TooManyArguments => ParsingErrorKind::TooManyArguments,
                SyntaxErrorKind::EmptyParenthesis => ParsingErrorKind::EmptyParenthesis,
                SyntaxErrorKind::EmptyArgument => ParsingErrorKind::EmptyArgument,
                SyntaxErrorKind::MissingOpeningParenthesis => {
                    ParsingErrorKind::MissingOpenParenthesis
                }
                SyntaxErrorKind::MissingClosingParenthesis => {
                    ParsingErrorKind::MissingCloseParenthesis
                }
                SyntaxErrorKind::CommaOutsideFunction => ParsingErrorKind::CommaOutsideFunction,
                SyntaxErrorKind::EmptyInput => ParsingErrorKind::EmptyInput,
                SyntaxErrorKind::PipeAbsNotClosed => ParsingErrorKind::PipeAbsNotClosed,
                SyntaxErrorKind::NameTooLong => ParsingErrorKind::NameTooLong,
            },
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SYFunction<F>
where
    F: FunctionIdentifier,
{
    NativeFunction(NativeFunction),
    CustomFunction(F, u8, Option<u8>),
    PipeAbs,
}

// FIX: check size_of and reduce size if it's too large
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SYOperator<F>
where
    F: FunctionIdentifier,
{
    BiOperation(BiOperation),
    UnOperation(UnOperation),
    Function(SYFunction<F>, u8),
    Parentheses,
}

impl<F> SYOperator<F>
where
    F: FunctionIdentifier,
{
    fn precedence(&self) -> u8 {
        match self {
            SYOperator::BiOperation(BiOperation::Add) => 0,
            SYOperator::BiOperation(BiOperation::Sub) => 0,
            SYOperator::BiOperation(BiOperation::Mul) => 1,
            SYOperator::BiOperation(BiOperation::Div) => 1,
            SYOperator::BiOperation(BiOperation::Mod) => 1,
            SYOperator::UnOperation(UnOperation::Neg) => 2,
            SYOperator::BiOperation(BiOperation::Pow) => 3,
            SYOperator::UnOperation(UnOperation::Fac) => 4,
            SYOperator::Function(_, _) | SYOperator::Parentheses => {
                unreachable!()
            }
        }
    }
    fn is_right_associative(&self) -> bool {
        matches!(self, SYOperator::BiOperation(BiOperation::Pow))
    }
    fn opr2syn<N, V>(self) -> SyntaxNode<N, V, F>
    where
        N: MathEvalNumber,
        V: VariableIdentifier,
    {
        match self {
            SYOperator::BiOperation(opr) => SyntaxNode::BiOperation(opr),
            SYOperator::UnOperation(opr) => SyntaxNode::UnOperation(opr),
            SYOperator::Function(SYFunction::NativeFunction(nf), _) => {
                SyntaxNode::NativeFunction(nf)
            }
            SYOperator::Function(SYFunction::CustomFunction(cf, _, _), _) => {
                SyntaxNode::CustomFunction(cf)
            }
            SYOperator::Function(SYFunction::PipeAbs, _) => {
                SyntaxNode::NativeFunction(NativeFunction::Abs)
            }
            SYOperator::Parentheses => unreachable!(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SyntaxTree<N: MathEvalNumber, V: VariableIdentifier, F: FunctionIdentifier>(
    pub Tree<SyntaxNode<N, V, F>>,
);

fn after_implies_neg<F>(token: Token<'_>, operator_stack: &[SYOperator<F>]) -> bool
where
    F: FunctionIdentifier,
{
    matches!(
        token,
        Token::Operation('*' | '/' | '%' | '^' | '-' | '+')
            | Token::OpenParen
            | Token::Comma
            | Token::Function(_)
    ) || matches!(token, Token::Pipe) && inside_pipe_abs(operator_stack).is_some()
}

fn shunting_yard_pop_opr<N, V, F>(
    operator_stack: &mut Vec<SYOperator<F>>,
    output_stack: &mut Vec<NodeId>,
    arena: &mut Arena<SyntaxNode<N, V, F>>,
) where
    N: MathEvalNumber,
    V: VariableIdentifier,
    F: FunctionIdentifier,
{
    let opr = operator_stack.pop().unwrap();
    let node = arena.new_node(opr.opr2syn());
    // output_stack can be empty in case of an incorrect syntax. But not enough info is available
    // in this function to find the source of the syntax error. So either this function must report
    // the error to the caller so it searches for the source itself, or errors must be detected
    // before calling this function. The latter solution is used for now.
    let child = output_stack.pop().unwrap();
    if matches!(opr, SYOperator::BiOperation(_)) {
        node.append(output_stack.pop().unwrap(), arena);
    }
    if let SyntaxNode::Number(num) = arena[child].get_mut()
        && opr == SYOperator::UnOperation(UnOperation::Neg)
    {
        *num = -num.clone();
        output_stack.push(child);
    } else {
        node.append(child, arena);
        output_stack.push(node);
    }
}

fn shunting_yard_push_opr<N, V, F>(
    operator: SYOperator<F>,
    operator_stack: &mut Vec<SYOperator<F>>,
    output_stack: &mut Vec<NodeId>,
    arena: &mut Arena<SyntaxNode<N, V, F>>,
) where
    N: MathEvalNumber,
    V: VariableIdentifier,
    F: FunctionIdentifier,
{
    while let Some(top_opr) = operator_stack.last()
        && matches!(
            top_opr,
            SYOperator::BiOperation(_) | SYOperator::UnOperation(_)
        )
        && (operator.precedence() < top_opr.precedence()
            || operator.precedence() == top_opr.precedence() && !operator.is_right_associative())
    {
        shunting_yard_pop_opr(operator_stack, output_stack, arena);
    }
    operator_stack.push(operator);
}

fn shunting_yard_flush<N, V, F>(
    operator_stack: &mut Vec<SYOperator<F>>,
    output_stack: &mut Vec<NodeId>,
    arena: &mut Arena<SyntaxNode<N, V, F>>,
) where
    N: MathEvalNumber,
    V: VariableIdentifier,
    F: FunctionIdentifier,
{
    while let Some(opr) = operator_stack.last()
        && matches!(opr, SYOperator::BiOperation(_) | SYOperator::UnOperation(_))
    {
        shunting_yard_pop_opr(operator_stack, output_stack, arena)
    }
}

fn validate_consecutive_tokens<'a>(
    last: Option<Token<'a>>,
    current: Option<Token<'a>>,
    pos: usize,
) -> Result<(), SyntaxError> {
    match (last, current) {
        (
            None | Some(Token::OpenParen | Token::Function(_)),
            Some(Token::Operation('!' | '*' | '/' | '^' | '%')),
        ) => Err(SyntaxError(SyntaxErrorKind::MisplacedOperator, pos..=pos)),
        (
            Some(Token::Operation('+' | '-' | '*' | '/' | '^' | '%')),
            None | Some(Token::CloseParen),
        ) => Err(SyntaxError(
            SyntaxErrorKind::MisplacedOperator,
            pos - 1..=pos - 1,
        )),
        (
            Some(Token::Operation('*' | '/' | '%' | '^')),
            Some(Token::Operation('*' | '/' | '%' | '^')),
        ) => Err(SyntaxError(
            SyntaxErrorKind::MisplacedOperator,
            pos - 1..=pos,
        )),
        (Some(Token::OpenParen), Some(Token::CloseParen))
        | (Some(Token::Pipe), Some(Token::Pipe)) => Err(SyntaxError(
            SyntaxErrorKind::EmptyParenthesis,
            pos - 1..=pos,
        )),
        (Some(Token::Comma | Token::Function(_)), Some(Token::CloseParen | Token::Comma)) => {
            Err(SyntaxError(SyntaxErrorKind::EmptyArgument, pos - 1..=pos))
        }
        (None, None) => Err(SyntaxError(SyntaxErrorKind::EmptyInput, 0..=0)),
        _ => Ok(()),
    }
}

fn find_opening_paren(tokens: &[Token<'_>]) -> Option<usize> {
    let mut nesting = 1;
    for (i, tk) in tokens.iter().enumerate().rev() {
        match tk {
            Token::CloseParen => nesting += 1,
            Token::OpenParen | Token::Function(_) => {
                nesting -= 1;
                if nesting == 0 {
                    return Some(i);
                }
            }
            _ => (),
        }
    }
    None
}

fn find_opening_pipe(tokens: &[Token<'_>]) -> Option<usize> {
    let mut idx = tokens.len() - 1;
    loop {
        match tokens[idx] {
            Token::CloseParen => {
                idx = find_opening_paren(&tokens[..idx]).unwrap();
            }
            Token::Pipe => return Some(idx),
            Token::OpenParen | Token::Function(_) => unreachable!(),
            _ => (),
        }
        if idx == 0 {
            return None;
        } else {
            idx -= 1;
        }
    }
}

fn inside_pipe_abs<F>(operator_stack: &[SYOperator<F>]) -> Option<usize>
where
    F: FunctionIdentifier,
{
    for (i, opr) in operator_stack.iter().enumerate().rev() {
        match opr {
            SYOperator::BiOperation(_) | SYOperator::UnOperation(_) => (),
            SYOperator::Function(SYFunction::PipeAbs, _) => return Some(i),
            SYOperator::Function(_, _) | SYOperator::Parentheses => return None,
        }
    }
    None
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Segment<N, V>
where
    N: MathEvalNumber,
    V: VariableIdentifier,
{
    Constant(N),
    Variable(V),
}

fn segment_variable<N, V>(
    input: &str,
    constant_parser: &impl Fn(&str) -> Option<N>,
    variable_parser: &impl Fn(&str) -> Option<V>,
) -> Option<Vec<Segment<N, V>>>
where
    N: MathEvalNumber,
    V: VariableIdentifier,
{
    if input.len() > NAME_LIMIT as usize {
        panic!();
    }
    let mut can_segment: Vec<Option<(Option<u8>, Segment<N, V>)>> = vec![None; input.len()];
    for i in 1..=input.len() {
        for j in 0..i {
            if j == 0 || can_segment[j - 1].is_some() {
                let prev = (j != 0).then(|| j as u8 - 1);
                let seg = constant_parser(&input[j..i])
                    .map(Segment::Constant)
                    .or_else(|| variable_parser(&input[j..i]).map(Segment::Variable))
                    .or_else(|| input[j..i].parse().ok().map(Segment::Constant));
                if let Some(seg) = seg {
                    can_segment[i - 1] = Some((prev, seg));
                    break;
                }
            }
        }
    }
    if can_segment[input.len() - 1].is_some() {
        let mut result = Vec::with_capacity(NAME_LIMIT as usize);
        let mut idx = Some(input.len() as u8 - 1);
        while let Some(i) = idx {
            let (prev, seg) = can_segment.swap_remove(i as usize).unwrap();
            result.push(seg);
            idx = prev;
        }
        result.reverse();
        Some(result)
    } else {
        None
    }
}

fn segment_function<N, V, F>(
    input: &str,
    constant_parser: &impl Fn(&str) -> Option<N>,
    variable_parser: &impl Fn(&str) -> Option<V>,
    function_parser: &impl Fn(&str) -> Option<(F, u8, Option<u8>)>,
) -> Option<(Vec<Segment<N, V>>, SYFunction<F>)>
where
    N: MathEvalNumber,
    V: VariableIdentifier,
    F: FunctionIdentifier,
{
    if input.len() > NAME_LIMIT as usize {
        panic!();
    }
    let mut can_segment: Vec<Option<(Option<u8>, Segment<N, V>)>> = vec![None; input.len() - 1];
    let mut function: Option<(u8, SYFunction<F>)> = None;
    for i in 1..input.len() {
        for j in 0..i {
            if j == 0 || can_segment[j - 1].is_some() {
                let prev = (j != 0).then(|| j as u8 - 1);
                let seg = constant_parser(&input[j..i])
                    .map(Segment::Constant)
                    .or_else(|| variable_parser(&input[j..i]).map(Segment::Variable))
                    .or_else(|| input[j..i].parse().ok().map(Segment::Constant));
                if let Some(seg) = seg {
                    can_segment[i - 1] = Some((prev, seg));
                    break;
                }
            }
        }
    }
    for j in 1..input.len() {
        if can_segment[j - 1].is_some() {
            function = NativeFunction::parse(&input[j..])
                .map(SYFunction::NativeFunction)
                .or_else(|| {
                    function_parser(&input[j..])
                        .map(|(cf, min, max)| SYFunction::CustomFunction(cf, min, max))
                })
                .map(|func| (j as u8 - 1, func));
            if function.is_some() {
                break;
            }
        }
    }
    if let Some((start, func)) = function {
        let mut result = Vec::with_capacity(NAME_LIMIT as usize);
        let mut idx = Some(start);
        while let Some(i) = idx {
            let (prev, seg) = can_segment.swap_remove(i as usize).unwrap();
            result.push(seg);
            idx = prev;
        }
        result.reverse();
        Some((result, func))
    } else {
        None
    }
}

impl<V, N, F> SyntaxTree<N, V, F>
where
    N: MathEvalNumber,
    V: VariableIdentifier,
    F: FunctionIdentifier,
{
    pub fn new<'a>(
        token_stream: &'a TokenStream<'a>,
        custom_constant_parser: impl Fn(&str) -> Option<N>,
        custom_function_parser: impl Fn(&str) -> Option<(F, u8, Option<u8>)>,
        custom_variable_parser: impl Fn(&str) -> Option<V>,
    ) -> Result<SyntaxTree<N, V, F>, SyntaxError> {
        // Dijkstra's shunting yard algorithm
        let mut arena: Arena<SyntaxNode<N, V, F>> = Arena::new();
        let mut output_stack: Vec<NodeId> = Vec::new();
        let mut operator_stack: Vec<SYOperator<F>> = Vec::new();
        let mut last_tk: Option<Token<'a>> = None;
        for (pos, &token) in token_stream.0.iter().enumerate() {
            validate_consecutive_tokens(last_tk, Some(token), pos)?;
            // for detecting implied multiplication
            if matches!(
                (last_tk, token),
                (
                    Some(
                        Token::Operation('!')
                            | Token::Number(_)
                            | Token::Variable(_)
                            | Token::CloseParen,
                    ),
                    Token::Number(_) | Token::Variable(_) | Token::OpenParen | Token::Function(_),
                ),
            ) || matches!(
                (last_tk, token),
                (
                    Some(Token::Pipe),
                    Token::Number(_) | Token::Variable(_) | Token::OpenParen | Token::Function(_)
                )
            ) && inside_pipe_abs(&operator_stack).is_none()
                || matches!(
                    (last_tk, token),
                    (
                        Some(
                            Token::Operation('!')
                                | Token::Number(_)
                                | Token::Variable(_)
                                | Token::CloseParen,
                        ),
                        Token::Pipe
                    )
                ) && inside_pipe_abs(&operator_stack).is_none()
            {
                shunting_yard_push_opr(
                    SYOperator::BiOperation(BiOperation::Mul),
                    &mut operator_stack,
                    &mut output_stack,
                    &mut arena,
                );
            }
            match token {
                Token::Number(num) => output_stack.push(
                    arena.new_node(num.parse::<N>().map(SyntaxNode::Number).map_err(|_| {
                        SyntaxError(SyntaxErrorKind::NumberParsingError, pos..=pos)
                    })?),
                ),
                Token::Variable(var) => {
                    if var.len() > NAME_LIMIT as usize {
                        return Err(SyntaxError(SyntaxErrorKind::NameTooLong, pos..=pos));
                    }
                    if let Some(node) = N::parse_constant(var)
                        .or_else(|| custom_constant_parser(var))
                        .map(|c| SyntaxNode::Number(c))
                        .or_else(|| {
                            custom_variable_parser(var).map(|var| SyntaxNode::Variable(var))
                        })
                    {
                        output_stack.push(arena.new_node(node))
                    } else if let Some(segments) =
                        segment_variable(var, &custom_constant_parser, &custom_variable_parser)
                    {
                        let mut first = true;
                        for seg in segments {
                            output_stack.push(arena.new_node(match seg {
                                Segment::Constant(c) => SyntaxNode::Number(c),
                                Segment::Variable(v) => SyntaxNode::Variable(v),
                            }));
                            if first {
                                first = false;
                            } else {
                                shunting_yard_push_opr(
                                    SYOperator::BiOperation(BiOperation::Mul),
                                    &mut operator_stack,
                                    &mut output_stack,
                                    &mut arena,
                                );
                            }
                        }
                    } else {
                        return Err(SyntaxError(
                            SyntaxErrorKind::UnknownVariableOrConstant,
                            pos..=pos,
                        ));
                    }
                }
                Token::Operation(opr) => {
                    let mut sy_opr: SYOperator<F> = BiOperation::parse(opr)
                        .map(|biopr| SYOperator::BiOperation(biopr))
                        .or_else(|| {
                            UnOperation::parse(opr).map(|unopr| SYOperator::UnOperation(unopr))
                        })
                        .unwrap();
                    if opr == '-' && last_tk.is_none_or(|tk| after_implies_neg(tk, &operator_stack))
                    {
                        sy_opr = SYOperator::UnOperation(UnOperation::Neg);
                    }
                    if opr != '+'
                        || last_tk.is_some_and(|tk| !after_implies_neg(tk, &operator_stack))
                    {
                        shunting_yard_push_opr(
                            sy_opr,
                            &mut operator_stack,
                            &mut output_stack,
                            &mut arena,
                        );
                    }
                }
                Token::Function(name) => {
                    if let Some(func) = NativeFunction::parse(name)
                        .map(|nf| SYOperator::Function(SYFunction::NativeFunction(nf), 1))
                        .or_else(|| {
                            custom_function_parser(name).map(|(cf, min, max)| {
                                SYOperator::Function(SYFunction::CustomFunction(cf, min, max), 1)
                            })
                        })
                    {
                        operator_stack.push(func);
                    } else if let Some((segments, func)) = segment_function(
                        name,
                        &custom_constant_parser,
                        &custom_variable_parser,
                        &custom_function_parser,
                    ) {
                        for seg in segments {
                            output_stack.push(arena.new_node(match seg {
                                Segment::Constant(c) => SyntaxNode::Number(c),
                                Segment::Variable(v) => SyntaxNode::Variable(v),
                            }));
                            shunting_yard_push_opr(
                                SYOperator::BiOperation(BiOperation::Mul),
                                &mut operator_stack,
                                &mut output_stack,
                                &mut arena,
                            );
                        }
                        operator_stack.push(SYOperator::Function(func, 1));
                    } else {
                        return Err(SyntaxError(SyntaxErrorKind::UnknownFunction, pos..=pos));
                    }
                }
                Token::OpenParen => {
                    operator_stack.push(SYOperator::Parentheses);
                }
                Token::Comma => {
                    shunting_yard_flush(&mut operator_stack, &mut output_stack, &mut arena);
                    match operator_stack.last_mut() {
                        Some(SYOperator::Function(_, args)) => {
                            *args += 1;
                        }
                        _ => {
                            return Err(SyntaxError(
                                SyntaxErrorKind::CommaOutsideFunction,
                                pos..=pos,
                            ));
                        }
                    }
                }
                Token::CloseParen => {
                    shunting_yard_flush(&mut operator_stack, &mut output_stack, &mut arena);
                    match operator_stack.pop() {
                        Some(SYOperator::Function(SYFunction::PipeAbs, _)) => {
                            let opening = find_opening_pipe(&token_stream.0[..pos]).unwrap();
                            return Err(SyntaxError(
                                SyntaxErrorKind::PipeAbsNotClosed,
                                opening..=opening,
                            ));
                        }
                        Some(SYOperator::Function(SYFunction::NativeFunction(nf), args)) => {
                            if args < nf.min_args() && nf != NativeFunction::Log {
                                Err(SyntaxErrorKind::NotEnoughArguments)
                            } else if nf.max_args().is_some_and(|m| args > m) {
                                Err(SyntaxErrorKind::TooManyArguments)
                            } else {
                                let node = arena.new_node(SyntaxNode::NativeFunction(nf));
                                for _ in 0..args {
                                    node.prepend(output_stack.pop().unwrap(), &mut arena);
                                }
                                if nf == NativeFunction::Log {
                                    match node.children(&arena).nth(1) {
                                        Some(base)
                                            if SyntaxNode::Number(N::from(10))
                                                == *arena[base].get() =>
                                        {
                                            *arena[node].get_mut() =
                                                SyntaxNode::NativeFunction(NativeFunction::Log10);
                                            base.remove(&mut arena);
                                        }
                                        Some(base)
                                            if SyntaxNode::Number(N::from(2))
                                                == *arena[base].get() =>
                                        {
                                            *arena[node].get_mut() =
                                                SyntaxNode::NativeFunction(NativeFunction::Log2);
                                            base.remove(&mut arena);
                                        }
                                        None => {
                                            *arena[node].get_mut() =
                                                SyntaxNode::NativeFunction(NativeFunction::Log10);
                                        }
                                        _ => (),
                                    }
                                }
                                output_stack.push(node);
                                Ok(())
                            }
                            .map_err(|e| {
                                SyntaxError(
                                    e,
                                    find_opening_paren(&token_stream.0[..pos]).unwrap()..=pos,
                                )
                            })?
                        }
                        Some(SYOperator::Function(
                            SYFunction::CustomFunction(cf, min_args, max_args),
                            args,
                        )) => if args < min_args {
                            Err(SyntaxErrorKind::NotEnoughArguments)
                        } else if max_args.is_some_and(|m| args > m) {
                            Err(SyntaxErrorKind::TooManyArguments)
                        } else {
                            let node = arena.new_node(SyntaxNode::CustomFunction(cf));
                            for _ in 0..args {
                                node.prepend(output_stack.pop().unwrap(), &mut arena);
                            }
                            output_stack.push(node);
                            Ok(())
                        }
                        .map_err(|e| {
                            SyntaxError(
                                e,
                                find_opening_paren(&token_stream.0[..pos]).unwrap()..=pos,
                            )
                        })?,
                        Some(SYOperator::Parentheses) => (),
                        _ => {
                            return Err(SyntaxError(
                                SyntaxErrorKind::MissingOpeningParenthesis,
                                pos..=pos,
                            ));
                        }
                    }
                }
                Token::Pipe => {
                    if let Some(opening_pipe) = inside_pipe_abs(&operator_stack) {
                        shunting_yard_flush(&mut operator_stack, &mut output_stack, &mut arena);
                        if let Some(SYOperator::Function(SYFunction::PipeAbs, args)) =
                            operator_stack.pop()
                            && args > 1
                        {
                            return Err(SyntaxError(
                                SyntaxErrorKind::TooManyArguments,
                                opening_pipe..=pos,
                            ));
                        }
                        let node = arena.new_node(SyntaxNode::NativeFunction(NativeFunction::Abs));
                        node.append(output_stack.pop().unwrap(), &mut arena);
                        output_stack.push(node);
                    } else {
                        operator_stack.push(SYOperator::Function(SYFunction::PipeAbs, 0));
                    }
                }
            }
            last_tk = Some(token);
        }
        validate_consecutive_tokens(last_tk, None, token_stream.0.len())?;
        shunting_yard_flush(&mut operator_stack, &mut output_stack, &mut arena);
        match operator_stack.last() {
            Some(SYOperator::Function(SYFunction::PipeAbs, _)) => {
                let opening = find_opening_pipe(&token_stream.0).unwrap();
                Err(SyntaxError(
                    SyntaxErrorKind::PipeAbsNotClosed,
                    opening..=opening,
                ))
            }
            Some(SYOperator::Function(_, _) | SYOperator::Parentheses) => {
                let unclosed_paren_pos = find_opening_paren(&token_stream.0).unwrap();
                Err(SyntaxError(
                    SyntaxErrorKind::MissingClosingParenthesis,
                    unclosed_paren_pos..=unclosed_paren_pos,
                ))
            }
            _ => Ok(SyntaxTree(Tree {
                arena,
                root: output_stack.pop().unwrap(),
            })),
        }
    }

    pub fn eval<'a>(
        &self,
        function_to_pointer: impl Fn(F) -> CFPointer<'a, N>,
        variable_values: &impl crate::VariableStore<N, V>,
    ) -> N {
        let mut stack: Stack<N> = Stack::new();
        let is_fixed_input = |node: Option<NodeId>| match node.map(|id| self.0.arena[id].get()) {
            Some(SyntaxNode::BiOperation(_) | SyntaxNode::UnOperation(_)) => true,
            Some(SyntaxNode::NativeFunction(nf)) => nf.is_fixed(),
            Some(SyntaxNode::CustomFunction(cf)) => {
                !matches!(function_to_pointer(*cf), CFPointer::Flexible(_))
            }
            _ => false,
        };

        for current in self.0.root.traverse(&self.0.arena).filter_map(|n| match n {
            NodeEdge::Start(_) => None,
            NodeEdge::End(id) => Some(id),
        }) {
            let mut argnum = stack.len();
            macro_rules! get {
                ($node: expr) => {
                    match self.0.arena[$node.unwrap()].get() {
                        SyntaxNode::Number(num) => num.asarg(),
                        SyntaxNode::Variable(var) => variable_values.get(*var),
                        _ => {
                            argnum -= 1;
                            stack[argnum].asarg()
                        }
                    }
                };
            }

            let mut children = current.children(&self.0.arena);
            let parent = current.ancestors(&self.0.arena).nth(1);

            let result = match self.0.arena[current].get() {
                SyntaxNode::Number(num) => {
                    if is_fixed_input(parent) {
                        continue;
                    } else {
                        num.clone()
                    }
                }
                SyntaxNode::Variable(var) => {
                    if is_fixed_input(parent) {
                        continue;
                    } else {
                        variable_values.get(*var).to_owned()
                    }
                }
                SyntaxNode::UnOperation(opr) => opr.eval(get!(children.next())),
                SyntaxNode::BiOperation(opr) => {
                    opr.eval(get!(children.next()), get!(children.next()))
                }
                SyntaxNode::NativeFunction(nf) => match nf.to_pointer() {
                    NFPointer::Single(func) => func(get!(children.next())),
                    NFPointer::Dual(func) => func(get!(children.next()), get!(children.next())),
                    NFPointer::Flexible(func) => {
                        argnum -= children.count();
                        func(&stack[argnum..])
                    }
                },
                SyntaxNode::CustomFunction(cf) => match function_to_pointer(*cf) {
                    CFPointer::Single(func) => func(get!(children.next())),
                    CFPointer::Dual(func) => func(get!(children.next()), get!(children.next())),
                    CFPointer::Triple(func) => func(
                        get!(children.next()),
                        get!(children.next()),
                        get!(children.next()),
                    ),
                    CFPointer::Flexible(func) => {
                        argnum -= children.count();
                        func(&stack[argnum..])
                    }
                },
            };
            stack.truncate(argnum);
            stack.push(result);
        }
        stack.pop().unwrap()
    }

    pub fn to_asm<'a>(
        &self,
        function_to_pointer: impl Fn(F) -> CFPointer<'a, N>,
        variable_order: &[V],
    ) -> MathAssembly<'a, N, F> {
        MathAssembly::new(
            &self.0.arena,
            self.0.root,
            function_to_pointer,
            variable_order,
        )
    }

    pub fn aot_evaluation<'a>(&mut self, function_to_pointer: impl Fn(F) -> CFPointer<'a, N>) {
        let mut examin: Vec<NodeId> = Vec::new();
        for node in self.0.root.traverse(&self.0.arena) {
            if let NodeEdge::End(node) = node {
                match self.0.arena[node].get() {
                    SyntaxNode::Number(_) | SyntaxNode::Variable(_) => (),
                    _ => examin.push(node),
                }
            }
        }
        for node in examin {
            if node.children(&self.0.arena).all(|c| self.is_number(c)) {
                let answer = MathAssembly::new(&self.0.arena, node, &function_to_pointer, &[])
                    .eval(&[], &mut crate::asm::Stack::new());
                *self.0.arena[node].get_mut() = SyntaxNode::Number(answer);
                while let Some(c) = self.0.arena[node].first_child() {
                    c.remove(&mut self.0.arena);
                }
            }
        }
    }

    fn is_number(&self, node: NodeId) -> bool {
        matches!(self.0.arena[node].get(), SyntaxNode::Number(_))
    }

    pub fn displacing_simplification(&mut self) {
        self._displacing_simplification(BiOperation::Add, BiOperation::Sub, 0.into());
        self._displacing_simplification(BiOperation::Mul, BiOperation::Div, 1.into());
    }

    fn _displacing_simplification(&mut self, pos: BiOperation, neg: BiOperation, inital_value: N) {
        let is_targeting_opr = |node: NodeId| matches!(self.0.arena[node].get(), SyntaxNode::BiOperation(opr) if *opr == pos || *opr == neg);
        let mut found: Vec<NodeId> = Vec::new();
        let mul_opr = |target: BiOperation, side: usize, parent: BiOperation| {
            if side == 0 {
                parent
            } else if target == parent {
                pos
            } else {
                neg
            }
        };
        for node in self.0.root.traverse(&self.0.arena) {
            if let NodeEdge::End(upper) = node
                && is_targeting_opr(upper)
                && upper.children(&self.0.arena).all(|lower| {
                    is_targeting_opr(lower)
                        && lower
                            .children(&self.0.arena)
                            .any(|lowest| self.is_number(lowest))
                        || self.is_number(lower)
                })
            {
                found.push(upper);
            }
        }
        for upper in found {
            let SyntaxNode::BiOperation(upper_opr) = self.0.arena[upper].get() else {
                panic!();
            };
            let mut symbols: [Option<(NodeId, bool)>; 2] = [None, None];
            let mut lhs = inital_value.clone();
            for (upper_side, lower) in upper.children(&self.0.arena).enumerate() {
                match self.0.arena[lower].get() {
                    SyntaxNode::BiOperation(lower_opr) => {
                        for (lower_side, lowest) in lower.children(&self.0.arena).enumerate() {
                            let opr = mul_opr(
                                *lower_opr,
                                lower_side,
                                if upper_side == 0 { pos } else { *upper_opr },
                            );
                            match self.0.arena[lowest].get() {
                                SyntaxNode::Number(value) => {
                                    lhs = opr.eval(lhs.asarg(), value.asarg())
                                }
                                _ => {
                                    symbols[symbols[0].is_some() as usize] =
                                        Some((lowest, opr == neg))
                                }
                            }
                        }
                    }
                    SyntaxNode::Number(value) => {
                        lhs =
                            (mul_opr(*upper_opr, upper_side, pos)).eval(lhs.asarg(), value.asarg())
                    }
                    _ => panic!(),
                }
            }
            let symb1 = symbols[0].unwrap();
            symb1.0.detach(&mut self.0.arena);
            if let Some((sym, _)) = symbols[1] {
                sym.detach(&mut self.0.arena);
            }
            while let Some(child) = upper.children(&self.0.arena).next() {
                child.remove_subtree(&mut self.0.arena);
            }
            upper.append_value(SyntaxNode::Number(lhs), &mut self.0.arena);
            if let Some(symb2) = symbols[1] {
                if symb1.1 == symb2.1 {
                    *self.0.arena[upper].get_mut() =
                        SyntaxNode::BiOperation(if symb1.1 { neg } else { pos });
                    let lower = upper.append_value(SyntaxNode::BiOperation(pos), &mut self.0.arena);
                    lower.append(symb1.0, &mut self.0.arena);
                    lower.append(symb2.0, &mut self.0.arena);
                } else {
                    *self.0.arena[upper].get_mut() = SyntaxNode::BiOperation(pos);
                    let lower = upper.append_value(SyntaxNode::BiOperation(neg), &mut self.0.arena);
                    if symb2.1 {
                        lower.append(symb1.0, &mut self.0.arena);
                        lower.append(symb2.0, &mut self.0.arena);
                    } else {
                        lower.append(symb2.0, &mut self.0.arena);
                        lower.append(symb1.0, &mut self.0.arena);
                    }
                }
            } else {
                *self.0.arena[upper].get_mut() =
                    SyntaxNode::BiOperation(if symb1.1 { neg } else { pos });
                upper.append(symb1.0, &mut self.0.arena);
            }
        }
    }

    #[allow(dead_code)]
    fn verify(
        arena: &indextree::Arena<SyntaxNode<N, V, F>>,
        root: NodeId,
        cf_bounds: impl Fn(F) -> (u8, Option<u8>),
    ) -> bool {
        macro_rules! short {
            ($res: expr) => {
                if !($res) {
                    return false;
                }
            };
        }

        for id in root.traverse(arena).filter_map(|e| match e {
            NodeEdge::Start(_) => None,
            NodeEdge::End(id) => Some(id),
        }) {
            let child_count = id.children(arena).count();
            match arena[id].get() {
                SyntaxNode::Number(_) | SyntaxNode::Variable(_) => short!(child_count == 0),
                SyntaxNode::BiOperation(_) => short!(child_count == 2),
                SyntaxNode::UnOperation(_) => short!(child_count == 1),
                SyntaxNode::NativeFunction(nf) => short!(
                    child_count >= nf.min_args() as usize
                        && nf.max_args().is_none_or(|m| child_count <= m as usize)
                ),
                SyntaxNode::CustomFunction(cf) => {
                    let (min, max) = cf_bounds(*cf);
                    short!(
                        child_count >= min as usize
                            && max.is_none_or(|m| child_count <= m as usize)
                    )
                }
            }
        }
        true
    }
}

impl<V, N, F> Display for SyntaxTree<N, V, F>
where
    N: MathEvalNumber + Display,
    V: VariableIdentifier + Display,
    F: FunctionIdentifier + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let arena = &self.0.arena;
        for edge in self.0.root.traverse(arena) {
            match edge {
                NodeEdge::Start(node) => match arena[node].get() {
                    SyntaxNode::Number(num) => Display::fmt(num, f)?,
                    SyntaxNode::Variable(var) => Display::fmt(&var, f)?,
                    SyntaxNode::BiOperation(opr) => {
                        if matches!(
                            node.ancestors(arena).nth(1).map(|p| arena[p].get()),
                            Some(SyntaxNode::BiOperation(paopr))
                                if opr.precedence() < paopr.precedence()
                                || !paopr.is_commutative() && opr.precedence() <= paopr.precedence()
                        ) {
                            f.write_str("(")?;
                        }
                    }
                    SyntaxNode::UnOperation(UnOperation::Neg) => {
                        // parent is BiOperation, and Neg is the rhs
                        if node
                            .ancestors(arena)
                            .nth(1)
                            .filter(|p| matches!(arena[*p].get(), SyntaxNode::BiOperation(_)))
                            .map(|p| p.children(arena).nth(1))
                            .is_some()
                        {
                            f.write_str("(")?;
                        }
                        Display::fmt(&UnOperation::Neg, f)?;
                    }
                    SyntaxNode::NativeFunction(nf) => {
                        Display::fmt(nf, f)?;
                        f.write_str("(")?
                    }
                    SyntaxNode::CustomFunction(cf) => {
                        Display::fmt(&cf, f)?;
                        f.write_str("(")?
                    }
                    _ => (),
                },
                NodeEdge::End(node) => {
                    match arena[node].get() {
                        SyntaxNode::BiOperation(opr) => {
                            if matches!(
                                node.ancestors(arena).nth(1).map(|p| arena[p].get()),
                                Some(SyntaxNode::BiOperation(paopr))
                                    if opr.precedence() < paopr.precedence()
                                    || !paopr.is_commutative() && opr.precedence() <= paopr.precedence()
                            ) {
                                f.write_str(")")?
                            }
                        }
                        SyntaxNode::UnOperation(UnOperation::Neg) => {
                            if node
                                .ancestors(arena)
                                .nth(1)
                                .filter(|p| matches!(arena[*p].get(), SyntaxNode::BiOperation(_)))
                                .map(|p| p.children(arena).nth(1))
                                .is_some()
                            {
                                f.write_str(")")?;
                            }
                        }
                        SyntaxNode::UnOperation(UnOperation::Fac) => {
                            f.write_str("!")?;
                        }
                        SyntaxNode::NativeFunction(_) | SyntaxNode::CustomFunction(_) => {
                            f.write_str(")")?
                        }
                        _ => (),
                    };
                    if node.following_siblings(arena).nth(1).is_some() {
                        match node.ancestors(arena).nth(1).map(|p| arena[p].get()) {
                            Some(SyntaxNode::NativeFunction(_) | SyntaxNode::CustomFunction(_)) => {
                                f.write_str(", ")?
                            }
                            Some(SyntaxNode::BiOperation(opr))
                                if *opr == BiOperation::Add || *opr == BiOperation::Sub =>
                            {
                                write!(f, " {} ", opr.as_char())?;
                            }
                            Some(SyntaxNode::BiOperation(opr)) => Display::fmt(&opr, f)?,
                            _ => (),
                        }
                    }
                }
            };
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use indextree::Arena;
    use std::f64::consts::*;

    use super::*;
    use crate::VariableStore;
    use crate::tokenizer::TokenStream;
    use crate::tree_utils::VecTree::{self, Leaf};

    macro_rules! branch {
        ($node:expr, $($children:expr),+ $(,)?) => {
            VecTree::Branch($node,vec![$($children),+])
        };
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    enum TestVar {
        X,
        Y,
        T,
    }

    impl Display for TestVar {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                TestVar::X => f.write_str("x"),
                TestVar::Y => f.write_str("y"),
                TestVar::T => f.write_str("t"),
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum TestFunc {
        Deg2Rad,
        ExpD,
        Clamp,
        Digits,
    }

    impl Display for TestFunc {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                TestFunc::Deg2Rad => f.write_str("deg2rad"),
                TestFunc::ExpD => f.write_str("expd"),
                TestFunc::Clamp => f.write_str("clamp"),
                TestFunc::Digits => f.write_str("digits"),
            }
        }
    }

    #[test]
    fn test_segment_variable() {
        macro_rules! seg_var {
            ($input: expr) => {
                segment_variable(
                    $input,
                    &|input| match input {
                        "c" => Some(299792458.0),
                        "pi2" => Some(FRAC_2_PI),
                        _ => None,
                    },
                    &|input| match input {
                        "x" => Some(0),
                        "y" => Some(1),
                        "var5" => Some(2),
                        "shallnotbenamed" => Some(3),
                        _ => None,
                    },
                )
            };
        }
        assert_eq!(
            seg_var!("xy"),
            Some(vec![Segment::Variable(0), Segment::Variable(1)])
        );
        assert_eq!(
            seg_var!("cx"),
            Some(vec![Segment::Constant(299792458.0), Segment::Variable(0)])
        );
        assert_eq!(
            seg_var!("x2var5"),
            Some(vec![
                Segment::Variable(0),
                Segment::Constant(2.0),
                Segment::Variable(2)
            ])
        );
        assert_eq!(
            seg_var!("x2yy"),
            Some(vec![
                Segment::Variable(0),
                Segment::Constant(2.0),
                Segment::Variable(1),
                Segment::Variable(1)
            ])
        );
        assert_eq!(
            seg_var!("pi2x"),
            Some(vec![Segment::Constant(FRAC_2_PI), Segment::Variable(0)])
        );
        assert_eq!(
            seg_var!("x8759y"),
            Some(vec![
                Segment::Variable(0),
                Segment::Constant(8759.0),
                Segment::Variable(1)
            ])
        );
        assert_eq!(
            seg_var!("x9shallnotbenamedy"),
            Some(vec![
                Segment::Variable(0),
                Segment::Constant(9.0),
                Segment::Variable(3),
                Segment::Variable(1),
            ])
        );
    }

    #[test]
    fn test_segment_function() {
        macro_rules! seg_func {
            ($input: expr) => {
                segment_function(
                    $input,
                    &|input| match input {
                        "c" => Some(299792458.0),
                        "pi4" => Some(FRAC_PI_2),
                        _ => None,
                    },
                    &|input| match input {
                        "x" => Some(0),
                        "y" => Some(1),
                        "var5" => Some(2),
                        "shallnotbenamed" => Some(3),
                        _ => None,
                    },
                    &|input| match input {
                        "func1" => Some((0, 0, None)),
                        "f2" => Some((1, 0, None)),
                        "verylongfunction" => Some((2, 0, None)),
                        _ => None,
                    },
                )
            };
        }
        assert_eq!(
            seg_func!("cmin"),
            Some((
                vec![Segment::Constant(299792458.0)],
                SYFunction::NativeFunction(NativeFunction::Min)
            ))
        );
        assert_eq!(
            seg_func!("x55sin"),
            Some((
                vec![Segment::Variable(0), Segment::Constant(55.0)],
                SYFunction::NativeFunction(NativeFunction::Sin)
            ))
        );
        assert_eq!(
            seg_func!("xyf2"),
            Some((
                vec![Segment::Variable(0), Segment::Variable(1)],
                SYFunction::CustomFunction(1, 0, None)
            ))
        );
        assert_eq!(
            seg_func!("xvar5func1"),
            Some((
                vec![Segment::Variable(0), Segment::Variable(2)],
                SYFunction::CustomFunction(0, 0, None)
            ))
        );
        assert_eq!(
            seg_func!("xxxxvar5verylongfunction"),
            Some((
                vec![
                    Segment::Variable(0),
                    Segment::Variable(0),
                    Segment::Variable(0),
                    Segment::Variable(0),
                    Segment::Variable(2)
                ],
                SYFunction::CustomFunction(2, 0, None)
            ))
        );
    }

    fn parse(input: &str) -> Result<SyntaxTree<f64, TestVar, TestFunc>, ParsingError> {
        let token_stream = TokenStream::new(input).map_err(|e| e.to_general())?;
        SyntaxTree::new(
            &token_stream,
            |inp| match inp {
                "c" => Some(299792458.0),
                _ => None,
            },
            |input| match input {
                "deg2rad" => Some((TestFunc::Deg2Rad, 1, Some(1))),
                "expd" => Some((TestFunc::ExpD, 2, Some(2))),
                "clamp" => Some((TestFunc::Clamp, 3, Some(3))),
                "digits" => Some((TestFunc::Digits, 1, None)),
                _ => None,
            },
            |input| match input {
                "x" => Some(TestVar::X),
                "y" => Some(TestVar::Y),
                "t" => Some(TestVar::T),
                _ => None,
            },
        )
        .map_err(|e| e.to_general(input, &token_stream))
    }

    #[test]
    fn test_syntaxify() {
        fn syntaxify(
            input: &str,
        ) -> Result<VecTree<SyntaxNode<f64, TestVar, TestFunc>>, ParsingError> {
            parse(input).map(|st| VecTree::new(&st.0.arena, st.0.root))
        }
        assert_eq!(syntaxify("0"), Ok(Leaf(SyntaxNode::Number(0.0))));
        assert_eq!(syntaxify("(0)"), Ok(Leaf(SyntaxNode::Number(0.0))));
        assert_eq!(syntaxify("((0))"), Ok(Leaf(SyntaxNode::Number(0.0))));
        assert_eq!(syntaxify("pi"), Ok(Leaf(SyntaxNode::Number(PI))));
        assert_eq!(
            syntaxify("1+1"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Add),
                Leaf(SyntaxNode::Number(1.0)),
                Leaf(SyntaxNode::Number(1.0))
            ))
        );
        assert_eq!(syntaxify("-0.5"), Ok(Leaf(SyntaxNode::Number(-0.5))));
        assert_eq!(
            syntaxify("5-3"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Sub),
                Leaf(SyntaxNode::Number(5.0)),
                Leaf(SyntaxNode::Number(3.0))
            ))
        );
        assert_eq!(
            syntaxify("-y!"),
            Ok(branch!(
                SyntaxNode::UnOperation(UnOperation::Neg),
                branch!(
                    SyntaxNode::UnOperation(UnOperation::Fac),
                    Leaf(SyntaxNode::Variable(TestVar::Y))
                )
            ))
        );
        assert_eq!(
            syntaxify("8*3+1"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Add),
                branch!(
                    SyntaxNode::BiOperation(BiOperation::Mul),
                    Leaf(SyntaxNode::Number(8.0)),
                    Leaf(SyntaxNode::Number(3.0))
                ),
                Leaf(SyntaxNode::Number(1.0))
            ))
        );
        assert_eq!(
            syntaxify("12/3/2"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Div),
                branch!(
                    SyntaxNode::BiOperation(BiOperation::Div),
                    Leaf(SyntaxNode::Number(12.0)),
                    Leaf(SyntaxNode::Number(3.0))
                ),
                Leaf(SyntaxNode::Number(2.0))
            ))
        );
        assert_eq!(
            syntaxify("8*(3+1)"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Mul),
                Leaf(SyntaxNode::Number(8.0)),
                branch!(
                    SyntaxNode::BiOperation(BiOperation::Add),
                    Leaf(SyntaxNode::Number(3.0)),
                    Leaf(SyntaxNode::Number(1.0))
                )
            ))
        );
        assert_eq!(
            syntaxify("8*3^2-1"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Sub),
                branch!(
                    SyntaxNode::BiOperation(BiOperation::Mul),
                    Leaf(SyntaxNode::Number(8.0)),
                    branch!(
                        SyntaxNode::BiOperation(BiOperation::Pow),
                        Leaf(SyntaxNode::Number(3.0)),
                        Leaf(SyntaxNode::Number(2.0))
                    )
                ),
                Leaf(SyntaxNode::Number(1.0))
            ))
        );
        assert_eq!(
            syntaxify("2x"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Mul),
                Leaf(SyntaxNode::Number(2.0)),
                Leaf(SyntaxNode::Variable(TestVar::X))
            ))
        );
        assert_eq!(
            syntaxify("sin(14)"),
            Ok(branch!(
                SyntaxNode::NativeFunction(NativeFunction::Sin),
                Leaf(SyntaxNode::Number(14.0))
            ))
        );
        assert_eq!(
            syntaxify("deg2rad(80)"),
            Ok(branch!(
                SyntaxNode::CustomFunction(TestFunc::Deg2Rad),
                Leaf(SyntaxNode::Number(80.0))
            ))
        );
        assert_eq!(
            syntaxify("expd(0.2, x)"),
            Ok(branch!(
                SyntaxNode::CustomFunction(TestFunc::ExpD),
                Leaf(SyntaxNode::Number(0.2)),
                Leaf(SyntaxNode::Variable(TestVar::X))
            ))
        );
        assert_eq!(
            syntaxify("clamp(t, y, x)"),
            Ok(branch!(
                SyntaxNode::CustomFunction(TestFunc::Clamp),
                Leaf(SyntaxNode::Variable(TestVar::T)),
                Leaf(SyntaxNode::Variable(TestVar::Y)),
                Leaf(SyntaxNode::Variable(TestVar::X))
            ))
        );
        assert_eq!(
            syntaxify("digits(3, 1, 5, 7, 2, x)"),
            Ok(branch!(
                SyntaxNode::CustomFunction(TestFunc::Digits),
                Leaf(SyntaxNode::Number(3.0)),
                Leaf(SyntaxNode::Number(1.0)),
                Leaf(SyntaxNode::Number(5.0)),
                Leaf(SyntaxNode::Number(7.0)),
                Leaf(SyntaxNode::Number(2.0)),
                Leaf(SyntaxNode::Variable(TestVar::X))
            ))
        );
        assert_eq!(
            syntaxify("log2(8)"),
            Ok(branch!(
                SyntaxNode::NativeFunction(NativeFunction::Log2),
                Leaf(SyntaxNode::Number(8.0))
            ))
        );
        assert_eq!(
            syntaxify("log(100)"),
            Ok(branch!(
                SyntaxNode::NativeFunction(NativeFunction::Log10),
                Leaf(SyntaxNode::Number(100.0))
            ))
        );
        assert_eq!(
            syntaxify("sin(cos(0))"),
            Ok(branch!(
                SyntaxNode::NativeFunction(NativeFunction::Sin),
                branch!(
                    SyntaxNode::NativeFunction(NativeFunction::Cos),
                    Leaf(SyntaxNode::Number(0.0))
                )
            ))
        );
        assert_eq!(
            syntaxify("x^2 + sin(y)"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Add),
                branch!(
                    SyntaxNode::BiOperation(BiOperation::Pow),
                    Leaf(SyntaxNode::Variable(TestVar::X)),
                    Leaf(SyntaxNode::Number(2.0))
                ),
                branch!(
                    SyntaxNode::NativeFunction(NativeFunction::Sin),
                    Leaf(SyntaxNode::Variable(TestVar::Y))
                )
            ))
        );
        assert_eq!(
            syntaxify("sqrt(max(4, 9))"),
            Ok(branch!(
                SyntaxNode::NativeFunction(NativeFunction::Sqrt),
                branch!(
                    SyntaxNode::NativeFunction(NativeFunction::Max),
                    Leaf(SyntaxNode::Number(4.0)),
                    Leaf(SyntaxNode::Number(9.0))
                )
            ))
        );
        assert_eq!(
            syntaxify("max(2, x, 8y, x*y+1)"),
            Ok(branch!(
                SyntaxNode::NativeFunction(NativeFunction::Max),
                Leaf(SyntaxNode::Number(2.0)),
                Leaf(SyntaxNode::Variable(TestVar::X)),
                branch!(
                    SyntaxNode::BiOperation(BiOperation::Mul),
                    Leaf(SyntaxNode::Number(8.0)),
                    Leaf(SyntaxNode::Variable(TestVar::Y))
                ),
                branch!(
                    SyntaxNode::BiOperation(BiOperation::Add),
                    branch!(
                        SyntaxNode::BiOperation(BiOperation::Mul),
                        Leaf(SyntaxNode::Variable(TestVar::X)),
                        Leaf(SyntaxNode::Variable(TestVar::Y))
                    ),
                    Leaf(SyntaxNode::Number(1.0))
                )
            ))
        );
        assert_eq!(
            syntaxify("2*x + 3*y"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Add),
                branch!(
                    SyntaxNode::BiOperation(BiOperation::Mul),
                    Leaf(SyntaxNode::Number(2.0)),
                    Leaf(SyntaxNode::Variable(TestVar::X))
                ),
                branch!(
                    SyntaxNode::BiOperation(BiOperation::Mul),
                    Leaf(SyntaxNode::Number(3.0)),
                    Leaf(SyntaxNode::Variable(TestVar::Y))
                )
            ))
        );
        assert_eq!(
            syntaxify("log10(1000)"),
            Ok(branch!(
                SyntaxNode::NativeFunction(NativeFunction::Log10),
                Leaf(SyntaxNode::Number(1000.0))
            ))
        );
        assert_eq!(
            syntaxify("1/(2+3)"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Div),
                Leaf(SyntaxNode::Number(1.0)),
                branch!(
                    SyntaxNode::BiOperation(BiOperation::Add),
                    Leaf(SyntaxNode::Number(2.0)),
                    Leaf(SyntaxNode::Number(3.0))
                )
            ))
        );
        assert_eq!(
            syntaxify("e^x"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Pow),
                Leaf(SyntaxNode::Number(E)),
                Leaf(SyntaxNode::Variable(TestVar::X))
            ))
        );
        assert_eq!(
            syntaxify("x * -2"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Mul),
                Leaf(SyntaxNode::Variable(TestVar::X)),
                Leaf(SyntaxNode::Number(-2.0))
            ))
        );
        assert_eq!(
            syntaxify("4/-1.33"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Div),
                Leaf(SyntaxNode::Number(4.0)),
                Leaf(SyntaxNode::Number(-1.33))
            ))
        );
        assert_eq!(
            syntaxify("sqrt(16)"),
            Ok(branch!(
                SyntaxNode::NativeFunction(NativeFunction::Sqrt),
                Leaf(SyntaxNode::Number(16.0))
            ))
        );
        assert_eq!(
            syntaxify("abs(-5)"),
            Ok(branch!(
                SyntaxNode::NativeFunction(NativeFunction::Abs),
                Leaf(SyntaxNode::Number(-5.0))
            ))
        );
        assert_eq!(
            syntaxify("x^2 - y^2"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Sub),
                branch!(
                    SyntaxNode::BiOperation(BiOperation::Pow),
                    Leaf(SyntaxNode::Variable(TestVar::X)),
                    Leaf(SyntaxNode::Number(2.0))
                ),
                branch!(
                    SyntaxNode::BiOperation(BiOperation::Pow),
                    Leaf(SyntaxNode::Variable(TestVar::Y)),
                    Leaf(SyntaxNode::Number(2.0))
                )
            ))
        );
        assert_eq!(
            syntaxify("|x|"),
            Ok(branch!(
                SyntaxNode::NativeFunction(NativeFunction::Abs),
                Leaf(SyntaxNode::Variable(TestVar::X)),
            ))
        );
        assert_eq!(
            syntaxify("|-x|"),
            Ok(branch!(
                SyntaxNode::NativeFunction(NativeFunction::Abs),
                branch!(
                    SyntaxNode::UnOperation(UnOperation::Neg),
                    Leaf(SyntaxNode::Variable(TestVar::X))
                )
            ))
        );
        assert_eq!(
            syntaxify("4*-x"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Mul),
                Leaf(SyntaxNode::Number(4.0)),
                branch!(
                    SyntaxNode::UnOperation(UnOperation::Neg),
                    Leaf(SyntaxNode::Variable(TestVar::X))
                )
            ))
        );
        assert_eq!(
            syntaxify("3!"),
            Ok(branch!(
                SyntaxNode::UnOperation(UnOperation::Fac),
                Leaf(SyntaxNode::Number(3.0))
            ))
        );
        assert_eq!(
            syntaxify("8.3 + -1"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Add),
                Leaf(SyntaxNode::Number(8.3)),
                Leaf(SyntaxNode::Number(-1.0))
            ))
        );
        assert_eq!(syntaxify("++1"), Ok(Leaf(SyntaxNode::Number(1.0))));
        assert_eq!(syntaxify("+-1"), Ok(Leaf(SyntaxNode::Number(-1.0))));
        assert_eq!(
            syntaxify("x + +1"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Add),
                Leaf(SyntaxNode::Variable(TestVar::X)),
                Leaf(SyntaxNode::Number(1.0))
            ))
        );
        assert_eq!(
            syntaxify("-1^2"),
            Ok(branch!(
                SyntaxNode::UnOperation(UnOperation::Neg),
                branch!(
                    SyntaxNode::BiOperation(BiOperation::Pow),
                    Leaf(SyntaxNode::Number(1.0)),
                    Leaf(SyntaxNode::Number(2.0))
                )
            ))
        );
        assert_eq!(
            syntaxify("x!y"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Mul),
                branch!(
                    SyntaxNode::UnOperation(UnOperation::Fac),
                    Leaf(SyntaxNode::Variable(TestVar::X)),
                ),
                Leaf(SyntaxNode::Variable(TestVar::Y)),
            ))
        );
        assert_eq!(
            syntaxify("sin(x!-1)"),
            Ok(branch!(
                SyntaxNode::NativeFunction(NativeFunction::Sin),
                branch!(
                    SyntaxNode::BiOperation(BiOperation::Sub),
                    branch!(
                        SyntaxNode::UnOperation(UnOperation::Fac),
                        Leaf(SyntaxNode::Variable(TestVar::X))
                    ),
                    Leaf(SyntaxNode::Number(1.0))
                )
            ))
        );
        assert_eq!(
            syntaxify("3|x-1|/2 + 1"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Add),
                branch!(
                    SyntaxNode::BiOperation(BiOperation::Div),
                    branch!(
                        SyntaxNode::BiOperation(BiOperation::Mul),
                        Leaf(SyntaxNode::Number(3.0)),
                        branch!(
                            SyntaxNode::NativeFunction(NativeFunction::Abs),
                            branch!(
                                SyntaxNode::BiOperation(BiOperation::Sub),
                                Leaf(SyntaxNode::Variable(TestVar::X)),
                                Leaf(SyntaxNode::Number(1.0))
                            )
                        )
                    ),
                    Leaf(SyntaxNode::Number(2.0))
                ),
                Leaf(SyntaxNode::Number(1.0))
            ))
        );
        assert_eq!(
            syntaxify(""),
            Err(ParsingError {
                kind: ParsingErrorKind::EmptyInput,
                at: 0..=0
            })
        );
        assert_eq!(
            syntaxify("sin(x)+ja"),
            Err(ParsingError {
                kind: ParsingErrorKind::UnknownVariableOrConstant,
                at: 7..=8
            })
        );
        assert_eq!(
            syntaxify("x*()"),
            Err(ParsingError {
                kind: ParsingErrorKind::EmptyParenthesis,
                at: 2..=3
            })
        );
        assert_eq!(
            syntaxify("5+pi*sinj(x)"),
            Err(ParsingError {
                kind: ParsingErrorKind::UnknownFunction,
                at: 5..=9
            })
        );
        assert_eq!(
            syntaxify("1+expd(2y)"),
            Err(ParsingError {
                kind: ParsingErrorKind::NotEnoughArguments,
                at: 2..=9
            })
        );
        assert_eq!(
            syntaxify("5(1+clamp(2y, 1))"),
            Err(ParsingError {
                kind: ParsingErrorKind::NotEnoughArguments,
                at: 4..=15
            })
        );
        assert_eq!(
            syntaxify("deg2rad(1, pi)"),
            Err(ParsingError {
                kind: ParsingErrorKind::TooManyArguments,
                at: 0..=13
            })
        );
        assert_eq!(
            syntaxify("-expd(1, pi, sin(x))"),
            Err(ParsingError {
                kind: ParsingErrorKind::TooManyArguments,
                at: 1..=19
            })
        );
        assert_eq!(
            syntaxify("9t-clamp(1, pi, sin(x), 15tan(y))"),
            Err(ParsingError {
                kind: ParsingErrorKind::TooManyArguments,
                at: 3..=32
            })
        );
        assert_eq!(
            syntaxify("x*sin()"),
            Err(ParsingError {
                kind: ParsingErrorKind::EmptyArgument,
                at: 2..=6
            })
        );
        assert_eq!(
            syntaxify("x*log(y,)"),
            Err(ParsingError {
                kind: ParsingErrorKind::EmptyArgument,
                at: 7..=8
            })
        );
        assert_eq!(
            syntaxify("x*expd(,5y)"),
            Err(ParsingError {
                kind: ParsingErrorKind::EmptyArgument,
                at: 2..=7
            })
        );
        assert_eq!(
            syntaxify("x*clamp(x,,5y)"),
            Err(ParsingError {
                kind: ParsingErrorKind::EmptyArgument,
                at: 9..=10
            })
        );
        assert_eq!(
            syntaxify("x)"),
            Err(ParsingError {
                kind: ParsingErrorKind::MissingOpenParenthesis,
                at: 1..=1
            })
        );
        assert_eq!(
            syntaxify("(x"),
            Err(ParsingError {
                kind: ParsingErrorKind::MissingCloseParenthesis,
                at: 0..=0
            })
        );
        assert_eq!(
            syntaxify("(x + 1)*sin(y"),
            Err(ParsingError {
                kind: ParsingErrorKind::MissingCloseParenthesis,
                at: 8..=11
            })
        );
        assert_eq!(
            syntaxify("(x + (y)"),
            Err(ParsingError {
                kind: ParsingErrorKind::MissingCloseParenthesis,
                at: 0..=0
            })
        );
        assert_eq!(
            syntaxify("(10) * y) + 4x"),
            Err(ParsingError {
                kind: ParsingErrorKind::MissingOpenParenthesis,
                at: 8..=8
            })
        );
        assert_eq!(
            syntaxify("*10 2"),
            Err(ParsingError {
                kind: ParsingErrorKind::MisplacedOperator,
                at: 0..=0
            })
        );
        assert_eq!(
            syntaxify("(*x 2)"),
            Err(ParsingError {
                kind: ParsingErrorKind::MisplacedOperator,
                at: 1..=1
            })
        );
        assert_eq!(
            syntaxify("1+(x 2-)"),
            Err(ParsingError {
                kind: ParsingErrorKind::MisplacedOperator,
                at: 6..=6
            })
        );
        assert_eq!(
            syntaxify("(+)"),
            Err(ParsingError {
                kind: ParsingErrorKind::MisplacedOperator,
                at: 1..=1
            })
        );
        assert_eq!(
            syntaxify("(!)"),
            Err(ParsingError {
                kind: ParsingErrorKind::MisplacedOperator,
                at: 1..=1
            })
        );
        assert_eq!(
            syntaxify("|x"),
            Err(ParsingError {
                kind: ParsingErrorKind::PipeAbsNotClosed,
                at: 0..=0
            })
        );
        assert_eq!(
            syntaxify("3+|(|y|+1)/2"),
            Err(ParsingError {
                kind: ParsingErrorKind::PipeAbsNotClosed,
                at: 2..=2
            })
        );
        assert_eq!(
            syntaxify("|sin(|x)|"),
            Err(ParsingError {
                kind: ParsingErrorKind::PipeAbsNotClosed,
                at: 5..=5
            })
        );
    }

    #[test]
    fn test_aot_evaluation() {
        macro_rules! compare {
            ($i1:literal, $i2:literal) => {
                let mut syn1 = parse($i1).unwrap();
                syn1.aot_evaluation(|_| CFPointer::Single(&|_| 0.0));
                let syn2 = parse($i2).unwrap();
                assert_eq!(format!("{}", syn1), format!("{}", syn2));
            };
        }
        compare!("16/8+11", "13");
        compare!("sqrt(0)", "0");
        compare!("sin(1/8+t)", "sin(0.125+t)");
        compare!(
            "max(80/5, x^2, min(1,sin(0)))+sqrt(121)",
            "max(16, x^2, 0)+11"
        );
    }

    #[test]
    fn test_displacing_simplification() {
        macro_rules! compare {
            ($i1:literal, $i2:literal) => {
                let mut syn1 = parse($i1).unwrap();
                syn1.displacing_simplification();
                assert_eq!(format!("{}", syn1), $i2);
            };
        }
        compare!("x/1/8", "0.125*x");
        compare!("(x/16)/(y*4)", "0.015625*x/y");
        compare!("(7/x)/(y/2)", "14/(x*y)");
        compare!("(x/4)/(4/y)", "0.0625*x*y");
        compare!("10-x+12", "22 - x");
        compare!("x*pi*2", "6.283185307179586*x");
    }

    #[test]
    fn test_syntax_display() {
        let cases = [
            "x",
            "-y!",
            "1 + x",
            "10 - t + 12",
            "3*x - 2*y",
            "y^3 - 4*y^2 + y - 7",
            "x*5*y/4*3",
            "(x + 1)*(y - 1)",
            "1/(y - 1)",
            "1/(2*sqrt(y))",
            "y - (x + 1)",
            "x - y!",
            "x/(-y)",
            "x/y!",
            "(x^2)^y",
            "(7/x)/(y/2)",
            "(t^2 + 3*t + 2)/(t + 1)",
            "sin(x)",
            "x/sin(x + 1)",
            "clamp(x, 0, 1)",
            "min(1, x, y^2, x*y + 1)",
            "max(1, x, y^2, x*y + 1, sin(x*cos(y) + 1))",
            "digits(1, deg2rad(t), y^2, x*y + 1, sin(x*cos(y) + 1))",
        ];

        for c in cases {
            assert_eq!(c, parse(c).unwrap().to_string());
        }
    }

    fn random_syntax_tree(branch_count: usize) -> SyntaxTree<f64, TestVar, TestFunc> {
        const BRANCH_NODES: [(SyntaxNode<f64, TestVar, TestFunc>, RangeInclusive<u8>); 14] = [
            (SyntaxNode::BiOperation(BiOperation::Add), 2..=2),
            (SyntaxNode::BiOperation(BiOperation::Sub), 2..=2),
            (SyntaxNode::BiOperation(BiOperation::Mul), 2..=2),
            (SyntaxNode::BiOperation(BiOperation::Div), 2..=2),
            (SyntaxNode::UnOperation(UnOperation::Neg), 1..=1),
            (SyntaxNode::UnOperation(UnOperation::Fac), 1..=1),
            (SyntaxNode::NativeFunction(NativeFunction::Sin), 1..=1),
            (SyntaxNode::NativeFunction(NativeFunction::Sqrt), 1..=1),
            (SyntaxNode::NativeFunction(NativeFunction::Log), 2..=2),
            (SyntaxNode::NativeFunction(NativeFunction::Max), 2..=10),
            (SyntaxNode::CustomFunction(TestFunc::Deg2Rad), 1..=1),
            (SyntaxNode::CustomFunction(TestFunc::ExpD), 2..=2),
            (SyntaxNode::CustomFunction(TestFunc::Clamp), 3..=3),
            (SyntaxNode::CustomFunction(TestFunc::Digits), 2..=10),
        ];

        let mut arena: Arena<SyntaxNode<f64, TestVar, TestFunc>> = Arena::new();
        let mut dangling_branches: Vec<(NodeId, RangeInclusive<u8>)> = {
            let root = fastrand::choice(BRANCH_NODES).unwrap();
            vec![(arena.new_node(root.0), root.1)]
        };
        let root = dangling_branches[0].0;
        while arena.count() < branch_count {
            let (node, child_range) =
                dangling_branches.swap_remove(fastrand::usize(0..dangling_branches.len()));
            for _ in 0..fastrand::u8(child_range) {
                let (new_node, child_range) = fastrand::choice(BRANCH_NODES).unwrap();
                let id = node.append_value(new_node, &mut arena);
                dangling_branches.push((id, child_range));
            }
        }
        for (node, child_range) in dangling_branches {
            for _ in 0..fastrand::u8(child_range) {
                node.append_value(
                    if fastrand::bool() {
                        SyntaxNode::Variable(
                            fastrand::choice([TestVar::X, TestVar::Y, TestVar::T]).unwrap(),
                        )
                    } else {
                        SyntaxNode::Number(fastrand::i16(i16::MIN..i16::MAX) as f64 / 128.0)
                    },
                    &mut arena,
                );
            }
        }
        assert!(SyntaxTree::verify(&arena, root, |cf| match cf {
            TestFunc::Deg2Rad => (1, Some(1)),
            TestFunc::ExpD => (2, Some(2)),
            TestFunc::Clamp => (3, Some(3)),
            TestFunc::Digits => (2, None),
        }));
        SyntaxTree(Tree { arena, root })
    }

    #[test]
    #[ignore]
    fn test_syntax_display_random() {
        for size in [1, 10, 50] {
            for _ in 0..200 {
                let original_ast = random_syntax_tree(size);
                let expr = original_ast.to_string();
                let parsed_ast = parse(&expr).unwrap();
                let expr2 = parsed_ast.to_string();
                assert_eq!(
                    expr, expr2,
                    "\noriginal syntax tree: {:?}\nparsed syntax tree: {:?}",
                    original_ast.0, parsed_ast.0
                );
            }
        }
    }

    #[test]
    #[ignore]
    fn test_ast_eval() {
        struct VarStore;

        impl VariableStore<f64, TestVar> for VarStore {
            fn get(&self, var: TestVar) -> f64 {
                match var {
                    TestVar::X => 1.0,
                    TestVar::Y => 5.0,
                    TestVar::T => 0.1,
                }
            }
        }

        let cf2p = |cf: TestFunc| -> CFPointer<'_, f64> {
            match cf {
                TestFunc::Deg2Rad => CFPointer::Single(&|x: f64| x.to_radians()),
                TestFunc::ExpD => CFPointer::Dual(&|l: f64, x: f64| l * (-l * x).exp()),
                TestFunc::Clamp => {
                    CFPointer::Triple(&|x: f64, min: f64, max: f64| x.min(max).max(min))
                }
                TestFunc::Digits => CFPointer::Flexible(&|values: &[f64]| {
                    values.iter().enumerate().map(|(i, &v)| i as f64 * v).sum()
                }),
            }
        };
        macro_rules! assert_eval {
            ($expr: literal, $res: literal) => {
                assert_eq!(parse($expr).unwrap().eval(cf2p, &VarStore), $res);
            };
        }

        assert_eval!("1", 1.0);
        assert_eval!("x*3", 3.0);
        assert_eval!("3!", 6.0);
        assert_eval!("-t", -0.1);
        assert_eval!("y+100*t", 15.0);
        assert_eval!("sin(pi*t)", 0.3090169943749474);
        assert_eval!("log(6561, 3)", 8.0);
        assert_eval!("max(x, y, -18)*t", 0.5);
        assert_eval!("clamp(x + y, -273.15, t)", 0.1);
        assert_eval!("dist(1, 3, 4, 7)/y", 1.0);
        assert_eval!("digits(y, 1)", 15.0);
    }

    #[test]
    fn test_token2range() {
        let input = " max(pi, 1, -4)*3";
        let ts = TokenStream::new(input).unwrap();
        assert_eq!(token_range_to_str_range(input, &ts, 0..=0), 1..=4);
        assert_eq!(token_range_to_str_range(input, &ts, 1..=1), 5..=6);
        assert_eq!(token_range_to_str_range(input, &ts, 2..=2), 7..=7);
        assert_eq!(token_range_to_str_range(input, &ts, 3..=3), 9..=9);
    }
}
