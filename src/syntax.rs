use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::RangeInclusive;

use crate::asm::{CFPointer, Stack};
use crate::number::{MathEvalNumber, NFPointer, NativeFunction};
use crate::tokenizer::Token;
use crate::{
    BinaryOp, FunctionIdentifier, NAME_LIMIT, ParsingError, ParsingErrorKind, UnaryOp,
    VariableIdentifier, VariableStore,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AstNodeKind<N, V, F>
where
    N: MathEvalNumber,
    V: VariableIdentifier,
    F: FunctionIdentifier,
{
    Number(N),
    Variable(V),
    BinaryOp(BinaryOp),
    UnaryOp(UnaryOp),
    NativeFunction(NativeFunction, u8),
    CustomFunction(F, u8),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AstNode<N, V, F>
where
    N: MathEvalNumber,
    V: VariableIdentifier,
    F: FunctionIdentifier,
{
    pub kind: AstNodeKind<N, V, F>,
    pub descendants_count: usize,
}

impl<N, V, F> AstNode<N, V, F>
where
    N: MathEvalNumber,
    V: VariableIdentifier,
    F: FunctionIdentifier,
{
    fn new(kind: AstNodeKind<N, V, F>, descendants_count: usize) -> Self {
        AstNode {
            kind,
            descendants_count,
        }
    }
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

fn token_range_to_str_range(
    input: &str,
    tokens: &[Token<'_>],
    token_range: RangeInclusive<usize>,
) -> RangeInclusive<usize> {
    let mut start = 0;
    let mut index = 0;
    for (tk_idx, token) in tokens[..=*token_range.end()].iter().enumerate() {
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
    pub fn to_general(self, input: &str, tokens: &[Token<'_>]) -> ParsingError {
        ParsingError {
            at: if self.0 == SyntaxErrorKind::EmptyInput {
                0..=0
            } else {
                token_range_to_str_range(input, tokens, self.1)
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SYOperator<F>
where
    F: FunctionIdentifier,
{
    BinaryOp(BinaryOp),
    UnaryOp(UnaryOp),
    Function(SYFunction<F>, u8),
    Parentheses,
}

impl<F> SYOperator<F>
where
    F: FunctionIdentifier,
{
    fn precedence(&self) -> u8 {
        match self {
            SYOperator::BinaryOp(BinaryOp::Add) => 0,
            SYOperator::BinaryOp(BinaryOp::Sub) => 0,
            SYOperator::BinaryOp(BinaryOp::Mul) => 1,
            SYOperator::BinaryOp(BinaryOp::Div) => 1,
            SYOperator::BinaryOp(BinaryOp::Mod) => 1,
            SYOperator::UnaryOp(UnaryOp::Neg) => 2,
            SYOperator::BinaryOp(BinaryOp::Pow) => 3,
            SYOperator::UnaryOp(UnaryOp::Fac) => 4,
            SYOperator::UnaryOp(UnaryOp::DoubleFac) => 4,
            SYOperator::Function(_, _) | SYOperator::Parentheses => {
                unreachable!()
            }
        }
    }
    fn is_right_associative(&self) -> bool {
        matches!(self, SYOperator::BinaryOp(BinaryOp::Pow))
    }
    fn to_syn<N, V>(self) -> AstNodeKind<N, V, F>
    where
        N: MathEvalNumber,
        V: VariableIdentifier,
    {
        match self {
            SYOperator::BinaryOp(opr) => AstNodeKind::BinaryOp(opr),
            SYOperator::UnaryOp(opr) => AstNodeKind::UnaryOp(opr),
            SYOperator::Function(SYFunction::NativeFunction(nf), args) => {
                AstNodeKind::NativeFunction(nf, args)
            }
            SYOperator::Function(SYFunction::CustomFunction(cf, _, _), args) => {
                AstNodeKind::CustomFunction(cf, args)
            }
            SYOperator::Function(SYFunction::PipeAbs, _) => {
                AstNodeKind::NativeFunction(NativeFunction::Abs, 1)
            }
            SYOperator::Parentheses => unreachable!(),
        }
    }
}

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

trait ShuntingYardOutput<N, V, F>
where
    N: MathEvalNumber,
    V: VariableIdentifier,
    F: FunctionIdentifier,
{
    type Output;

    fn pop_opr(&mut self, operator_stack: &mut Vec<SYOperator<F>>);
    fn make(self) -> Self::Output;
    fn push(&mut self, node: AstNodeKind<N, V, F>);
    fn pop_arg(&mut self) -> Option<AstNodeKind<N, V, F>>;
    fn last_num<'a>(&'a self) -> Option<N::AsArg<'a>>;

    fn push_opr(&mut self, operator: SYOperator<F>, operator_stack: &mut Vec<SYOperator<F>>) {
        while let Some(top_opr) = operator_stack.last()
            && matches!(top_opr, SYOperator::BinaryOp(_) | SYOperator::UnaryOp(_))
            && (operator.precedence() < top_opr.precedence()
                || operator.precedence() == top_opr.precedence()
                    && !operator.is_right_associative())
        {
            self.pop_opr(operator_stack);
        }
        operator_stack.push(operator);
    }
    fn flush(&mut self, operator_stack: &mut Vec<SYOperator<F>>) {
        while let Some(opr) = operator_stack.last()
            && matches!(opr, SYOperator::BinaryOp(_) | SYOperator::UnaryOp(_))
        {
            self.pop_opr(operator_stack)
        }
    }
}

fn calc_child_count<N: MathEvalNumber, V: VariableIdentifier, F: FunctionIdentifier>(
    nodes: &[AstNode<N, V, F>],
    kind: AstNodeKind<N, V, F>,
) -> AstNode<N, V, F>
where
{
    let arg_cons = match kind {
        AstNodeKind::Number(_) | AstNodeKind::Variable(_) => 0,
        AstNodeKind::UnaryOp(_) => 1,
        AstNodeKind::BinaryOp(_) => 2,
        AstNodeKind::NativeFunction(_, a) | AstNodeKind::CustomFunction(_, a) => a,
    };
    let mut dc = 0;
    for ac in 0..arg_cons {
        dc += nodes[nodes.len() - dc - 1].descendants_count + 1;
    }
    AstNode::new(kind, dc)
}

struct SyAstOutput<N, V, F>(Vec<AstNode<N, V, F>>)
where
    N: MathEvalNumber,
    V: VariableIdentifier,
    F: FunctionIdentifier;

impl<N, V, F> ShuntingYardOutput<N, V, F> for SyAstOutput<N, V, F>
where
    N: MathEvalNumber,
    V: VariableIdentifier,
    F: FunctionIdentifier,
{
    type Output = PostfixMathAst<N, V, F>;

    fn pop_opr(&mut self, operator_stack: &mut Vec<SYOperator<F>>) {
        let opr = operator_stack.pop().unwrap();
        if let Some(AstNodeKind::Number(num)) = self.0.last_mut().map(|n| &mut n.kind)
            && opr == SYOperator::UnaryOp(UnaryOp::Neg)
        {
            *num = -num.clone();
        } else {
            self.push(opr.to_syn());
        }
    }
    fn make(self) -> Self::Output {
        PostfixMathAst(self.0)
    }
    fn push(&mut self, kind: AstNodeKind<N, V, F>) {
        self.0.push(calc_child_count(&self.0, kind))
    }
    fn pop_arg(&mut self) -> Option<AstNodeKind<N, V, F>> {
        self.0.pop().map(|n| n.kind)
    }
    fn last_num<'a>(&'a self) -> Option<<N as MathEvalNumber>::AsArg<'a>> {
        if let Some(AstNodeKind::Number(num)) = self.0.last().map(|n| &n.kind) {
            Some(num.asarg())
        } else {
            None
        }
    }
}

struct SyNumberOutput<'a, 'b, N, V, F, S, C>
where
    N: MathEvalNumber,
    V: VariableIdentifier,
    F: FunctionIdentifier,
    S: VariableStore<N, V>,
    C: Fn(F) -> CFPointer<'a, N>,
{
    args: Vec<N>,
    variable_store: &'b S,
    cf2pointer: C,
    var_ident: PhantomData<V>,
    func_ident: PhantomData<F>,
}

impl<'a, 'b, N, V, F, S, C> ShuntingYardOutput<N, V, F> for SyNumberOutput<'a, 'b, N, V, F, S, C>
where
    N: MathEvalNumber,
    V: VariableIdentifier,
    F: FunctionIdentifier,
    S: VariableStore<N, V>,
    C: Fn(F) -> CFPointer<'a, N>,
{
    type Output = N;

    fn pop_opr(&mut self, operator_stack: &mut Vec<SYOperator<F>>) {
        let res = match operator_stack.pop().unwrap() {
            SYOperator::BinaryOp(opr) => {
                let rhs = self.args.pop().unwrap();
                opr.eval(self.args.pop().unwrap().asarg(), rhs.asarg())
            }
            SYOperator::UnaryOp(opr) => opr.eval(self.args.pop().unwrap().asarg()),
            _ => panic!(),
        };
        self.args.push(res);
    }
    fn make(mut self) -> Self::Output {
        debug_assert_eq!(self.args.len(), 1);
        self.args.pop().unwrap()
    }
    fn push(&mut self, node: AstNodeKind<N, V, F>) {
        let res = match node {
            AstNodeKind::Number(num) => num,
            AstNodeKind::Variable(var) => self.variable_store.get(var).to_owned(),
            AstNodeKind::NativeFunction(nf, args) => match nf.to_pointer::<N>() {
                NFPointer::Single(func) => func(self.args.pop().unwrap().asarg()),
                NFPointer::Dual(func) => {
                    let rhs = self.args.pop().unwrap();
                    func(self.args.pop().unwrap().asarg(), rhs.asarg())
                }
                NFPointer::Flexible(func) => {
                    let res = func(&self.args[self.args.len() - args as usize..]);
                    self.args.truncate(self.args.len() - args as usize);
                    res
                }
            },
            AstNodeKind::CustomFunction(cf, args) => match (self.cf2pointer)(cf) {
                CFPointer::Single(func) => func(self.args.pop().unwrap().asarg()),
                CFPointer::Dual(func) => {
                    let rhs = self.args.pop().unwrap();
                    func(self.args.pop().unwrap().asarg(), rhs.asarg())
                }
                CFPointer::Triple(func) => {
                    let a3 = self.args.pop().unwrap();
                    let a2 = self.args.pop().unwrap();
                    func(self.args.pop().unwrap().asarg(), a2.asarg(), a3.asarg())
                }
                CFPointer::Flexible(func) => {
                    let res = func(&self.args[self.args.len() - args as usize..]);
                    self.args.truncate(self.args.len() - args as usize);
                    res
                }
            },
            AstNodeKind::BinaryOp(_) | AstNodeKind::UnaryOp(_) => panic!(),
        };
        self.args.push(res);
    }
    fn pop_arg(&mut self) -> Option<AstNodeKind<N, V, F>> {
        self.args.pop().map(|num| AstNodeKind::Number(num))
    }
    fn last_num<'c>(&'c self) -> Option<N::AsArg<'c>> {
        self.args.last().map(|num| num.asarg())
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
            SYOperator::BinaryOp(_) | SYOperator::UnaryOp(_) => (),
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

fn parse_or_eval<'a, O, N, V, F>(
    mut output_queue: O,
    tokens: &'a [Token<'a>],
    custom_constant_parser: impl Fn(&str) -> Option<N>,
    custom_function_parser: impl Fn(&str) -> Option<(F, u8, Option<u8>)>,
    custom_variable_parser: impl Fn(&str) -> Option<V>,
) -> Result<O::Output, SyntaxError>
where
    O: ShuntingYardOutput<N, V, F>,
    N: MathEvalNumber,
    V: VariableIdentifier,
    F: FunctionIdentifier,
{
    // Dijkstra's shunting yard algorithm
    let mut operator_stack: Vec<SYOperator<F>> = Vec::new();
    let mut last_tk: Option<Token<'a>> = None;
    for (pos, &token) in tokens.iter().enumerate() {
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
            output_queue.push_opr(SYOperator::BinaryOp(BinaryOp::Mul), &mut operator_stack);
        }
        match token {
            Token::Number(num) => output_queue.push(
                num.parse::<N>()
                    .map(AstNodeKind::Number)
                    .map_err(|_| SyntaxError(SyntaxErrorKind::NumberParsingError, pos..=pos))?,
            ),
            Token::Variable(var) => {
                if var.len() > NAME_LIMIT as usize {
                    return Err(SyntaxError(SyntaxErrorKind::NameTooLong, pos..=pos));
                }
                if let Some(node) = N::parse_constant(var)
                    .or_else(|| custom_constant_parser(var))
                    .map(|c| AstNodeKind::Number(c))
                    .or_else(|| custom_variable_parser(var).map(|var| AstNodeKind::Variable(var)))
                {
                    output_queue.push(node)
                } else if let Some(segments) =
                    segment_variable(var, &custom_constant_parser, &custom_variable_parser)
                {
                    let mut first = true;
                    for seg in segments {
                        output_queue.push(match seg {
                            Segment::Constant(c) => AstNodeKind::Number(c),
                            Segment::Variable(v) => AstNodeKind::Variable(v),
                        });
                        if first {
                            first = false;
                        } else {
                            output_queue
                                .push_opr(SYOperator::BinaryOp(BinaryOp::Mul), &mut operator_stack);
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
                let mut sy_opr: SYOperator<F> = BinaryOp::parse(opr)
                    .map(|biopr| SYOperator::BinaryOp(biopr))
                    .or_else(|| UnaryOp::parse(opr).map(|unopr| SYOperator::UnaryOp(unopr)))
                    .unwrap();
                if opr == '-' && last_tk.is_none_or(|tk| after_implies_neg(tk, &operator_stack)) {
                    sy_opr = SYOperator::UnaryOp(UnaryOp::Neg);
                } else if opr == '!' && last_tk.is_some_and(|tk| tk == Token::Operation('!')) {
                    let fac = operator_stack.pop();
                    debug_assert_eq!(fac, Some(SYOperator::UnaryOp(UnaryOp::Fac)));
                    sy_opr = SYOperator::UnaryOp(UnaryOp::DoubleFac)
                }
                if opr != '+' || last_tk.is_some_and(|tk| !after_implies_neg(tk, &operator_stack)) {
                    output_queue.push_opr(sy_opr, &mut operator_stack);
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
                        output_queue.push(match seg {
                            Segment::Constant(c) => AstNodeKind::Number(c),
                            Segment::Variable(v) => AstNodeKind::Variable(v),
                        });
                        output_queue
                            .push_opr(SYOperator::BinaryOp(BinaryOp::Mul), &mut operator_stack);
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
                output_queue.flush(&mut operator_stack);
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
                output_queue.flush(&mut operator_stack);
                match operator_stack.pop() {
                    Some(SYOperator::Function(SYFunction::PipeAbs, _)) => {
                        let opening = find_opening_pipe(&tokens[..pos]).unwrap();
                        return Err(SyntaxError(
                            SyntaxErrorKind::PipeAbsNotClosed,
                            opening..=opening,
                        ));
                    }
                    Some(SYOperator::Function(SYFunction::NativeFunction(mut nf), args)) => {
                        if args < nf.min_args() && nf != NativeFunction::Log {
                            Err(SyntaxErrorKind::NotEnoughArguments)
                        } else if nf.max_args().is_some_and(|m| args > m) {
                            Err(SyntaxErrorKind::TooManyArguments)
                        } else {
                            if nf == NativeFunction::Log {
                                let ten = N::from(10);
                                let two = N::from(2);
                                match output_queue.last_num().unwrap() {
                                    num if num == ten.asarg() => {
                                        output_queue.pop_arg();
                                        nf = NativeFunction::Log10;
                                    }
                                    num if num == two.asarg() => {
                                        output_queue.pop_arg();
                                        nf = NativeFunction::Log2;
                                    }
                                    _ if args == 1 => {
                                        nf = NativeFunction::Log10;
                                    }
                                    _ => (),
                                }
                            }
                            output_queue.push(AstNodeKind::NativeFunction(nf, args));
                            Ok(())
                        }
                        .map_err(|e| {
                            SyntaxError(e, find_opening_paren(&tokens[..pos]).unwrap()..=pos)
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
                        output_queue.push(AstNodeKind::CustomFunction(cf, args));
                        Ok(())
                    }
                    .map_err(|e| {
                        SyntaxError(e, find_opening_paren(&tokens[..pos]).unwrap()..=pos)
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
                    output_queue.flush(&mut operator_stack);
                    if let Some(SYOperator::Function(SYFunction::PipeAbs, args)) =
                        operator_stack.pop()
                        && args > 1
                    {
                        return Err(SyntaxError(
                            SyntaxErrorKind::TooManyArguments,
                            opening_pipe..=pos,
                        ));
                    }
                    output_queue.push(AstNodeKind::NativeFunction(NativeFunction::Abs, 1));
                } else {
                    operator_stack.push(SYOperator::Function(SYFunction::PipeAbs, 0));
                }
            }
        }
        last_tk = Some(token);
    }
    validate_consecutive_tokens(last_tk, None, tokens.len())?;
    output_queue.flush(&mut operator_stack);
    match operator_stack.last() {
        Some(SYOperator::Function(SYFunction::PipeAbs, _)) => {
            let opening = find_opening_pipe(tokens).unwrap();
            Err(SyntaxError(
                SyntaxErrorKind::PipeAbsNotClosed,
                opening..=opening,
            ))
        }
        Some(SYOperator::Function(_, _) | SYOperator::Parentheses) => {
            let unclosed_paren_pos = find_opening_paren(tokens).unwrap();
            Err(SyntaxError(
                SyntaxErrorKind::MissingClosingParenthesis,
                unclosed_paren_pos..=unclosed_paren_pos,
            ))
        }
        _ => Ok(output_queue.make()),
    }
}

#[derive(Debug, Clone)]
pub struct ChildNodeIter<'a, N: MathEvalNumber, V: VariableIdentifier, F: FunctionIdentifier> {
    tree: &'a PostfixMathAst<N, V, F>,
    pos: usize,
    count: u8,
}

impl<'a, N, V, F> Iterator for ChildNodeIter<'a, N, V, F>
where
    N: MathEvalNumber,
    V: VariableIdentifier,
    F: FunctionIdentifier,
{
    type Item = (&'a AstNode<N, V, F>, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.count == 0 {
            return None;
        }
        self.count -= 1;
        let idx = self.pos;
        let cur = &self.tree.0[self.pos];
        if cur.descendants_count < self.pos {
            self.pos -= cur.descendants_count + 1;
        } else {
            debug_assert_eq!(cur.descendants_count, self.pos);
            debug_assert_eq!(self.count, 0);
        }
        Some((cur, idx))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PostfixMathAst<N: MathEvalNumber, V: VariableIdentifier, F: FunctionIdentifier>(
    pub(crate) Vec<AstNode<N, V, F>>,
);

impl<V, N, F> PostfixMathAst<N, V, F>
where
    N: MathEvalNumber,
    V: VariableIdentifier,
    F: FunctionIdentifier,
{
    pub fn new<'a>(
        tokens: &'a [Token<'a>],
        custom_constant_parser: impl Fn(&str) -> Option<N>,
        custom_function_parser: impl Fn(&str) -> Option<(F, u8, Option<u8>)>,
        custom_variable_parser: impl Fn(&str) -> Option<V>,
    ) -> Result<PostfixMathAst<N, V, F>, SyntaxError> {
        parse_or_eval(
            SyAstOutput(Vec::new()),
            tokens,
            custom_constant_parser,
            custom_function_parser,
            custom_variable_parser,
        )
    }
    pub fn parse_and_eval<'a, 'b, 'c, S: VariableStore<N, V>>(
        tokens: &'a [Token<'a>],
        custom_constant_parser: impl Fn(&str) -> Option<N>,
        custom_function_parser: impl Fn(&str) -> Option<(F, u8, Option<u8>)>,
        custom_variable_parser: impl Fn(&str) -> Option<V>,
        variable_values: &'b S,
        function_to_pointer: impl Fn(F) -> CFPointer<'c, N>,
    ) -> Result<N, SyntaxError> {
        parse_or_eval(
            SyNumberOutput {
                args: Vec::new(),
                variable_store: variable_values,
                cf2pointer: function_to_pointer,
                var_ident: PhantomData,
                func_ident: PhantomData,
            },
            tokens,
            custom_constant_parser,
            custom_function_parser,
            custom_variable_parser,
        )
    }

    fn eval_stack_capacity(tree: &[AstNode<N, V, F>]) -> usize {
        let mut stack_capacity = 0usize;
        let mut stack_len = 0i64;
        for node in tree {
            stack_len += match node.kind {
                AstNodeKind::Number(_) | AstNodeKind::Variable(_) => 1,
                AstNodeKind::BinaryOp(_) => -1,
                AstNodeKind::UnaryOp(_) => 0,
                AstNodeKind::NativeFunction(_, args) => -(args as i64) + 1,
                AstNodeKind::CustomFunction(_, args) => -(args as i64) + 1,
            };
            if stack_len as usize > stack_capacity {
                stack_capacity = stack_len as usize;
            }
        }
        stack_capacity
    }

    pub fn eval<'a>(
        &self,
        function_to_pointer: impl Fn(F) -> CFPointer<'a, N>,
        variable_values: &impl crate::VariableStore<N, V>,
    ) -> N {
        Self::_eval(
            &self.0,
            function_to_pointer,
            variable_values,
            Stack::with_capacity(Self::eval_stack_capacity(&self.0)),
        ).unwrap()
    }

    fn _eval<'a>(
        tree: &[AstNode<N, V, F>],
        functibn_to_pointer: impl Fn(F) -> CFPointer<'a, N>,
        variable_values: &impl crate::VariableStore<N, V>,
        mut stack: Stack<N>,
    ) -> Result<N, usize> {
        for (idx, node) in tree.iter().enumerate() {
            let mut pop = || stack.pop().ok_or(idx);
            let result: N = match &node.kind {
                AstNodeKind::Number(num) => num.clone(),
                AstNodeKind::Variable(var) => variable_values.get(*var).to_owned(),
                AstNodeKind::BinaryOp(opr) => {
                    let rhs = pop()?;
                    opr.eval(pop()?.asarg(), rhs.asarg())
                }
                AstNodeKind::UnaryOp(opr) => opr.eval(pop()?.asarg()),
                AstNodeKind::NativeFunction(nf, argc) => match nf.to_pointer() {
                    NFPointer::Single(func) => func(pop()?.asarg()),
                    NFPointer::Dual(func) => {
                        let arg2 = pop()?;
                        func(pop()?.asarg(), arg2.asarg())
                    }
                    NFPointer::Flexible(func) => {
                        let new_len = stack.len() - *argc as usize;
                        let res = func(&stack[new_len..]);
                        stack.truncate(new_len);
                        res
                    }
                },
                AstNodeKind::CustomFunction(cf, argc) => match functibn_to_pointer(*cf) {
                    CFPointer::Single(func) => func(pop()?.asarg()),
                    CFPointer::Dual(func) => {
                        let arg2 = pop()?;
                        func(pop()?.asarg(), arg2.asarg())
                    }
                    CFPointer::Triple(func) => {
                        let arg3 = pop()?;
                        let arg2 = pop()?;
                        func(pop()?.asarg(), arg2.asarg(), arg3.asarg())
                    }
                    CFPointer::Flexible(func) => {
                        let new_len = stack.len() - *argc as usize;
                        let res = func(&stack[new_len..]);
                        stack.truncate(new_len);
                        res
                    }
                },
            };
            stack.push(result);
        }
        stack.pop().ok_or(0)
    }

    pub fn iter_children<'a>(&'a self, target: usize) -> ChildNodeIter<'a, N, V, F> {
        let cc = match self.0[target].kind {
            AstNodeKind::Number(_) | AstNodeKind::Variable(_) => 0,
            AstNodeKind::BinaryOp(_) => 2,
            AstNodeKind::UnaryOp(_) => 1,
            AstNodeKind::NativeFunction(_, argc) | AstNodeKind::CustomFunction(_, argc) => argc,
        };
        ChildNodeIter {
            tree: self,
            pos: target.wrapping_sub(1),
            count: cc,
        }
    }

    pub fn aot_evaluation<'a>(&mut self, function_to_pointer: impl Fn(F) -> CFPointer<'a, N>) {
        todo!()
    }

    pub fn displacing_simplification(&mut self) {
        self._displacing_simplification(BinaryOp::Add, BinaryOp::Sub, 0.into());
        self._displacing_simplification(BinaryOp::Mul, BinaryOp::Div, 1.into());
    }

    fn _displacing_simplification(&mut self, pos: BinaryOp, neg: BinaryOp, inital_value: N) {
        todo!()
    }

    #[allow(dead_code)]
    fn verify(tree: &[AstNodeKind<N, V, F>], cf_bounds: impl Fn(F) -> (u8, Option<u8>)) -> bool {
        todo!()
    }
}

impl<V, N, F> Display for PostfixMathAst<N, V, F>
where
    N: MathEvalNumber + Display,
    V: VariableIdentifier + Display,
    F: FunctionIdentifier + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
    }
}

#[cfg(test)]
mod test {
    use std::f64::consts::*;

    use super::*;
    use crate::VariableStore;
    use crate::tokenizer::TokenStream;

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

    fn parse(input: &str) -> Result<PostfixMathAst<f64, TestVar, TestFunc>, ParsingError> {
        let tokens = TokenStream::new(input).map_err(|e| e.to_general())?.0;
        PostfixMathAst::new(
            &tokens,
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
        .map_err(|e| e.to_general(input, &tokens))
    }

    #[test]
    fn test_syntaxify() {
        fn syntaxify(
            input: &str,
        ) -> Result<Vec<AstNodeKind<f64, TestVar, TestFunc>>, ParsingError> {
            parse(input).map(|st| st.0.into_iter().map(|n| n.kind).collect())
        }
        assert_eq!(syntaxify("0"), Ok(vec![AstNodeKind::Number(0.0)]));
        assert_eq!(syntaxify("(0)"), Ok(vec![AstNodeKind::Number(0.0)]));
        assert_eq!(syntaxify("((0))"), Ok(vec![AstNodeKind::Number(0.0)]));
        assert_eq!(syntaxify("pi"), Ok(vec![AstNodeKind::Number(PI)]));
        assert_eq!(
            syntaxify("1+1"),
            Ok(vec![
                AstNodeKind::Number(1.0),
                AstNodeKind::Number(1.0),
                AstNodeKind::BinaryOp(BinaryOp::Add),
            ])
        );
        assert_eq!(syntaxify("-0.5"), Ok(vec![AstNodeKind::Number(-0.5)]));
        assert_eq!(
            syntaxify("5-3"),
            Ok(vec![
                AstNodeKind::Number(5.0),
                AstNodeKind::Number(3.0),
                AstNodeKind::BinaryOp(BinaryOp::Sub),
            ])
        );
        assert_eq!(
            syntaxify("-y!"),
            Ok(vec![
                AstNodeKind::Variable(TestVar::Y),
                AstNodeKind::UnaryOp(UnaryOp::Fac),
                AstNodeKind::UnaryOp(UnaryOp::Neg),
            ])
        );
        assert_eq!(
            syntaxify("t!!"),
            Ok(vec![
                AstNodeKind::Variable(TestVar::T),
                AstNodeKind::UnaryOp(UnaryOp::DoubleFac)
            ])
        );
        assert_eq!(
            syntaxify("8*3+1"),
            Ok(vec![
                AstNodeKind::Number(8.0),
                AstNodeKind::Number(3.0),
                AstNodeKind::BinaryOp(BinaryOp::Mul),
                AstNodeKind::Number(1.0),
                AstNodeKind::BinaryOp(BinaryOp::Add),
            ])
        );
        assert_eq!(
            syntaxify("12/3/2"),
            Ok(vec![
                AstNodeKind::Number(12.0),
                AstNodeKind::Number(3.0),
                AstNodeKind::BinaryOp(BinaryOp::Div),
                AstNodeKind::Number(2.0),
                AstNodeKind::BinaryOp(BinaryOp::Div),
            ])
        );
        assert_eq!(
            syntaxify("8*(3+1)"),
            Ok(vec![
                AstNodeKind::Number(8.0),
                AstNodeKind::Number(3.0),
                AstNodeKind::Number(1.0),
                AstNodeKind::BinaryOp(BinaryOp::Add),
                AstNodeKind::BinaryOp(BinaryOp::Mul),
            ])
        );
        assert_eq!(
            syntaxify("8*3^2-1"),
            Ok(vec![
                AstNodeKind::Number(8.0),
                AstNodeKind::Number(3.0),
                AstNodeKind::Number(2.0),
                AstNodeKind::BinaryOp(BinaryOp::Pow),
                AstNodeKind::BinaryOp(BinaryOp::Mul),
                AstNodeKind::Number(1.0),
                AstNodeKind::BinaryOp(BinaryOp::Sub),
            ])
        );
        assert_eq!(
            syntaxify("2x"),
            Ok(vec![
                AstNodeKind::Number(2.0),
                AstNodeKind::Variable(TestVar::X),
                AstNodeKind::BinaryOp(BinaryOp::Mul),
            ])
        );
        assert_eq!(
            syntaxify("sin(14)"),
            Ok(vec![
                AstNodeKind::Number(14.0),
                AstNodeKind::NativeFunction(NativeFunction::Sin, 1),
            ])
        );
        assert_eq!(
            syntaxify("deg2rad(80)"),
            Ok(vec![
                AstNodeKind::Number(80.0),
                AstNodeKind::CustomFunction(TestFunc::Deg2Rad, 1),
            ])
        );
        assert_eq!(
            syntaxify("expd(0.2, x)"),
            Ok(vec![
                AstNodeKind::Number(0.2),
                AstNodeKind::Variable(TestVar::X),
                AstNodeKind::CustomFunction(TestFunc::ExpD, 2),
            ])
        );
        assert_eq!(
            syntaxify("clamp(t, y, x)"),
            Ok(vec![
                AstNodeKind::Variable(TestVar::T),
                AstNodeKind::Variable(TestVar::Y),
                AstNodeKind::Variable(TestVar::X),
                AstNodeKind::CustomFunction(TestFunc::Clamp, 3),
            ])
        );
        assert_eq!(
            syntaxify("digits(3, 1, 5, 7, 2, x)"),
            Ok(vec![
                AstNodeKind::Number(3.0),
                AstNodeKind::Number(1.0),
                AstNodeKind::Number(5.0),
                AstNodeKind::Number(7.0),
                AstNodeKind::Number(2.0),
                AstNodeKind::Variable(TestVar::X),
                AstNodeKind::CustomFunction(TestFunc::Digits, 6),
            ])
        );
        assert_eq!(
            syntaxify("lb(8)"),
            Ok(vec![
                AstNodeKind::Number(8.0),
                AstNodeKind::NativeFunction(NativeFunction::Log2, 1),
            ])
        );
        assert_eq!(
            syntaxify("log(100)"),
            Ok(vec![
                AstNodeKind::Number(100.0),
                AstNodeKind::NativeFunction(NativeFunction::Log10, 1),
            ])
        );
        assert_eq!(
            syntaxify("sin(cos(0))"),
            Ok(vec![
                AstNodeKind::Number(0.0),
                AstNodeKind::NativeFunction(NativeFunction::Cos, 1),
                AstNodeKind::NativeFunction(NativeFunction::Sin, 1),
            ])
        );
        assert_eq!(
            syntaxify("x^2 + sin(y)"),
            Ok(vec![
                AstNodeKind::Variable(TestVar::X),
                AstNodeKind::Number(2.0),
                AstNodeKind::BinaryOp(BinaryOp::Pow),
                AstNodeKind::Variable(TestVar::Y),
                AstNodeKind::NativeFunction(NativeFunction::Sin, 1),
                AstNodeKind::BinaryOp(BinaryOp::Add),
            ])
        );
        assert_eq!(
            syntaxify("sqrt(max(4, 9))"),
            Ok(vec![
                AstNodeKind::Number(4.0),
                AstNodeKind::Number(9.0),
                AstNodeKind::NativeFunction(NativeFunction::Max, 2),
                AstNodeKind::NativeFunction(NativeFunction::Sqrt, 1),
            ])
        );
        assert_eq!(
            syntaxify("max(2, x, 8y, xy+1)"),
            Ok(vec![
                AstNodeKind::Number(2.0),
                AstNodeKind::Variable(TestVar::X),
                AstNodeKind::Number(8.0),
                AstNodeKind::Variable(TestVar::Y),
                AstNodeKind::BinaryOp(BinaryOp::Mul),
                AstNodeKind::Variable(TestVar::X),
                AstNodeKind::Variable(TestVar::Y),
                AstNodeKind::BinaryOp(BinaryOp::Mul),
                AstNodeKind::Number(1.0),
                AstNodeKind::BinaryOp(BinaryOp::Add),
                AstNodeKind::NativeFunction(NativeFunction::Max, 4),
            ])
        );
        assert_eq!(
            syntaxify("2*x + 3*y"),
            Ok(vec![
                AstNodeKind::Number(2.0),
                AstNodeKind::Variable(TestVar::X),
                AstNodeKind::BinaryOp(BinaryOp::Mul),
                AstNodeKind::Number(3.0),
                AstNodeKind::Variable(TestVar::Y),
                AstNodeKind::BinaryOp(BinaryOp::Mul),
                AstNodeKind::BinaryOp(BinaryOp::Add),
            ])
        );
        assert_eq!(
            syntaxify("log10(1000)"),
            Ok(vec![
                AstNodeKind::Number(1000.0),
                AstNodeKind::NativeFunction(NativeFunction::Log10, 1),
            ])
        );
        assert_eq!(
            syntaxify("1/(2+3)"),
            Ok(vec![
                AstNodeKind::Number(1.0),
                AstNodeKind::Number(2.0),
                AstNodeKind::Number(3.0),
                AstNodeKind::BinaryOp(BinaryOp::Add),
                AstNodeKind::BinaryOp(BinaryOp::Div),
            ])
        );
        assert_eq!(
            syntaxify("e^x"),
            Ok(vec![
                AstNodeKind::Number(E),
                AstNodeKind::Variable(TestVar::X),
                AstNodeKind::BinaryOp(BinaryOp::Pow),
            ])
        );
        assert_eq!(
            syntaxify("x * -2"),
            Ok(vec![
                AstNodeKind::Variable(TestVar::X),
                AstNodeKind::Number(-2.0),
                AstNodeKind::BinaryOp(BinaryOp::Mul),
            ])
        );
        assert_eq!(
            syntaxify("4/-1.33"),
            Ok(vec![
                AstNodeKind::Number(4.0),
                AstNodeKind::Number(-1.33),
                AstNodeKind::BinaryOp(BinaryOp::Div),
            ])
        );
        assert_eq!(
            syntaxify("sqrt(16)"),
            Ok(vec![
                AstNodeKind::Number(16.0),
                AstNodeKind::NativeFunction(NativeFunction::Sqrt, 1),
            ])
        );
        assert_eq!(
            syntaxify("abs(-5)"),
            Ok(vec![
                AstNodeKind::Number(-5.0),
                AstNodeKind::NativeFunction(NativeFunction::Abs, 1),
            ])
        );
        assert_eq!(
            syntaxify("x^2 - y^2"),
            Ok(vec![
                AstNodeKind::Variable(TestVar::X),
                AstNodeKind::Number(2.0),
                AstNodeKind::BinaryOp(BinaryOp::Pow),
                AstNodeKind::Variable(TestVar::Y),
                AstNodeKind::Number(2.0),
                AstNodeKind::BinaryOp(BinaryOp::Pow),
                AstNodeKind::BinaryOp(BinaryOp::Sub),
            ])
        );
        assert_eq!(
            syntaxify("|x|"),
            Ok(vec![
                AstNodeKind::Variable(TestVar::X),
                AstNodeKind::NativeFunction(NativeFunction::Abs, 1),
            ])
        );
        assert_eq!(
            syntaxify("|-x|"),
            Ok(vec![
                AstNodeKind::Variable(TestVar::X),
                AstNodeKind::UnaryOp(UnaryOp::Neg),
                AstNodeKind::NativeFunction(NativeFunction::Abs, 1),
            ])
        );
        assert_eq!(
            syntaxify("4*-x"),
            Ok(vec![
                AstNodeKind::Number(4.0),
                AstNodeKind::Variable(TestVar::X),
                AstNodeKind::UnaryOp(UnaryOp::Neg),
                AstNodeKind::BinaryOp(BinaryOp::Mul),
            ])
        );
        assert_eq!(
            syntaxify("8.3 + -1"),
            Ok(vec![
                AstNodeKind::Number(8.3),
                AstNodeKind::Number(-1.0),
                AstNodeKind::BinaryOp(BinaryOp::Add),
            ])
        );
        assert_eq!(syntaxify("++1"), Ok(vec![AstNodeKind::Number(1.0)]));
        assert_eq!(syntaxify("+-1"), Ok(vec![AstNodeKind::Number(-1.0)]));
        assert_eq!(
            syntaxify("x + +1"),
            Ok(vec![
                AstNodeKind::Variable(TestVar::X),
                AstNodeKind::Number(1.0),
                AstNodeKind::BinaryOp(BinaryOp::Add),
            ])
        );
        assert_eq!(
            syntaxify("-1^2"),
            Ok(vec![
                AstNodeKind::Number(1.0),
                AstNodeKind::Number(2.0),
                AstNodeKind::BinaryOp(BinaryOp::Pow),
                AstNodeKind::UnaryOp(UnaryOp::Neg),
            ])
        );
        assert_eq!(
            syntaxify("x!y"),
            Ok(vec![
                AstNodeKind::Variable(TestVar::X),
                AstNodeKind::UnaryOp(UnaryOp::Fac),
                AstNodeKind::Variable(TestVar::Y),
                AstNodeKind::BinaryOp(BinaryOp::Mul),
            ])
        );
        assert_eq!(
            syntaxify("sin(x!-1)"),
            Ok(vec![
                AstNodeKind::Variable(TestVar::X),
                AstNodeKind::UnaryOp(UnaryOp::Fac),
                AstNodeKind::Number(1.0),
                AstNodeKind::BinaryOp(BinaryOp::Sub),
                AstNodeKind::NativeFunction(NativeFunction::Sin, 1),
            ])
        );
        assert_eq!(
            syntaxify("3|x-1|/2 + 1"),
            Ok(vec![
                AstNodeKind::Number(3.0),
                AstNodeKind::Variable(TestVar::X),
                AstNodeKind::Number(1.0),
                AstNodeKind::BinaryOp(BinaryOp::Sub),
                AstNodeKind::NativeFunction(NativeFunction::Abs, 1),
                AstNodeKind::BinaryOp(BinaryOp::Mul),
                AstNodeKind::Number(2.0),
                AstNodeKind::BinaryOp(BinaryOp::Div),
                AstNodeKind::Number(1.0),
                AstNodeKind::BinaryOp(BinaryOp::Add),
            ])
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
    fn test_descendants_count() {
        macro_rules! test {
            ($input: expr, $cc: expr) => {{
                let ast = parse($input).unwrap();
                assert_eq!(
                    ast.0
                        .iter()
                        .map(|n| n.descendants_count)
                        .collect::<Vec<_>>(),
                    $cc,
                    "{ast:?}"
                );
            }};
        }
        test!("1", vec![0]);
        test!("1+2", vec![0, 0, 2]);
        test!("3*5+5", vec![0, 0, 2, 0, 4]);
        test!("-x!", vec![0, 1, 2]);
        test!("8*3^2-1", vec![0, 0, 0, 2, 4, 0, 6]);
        test!("sin(x)", vec![0, 1]);
        test!("deg2rad(170)", vec![0, 1]);
        test!("expd(0.05, x)", vec![0, 0, 2]);
        test!("clamp(t, x, y)", vec![0, 0, 0, 3]);
        test!("digits(3, 3, 4, 4, 5)", vec![0, 0, 0, 0, 0, 5]);
        test!("sin(cos(1))", vec![0, 1, 2]);
        test!("x^2 + sin(y)", vec![0, 0, 2, 0, 1, 5]);
        test!("sqrt(max(x, 9))", vec![0, 0, 2, 3]);
        test!(
            "1+max(2, x, 8y, xy+1)",
            vec![0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 4, 10, 12]
        )
    }

    #[test]
    fn test_child_node_iter() {
        let ast = PostfixMathAst::<f64, TestVar, TestFunc>(
            [
                (AstNodeKind::Variable(TestVar::T), 0),
                (AstNodeKind::Number(100.0), 0),
                (AstNodeKind::BinaryOp(BinaryOp::Mul), 2),
                (AstNodeKind::Number(2.0), 0),
                (AstNodeKind::Variable(TestVar::X), 0),
                (AstNodeKind::BinaryOp(BinaryOp::Mul), 2),
                (AstNodeKind::Variable(TestVar::Y), 0),
                (AstNodeKind::Number(7.1), 0),
                (AstNodeKind::BinaryOp(BinaryOp::Mul), 2),
                (AstNodeKind::BinaryOp(BinaryOp::Sub), 6),
                (AstNodeKind::Number(0.0), 0),
                (AstNodeKind::NativeFunction(NativeFunction::Max, 3), 11),
            ]
            .into_iter()
            .map(|(k, cc)| AstNode::new(k, cc))
            .collect(),
        );
        let iter_children = |idx| {
            ast.iter_children(idx)
                .map(|(n, i)| (n.kind, i))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            iter_children(2),
            vec![
                (AstNodeKind::Number(100.0), 1),
                (AstNodeKind::Variable(TestVar::T), 0),
            ]
        );
        assert_eq!(
            iter_children(5),
            vec![
                (AstNodeKind::Variable(TestVar::X), 4),
                (AstNodeKind::Number(2.0), 3),
            ]
        );
        assert_eq!(
            iter_children(9),
            vec![
                (AstNodeKind::BinaryOp(BinaryOp::Mul), 8),
                (AstNodeKind::BinaryOp(BinaryOp::Mul), 5),
            ]
        );
        assert_eq!(
            iter_children(11),
            vec![
                (AstNodeKind::Number(0.0), 10),
                (AstNodeKind::BinaryOp(BinaryOp::Sub), 9),
                (AstNodeKind::BinaryOp(BinaryOp::Mul), 2)
            ]
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

    #[test]
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
                    values
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| 10f64.powi(i as i32) * v)
                        .sum()
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
        assert_eval!("digits(y, 1)", 15.0);
        assert_eval!("digits(5, 4, 9)*t", 94.5);
    }

    #[test]
    fn test_token2range() {
        let input = " max(pi, 1, -4)*3";
        let ts = TokenStream::new(input).unwrap().0;
        assert_eq!(token_range_to_str_range(input, &ts, 0..=0), 1..=4);
        assert_eq!(token_range_to_str_range(input, &ts, 1..=1), 5..=6);
        assert_eq!(token_range_to_str_range(input, &ts, 2..=2), 7..=7);
        assert_eq!(token_range_to_str_range(input, &ts, 3..=3), 9..=9);
    }
}
