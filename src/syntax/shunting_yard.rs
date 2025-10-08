use std::marker::PhantomData;

use super::AstNode;
use crate::{
    BinaryOp, FunctionIdentifier, NAME_LIMIT, UnaryOp, VariableIdentifier, VariableStore,
    asm::CFPointer,
    number::{MathEvalNumber, NFPointer, NativeFunction},
    postfix_tree::subtree_collection::SubtreeCollection,
    syntax::{MathAst, SyntaxError, SyntaxErrorKind},
    tokenizer::Token,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SyFunction<F>
where
    F: FunctionIdentifier,
{
    NativeFunction(NativeFunction),
    CustomFunction(F, u8, Option<u8>),
    PipeAbs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SyOperator<F>
where
    F: FunctionIdentifier,
{
    BinaryOp(BinaryOp),
    UnaryOp(UnaryOp),
    Function(SyFunction<F>, u8),
    Parentheses,
}

impl<F> SyOperator<F>
where
    F: FunctionIdentifier,
{
    fn precedence(&self) -> u8 {
        match self {
            SyOperator::BinaryOp(BinaryOp::Add) => 0,
            SyOperator::BinaryOp(BinaryOp::Sub) => 0,
            SyOperator::BinaryOp(BinaryOp::Mul) => 1,
            SyOperator::BinaryOp(BinaryOp::Div) => 1,
            SyOperator::BinaryOp(BinaryOp::Mod) => 1,
            SyOperator::UnaryOp(UnaryOp::Neg) => 2,
            SyOperator::BinaryOp(BinaryOp::Pow) => 3,
            SyOperator::UnaryOp(UnaryOp::Fac) => 4,
            SyOperator::UnaryOp(UnaryOp::DoubleFac) => 4,
            SyOperator::Function(_, _) | SyOperator::Parentheses => {
                unreachable!()
            }
        }
    }
    fn is_right_binding(&self) -> bool {
        matches!(
            self,
            SyOperator::BinaryOp(BinaryOp::Pow) | SyOperator::UnaryOp(UnaryOp::Neg)
        )
    }
    fn to_syn<N, V>(self) -> AstNode<N, V, F>
    where
        N: MathEvalNumber,
        V: VariableIdentifier,
    {
        match self {
            SyOperator::BinaryOp(opr) => AstNode::BinaryOp(opr),
            SyOperator::UnaryOp(opr) => AstNode::UnaryOp(opr),
            SyOperator::Function(SyFunction::NativeFunction(nf), args) => {
                AstNode::NativeFunction(nf, args)
            }
            SyOperator::Function(SyFunction::CustomFunction(cf, _, _), args) => {
                AstNode::CustomFunction(cf, args)
            }
            SyOperator::Function(SyFunction::PipeAbs, _) => {
                AstNode::NativeFunction(NativeFunction::Abs, 1)
            }
            SyOperator::Parentheses => unreachable!(),
        }
    }
}

fn after_implies_neg<F>(token: Token<'_>, operator_stack: &[SyOperator<F>]) -> bool
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

pub(super) trait ShuntingYardOutput<N, V, F>
where
    N: MathEvalNumber,
    V: VariableIdentifier,
    F: FunctionIdentifier,
{
    type Output;

    fn pop_opr(&mut self, operator_stack: &mut Vec<SyOperator<F>>);
    fn make(self) -> Self::Output;
    fn push(&mut self, node: AstNode<N, V, F>);
    fn pop_arg(&mut self) -> Option<AstNode<N, V, F>>;
    fn last_num<'a>(&'a self) -> Option<N::AsArg<'a>>;

    fn push_opr(&mut self, operator: SyOperator<F>, operator_stack: &mut Vec<SyOperator<F>>) {
        while let Some(top_opr) = operator_stack.last()
            && matches!(top_opr, SyOperator::BinaryOp(_) | SyOperator::UnaryOp(_))
            && (operator.precedence() < top_opr.precedence()
                || operator.precedence() == top_opr.precedence() && !operator.is_right_binding())
        {
            self.pop_opr(operator_stack);
        }
        operator_stack.push(operator);
    }
    fn flush(&mut self, operator_stack: &mut Vec<SyOperator<F>>) {
        while let Some(opr) = operator_stack.last()
            && matches!(opr, SyOperator::BinaryOp(_) | SyOperator::UnaryOp(_))
        {
            self.pop_opr(operator_stack)
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct SyAstOutput<N, V, F>(pub(super) SubtreeCollection<AstNode<N, V, F>>)
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
    type Output = MathAst<N, V, F>;

    fn pop_opr(&mut self, operator_stack: &mut Vec<SyOperator<F>>) {
        let opr = operator_stack.pop().unwrap();
        if let Some(AstNode::Number(num)) = self.0.last_mut()
            && opr == SyOperator::UnaryOp(UnaryOp::Neg)
        {
            *num = -num.asarg();
        } else {
            self.push(opr.to_syn());
        }
    }
    fn make(self) -> Self::Output {
        MathAst(self.0.into_tree())
    }
    fn push(&mut self, kind: AstNode<N, V, F>) {
        self.0.push(kind)
    }
    fn pop_arg(&mut self) -> Option<AstNode<N, V, F>> {
        self.0.pop()
    }
    fn last_num<'a>(&'a self) -> Option<<N as MathEvalNumber>::AsArg<'a>> {
        if let Some(AstNode::Number(num)) = self.0.last() {
            Some(num.asarg())
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub(super) struct SyNumberOutput<'a, 'b, N, V, F, S, C>
where
    N: MathEvalNumber,
    V: VariableIdentifier,
    F: FunctionIdentifier,
    S: VariableStore<N, V>,
    C: Fn(F) -> CFPointer<'a, N>,
{
    pub(super) args: Vec<N>,
    pub(super) variable_store: &'b S,
    pub(super) cf2pointer: C,
    pub(super) var_ident: PhantomData<V>,
    pub(super) func_ident: PhantomData<F>,
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

    fn pop_opr(&mut self, operator_stack: &mut Vec<SyOperator<F>>) {
        let res = match operator_stack.pop().unwrap() {
            SyOperator::BinaryOp(opr) => {
                let rhs = self.args.pop().unwrap();
                opr.eval(self.args.pop().unwrap().asarg(), rhs.asarg())
            }
            SyOperator::UnaryOp(opr) => opr.eval(self.args.pop().unwrap().asarg()),
            _ => panic!(),
        };
        self.args.push(res);
    }
    fn make(mut self) -> Self::Output {
        debug_assert_eq!(self.args.len(), 1);
        self.args.pop().unwrap()
    }
    fn push(&mut self, node: AstNode<N, V, F>) {
        let res = match node {
            AstNode::Number(num) => num,
            AstNode::Variable(var) => self.variable_store.get(var).to_owned(),
            AstNode::NativeFunction(nf, args) => match nf.to_pointer::<N>() {
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
            AstNode::CustomFunction(cf, args) => match (self.cf2pointer)(cf) {
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
            AstNode::BinaryOp(_) | AstNode::UnaryOp(_) => panic!(),
        };
        self.args.push(res);
    }
    fn pop_arg(&mut self) -> Option<AstNode<N, V, F>> {
        self.args.pop().map(|num| AstNode::Number(num))
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

fn inside_pipe_abs<F>(operator_stack: &[SyOperator<F>]) -> Option<usize>
where
    F: FunctionIdentifier,
{
    for (i, opr) in operator_stack.iter().enumerate().rev() {
        match opr {
            SyOperator::BinaryOp(_) | SyOperator::UnaryOp(_) => (),
            SyOperator::Function(SyFunction::PipeAbs, _) => return Some(i),
            SyOperator::Function(_, _) | SyOperator::Parentheses => return None,
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
                    .or_else(|| N::parse_constant(&input[j..i]))
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
) -> Option<(Vec<Segment<N, V>>, SyFunction<F>)>
where
    N: MathEvalNumber,
    V: VariableIdentifier,
    F: FunctionIdentifier,
{
    if input.len() > NAME_LIMIT as usize {
        panic!();
    }
    let mut can_segment: Vec<Option<(Option<u8>, Segment<N, V>)>> = vec![None; input.len() - 1];
    let mut function: Option<(u8, SyFunction<F>)> = None;
    for i in 1..input.len() {
        for j in 0..i {
            if j == 0 || can_segment[j - 1].is_some() {
                let prev = (j != 0).then(|| j as u8 - 1);
                let seg = constant_parser(&input[j..i])
                    .or_else(|| N::parse_constant(&input[j..i]))
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
                .map(SyFunction::NativeFunction)
                .or_else(|| {
                    function_parser(&input[j..])
                        .map(|(cf, min, max)| SyFunction::CustomFunction(cf, min, max))
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

pub(super) fn parse_or_eval<'a, O, N, V, F>(
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
    let mut operator_stack: Vec<SyOperator<F>> = Vec::new();
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
            output_queue.push_opr(SyOperator::BinaryOp(BinaryOp::Mul), &mut operator_stack);
        }
        match token {
            Token::Number(num) => output_queue.push(
                num.parse::<N>()
                    .map(AstNode::Number)
                    .map_err(|_| SyntaxError(SyntaxErrorKind::NumberParsingError, pos..=pos))?,
            ),
            Token::Variable(var) => {
                if var.len() > NAME_LIMIT as usize {
                    return Err(SyntaxError(SyntaxErrorKind::NameTooLong, pos..=pos));
                }
                if let Some(node) = N::parse_constant(var)
                    .or_else(|| custom_constant_parser(var))
                    .map(|c| AstNode::Number(c))
                    .or_else(|| custom_variable_parser(var).map(|var| AstNode::Variable(var)))
                {
                    output_queue.push(node)
                } else if let Some(segments) =
                    segment_variable(var, &custom_constant_parser, &custom_variable_parser)
                {
                    let mut first = true;
                    for seg in segments {
                        output_queue.push(match seg {
                            Segment::Constant(c) => AstNode::Number(c),
                            Segment::Variable(v) => AstNode::Variable(v),
                        });
                        if first {
                            first = false;
                        } else {
                            output_queue
                                .push_opr(SyOperator::BinaryOp(BinaryOp::Mul), &mut operator_stack);
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
                let mut sy_opr: SyOperator<F> = BinaryOp::parse(opr)
                    .map(|biopr| SyOperator::BinaryOp(biopr))
                    .or_else(|| UnaryOp::parse(opr).map(|unopr| SyOperator::UnaryOp(unopr)))
                    .unwrap();
                if opr == '-' && last_tk.is_none_or(|tk| after_implies_neg(tk, &operator_stack)) {
                    sy_opr = SyOperator::UnaryOp(UnaryOp::Neg);
                } else if opr == '!' && last_tk.is_some_and(|tk| tk == Token::Operation('!')) {
                    let fac = operator_stack.pop();
                    debug_assert_eq!(fac, Some(SyOperator::UnaryOp(UnaryOp::Fac)));
                    sy_opr = SyOperator::UnaryOp(UnaryOp::DoubleFac)
                }
                if opr != '+' || last_tk.is_some_and(|tk| !after_implies_neg(tk, &operator_stack)) {
                    output_queue.push_opr(sy_opr, &mut operator_stack);
                }
            }
            Token::Function(name) => {
                if let Some(func) = NativeFunction::parse(name)
                    .map(|nf| SyOperator::Function(SyFunction::NativeFunction(nf), 1))
                    .or_else(|| {
                        custom_function_parser(name).map(|(cf, min, max)| {
                            SyOperator::Function(SyFunction::CustomFunction(cf, min, max), 1)
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
                            Segment::Constant(c) => AstNode::Number(c),
                            Segment::Variable(v) => AstNode::Variable(v),
                        });
                        output_queue
                            .push_opr(SyOperator::BinaryOp(BinaryOp::Mul), &mut operator_stack);
                    }
                    operator_stack.push(SyOperator::Function(func, 1));
                } else {
                    return Err(SyntaxError(SyntaxErrorKind::UnknownFunction, pos..=pos));
                }
            }
            Token::OpenParen => {
                operator_stack.push(SyOperator::Parentheses);
            }
            Token::Comma => {
                output_queue.flush(&mut operator_stack);
                match operator_stack.last_mut() {
                    Some(SyOperator::Function(_, args)) => {
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
                    Some(SyOperator::Function(SyFunction::PipeAbs, _)) => {
                        let opening = find_opening_pipe(&tokens[..pos]).unwrap();
                        return Err(SyntaxError(
                            SyntaxErrorKind::PipeAbsNotClosed,
                            opening..=opening,
                        ));
                    }
                    Some(SyOperator::Function(SyFunction::NativeFunction(mut nf), args)) => {
                        if nf == NativeFunction::Log {
                            // FIX: recieve 10 and 2 from MathEvalNumber so it can be const
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
                            output_queue.push(AstNode::NativeFunction(nf, args));
                            Ok(())
                        } else if args < nf.min_args() {
                            Err(SyntaxErrorKind::NotEnoughArguments)
                        } else if nf.max_args().is_some_and(|m| args > m) {
                            Err(SyntaxErrorKind::TooManyArguments)
                        } else {
                            output_queue.push(AstNode::NativeFunction(nf, args));
                            Ok(())
                        }
                        .map_err(|e| {
                            SyntaxError(e, find_opening_paren(&tokens[..pos]).unwrap()..=pos)
                        })?
                    }
                    Some(SyOperator::Function(
                        SyFunction::CustomFunction(cf, min_args, max_args),
                        args,
                    )) => if args < min_args {
                        Err(SyntaxErrorKind::NotEnoughArguments)
                    } else if max_args.is_some_and(|m| args > m) {
                        Err(SyntaxErrorKind::TooManyArguments)
                    } else {
                        output_queue.push(AstNode::CustomFunction(cf, args));
                        Ok(())
                    }
                    .map_err(|e| {
                        SyntaxError(e, find_opening_paren(&tokens[..pos]).unwrap()..=pos)
                    })?,
                    Some(SyOperator::Parentheses) => (),
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
                    if let Some(SyOperator::Function(SyFunction::PipeAbs, args)) =
                        operator_stack.pop()
                        && args > 1
                    {
                        return Err(SyntaxError(
                            SyntaxErrorKind::TooManyArguments,
                            opening_pipe..=pos,
                        ));
                    }
                    output_queue.push(AstNode::NativeFunction(NativeFunction::Abs, 1));
                } else {
                    operator_stack.push(SyOperator::Function(SyFunction::PipeAbs, 0));
                }
            }
        }
        last_tk = Some(token);
    }
    validate_consecutive_tokens(last_tk, None, tokens.len())?;
    output_queue.flush(&mut operator_stack);
    match operator_stack.last() {
        Some(SyOperator::Function(SyFunction::PipeAbs, _)) => {
            let opening = find_opening_pipe(tokens).unwrap();
            Err(SyntaxError(
                SyntaxErrorKind::PipeAbsNotClosed,
                opening..=opening,
            ))
        }
        Some(SyOperator::Function(_, _) | SyOperator::Parentheses) => {
            let unclosed_paren_pos = find_opening_paren(tokens).unwrap();
            Err(SyntaxError(
                SyntaxErrorKind::MissingClosingParenthesis,
                unclosed_paren_pos..=unclosed_paren_pos,
            ))
        }
        _ => Ok(output_queue.make()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::*;

    #[test]
    fn segment_variable() {
        let seg_var = |input: &str| {
            super::segment_variable(
                input,
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
        assert_eq!(
            seg_var("xy"),
            Some(vec![Segment::Variable(0), Segment::Variable(1)])
        );
        assert_eq!(
            seg_var("cx"),
            Some(vec![Segment::Constant(299792458.0), Segment::Variable(0)])
        );
        assert_eq!(
            seg_var("pix"),
            Some(vec![Segment::Constant(PI), Segment::Variable(0)])
        );
        assert_eq!(
            seg_var("x2var5"),
            Some(vec![
                Segment::Variable(0),
                Segment::Constant(2.0),
                Segment::Variable(2)
            ])
        );
        assert_eq!(
            seg_var("x2yy"),
            Some(vec![
                Segment::Variable(0),
                Segment::Constant(2.0),
                Segment::Variable(1),
                Segment::Variable(1)
            ])
        );
        assert_eq!(
            seg_var("pi2x"),
            Some(vec![Segment::Constant(FRAC_2_PI), Segment::Variable(0)])
        );
        assert_eq!(
            seg_var("x8759y"),
            Some(vec![
                Segment::Variable(0),
                Segment::Constant(8759.0),
                Segment::Variable(1)
            ])
        );
        assert_eq!(
            seg_var("x9shallnotbenamedy"),
            Some(vec![
                Segment::Variable(0),
                Segment::Constant(9.0),
                Segment::Variable(3),
                Segment::Variable(1),
            ])
        );
    }

    #[test]
    fn segment_function() {
        let seg_func = |input: &str| {
            super::segment_function(
                input,
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
        assert_eq!(
            seg_func("cmin"),
            Some((
                vec![Segment::Constant(299792458.0)],
                SyFunction::NativeFunction(NativeFunction::Min)
            ))
        );
        assert_eq!(
            seg_func("x55sin"),
            Some((
                vec![Segment::Variable(0), Segment::Constant(55.0)],
                SyFunction::NativeFunction(NativeFunction::Sin)
            ))
        );
        assert_eq!(
            seg_func("xyf2"),
            Some((
                vec![Segment::Variable(0), Segment::Variable(1)],
                SyFunction::CustomFunction(1, 0, None)
            ))
        );
        assert_eq!(
            seg_func("xvar5func1"),
            Some((
                vec![Segment::Variable(0), Segment::Variable(2)],
                SyFunction::CustomFunction(0, 0, None)
            ))
        );
        assert_eq!(
            seg_func("xxxxvar5verylongfunction"),
            Some((
                vec![
                    Segment::Variable(0),
                    Segment::Variable(0),
                    Segment::Variable(0),
                    Segment::Variable(0),
                    Segment::Variable(2)
                ],
                SyFunction::CustomFunction(2, 0, None)
            ))
        );
    }
}
