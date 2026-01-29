use std::marker::PhantomData;

use super::{AstNode, FunctionType, MathAst, SyntaxError, SyntaxErrorKind};
use crate::{
    BinaryOp, FunctionIdentifier, FunctionPointer, NAME_LIMIT, UnaryOp, VariableIdentifier,
    VariableStore,
    number::{NFPointer, NativeFunction, Number},
    postfix_tree::subtree_collection::SubtreeCollection,
    tokenizer::{OprToken, Token},
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
            SyOperator::BinaryOp(BinaryOp::NegExp) => 3,
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
        N: Number,
        V: VariableIdentifier,
    {
        match self {
            SyOperator::BinaryOp(opr) => AstNode::BinaryOp(opr),
            SyOperator::UnaryOp(opr) => AstNode::UnaryOp(opr),
            SyOperator::Function(SyFunction::NativeFunction(nf), args) => {
                AstNode::Function(nf.into(), args)
            }
            SyOperator::Function(SyFunction::CustomFunction(cf, _, _), args) => {
                AstNode::Function(FunctionType::Custom(cf), args)
            }
            SyOperator::Function(SyFunction::PipeAbs, _) => {
                AstNode::Function(NativeFunction::Abs.into(), 1)
            }
            SyOperator::Parentheses => unreachable!(),
        }
    }
}

impl<F> From<OprToken> for SyOperator<F>
where
    F: FunctionIdentifier,
{
    fn from(value: OprToken) -> Self {
        match value {
            OprToken::Plus => Self::BinaryOp(BinaryOp::Add),
            OprToken::Minus => Self::BinaryOp(BinaryOp::Sub),
            OprToken::Multiply => Self::BinaryOp(BinaryOp::Mul),
            OprToken::Divide => Self::BinaryOp(BinaryOp::Div),
            OprToken::Power => Self::BinaryOp(BinaryOp::Pow),
            OprToken::Modulo => Self::BinaryOp(BinaryOp::Mod),
            OprToken::NegExp => Self::BinaryOp(BinaryOp::NegExp),
            OprToken::DoubleStar => Self::BinaryOp(BinaryOp::Mul),
            OprToken::Factorial => Self::UnaryOp(UnaryOp::Fac),
            OprToken::DoubleFactorial => Self::UnaryOp(UnaryOp::DoubleFac),
        }
    }
}

pub(super) trait ShuntingYardOutput<N, V, F>
where
    N: Number,
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
    N: Number,
    V: VariableIdentifier,
    F: FunctionIdentifier;

impl<N, V, F> ShuntingYardOutput<N, V, F> for SyAstOutput<N, V, F>
where
    N: Number,
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
    fn last_num<'a>(&'a self) -> Option<<N as Number>::AsArg<'a>> {
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
    N: Number,
    V: VariableIdentifier,
    F: FunctionIdentifier,
    S: VariableStore<N, V>,
    C: Fn(F) -> FunctionPointer<'a, N>,
{
    pub(super) args: Vec<N>,
    pub(super) variable_store: &'b S,
    pub(super) cf2pointer: C,
    pub(super) var_ident: PhantomData<V>,
    pub(super) func_ident: PhantomData<F>,
}

impl<'a, 'b, N, V, F, S, C> ShuntingYardOutput<N, V, F> for SyNumberOutput<'a, 'b, N, V, F, S, C>
where
    N: Number,
    V: VariableIdentifier,
    F: FunctionIdentifier,
    S: VariableStore<N, V>,
    C: Fn(F) -> FunctionPointer<'a, N>,
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
        // FIX: unwrap in one place
        let res = match node {
            AstNode::Number(num) => num,
            AstNode::Variable(var) => self.variable_store.get(var).to_owned(),
            AstNode::Function(FunctionType::Native(nf), args) => match nf.as_pointer::<N>() {
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
            AstNode::Function(FunctionType::Custom(cf), args) => match (self.cf2pointer)(cf) {
                FunctionPointer::Single(func) => func(self.args.pop().unwrap().asarg()),
                FunctionPointer::Dual(func) => {
                    let rhs = self.args.pop().unwrap();
                    func(self.args.pop().unwrap().asarg(), rhs.asarg())
                }
                FunctionPointer::Triple(func) => {
                    let a3 = self.args.pop().unwrap();
                    let a2 = self.args.pop().unwrap();
                    func(self.args.pop().unwrap().asarg(), a2.asarg(), a3.asarg())
                }
                FunctionPointer::Flexible(func) => {
                    let res = func(&self.args[self.args.len() - args as usize..]);
                    self.args.truncate(self.args.len() - args as usize);
                    res
                }
                FunctionPointer::DynSingle(func) => func(self.args.pop().unwrap().asarg()),
                FunctionPointer::DynDual(func) => {
                    let rhs = self.args.pop().unwrap();
                    func(self.args.pop().unwrap().asarg(), rhs.asarg())
                }
                FunctionPointer::DynTriple(func) => {
                    let a3 = self.args.pop().unwrap();
                    let a2 = self.args.pop().unwrap();
                    func(self.args.pop().unwrap().asarg(), a2.asarg(), a3.asarg())
                }
                FunctionPointer::DynFlexible(func) => {
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

fn find_opening_paren<S: AsRef<str>>(tokens: &[Token<S>]) -> Option<usize> {
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

fn find_opening_pipe<S: AsRef<str>>(tokens: &[Token<S>]) -> Option<usize> {
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

fn inside_pipe_abs<F>(operator_stack: &[SyOperator<F>]) -> bool
where
    F: FunctionIdentifier,
{
    for opr in operator_stack.iter().rev() {
        match opr {
            SyOperator::BinaryOp(_) | SyOperator::UnaryOp(_) => (),
            SyOperator::Function(SyFunction::PipeAbs, _) => return true,
            SyOperator::Function(_, _) | SyOperator::Parentheses => return false,
        }
    }
    false
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Segment<N, V>
where
    N: Number,
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
    N: Number,
    V: VariableIdentifier,
{
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
    N: Number,
    V: VariableIdentifier,
    F: FunctionIdentifier,
{
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

// E -> PE | ES | EIE | (E) | |E| | F(A) | T
// P -> + | -
// S -> ! | !!
// I -> + | - | * | / | ^
// A -> E,A | E
// T -> N | V

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExprState {
    Prefix,
    Suffix,
}

impl ExprState {
    fn new() -> Self {
        Self::Prefix
    }

    fn next<S: AsRef<str>, F: FunctionIdentifier>(
        &mut self,
        token: &Token<S>,
        operator_stack: &[SyOperator<F>],
    ) -> Result<bool, SyntaxErrorKind> {
        match self {
            ExprState::Prefix => match token {
                // prefix
                Token::Operator(OprToken::Plus | OprToken::Minus)
                | Token::OpenParen
                | Token::Function(_) => Ok(false),
                // suffix & infix
                Token::CloseParen | Token::Comma => Err(SyntaxErrorKind::UnexpectedToken),
                Token::Operator(
                    OprToken::Factorial
                    | OprToken::DoubleFactorial
                    | OprToken::Multiply
                    | OprToken::Divide
                    | OprToken::Power
                    | OprToken::NegExp
                    | OprToken::Modulo
                    | OprToken::DoubleStar,
                ) => Err(SyntaxErrorKind::MisplacedOperator),
                // pipe which may be prefix or suffix depending on the state
                Token::Pipe => {
                    if inside_pipe_abs(operator_stack) {
                        // suffix
                        Err(SyntaxErrorKind::UnexpectedToken)
                    } else {
                        // prefix
                        Ok(false)
                    }
                }
                // terminal
                Token::Number(_) | Token::Variable(_) => {
                    *self = ExprState::Suffix;
                    Ok(false)
                }
            },
            ExprState::Suffix => match token {
                // suffix
                Token::Operator(OprToken::Factorial | OprToken::DoubleFactorial)
                | Token::CloseParen => Ok(false),
                // prefix
                Token::OpenParen | Token::Function(_) => Ok(true),
                // pipe which may be prefix or suffix depending on the state
                Token::Pipe => {
                    if inside_pipe_abs(operator_stack) {
                        Ok(false)
                    } else {
                        *self = ExprState::Prefix;
                        Ok(true)
                    }
                }
                // infix
                Token::Operator(
                    OprToken::Plus
                    | OprToken::Minus
                    | OprToken::Multiply
                    | OprToken::Divide
                    | OprToken::Power
                    | OprToken::NegExp
                    | OprToken::Modulo
                    | OprToken::DoubleStar,
                )
                | Token::Comma => {
                    *self = ExprState::Prefix;
                    Ok(false)
                }
                // terminal
                Token::Number(_) | Token::Variable(_) => Ok(true),
            },
        }
    }
}

pub(super) fn parse_or_eval<O, N, V, F, S>(
    mut output_queue: O,
    tokens: impl AsRef<[Token<S>]>,
    custom_constant_parser: impl Fn(&str) -> Option<N>,
    custom_function_parser: impl Fn(&str) -> Option<(F, u8, Option<u8>)>,
    custom_variable_parser: impl Fn(&str) -> Option<V>,
) -> Result<O::Output, SyntaxError>
where
    O: ShuntingYardOutput<N, V, F>,
    N: Number,
    V: VariableIdentifier,
    F: FunctionIdentifier,
    S: AsRef<str>,
{
    // Dijkstra's shunting yard algorithm
    let tokens = tokens.as_ref();
    let clarify_err = |kind: SyntaxErrorKind, pos: usize| {
        if kind == SyntaxErrorKind::UnexpectedToken && pos > 0 {
            match tokens[pos - 1..=pos] {
                [Token::Comma, Token::CloseParen]
                | [Token::OpenParen | Token::Function(_), Token::Comma]
                | [Token::Comma, Token::Comma]
                | [Token::Function(_), Token::CloseParen] => {
                    SyntaxError(SyntaxErrorKind::EmptyArgument, pos - 1..=pos)
                }
                [Token::OpenParen, Token::CloseParen] => {
                    SyntaxError(SyntaxErrorKind::EmptyParenthesis, pos - 1..=pos)
                }
                [Token::Pipe, Token::Pipe] => {
                    SyntaxError(SyntaxErrorKind::EmptyPipeAbs, pos - 1..=pos)
                }
                [Token::Operator(_), Token::CloseParen | Token::Comma] => {
                    SyntaxError(SyntaxErrorKind::MisplacedOperator, pos - 1..=pos - 1)
                }
                _ => SyntaxError(kind, pos..=pos),
            }
        } else {
            return SyntaxError(kind, pos..=pos);
        }
    };
    let mut state = ExprState::new();
    let mut operator_stack: Vec<SyOperator<F>> = Vec::new();
    for (pos, token) in tokens.iter().enumerate() {
        let last_state = state;
        let implied_mult = state
            .next(token, &operator_stack)
            .map_err(|e| clarify_err(e, pos))?;
        if implied_mult {
            output_queue.push_opr(SyOperator::BinaryOp(BinaryOp::Mul), &mut operator_stack);
        }
        match token {
            Token::Number(num) => output_queue.push(
                num.as_ref()
                    .parse::<N>()
                    .map(AstNode::Number)
                    .map_err(|_| SyntaxError(SyntaxErrorKind::NumberParsingError, pos..=pos))?,
            ),
            Token::Variable(var) => {
                let var = var.as_ref();
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
            Token::Operator(opr) => {
                let opr = *opr;
                if opr == OprToken::Minus && last_state == ExprState::Prefix {
                    output_queue.push_opr(SyOperator::UnaryOp(UnaryOp::Neg), &mut operator_stack)
                } else if opr != OprToken::Plus || last_state != ExprState::Prefix {
                    output_queue.push_opr(opr.into(), &mut operator_stack)
                }
            }
            Token::Function(name) => {
                let name = name.as_ref();
                if name.len() > NAME_LIMIT as usize {
                    return Err(SyntaxError(SyntaxErrorKind::NameTooLong, pos..=pos));
                }
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
                } else if let Some(segments) =
                    segment_variable(name, &custom_constant_parser, &custom_variable_parser)
                {
                    for seg in segments {
                        output_queue.push(match seg {
                            Segment::Constant(c) => AstNode::Number(c),
                            Segment::Variable(v) => AstNode::Variable(v),
                        });
                        output_queue
                            .push_opr(SyOperator::BinaryOp(BinaryOp::Mul), &mut operator_stack);
                    }
                    operator_stack.push(SyOperator::Parentheses);
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
                            match output_queue.last_num() {
                                Some(num) if num == ten.asarg() => {
                                    output_queue.pop_arg();
                                    nf = NativeFunction::Log10;
                                }
                                Some(num) if num == two.asarg() => {
                                    output_queue.pop_arg();
                                    nf = NativeFunction::Log2;
                                }
                                _ if args == 1 => {
                                    nf = NativeFunction::Log10;
                                }
                                _ => (),
                            }
                            output_queue.push(AstNode::Function(nf.into(), args));
                            Ok(())
                        } else if args < nf.min_args() {
                            Err(SyntaxErrorKind::NotEnoughArguments)
                        } else if nf.max_args().is_some_and(|m| args > m) {
                            Err(SyntaxErrorKind::TooManyArguments)
                        } else {
                            output_queue.push(AstNode::Function(nf.into(), args));
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
                        output_queue.push(AstNode::Function(FunctionType::Custom(cf), args));
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
                if inside_pipe_abs(&operator_stack) {
                    output_queue.flush(&mut operator_stack);
                    if let Some(SyOperator::Function(SyFunction::PipeAbs, args)) =
                        operator_stack.pop()
                        && args > 1
                    {
                        let mut opening_pipe = 0;
                        for (i, tk) in tokens[..pos].iter().enumerate().rev() {
                            if matches!(tk, Token::Pipe) {
                                opening_pipe = i;
                                break;
                            }
                        }
                        return Err(SyntaxError(
                            SyntaxErrorKind::TooManyArguments,
                            opening_pipe..=pos,
                        ));
                    }
                    output_queue.push(AstNode::Function(NativeFunction::Abs.into(), 1));
                } else {
                    operator_stack.push(SyOperator::Function(SyFunction::PipeAbs, 0));
                }
            }
        }
    }
    if let Some(last) = tokens.last() {
        if state == ExprState::Prefix {
            let pos = tokens.len() - 1;
            if matches!(last, Token::Operator(_)) {
                return Err(SyntaxError(SyntaxErrorKind::MisplacedOperator, pos..=pos));
            } else {
                return Err(SyntaxError(SyntaxErrorKind::UnexpectedToken, pos..=pos));
            }
        }
    } else {
        return Err(SyntaxError(SyntaxErrorKind::EmptyInput, 0..=0));
    }
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
