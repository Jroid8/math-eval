use std::{fmt::Debug, marker::PhantomData};

use super::{AstNode, FunctionType, MathAst, SyntaxError, SyntaxErrorKind};
use crate::{
    BinaryOp, FunctionIdentifier, FunctionPointer, NAME_LIMIT, UnaryOp, VariableIdentifier,
    VariableStore,
    number::{NFPointer, NativeFunction, Number},
    postfix_tree::subtree_collection::{MultipleRoots, NotEnoughOrphans, SubtreeCollection},
    tokenizer::{OprToken, Token},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SyFunction<F>
where
    F: FunctionIdentifier,
{
    Native(NativeFunction),
    Custom(F, u8, Option<u8>),
    PipeAbs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SyOperator<F>
where
    F: FunctionIdentifier,
{
    BinaryOp(BinaryOp),
    UnaryOp(UnaryOp),
    HpNeg,
    Function(SyFunction<F>, u8),
    FuncNoParen(FunctionType<F>),
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
            SyOperator::FuncNoParen(_) => 3,
            SyOperator::BinaryOp(BinaryOp::Pow) => 4,
            SyOperator::HpNeg => 5,
            SyOperator::UnaryOp(UnaryOp::Fac) => 5,
            SyOperator::UnaryOp(UnaryOp::DoubleFac) => 5,
            SyOperator::Function(_, _) | SyOperator::Parentheses => {
                unreachable!()
            }
        }
    }
    fn is_right_binding(&self) -> bool {
        matches!(
            self,
            SyOperator::BinaryOp(BinaryOp::Pow)
                | SyOperator::UnaryOp(UnaryOp::Neg)
                | SyOperator::FuncNoParen(_)
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
            SyOperator::HpNeg => AstNode::UnaryOp(UnaryOp::Neg),
            SyOperator::Function(SyFunction::Native(nf), args) => {
                AstNode::Function(nf.into(), args)
            }
            SyOperator::Function(SyFunction::Custom(cf, _, _), args) => {
                AstNode::Function(FunctionType::Custom(cf), args)
            }
            SyOperator::FuncNoParen(FunctionType::Native(nf)) => AstNode::Function(nf.into(), 1),
            SyOperator::FuncNoParen(FunctionType::Custom(cf)) => {
                AstNode::Function(FunctionType::Custom(cf), 1)
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
            OprToken::DoubleStar => Self::BinaryOp(BinaryOp::Mul),
            OprToken::Factorial => Self::UnaryOp(UnaryOp::Fac),
            OprToken::DoubleFactorial => Self::UnaryOp(UnaryOp::DoubleFac),
        }
    }
}

pub(super) trait ShuntingYardOutput<N, V, F>: Debug
where
    N: Number,
    V: VariableIdentifier,
    F: FunctionIdentifier,
{
    type Output;

    fn pop_opr(&mut self, operator_stack: &mut Vec<SyOperator<F>>) -> Result<(), NotEnoughOrphans>;
    fn make(self) -> Result<Self::Output, MultipleRoots>;
    fn push(&mut self, node: AstNode<N, V, F>) -> Result<(), NotEnoughOrphans>;
    fn pop_arg(&mut self) -> Option<AstNode<N, V, F>>;
    fn last_num<'a>(&'a self) -> Option<N::AsArg<'a>>;

    fn push_opr(
        &mut self,
        operator: SyOperator<F>,
        operator_stack: &mut Vec<SyOperator<F>>,
    ) -> Result<(), NotEnoughOrphans> {
        while let Some(top_opr) = operator_stack.last()
            && matches!(top_opr, SyOperator::BinaryOp(_) | SyOperator::UnaryOp(_))
            && (operator.precedence() < top_opr.precedence()
                || operator.precedence() == top_opr.precedence() && !operator.is_right_binding())
        {
            self.pop_opr(operator_stack)?;
        }
        operator_stack.push(operator);
        Ok(())
    }
    fn flush(&mut self, operator_stack: &mut Vec<SyOperator<F>>) -> Result<(), NotEnoughOrphans> {
        while let Some(opr) = operator_stack.last()
            && matches!(
                opr,
                SyOperator::BinaryOp(_)
                    | SyOperator::UnaryOp(_)
                    | SyOperator::HpNeg
                    | SyOperator::FuncNoParen(_)
            )
        {
            self.pop_opr(operator_stack)?;
        }
        Ok(())
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

    fn pop_opr(&mut self, operator_stack: &mut Vec<SyOperator<F>>) -> Result<(), NotEnoughOrphans> {
        let opr = operator_stack.pop().unwrap();
        if let Some(AstNode::Number(num)) = self.0.last_mut()
            && matches!(opr, SyOperator::UnaryOp(UnaryOp::Neg) | SyOperator::HpNeg)
        {
            *num = -num.asarg();
            Ok(())
        } else {
            self.push(opr.to_syn())
        }
    }
    fn make(self) -> Result<Self::Output, MultipleRoots> {
        self.0.into_tree().map(|t| MathAst(t))
    }
    fn push(&mut self, kind: AstNode<N, V, F>) -> Result<(), NotEnoughOrphans> {
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

#[derive(Clone)]
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

impl<'a, 'b, N, V, F, S, C> SyNumberOutput<'a, 'b, N, V, F, S, C>
where
    N: Number,
    V: VariableIdentifier,
    F: FunctionIdentifier,
    S: VariableStore<N, V>,
    C: Fn(F) -> FunctionPointer<'a, N>,
{
    fn args_pop(&mut self) -> Result<N, NotEnoughOrphans> {
        self.args.pop().ok_or(NotEnoughOrphans)
    }
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

    fn pop_opr(&mut self, operator_stack: &mut Vec<SyOperator<F>>) -> Result<(), NotEnoughOrphans> {
        let res = match operator_stack.pop().unwrap() {
            SyOperator::BinaryOp(opr) => {
                let rhs = self.args_pop()?;
                opr.eval(self.args_pop()?.asarg(), rhs.asarg())
            }
            SyOperator::UnaryOp(opr) => opr.eval(self.args_pop()?.asarg()),
            SyOperator::HpNeg => -self.args_pop()?,
            SyOperator::FuncNoParen(FunctionType::Native(nf)) => match nf.as_pointer() {
                NFPointer::Single(func) => func(self.args_pop()?.asarg()),
                NFPointer::Flexible(func) => func(&[self.args_pop()?]),
                NFPointer::Dual(_) => unreachable!(),
            },
            SyOperator::FuncNoParen(FunctionType::Custom(cf)) => match (self.cf2pointer)(cf) {
                FunctionPointer::Single(func) => func(self.args_pop()?.asarg()),
                FunctionPointer::Flexible(func) => func(&[self.args_pop()?]),
                FunctionPointer::DynSingle(func) => func(self.args_pop()?.asarg()),
                FunctionPointer::DynFlexible(func) => func(&[self.args_pop()?]),
                _ => unreachable!(),
            },
            _ => unreachable!(),
        };
        self.args.push(res);
        Ok(())
    }
    fn make(mut self) -> Result<Self::Output, MultipleRoots> {
        if self.args.len() > 1 {
            Err(MultipleRoots)
        } else {
            Ok(self.args.pop().unwrap())
        }
    }
    fn push(&mut self, node: AstNode<N, V, F>) -> Result<(), NotEnoughOrphans> {
        // FIX: unwrap in one place
        let res = match node {
            AstNode::Number(num) => num,
            AstNode::Variable(var) => self.variable_store.get(var).to_owned(),
            AstNode::Function(FunctionType::Native(nf), args) => match nf.as_pointer::<N>() {
                NFPointer::Single(func) => func(self.args_pop()?.asarg()),
                NFPointer::Dual(func) => {
                    let rhs = self.args_pop()?;
                    func(self.args_pop()?.asarg(), rhs.asarg())
                }
                NFPointer::Flexible(func) => {
                    let res = func(&self.args[self.args.len() - args as usize..]);
                    self.args.truncate(self.args.len() - args as usize);
                    res
                }
            },
            AstNode::Function(FunctionType::Custom(cf), args) => match (self.cf2pointer)(cf) {
                FunctionPointer::Single(func) => func(self.args_pop()?.asarg()),
                FunctionPointer::Dual(func) => {
                    let rhs = self.args_pop()?;
                    func(self.args_pop()?.asarg(), rhs.asarg())
                }
                FunctionPointer::Triple(func) => {
                    let a3 = self.args_pop()?;
                    let a2 = self.args_pop()?;
                    func(self.args_pop()?.asarg(), a2.asarg(), a3.asarg())
                }
                FunctionPointer::Flexible(func) => {
                    let res = func(&self.args[self.args.len() - args as usize..]);
                    self.args.truncate(self.args.len() - args as usize);
                    res
                }
                FunctionPointer::DynSingle(func) => func(self.args_pop()?.asarg()),
                FunctionPointer::DynDual(func) => {
                    let rhs = self.args_pop()?;
                    func(self.args_pop()?.asarg(), rhs.asarg())
                }
                FunctionPointer::DynTriple(func) => {
                    let a3 = self.args_pop()?;
                    let a2 = self.args_pop()?;
                    func(self.args_pop()?.asarg(), a2.asarg(), a3.asarg())
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
        Ok(())
    }
    fn pop_arg(&mut self) -> Option<AstNode<N, V, F>> {
        self.args.pop().map(|num| AstNode::Number(num))
    }
    fn last_num<'c>(&'c self) -> Option<N::AsArg<'c>> {
        self.args.last().map(|num| num.asarg())
    }
}

impl<'a, 'b, N, V, F, S, C> Debug for SyNumberOutput<'a, 'b, N, V, F, S, C>
where
    N: Number,
    V: VariableIdentifier,
    F: FunctionIdentifier,
    S: VariableStore<N, V>,
    C: Fn(F) -> FunctionPointer<'a, N>,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SyNumberOutput")
            .field("args", &self.args)
            .field("variable_store", &self.variable_store)
            .field("var_ident", &self.var_ident)
            .field("func_ident", &self.func_ident)
            .finish()
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
            SyOperator::BinaryOp(_)
            | SyOperator::UnaryOp(_)
            | SyOperator::HpNeg
            | SyOperator::FuncNoParen(_) => (),
            SyOperator::Function(SyFunction::PipeAbs, _) => return true,
            SyOperator::Function(_, _) | SyOperator::Parentheses => {
                return false;
            }
        }
    }
    false
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Segment<N, V, F>
where
    N: Number,
    V: VariableIdentifier,
    F: FunctionIdentifier,
{
    Constant(N),
    Variable(V),
    Function(FunctionType<F>, u8, Option<u8>),
}

fn push_segments<N, V, F, O>(
    segments: Vec<Segment<N, V, F>>,
    operator_stack: &mut Vec<SyOperator<F>>,
    output_queue: &mut O,
) -> Result<(), SyntaxErrorKind>
where
    N: Number,
    V: VariableIdentifier,
    F: FunctionIdentifier,
    O: ShuntingYardOutput<N, V, F>,
{
    let mut was_func = true;
    for seg in segments {
        let is_func = matches!(seg, Segment::Function(_, _, _));
        if !was_func {
            output_queue.push_opr(SyOperator::BinaryOp(BinaryOp::Mul), operator_stack)?;
        }
        match seg {
            Segment::Constant(num) => output_queue.push(AstNode::Number(num))?,
            Segment::Variable(var) => output_queue.push(AstNode::Variable(var))?,
            Segment::Function(func, min, _) => {
                if min > 1 {
                    return Err(SyntaxErrorKind::NotEnoughArguments);
                } else {
                    output_queue.push_opr(SyOperator::FuncNoParen(func), operator_stack)?;
                }
            }
        }
        was_func = is_func;
    }
    Ok(())
}

fn segment_token<N, V, F>(
    input: &str,
    constant_parser: &impl Fn(&str) -> Option<N>,
    variable_parser: &impl Fn(&str) -> Option<V>,
    function_parser: &impl Fn(&str) -> Option<(F, u8, Option<u8>)>,
) -> Option<Vec<Segment<N, V, F>>>
where
    N: Number,
    V: VariableIdentifier,
    F: FunctionIdentifier,
{
    let char_count = input.chars().count();
    let mut can_segment: Vec<Option<(Option<u8>, Segment<N, V, F>)>> = vec![None; char_count];
    for (i, ic) in input
        .char_indices()
        .enumerate()
        .map(|(i, (cb, c))| (i + 1, cb + c.len_utf8()))
    {
        for (j, (jc, _)) in input[..ic].char_indices().enumerate() {
            if j == 0 || can_segment[j - 1].is_some() {
                let seg = &input[jc..ic];
                let prev = (j != 0).then(|| j as u8 - 1);
                if let Some(seg) = constant_parser(seg)
                    .or_else(|| N::parse_constant(seg))
                    .or_else(|| seg.parse().ok())
                    .map(Segment::Constant)
                    .or_else(|| variable_parser(seg).map(Segment::Variable))
                    .or_else(|| {
                        NativeFunction::parse(seg)
                            .map(|nf| Segment::Function(nf.into(), nf.min_args(), nf.max_args()))
                    })
                    .or_else(|| {
                        function_parser(seg).map(|(cf, min, max)| {
                            Segment::Function(FunctionType::Custom(cf), min, max)
                        })
                    })
                {
                    can_segment[i - 1] = Some((prev, seg));
                    break;
                }
            }
        }
    }
    if can_segment[char_count - 1].is_some() {
        let mut result = Vec::with_capacity(NAME_LIMIT as usize);
        let mut idx = Some(char_count as u8 - 1);
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

// E -> PE | ES | EIE | (E) | |E| | F(A) | FE | T
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
                Token::OpenParen | Token::Function(_) => {
                    *self = ExprState::Prefix;
                    Ok(true)
                }
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
            SyntaxError(kind, pos..=pos)
        }
    };
    let mut state = ExprState::new();
    let mut was_pow = false;
    let mut operator_stack: Vec<SyOperator<F>> = Vec::new();
    for (pos, token) in tokens.iter().enumerate() {
        let last_state = state;
        let implied_mult = state
            .next(token, &operator_stack)
            .map_err(|e| clarify_err(e, pos))?;
        if implied_mult {
            output_queue.push_opr(SyOperator::BinaryOp(BinaryOp::Mul), &mut operator_stack)?;
        }
        match token {
            Token::Number(num) => output_queue.push(
                num.as_ref()
                    .parse::<N>()
                    .map(AstNode::Number)
                    .map_err(|_| SyntaxError(SyntaxErrorKind::NumberParsingError, pos..=pos))?,
            )?,
            Token::Operator(opr) => {
                let opr = *opr;
                if opr == OprToken::Minus && last_state == ExprState::Prefix {
                    if was_pow {
                        output_queue.push_opr(SyOperator::HpNeg, &mut operator_stack)?;
                    } else {
                        output_queue
                            .push_opr(SyOperator::UnaryOp(UnaryOp::Neg), &mut operator_stack)?;
                    }
                } else if opr != OprToken::Plus || last_state != ExprState::Prefix {
                    output_queue.push_opr(opr.into(), &mut operator_stack)?;
                }
            }
            Token::Variable(name) => {
                let name = name.as_ref();
                if name.len() > NAME_LIMIT as usize {
                    return Err(SyntaxError(SyntaxErrorKind::NameTooLong, pos..=pos));
                }
                if let Some(node) = N::parse_constant(name)
                    .or_else(|| custom_constant_parser(name))
                    .map(AstNode::Number)
                    .or_else(|| custom_variable_parser(name).map(|var| AstNode::Variable(var)))
                {
                    output_queue.push(node)?;
                } else if let Some((func, min)) = NativeFunction::parse(name)
                    .map(|nf| {
                        (
                            SyOperator::FuncNoParen(FunctionType::Native(nf)),
                            nf.min_args(),
                        )
                    })
                    .or_else(|| {
                        custom_function_parser(name).map(|(cf, min, _)| {
                            (SyOperator::FuncNoParen(FunctionType::Custom(cf)), min)
                        })
                    })
                {
                    if min > 1 {
                        return Err(SyntaxError(SyntaxErrorKind::NotEnoughArguments, pos..=pos));
                    }
                    state = ExprState::Prefix;
                    output_queue.push_opr(func, &mut operator_stack)?;
                } else if let Some(segments) = segment_token(
                    name,
                    &custom_constant_parser,
                    &custom_variable_parser,
                    &custom_function_parser,
                ) {
                    if matches!(segments.last().unwrap(), Segment::Function(_, _, _)) {
                        state = ExprState::Prefix;
                    }
                    push_segments(segments, &mut operator_stack, &mut output_queue)
                        .map_err(|e| SyntaxError(e, pos..=pos))?;
                } else {
                    return Err(SyntaxError(
                        SyntaxErrorKind::UnknownVariableOrConstant,
                        pos..=pos,
                    ));
                }
            }
            Token::Function(name) => {
                let name = name.as_ref();
                if name.len() > NAME_LIMIT as usize {
                    return Err(SyntaxError(SyntaxErrorKind::NameTooLong, pos..=pos));
                }
                if let Some(func) = NativeFunction::parse(name)
                    .map(|nf| SyOperator::Function(SyFunction::Native(nf), 1))
                    .or_else(|| {
                        custom_function_parser(name).map(|(cf, min, max)| {
                            SyOperator::Function(SyFunction::Custom(cf, min, max), 1)
                        })
                    })
                {
                    operator_stack.push(func);
                } else if let Some(node) = N::parse_constant(name)
                    .or_else(|| custom_constant_parser(name))
                    .map(AstNode::Number)
                    .or_else(|| custom_variable_parser(name).map(|var| AstNode::Variable(var)))
                {
                    output_queue.push(node)?;
                    output_queue
                        .push_opr(SyOperator::BinaryOp(BinaryOp::Mul), &mut operator_stack)?;
                    operator_stack.push(SyOperator::Parentheses);
                } else if let Some(mut segments) = segment_token(
                    name,
                    &custom_constant_parser,
                    &custom_variable_parser,
                    &custom_function_parser,
                ) {
                    if let Segment::Function(func, min, max) = segments.last().unwrap() {
                        let (func, min, max) = (*func, *min, *max);
                        if min > 1 {
                            return Err(SyntaxError(
                                SyntaxErrorKind::NotEnoughArguments,
                                pos..=pos,
                            ));
                        }
                        segments.pop();
                        let implied_mult = matches!(
                            segments.last(),
                            Some(Segment::Variable(_) | Segment::Constant(_))
                        );
                        push_segments(segments, &mut operator_stack, &mut output_queue)
                            .map_err(|e| SyntaxError(e, pos..=pos))?;
                        if implied_mult {
                            output_queue.push_opr(
                                SyOperator::BinaryOp(BinaryOp::Mul),
                                &mut operator_stack,
                            )?;
                        }
                        operator_stack.push(SyOperator::Function(
                            match func {
                                FunctionType::Native(nf) => SyFunction::Native(nf),
                                FunctionType::Custom(cf) => SyFunction::Custom(cf, min, max),
                            },
                            1,
                        ));
                    } else {
                        push_segments(segments, &mut operator_stack, &mut output_queue)
                            .map_err(|e| SyntaxError(e, pos..=pos))?;
                        output_queue
                            .push_opr(SyOperator::BinaryOp(BinaryOp::Mul), &mut operator_stack)?;
                        operator_stack.push(SyOperator::Parentheses);
                    }
                } else {
                    return Err(SyntaxError(SyntaxErrorKind::UnknownFunction, pos..=pos));
                }
            }
            Token::OpenParen => {
                operator_stack.push(SyOperator::Parentheses);
            }
            Token::Comma => {
                output_queue.flush(&mut operator_stack)?;
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
                output_queue.flush(&mut operator_stack)?;
                match operator_stack.pop() {
                    Some(SyOperator::Function(SyFunction::PipeAbs, _)) => {
                        let opening = find_opening_pipe(&tokens[..pos]).unwrap();
                        return Err(SyntaxError(
                            SyntaxErrorKind::PipeAbsNotClosed,
                            opening..=opening,
                        ));
                    }
                    Some(SyOperator::Function(SyFunction::Native(mut nf), args)) => {
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
                            output_queue.push(AstNode::Function(nf.into(), args))?;
                            Ok(())
                        } else if args < nf.min_args() {
                            Err(SyntaxErrorKind::NotEnoughArguments)
                        } else if nf.max_args().is_some_and(|m| args > m) {
                            Err(SyntaxErrorKind::TooManyArguments)
                        } else {
                            output_queue.push(AstNode::Function(nf.into(), args))?;
                            Ok(())
                        }
                        .map_err(|e| {
                            SyntaxError(e, find_opening_paren(&tokens[..pos]).unwrap()..=pos)
                        })?
                    }
                    Some(SyOperator::Function(
                        SyFunction::Custom(cf, min_args, max_args),
                        args,
                    )) => if args < min_args {
                        Err(SyntaxErrorKind::NotEnoughArguments)
                    } else if max_args.is_some_and(|m| args > m) {
                        Err(SyntaxErrorKind::TooManyArguments)
                    } else {
                        output_queue.push(AstNode::Function(FunctionType::Custom(cf), args))?;
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
                    output_queue.flush(&mut operator_stack)?;
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
                    output_queue.push(AstNode::Function(NativeFunction::Abs.into(), 1))?;
                } else {
                    operator_stack.push(SyOperator::Function(SyFunction::PipeAbs, 1));
                }
            }
        }
        was_pow = matches!(token, Token::Operator(OprToken::Power));
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
    output_queue.flush(&mut operator_stack)?;
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
        _ => Ok(output_queue.make()?),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum TestVar {
        X,
        Sigma,
        Var5,
        ShallNotBeNamed,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum TestFunc {
        Func1,
        F2,
        VeryLongFunction,
    }

    #[test]
    fn segment() {
        const C: f64 = 299792458.0;
        let segment = |input: &str| {
            super::segment_token(
                input,
                &|input| match input {
                    "c" => Some(C),
                    "pi2" => Some(FRAC_2_PI),
                    _ => None,
                },
                &|input| match input {
                    "x" => Some(TestVar::X),
                    "σ" => Some(TestVar::Sigma),
                    "var5" => Some(TestVar::Var5),
                    "shallnotbenamed" => Some(TestVar::ShallNotBeNamed),
                    _ => None,
                },
                &|input| match input {
                    "func1" => Some((TestFunc::Func1, 1, None)),
                    "f2" => Some((TestFunc::F2, 1, None)),
                    "verylongfunction" => Some((TestFunc::VeryLongFunction, 1, None)),
                    _ => None,
                },
            )
        };
        assert_eq!(
            segment("xσ"),
            Some(vec![
                Segment::Variable(TestVar::X),
                Segment::Variable(TestVar::Sigma)
            ])
        );
        assert_eq!(
            segment("cx"),
            Some(vec![Segment::Constant(C), Segment::Variable(TestVar::X)])
        );
        assert_eq!(
            segment("pix"),
            Some(vec![Segment::Constant(PI), Segment::Variable(TestVar::X)])
        );
        assert_eq!(
            segment("x2var5"),
            Some(vec![
                Segment::Variable(TestVar::X),
                Segment::Constant(2.0),
                Segment::Variable(TestVar::Var5)
            ])
        );
        assert_eq!(
            segment("x2σshallnotbenamedx"),
            Some(vec![
                Segment::Variable(TestVar::X),
                Segment::Constant(2.0),
                Segment::Variable(TestVar::Sigma),
                Segment::Variable(TestVar::ShallNotBeNamed),
                Segment::Variable(TestVar::X),
            ])
        );
        assert_eq!(
            segment("pi2x"),
            Some(vec![
                Segment::Constant(FRAC_2_PI),
                Segment::Variable(TestVar::X)
            ])
        );
        assert_eq!(
            segment("σf2"),
            Some(vec![
                Segment::Variable(TestVar::Sigma),
                Segment::Function(FunctionType::Custom(TestFunc::F2), 1, None)
            ])
        );
        assert_eq!(
            segment("cmin"),
            Some(vec![
                Segment::Constant(299792458.0),
                Segment::Function(NativeFunction::Min.into(), 2, None)
            ])
        );
        assert_eq!(
            segment("sinsinx"),
            Some(vec![
                Segment::Function(NativeFunction::Sin.into(), 1, Some(1)),
                Segment::Function(NativeFunction::Sin.into(), 1, Some(1)),
                Segment::Variable(TestVar::X),
            ])
        );
        assert_eq!(
            segment("x55sin"),
            Some(vec![
                Segment::Variable(TestVar::X),
                Segment::Constant(55.0),
                Segment::Function(NativeFunction::Sin.into(), 1, Some(1))
            ])
        );
        assert_eq!(
            segment("xsinvar5func1"),
            Some(vec![
                Segment::Variable(TestVar::X),
                Segment::Function(NativeFunction::Sin.into(), 1, Some(1)),
                Segment::Variable(TestVar::Var5),
                Segment::Function(FunctionType::Custom(TestFunc::Func1), 1, None),
            ])
        );
        assert_eq!(
            segment("xxlnxxvar5verylongfunction"),
            Some(vec![
                Segment::Variable(TestVar::X),
                Segment::Variable(TestVar::X),
                Segment::Function(NativeFunction::Ln.into(), 1, Some(1)),
                Segment::Variable(TestVar::X),
                Segment::Variable(TestVar::X),
                Segment::Variable(TestVar::Var5),
                Segment::Function(FunctionType::Custom(TestFunc::VeryLongFunction), 1, None),
            ])
        );
    }
}
