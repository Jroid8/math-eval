use std::{fmt::Debug, marker::PhantomData};

use super::{AstNode, FunctionType, MathAst, SyntaxError, SyntaxErrorKind};
use crate::{
    BinaryOp, FunctionIdentifier, FunctionPointer, UnaryOp, VariableIdentifier, VariableStore,
    number::{NFPointer, NativeFuncsNameTrie, NativeFunction, Number},
    postfix_tree::subtree_collection::{MultipleRoots, NotEnoughOrphans, SubtreeCollection},
    syntax::{
        grammar::{ResOprToken, ResToken, ResolvedTkStream},
        token_fragmentation::{FragKind, NAME_LIMIT, ParsedFragment, fragment_token},
    },
    trie::NameTrie,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum SyFunction<F>
where
    F: FunctionIdentifier,
{
    Native(NativeFunction),
    Custom(F, u8, Option<u8>),
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
            SyOperator::Parentheses => unreachable!(),
        }
    }
}

impl<F> From<ResOprToken> for SyOperator<F>
where
    F: FunctionIdentifier,
{
    fn from(value: ResOprToken) -> Self {
        match value {
            ResOprToken::Add => Self::BinaryOp(BinaryOp::Add),
            ResOprToken::Subtract => Self::BinaryOp(BinaryOp::Sub),
            ResOprToken::Multiply => Self::BinaryOp(BinaryOp::Mul),
            ResOprToken::Divide => Self::BinaryOp(BinaryOp::Div),
            ResOprToken::Power => Self::BinaryOp(BinaryOp::Pow),
            ResOprToken::Modulo => Self::BinaryOp(BinaryOp::Mod),
            ResOprToken::Factorial => Self::UnaryOp(UnaryOp::Fac),
            ResOprToken::DoubleFactorial => Self::UnaryOp(UnaryOp::DoubleFac),
            ResOprToken::Negative => Self::UnaryOp(UnaryOp::Neg),
            ResOprToken::Positive => unreachable!(),
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
    fn build(self) -> Result<Self::Output, MultipleRoots>;
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
    fn build(self) -> Result<Self::Output, MultipleRoots> {
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
    fn build(mut self) -> Result<Self::Output, MultipleRoots> {
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

fn find_opening<S: AsRef<str>>(stream: &ResolvedTkStream<'_, S>, target: usize) -> Option<usize> {
    let mut nesting = 1;
    for (i, tk) in stream.iter().take(target).enumerate().rev() {
        match tk {
            ResToken::CloseDelim | ResToken::ClosePipe => nesting += 1,
            ResToken::OpenDelim | ResToken::OpenPipe | ResToken::Function(_) => {
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

fn push_fragments<N, V, F, O>(
    fragments: &mut Vec<ParsedFragment<'_, N, V, F>>,
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
    for frag in fragments.drain(..) {
        let frag = frag.kind;
        let is_func = matches!(frag, FragKind::Function(_, _, _));
        if !was_func {
            output_queue.push_opr(SyOperator::BinaryOp(BinaryOp::Mul), operator_stack)?;
        }
        match frag {
            FragKind::Literal(num) => output_queue.push(AstNode::Number(num))?,
            FragKind::Constant(num) => output_queue.push(AstNode::Number(num.to_owned()))?,
            FragKind::Variable(var) => output_queue.push(AstNode::Variable(var))?,
            FragKind::Function(func, min, _) => {
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

pub(super) fn parse_or_eval<'a, O, N, V, F, S>(
    mut output_queue: O,
    stream: ResolvedTkStream<'_, S>,
    custom_constants: &impl NameTrie<&'a N>,
    custom_functions: &impl NameTrie<(F, u8, Option<u8>)>,
    custom_variables: &impl NameTrie<V>,
) -> Result<O::Output, SyntaxError>
where
    O: ShuntingYardOutput<N, V, F>,
    N: Number,
    V: VariableIdentifier,
    F: FunctionIdentifier,
    S: AsRef<str>,
{
    // Dijkstra's shunting yard algorithm
    let mut was_pow = false;
    let mut operator_stack: Vec<SyOperator<F>> = Vec::new();
    let mut fragments = Vec::with_capacity(0);
    for (pos, token) in stream.iter().enumerate() {
        if stream.has_implied_mult(pos) {
            output_queue.push_opr(SyOperator::BinaryOp(BinaryOp::Mul), &mut operator_stack)?;
        }
        match token {
            ResToken::Number(num) => output_queue.push(
                num.parse::<N>()
                    .map(AstNode::Number)
                    .map_err(|_| SyntaxError(SyntaxErrorKind::NumberParsingError, pos..=pos))?,
            )?,
            ResToken::Operator(opr) => {
                if was_pow && opr == ResOprToken::Negative {
                    output_queue.push_opr(SyOperator::HpNeg, &mut operator_stack)?;
                } else if opr != ResOprToken::Positive {
                    output_queue.push_opr(opr.into(), &mut operator_stack)?;
                }
            }
            ResToken::Variable(name) => {
                if name.chars().count() > NAME_LIMIT as usize {
                    return Err(SyntaxError(SyntaxErrorKind::NameTooLong, pos..=pos));
                }
                if let Some(node) = N::CONSTS_NAME_TRIE
                    .exact_match(name)
                    .or_else(|| custom_constants.exact_match(name))
                    .map(|c| c.to_owned())
                    .map(AstNode::Number)
                    .or_else(|| {
                        custom_variables
                            .exact_match(name)
                            .map(|var| AstNode::Variable(var))
                    })
                {
                    output_queue.push(node)?;
                } else if let Some((func, min)) = NativeFuncsNameTrie
                    .exact_match(name)
                    .map(|nf| {
                        (
                            SyOperator::FuncNoParen(FunctionType::Native(nf)),
                            nf.min_args(),
                        )
                    })
                    .or_else(|| {
                        custom_functions.exact_match(name).map(|(cf, min, _)| {
                            (SyOperator::FuncNoParen(FunctionType::Custom(cf)), min)
                        })
                    })
                {
                    if min > 1 {
                        return Err(SyntaxError(SyntaxErrorKind::NotEnoughArguments, pos..=pos));
                    }
                    output_queue.push_opr(func, &mut operator_stack)?;
                } else if fragment_token(
                    name,
                    &mut fragments,
                    custom_constants,
                    custom_variables,
                    custom_functions,
                ) {
                    push_fragments(&mut fragments, &mut operator_stack, &mut output_queue)
                        .map_err(|e| SyntaxError(e, pos..=pos))?;
                } else {
                    return Err(SyntaxError(
                        SyntaxErrorKind::UnknownVariableOrConstant,
                        pos..=pos,
                    ));
                }
            }
            ResToken::Function(name) => {
                if name.chars().count() > NAME_LIMIT as usize {
                    return Err(SyntaxError(SyntaxErrorKind::NameTooLong, pos..=pos));
                }
                if let Some(func) = NativeFuncsNameTrie
                    .exact_match(name)
                    .map(|nf| SyOperator::Function(SyFunction::Native(nf), 1))
                    .or_else(|| {
                        custom_functions.exact_match(name).map(|(cf, min, max)| {
                            SyOperator::Function(SyFunction::Custom(cf, min, max), 1)
                        })
                    })
                {
                    operator_stack.push(func);
                } else if let Some(node) = N::CONSTS_NAME_TRIE
                    .exact_match(name)
                    .or_else(|| custom_constants.exact_match(name))
                    .map(|c| c.to_owned())
                    .map(AstNode::Number)
                    .or_else(|| custom_variables.exact_match(name).map(AstNode::Variable))
                {
                    output_queue.push(node)?;
                    output_queue
                        .push_opr(SyOperator::BinaryOp(BinaryOp::Mul), &mut operator_stack)?;
                    operator_stack.push(SyOperator::Parentheses);
                } else if fragment_token(
                    name,
                    &mut fragments,
                    custom_constants,
                    custom_variables,
                    custom_functions,
                ) {
                    if let FragKind::Function(func, min, max) = fragments.last().unwrap().kind {
                        let (func, min, max) = (func, min, max);
                        if min > 1 {
                            return Err(SyntaxError(
                                SyntaxErrorKind::NotEnoughArguments,
                                pos..=pos,
                            ));
                        }
                        fragments.pop();
                        let implied_mult = matches!(
                            fragments.last().map(|f| &f.kind),
                            Some(FragKind::Variable(_) | FragKind::Constant(_))
                        );
                        push_fragments(&mut fragments, &mut operator_stack, &mut output_queue)
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
                        push_fragments(&mut fragments, &mut operator_stack, &mut output_queue)
                            .map_err(|e| SyntaxError(e, pos..=pos))?;
                        output_queue
                            .push_opr(SyOperator::BinaryOp(BinaryOp::Mul), &mut operator_stack)?;
                        operator_stack.push(SyOperator::Parentheses);
                    }
                } else {
                    return Err(SyntaxError(SyntaxErrorKind::UnknownFunction, pos..=pos));
                }
            }
            ResToken::OpenDelim => {
                operator_stack.push(SyOperator::Parentheses);
            }
            ResToken::Comma => {
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
            ResToken::CloseDelim => {
                output_queue.flush(&mut operator_stack)?;
                match operator_stack.pop().unwrap() {
                    SyOperator::Function(SyFunction::Native(mut nf), args) => {
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
                        .map_err(|e| SyntaxError(e, find_opening(&stream, pos).unwrap()..=pos))?
                    }
                    SyOperator::Function(SyFunction::Custom(cf, min_args, max_args), args) => {
                        if args < min_args {
                            Err(SyntaxErrorKind::NotEnoughArguments)
                        } else if max_args.is_some_and(|m| args > m) {
                            Err(SyntaxErrorKind::TooManyArguments)
                        } else {
                            output_queue.push(AstNode::Function(FunctionType::Custom(cf), args))?;
                            Ok(())
                        }
                        .map_err(|e| SyntaxError(e, find_opening(&stream, pos).unwrap()..=pos))?
                    }
                    SyOperator::Parentheses => (),
                    _ => unreachable!(),
                }
            }
            ResToken::OpenPipe => {
                operator_stack.push(SyOperator::Function(
                    SyFunction::Native(NativeFunction::Abs),
                    1,
                ));
            }
            ResToken::ClosePipe => {
                output_queue.flush(&mut operator_stack)?;
                if let Some(SyOperator::Function(SyFunction::Native(NativeFunction::Abs), args)) =
                    operator_stack.pop()
                {
                    if args == 1 {
                        output_queue.push(AstNode::Function(NativeFunction::Abs.into(), 1))?;
                    } else {
                        return Err(SyntaxError(
                            SyntaxErrorKind::TooManyArguments,
                            find_opening(&stream, pos).unwrap()..=pos,
                        ));
                    }
                } else {
                    // should be caught in syntax validation pass
                    unreachable!()
                }
            }
        }
        was_pow = matches!(token, ResToken::Operator(ResOprToken::Power));
    }
    output_queue.flush(&mut operator_stack)?;
    Ok(output_queue.build()?)
}
