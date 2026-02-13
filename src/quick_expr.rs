use std::{fmt::Debug, marker::PhantomData, slice::Iter};

use crate::{
    BinaryOp, FunctionIdentifier, FunctionPointer, UnaryOp, VariableIdentifier, VariableStore,
    number::{NativeFunction, Number},
    syntax::{AstNode, FunctionType, MathAst},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Source {
    Literal,
    Variable,
    Stack(u8),
}

impl Source {
    #[inline]
    fn resolve<'a, 'b, N: Number, V: VariableIdentifier>(
        self,
        literals: &mut Iter<'b, N>,
        variables: &mut Iter<'b, V>,
        variable_values: &'a impl VariableStore<N, V>,
        stack: &'a [N],
        removed: &mut u8,
    ) -> Result<N::AsArg<'a>, OptExprEvalError>
    where
        'b: 'a,
    {
        match self {
            Source::Literal => literals
                .next()
                .map(N::asarg)
                .ok_or(OptExprEvalError::NotEnoughLiterals),
            Source::Variable => variables
                .next()
                .map(|var| variable_values.get(*var))
                .ok_or(OptExprEvalError::NotEnoughVariables),
            Source::Stack(offset) => {
                *removed += 1;
                stack
                    .get(stack.len() - 1 - offset as usize)
                    .map(N::asarg)
                    .ok_or(OptExprEvalError::NotEnoughArgumentsInStack)
            }
        }
    }

    #[inline]
    fn resolve_owned<N: Number, V: VariableIdentifier>(
        self,
        literals: &mut Iter<'_, N>,
        variables: &mut Iter<'_, V>,
        variable_values: &'_ impl VariableStore<N, V>,
        stack: &[N],
        removed: &mut u8,
    ) -> Result<N, OptExprEvalError> {
        match self {
            Source::Literal => literals
                .next()
                .map(|num| num.to_owned())
                .ok_or(OptExprEvalError::NotEnoughLiterals),
            Source::Variable => variables
                .next()
                .map(|var| variable_values.get(*var).to_owned())
                .ok_or(OptExprEvalError::NotEnoughVariables),
            Source::Stack(offset) => {
                *removed += 1;
                stack
                    .get(stack.len() - 1 - offset as usize)
                    .map(|num| num.to_owned())
                    .ok_or(OptExprEvalError::NotEnoughArgumentsInStack)
            }
        }
    }

    pub fn is_stack(self) -> bool {
        matches!(self, Self::Stack(_))
    }
}

#[derive(Clone)]
pub enum CtxFuncPtr<'a, N>
where
    N: Number,
{
    Single(for<'b> fn(N::AsArg<'b>) -> N),
    Dual(for<'b, 'c> fn(N::AsArg<'b>, N::AsArg<'c>) -> N),
    Triple(for<'b, 'c, 'd> fn(N::AsArg<'b>, N::AsArg<'c>, N::AsArg<'d>) -> N),
    Flexible(fn(&[N]) -> N, u8),
    DynSingle(&'a dyn for<'b> Fn(N::AsArg<'b>) -> N),
    DynDual(&'a dyn for<'b> Fn(N::AsArg<'b>, N::AsArg<'b>) -> N),
    DynTriple(&'a dyn for<'b> Fn(N::AsArg<'b>, N::AsArg<'b>, N::AsArg<'b>) -> N),
    DynFlexible(&'a dyn Fn(&[N]) -> N, u8),
}

impl<'a, N: Number> CtxFuncPtr<'a, N> {
    fn from_ptr_args(func: FunctionPointer<'a, N>, argc: u8) -> Self {
        match func {
            FunctionPointer::Single(f) => Self::Single(f),
            FunctionPointer::Dual(f) => Self::Dual(f),
            FunctionPointer::Triple(f) => Self::Triple(f),
            FunctionPointer::Flexible(f) => Self::Flexible(f, argc),
            FunctionPointer::DynSingle(f) => Self::DynSingle(f),
            FunctionPointer::DynDual(f) => Self::DynDual(f),
            FunctionPointer::DynTriple(f) => Self::DynTriple(f),
            FunctionPointer::DynFlexible(f) => Self::DynFlexible(f, argc),
        }
    }
}

impl<'a, N> Copy for CtxFuncPtr<'a, N> where N: Number {}

impl<N: Number> Debug for CtxFuncPtr<'_, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Single(_) => f.write_str("Single"),
            Self::Dual(_) => f.write_str("Dual"),
            Self::Triple(_) => f.write_str("Triple"),
            Self::Flexible(_, a) => f.debug_tuple("Flexible").field(a).finish(),
            Self::DynSingle(_) => f.write_str("DynSingle"),
            Self::DynDual(_) => f.write_str("DynDual"),
            Self::DynTriple(_) => f.write_str("DynTriple"),
            Self::DynFlexible(_, a) => f.debug_tuple("DynFlexible").field(a).finish(),
        }
    }
}

impl<N: Number> From<BinaryOp> for CtxFuncPtr<'static, N> {
    fn from(value: BinaryOp) -> Self {
        Self::Dual(value.as_pointer())
    }
}

impl<N: Number> From<UnaryOp> for CtxFuncPtr<'static, N> {
    fn from(value: UnaryOp) -> Self {
        Self::Single(value.as_pointer())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FunctionSource<F: FunctionIdentifier> {
    BinaryOp(BinaryOp),
    UnaryOp(UnaryOp),
    NativeFunction(NativeFunction),
    CustomFunction(F),
}

#[derive(Clone)]
pub struct MarkedFunc<'a, N: Number, F: FunctionIdentifier> {
    pub func: CtxFuncPtr<'a, N>,
    pub _src: PhantomData<F>,
    #[cfg(debug_assertions)]
    pub src: FunctionSource<F>,
}

impl<'a, N: Number, F: FunctionIdentifier> MarkedFunc<'a, N, F> {
    pub fn new(func: CtxFuncPtr<'a, N>, src: FunctionSource<F>) -> Self {
        Self {
            func,
            _src: PhantomData,
            #[cfg(debug_assertions)]
            src,
        }
    }
}

#[cfg(debug_assertions)]
impl<'a, N: Number, F: FunctionIdentifier> PartialEq for MarkedFunc<'a, N, F> {
    fn eq(&self, other: &Self) -> bool {
        self.src == other.src
    }
}

#[cfg(debug_assertions)]
impl<'a, N: Number, F: FunctionIdentifier> Eq for MarkedFunc<'a, N, F> {}

impl<'a, N, F> std::fmt::Debug for MarkedFunc<'a, N, F>
where
    N: Number + std::fmt::Debug,
    F: FunctionIdentifier + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut dbg = f.debug_tuple("MarkedFunc");
        #[cfg(debug_assertions)]
        dbg.field(&self.src);
        #[cfg(not(debug_assertions))]
        dbg.field(&self.func);
        dbg.finish()
    }
}

impl<N: Number, F: FunctionIdentifier> From<BinaryOp> for MarkedFunc<'static, N, F> {
    fn from(value: BinaryOp) -> Self {
        Self::new(value.into(), FunctionSource::BinaryOp(value))
    }
}

impl<N: Number, F: FunctionIdentifier> From<UnaryOp> for MarkedFunc<'static, N, F> {
    fn from(value: UnaryOp) -> Self {
        Self::new(value.into(), FunctionSource::UnaryOp(value))
    }
}

#[derive(Clone, Debug)]
pub enum Instr<'a, N: Number, F: FunctionIdentifier> {
    Push(Source),
    Calculate(MarkedFunc<'a, N, F>),
}

#[cfg(debug_assertions)]
impl<'a, N: Number, F: FunctionIdentifier> PartialEq for Instr<'a, N, F> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Push(l0), Self::Push(r0)) => l0 == r0,
            (Self::Calculate(l0), Self::Calculate(r0)) => l0 == r0,
            _ => false,
        }
    }
}

#[cfg(debug_assertions)]
impl<'a, N: Number, F: FunctionIdentifier> Eq for Instr<'a, N, F> {}

#[derive(Clone, Debug, PartialEq, Eq)]
enum InstrArg<N: Number, V: VariableIdentifier> {
    Literal(N),
    Variable(V),
    Stack,
}

#[derive(Debug, Clone, Copy)]
pub enum OptExprEvalError {
    NotEnoughArgumentsInStack,
    EmptyExpr,
    NotEnoughLiterals,
    NotEnoughVariables,
    NotEnoughParams,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StackCapCalcError {
    StackUnderflow,
    NotEnoughParameters,
}

#[derive(Clone, Debug)]
pub struct QuickExpr<'a, N: Number, V: VariableIdentifier, F: FunctionIdentifier> {
    pub param_sources: Vec<Source>,
    pub literals: Vec<N>,
    pub variables: Vec<V>,
    pub instructions: Vec<Instr<'a, N, F>>,
}

impl<'a, N, V, F> QuickExpr<'a, N, V, F>
where
    N: Number,
    V: VariableIdentifier,
    F: FunctionIdentifier,
{
    pub fn new(
        ast: MathAst<N, V, F>,
        function_to_pointer: impl Fn(F) -> FunctionPointer<'a, N>,
    ) -> Self {
        let tree = ast.as_tree();
        let is_flex = |n: usize| match tree[n] {
            AstNode::Function(FunctionType::Native(nf), _) => !nf.is_fixed(),
            AstNode::Function(FunctionType::Custom(cf), _) => matches!(
                function_to_pointer(cf),
                FunctionPointer::Flexible(_) | FunctionPointer::DynFlexible(_)
            ),
            _ => false,
        };
        let mut must_push: Vec<bool> = Vec::with_capacity(
            tree.postorder_iter()
                .map(|n| matches!(n, AstNode::Number(_) | AstNode::Variable(_)) as usize)
                .sum::<usize>(),
        );
        for (idx, node) in tree.postorder_iter().enumerate() {
            if matches!(node, AstNode::Number(_) | AstNode::Variable(_)) {
                must_push.push(tree.parent(idx).is_none_or(is_flex));
            }
        }
        let mut param_sources = Vec::new();
        let mut literals = Vec::new();
        let mut variables = Vec::new();
        let mut instructions = Vec::new();
        let mut args: Vec<InstrArg<N, V>> = Vec::new();
        let mut must_push = must_push.into_iter();
        for node in ast.into_tree().into_postorder_iter() {
            let arg_stack = !matches!(node, AstNode::Number(_) | AstNode::Variable(_));
            let mut arg_cons = 0;
            match node {
                AstNode::Number(num) => {
                    if must_push.next().unwrap() {
                        literals.push(num);
                        instructions.push(Instr::Push(Source::Literal))
                    } else {
                        args.push(InstrArg::Literal(num));
                    }
                }
                AstNode::Variable(var) => {
                    if must_push.next().unwrap() {
                        variables.push(var);
                        instructions.push(Instr::Push(Source::Variable));
                    } else {
                        args.push(InstrArg::Variable(var));
                    }
                }
                AstNode::BinaryOp(opr) => {
                    arg_cons = 2;
                    instructions.push(Instr::Calculate(MarkedFunc::new(
                        CtxFuncPtr::from_ptr_args(opr.into(), 2),
                        FunctionSource::BinaryOp(opr),
                    )));
                }
                AstNode::UnaryOp(opr) => {
                    arg_cons = 1;
                    instructions.push(Instr::Calculate(MarkedFunc::new(
                        CtxFuncPtr::from_ptr_args(opr.into(), 1),
                        FunctionSource::UnaryOp(opr),
                    )));
                }
                AstNode::Function(FunctionType::Native(nf), argc) => {
                    if nf.is_fixed() {
                        arg_cons = argc;
                    }
                    instructions.push(Instr::Calculate(MarkedFunc::new(
                        CtxFuncPtr::from_ptr_args(nf.into(), argc),
                        FunctionSource::NativeFunction(nf),
                    )));
                }
                AstNode::Function(FunctionType::Custom(cf), argc) => {
                    let ptr = function_to_pointer(cf);
                    if ptr.is_fixed() {
                        arg_cons = argc;
                    }
                    instructions.push(Instr::Calculate(MarkedFunc::new(
                        CtxFuncPtr::from_ptr_args(ptr, argc),
                        FunctionSource::CustomFunction(cf),
                    )));
                }
            }
            if arg_cons > 0 {
                let range = args.len() - arg_cons as usize..;
                let mut stack_offset = args[range.clone()]
                    .iter()
                    .filter(|a| matches!(a, InstrArg::Stack))
                    .count() as u8;
                for arg in args.drain(range) {
                    match arg {
                        InstrArg::Literal(num) => {
                            param_sources.push(Source::Literal);
                            literals.push(num);
                        }
                        InstrArg::Variable(var) => {
                            param_sources.push(Source::Variable);
                            variables.push(var);
                        }
                        InstrArg::Stack => {
                            stack_offset -= 1;
                            param_sources.push(Source::Stack(stack_offset));
                        }
                    }
                }
            }
            if arg_stack {
                args.push(InstrArg::Stack);
            }
        }
        QuickExpr {
            param_sources,
            literals,
            variables,
            instructions,
        }
    }

    pub fn stack_req_capacity(&self) -> Result<usize, StackCapCalcError> {
        let mut p: usize = 0;
        let mut length: usize = 0;
        let mut capacity: usize = 0;
        for instr in &self.instructions {
            macro_rules! asc {
                ($argc: expr) => {{
                    if p + $argc > self.param_sources.len() {
                        return Err(StackCapCalcError::NotEnoughParameters);
                    }
                    length = length
                        .checked_sub(
                            self.param_sources[p..p + $argc]
                                .iter()
                                .filter(|s| s.is_stack())
                                .count(),
                        )
                        .ok_or(StackCapCalcError::StackUnderflow)?;
                    length += 1;
                    p += $argc;
                }};
            }
            match instr {
                Instr::Push(_) => length += 1,
                Instr::Calculate(func) => match func.func {
                    CtxFuncPtr::DynSingle(_) | CtxFuncPtr::Single(_) => asc!(1),
                    CtxFuncPtr::DynDual(_) | CtxFuncPtr::Dual(_) => asc!(2),
                    CtxFuncPtr::DynTriple(_) | CtxFuncPtr::Triple(_) => asc!(3),
                    CtxFuncPtr::DynFlexible(_, argc) | CtxFuncPtr::Flexible(_, argc) => {
                        length = length
                            .checked_sub(argc as usize)
                            .ok_or(StackCapCalcError::StackUnderflow)?;
                        length += 1;
                    }
                },
            }
            if length > capacity {
                capacity = length;
            }
        }
        Ok(capacity)
    }

    pub fn eval(
        &self,
        variable_values: impl VariableStore<N, V>,
        stack: &mut Vec<N>,
    ) -> Result<N, OptExprEvalError> {
        let mut param_sources = self.param_sources.iter().copied();
        let mut literals = self.literals.iter();
        let mut variables = self.variables.iter();
        for instr in &self.instructions {
            let mut removed = 0;
            macro_rules! arg {
                () => {
                    param_sources
                        .next()
                        .ok_or(OptExprEvalError::NotEnoughParams)?
                        .resolve(
                            &mut literals,
                            &mut variables,
                            &variable_values,
                            &stack,
                            &mut removed,
                        )?
                };
            }
            let result = match instr {
                Instr::Push(src) => src.resolve_owned(
                    &mut literals,
                    &mut variables,
                    &variable_values,
                    stack,
                    &mut removed,
                )?,
                Instr::Calculate(idfunc) => match idfunc.func {
                    CtxFuncPtr::Single(func) => func(arg!()),
                    CtxFuncPtr::Dual(func) => func(arg!(), arg!()),
                    CtxFuncPtr::Triple(func) => func(arg!(), arg!(), arg!()),
                    CtxFuncPtr::Flexible(func, argc) => {
                        // FIX: account for zero argc
                        removed += argc;
                        func(&stack[stack.len() - argc as usize..])
                    }
                    CtxFuncPtr::DynSingle(func) => func(arg!()),
                    CtxFuncPtr::DynDual(func) => func(arg!(), arg!()),
                    CtxFuncPtr::DynTriple(func) => func(arg!(), arg!(), arg!()),
                    CtxFuncPtr::DynFlexible(func, argc) => {
                        removed += argc;
                        func(&stack[stack.len() - argc as usize..])
                    }
                },
            };
            stack.truncate(stack.len() - removed as usize);
            stack.push(result);
        }
        stack.pop().ok_or(OptExprEvalError::EmptyExpr)
    }
}

#[cfg(debug_assertions)]
impl<'a, N, V, F> PartialEq for QuickExpr<'a, N, V, F>
where
    N: Number,
    V: VariableIdentifier,
    F: FunctionIdentifier,
{
    fn eq(&self, other: &Self) -> bool {
        self.param_sources == other.param_sources
            && self.literals == other.literals
            && self.variables == other.variables
            && self.instructions == other.instructions
    }
}

#[cfg(debug_assertions)]
impl<'a, N, V, F> Eq for QuickExpr<'a, N, V, F>
where
    N: Number,
    V: VariableIdentifier,
    F: FunctionIdentifier,
{
}

#[cfg(all(debug_assertions, test))]
mod tests {
    use crate::tokenizer::TokenStream;

    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum TestVar {
        X,
        Y,
        T,
    }

    impl TestVar {
        fn parse(input: &str) -> Option<Self> {
            match input {
                "x" => Some(TestVar::X),
                "y" => Some(TestVar::Y),
                "t" => Some(TestVar::T),
                _ => None,
            }
        }
    }

    struct TestStore;

    impl VariableStore<f64, TestVar> for TestStore {
        fn get<'a>(&'a self, var: TestVar) -> f64 {
            match var {
                TestVar::X => 2.0,
                TestVar::Y => 10.0,
                TestVar::T => 0.25,
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum TestFunc {
        Sigmoid,
        Hypot,
        F1,
        Digits,
    }

    impl TestFunc {
        fn parse(input: &str) -> Option<(Self, u8, Option<u8>)> {
            match input {
                "sigmoid" => Some((TestFunc::Sigmoid, 1, Some(1))),
                "hypot" => Some((TestFunc::Hypot, 2, Some(2))),
                "func1" => Some((TestFunc::F1, 3, Some(3))),
                "digits" => Some((TestFunc::Digits, 1, None)),
                _ => None,
            }
        }
        fn as_pointer(self) -> FunctionPointer<'static, f64> {
            match self {
                TestFunc::Sigmoid => FunctionPointer::<f64>::Single(sigmoid),
                TestFunc::Hypot => FunctionPointer::<f64>::Dual(hypot),
                TestFunc::F1 => FunctionPointer::<f64>::Triple(func1),
                TestFunc::Digits => FunctionPointer::Flexible(digits),
            }
        }
        fn as_marked(self, argc: u8) -> MarkedFunc<'static, f64, Self> {
            MarkedFunc::new(
                CtxFuncPtr::from_ptr_args(self.as_pointer(), argc),
                FunctionSource::CustomFunction(self),
            )
        }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn hypot(x: f64, y: f64) -> f64 {
        (x * x + y * y).sqrt()
    }

    fn func1(x: f64, y: f64, z: f64) -> f64 {
        x * x + 2.0 * y + 3.0 * z
    }

    fn digits(nums: &[f64]) -> f64 {
        nums.iter()
            .enumerate()
            .map(|(i, &v)| 10f64.powi(i as i32) * v)
            .sum()
    }

    #[test]
    fn convert() {
        fn convert(input: &str) -> QuickExpr<'_, f64, TestVar, TestFunc> {
            let tokens = TokenStream::new(input)
                .map_err(|e| e.to_general())
                .unwrap()
                .0;
            QuickExpr::new(
                MathAst::new(&tokens, |_| None, TestFunc::parse, TestVar::parse).unwrap(),
                TestFunc::as_pointer,
            )
        }
        assert_eq!(
            convert("2"),
            QuickExpr {
                param_sources: vec![],
                literals: vec![2.0],
                variables: vec![],
                instructions: vec![Instr::Push(Source::Literal)],
            },
        );
        assert_eq!(
            convert("1+2"),
            QuickExpr {
                param_sources: vec![Source::Literal, Source::Literal],
                literals: vec![1.0, 2.0],
                variables: vec![],
                instructions: vec![Instr::Calculate(BinaryOp::Add.into())],
            },
        );
        assert_eq!(
            convert("sin(x)"),
            QuickExpr {
                param_sources: vec![Source::Variable],
                literals: vec![],
                variables: vec![TestVar::X],
                instructions: vec![Instr::Calculate(NativeFunction::Sin.as_markedfunc(0))],
            },
        );
        assert_eq!(
            convert("sin(x)+1"),
            QuickExpr {
                param_sources: vec![Source::Variable, Source::Stack(0), Source::Literal],
                literals: vec![1.0],
                variables: vec![TestVar::X],
                instructions: vec![
                    Instr::Calculate(NativeFunction::Sin.as_markedfunc(0)),
                    Instr::Calculate(BinaryOp::Add.into())
                ],
            },
        );
        assert_eq!(
            convert("ysin(x)+1"),
            QuickExpr {
                param_sources: vec![
                    Source::Variable,
                    Source::Variable,
                    Source::Stack(0),
                    Source::Stack(0),
                    Source::Literal,
                ],
                literals: vec![1.0],
                variables: vec![TestVar::X, TestVar::Y],
                instructions: vec![
                    Instr::Calculate(NativeFunction::Sin.as_markedfunc(0)),
                    Instr::Calculate(BinaryOp::Mul.into()),
                    Instr::Calculate(BinaryOp::Add.into())
                ],
            },
        );
        assert_eq!(
            convert("func1(y,5x,1)"),
            QuickExpr {
                param_sources: vec![
                    Source::Literal,
                    Source::Variable,
                    Source::Variable,
                    Source::Stack(0),
                    Source::Literal,
                ],
                literals: vec![5.0, 1.0],
                variables: vec![TestVar::X, TestVar::Y],
                instructions: vec![
                    Instr::Calculate(BinaryOp::Mul.into()),
                    Instr::Calculate(TestFunc::F1.as_marked(0))
                ],
            },
        );
        assert_eq!(
            convert("max(y,5x,1)"),
            QuickExpr {
                param_sources: vec![Source::Literal, Source::Variable],
                literals: vec![5.0, 1.0],
                variables: vec![TestVar::Y, TestVar::X],
                instructions: vec![
                    Instr::Push(Source::Variable),
                    Instr::Calculate(BinaryOp::Mul.into()),
                    Instr::Push(Source::Literal),
                    Instr::Calculate(NativeFunction::Max.as_markedfunc(3)),
                ]
            }
        );
        assert_eq!(
            convert("x^2 + sin(y)"),
            QuickExpr {
                param_sources: vec![
                    Source::Variable,
                    Source::Literal,
                    Source::Variable,
                    Source::Stack(1),
                    Source::Stack(0)
                ],
                literals: vec![2.0],
                variables: vec![TestVar::X, TestVar::Y],
                instructions: vec![
                    Instr::Calculate(BinaryOp::Pow.into()),
                    Instr::Calculate(NativeFunction::Sin.as_markedfunc(0)),
                    Instr::Calculate(BinaryOp::Add.into())
                ]
            }
        );
        assert_eq!(
            convert("min(5, t, 3) + 2"),
            QuickExpr {
                param_sources: vec![Source::Stack(0), Source::Literal,],
                literals: vec![5.0, 3.0, 2.0],
                variables: vec![TestVar::T],
                instructions: vec![
                    Instr::Push(Source::Literal),
                    Instr::Push(Source::Variable),
                    Instr::Push(Source::Literal),
                    Instr::Calculate(NativeFunction::Min.as_markedfunc(3)),
                    Instr::Calculate(BinaryOp::Add.into()),
                ]
            }
        )
    }

    #[test]
    fn stack_req_capacity() {
        assert_eq!(
            QuickExpr::<f64, TestVar, TestFunc> {
                param_sources: vec![],
                literals: vec![2.0],
                variables: vec![],
                instructions: vec![Instr::Push(Source::Literal)],
            }
            .stack_req_capacity(),
            Ok(1)
        );
        assert_eq!(
            QuickExpr::<f64, TestVar, TestFunc> {
                param_sources: vec![Source::Variable, Source::Literal],
                literals: vec![2.0],
                variables: vec![TestVar::X],
                instructions: vec![Instr::Calculate(BinaryOp::Add.into())],
            }
            .stack_req_capacity(),
            Ok(1)
        );
        assert_eq!(
            QuickExpr::<f64, TestVar, TestFunc> {
                param_sources: vec![Source::Stack(1), Source::Stack(0)],
                literals: vec![],
                variables: vec![],
                instructions: vec![Instr::Calculate(BinaryOp::Add.into())]
            }
            .stack_req_capacity(),
            Err(StackCapCalcError::StackUnderflow)
        );
        assert_eq!(
            QuickExpr::<f64, TestVar, TestFunc> {
                param_sources: vec![Source::Literal],
                literals: vec![5.8],
                variables: vec![],
                instructions: vec![Instr::Calculate(BinaryOp::Add.into())]
            }
            .stack_req_capacity(),
            Err(StackCapCalcError::NotEnoughParameters)
        );
    }

    #[test]
    fn eval() {
        fn evaluate<'a>(
            param_sources: Vec<Source>,
            literals: Vec<f64>,
            variables: Vec<TestVar>,
            instructions: Vec<Instr<'a, f64, TestFunc>>,
        ) -> f64 {
            QuickExpr {
                param_sources,
                literals,
                variables,
                instructions,
            }
            .eval(TestStore, &mut Vec::new())
            .unwrap()
        }

        assert_eq!(
            evaluate(
                vec![],
                vec![1.0],
                vec![],
                vec![Instr::Push(Source::Literal)]
            ),
            1.0
        );
        assert_eq!(
            evaluate(
                vec![Source::Literal],
                vec![-3.0],
                vec![],
                vec![Instr::Calculate(UnaryOp::Neg.into())]
            ),
            3.0
        );
        assert_eq!(
            evaluate(
                vec![Source::Literal, Source::Variable],
                vec![5.0],
                vec![TestVar::X],
                vec![Instr::Calculate(BinaryOp::Add.into())]
            ),
            7.0
        );
        assert_eq!(
            evaluate(
                vec![Source::Literal, Source::Variable],
                vec![5.0],
                vec![TestVar::X],
                vec![Instr::Calculate(BinaryOp::Add.into())]
            ),
            7.0
        );
        assert_eq!(
            evaluate(
                vec![Source::Literal, Source::Stack(0), Source::Variable],
                vec![12.0, 5.0],
                vec![TestVar::X],
                vec![
                    Instr::Push(Source::Literal),
                    Instr::Calculate(TestFunc::F1.as_marked(1))
                ]
            ),
            55.0
        );
        assert_eq!(
            evaluate(
                vec![Source::Stack(2), Source::Stack(1), Source::Stack(0)],
                vec![8.0, 4.0, 2.0],
                vec![],
                vec![
                    Instr::Push(Source::Literal),
                    Instr::Push(Source::Literal),
                    Instr::Push(Source::Literal),
                    Instr::Calculate(TestFunc::Digits.as_marked(3))
                ]
            ),
            248.0
        );
    }
}
