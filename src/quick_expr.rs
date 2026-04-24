use std::{fmt::Debug, marker::PhantomData, num::NonZeroU8, slice::Iter};

use crate::{
    BinaryOp, FunctionIdentifier as FuncId, FunctionPointer, UnaryOp, VariableIdentifier as VarId,
    VariableStore,
    number::{BuiltinFunction, Number},
    nz,
    syntax::{AstNode, FunctionType, MathAst},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Source {
    Literal,
    Variable,
    Stack,
}

impl Source {
    #[inline]
    fn fetch_owned<N: Number, V: VarId>(
        self,
        literals: &mut Iter<'_, N>,
        variables: &mut Iter<'_, V>,
        variable_values: &impl VariableStore<N, V>,
        stack: &mut Vec<N>,
    ) -> N {
        match self {
            Source::Literal => literals.next().unwrap().clone(),
            Source::Variable => variable_values.get(*variables.next().unwrap()).to_owned(),
            Source::Stack => stack.pop().unwrap(),
        }
    }
}

fn fetch_args<'a, 'b: 'a, const A: usize, N: Number, V: VarId>(
    param_sources: &mut impl Iterator<Item = Source>,
    literals: &mut Iter<'b, N>,
    variables: &mut Iter<'b, V>,
    variable_values: &'a impl VariableStore<N, V>,
) -> [Option<N::AsArg<'a>>; A] {
    let mut res = [None; A];
    for (i, ps) in param_sources.enumerate().take(A) {
        match ps {
            Source::Literal => res[i] = Some(literals.next().unwrap().asarg()),
            Source::Variable => res[i] = Some(variable_values.get(*variables.next().unwrap())),
            Source::Stack => res[i] = None,
        }
    }
    res
}

#[derive(Clone)]
pub(crate) enum CtxFuncPtr<'a, N>
where
    N: Number,
{
    Single(fn(N) -> N),
    Dual(for<'b> fn(N, N::AsArg<'b>) -> N),
    Triple(for<'b, 'c> fn(N, N::AsArg<'b>, N::AsArg<'c>) -> N),
    Flexible(fn(&[N]) -> N, NonZeroU8),
    DynSingle(&'a dyn Fn(N) -> N),
    DynDual(&'a dyn for<'b> Fn(N, N::AsArg<'b>) -> N),
    DynTriple(&'a dyn for<'b, 'c> Fn(N, N::AsArg<'b>, N::AsArg<'c>) -> N),
    DynFlexible(&'a dyn Fn(&[N]) -> N, NonZeroU8),
}

impl<'a, N: Number> CtxFuncPtr<'a, N> {
    fn from_ptr_args(func: FunctionPointer<'a, N>, argc: NonZeroU8) -> Self {
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

impl<'a, N: Number> Copy for CtxFuncPtr<'a, N> {}

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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FunctionSource<F: FuncId> {
    BinaryOp(BinaryOp),
    UnaryOp(UnaryOp),
    BuiltinFunction(BuiltinFunction),
    CustomFunction(F),
}

#[derive(Clone, Copy)]
pub(crate) struct MarkedFunc<'a, N: Number, F: FuncId> {
    pub(crate) func: CtxFuncPtr<'a, N>,
    pub(crate) _src: PhantomData<F>,
    #[cfg(debug_assertions)]
    pub(crate) src: FunctionSource<F>,
}

impl<'a, N: Number, F: FuncId> MarkedFunc<'a, N, F> {
    #[allow(unused_variables)]
    pub(crate) fn new(func: CtxFuncPtr<'a, N>, src: FunctionSource<F>) -> Self {
        Self {
            func,
            _src: PhantomData,
            #[cfg(debug_assertions)]
            src,
        }
    }
}

#[cfg(debug_assertions)]
impl<'a, N: Number, F: FuncId> PartialEq for MarkedFunc<'a, N, F> {
    fn eq(&self, other: &Self) -> bool {
        self.src == other.src
    }
}

#[cfg(debug_assertions)]
impl<'a, N: Number, F: FuncId> Eq for MarkedFunc<'a, N, F> {}

impl<'a, N, F> std::fmt::Debug for MarkedFunc<'a, N, F>
where
    N: Number + std::fmt::Debug,
    F: FuncId + std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut dbg = f.debug_tuple("MarkedFunc");
        dbg.field(&self.func);
        #[cfg(debug_assertions)]
        dbg.field(&self.src);
        dbg.finish()
    }
}

impl<N: Number, F: FuncId> From<BinaryOp> for MarkedFunc<'static, N, F> {
    fn from(value: BinaryOp) -> Self {
        Self::new(value.into(), FunctionSource::BinaryOp(value))
    }
}

impl<N: Number, F: FuncId> From<UnaryOp> for MarkedFunc<'static, N, F> {
    fn from(value: UnaryOp) -> Self {
        Self::new(value.into(), FunctionSource::UnaryOp(value))
    }
}

#[derive(Clone, Copy, Debug)]
enum Instr<'a, N: Number, F: FuncId> {
    Push(Source),
    Calculate(MarkedFunc<'a, N, F>),
}

#[cfg(debug_assertions)]
impl<'a, N: Number, F: FuncId> PartialEq for Instr<'a, N, F> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Push(l0), Self::Push(r0)) => l0 == r0,
            (Self::Calculate(l0), Self::Calculate(r0)) => l0 == r0,
            _ => false,
        }
    }
}

#[cfg(debug_assertions)]
impl<'a, N: Number, F: FuncId> Eq for Instr<'a, N, F> {}

#[derive(Clone, Debug, PartialEq, Eq)]
enum InstrArg<N: Number, V: VarId> {
    Literal(N),
    Variable(V),
    Stack,
}

#[derive(Clone, Debug)]
pub struct QuickExpr<'a, N: Number, V: VarId, F: FuncId> {
    param_sources: Vec<Source>,
    literals: Vec<N>,
    variables: Vec<V>,
    instructions: Vec<Instr<'a, N, F>>,
}

impl<'a, N: Number, V: VarId, F: FuncId> QuickExpr<'a, N, V, F> {
    pub fn new(
        ast: MathAst<N, V, F>,
        function_to_pointer: impl Fn(F) -> FunctionPointer<'a, N>,
    ) -> Self {
        let tree = ast.as_tree();
        let is_flex = |n: usize| match tree[n] {
            AstNode::Function(FunctionType::Builtin(bf), _) => bf.is_flex(),
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
            let calculates = !matches!(node, AstNode::Number(_) | AstNode::Variable(_));
            let mut stack_exclusive = false;
            let mut arg_cons = 0;
            match node {
                AstNode::Number(num) => {
                    if must_push.next().unwrap() {
                        literals.push(num);
                        instructions.push(Instr::Push(Source::Literal));
                        args.push(InstrArg::Stack);
                    } else {
                        args.push(InstrArg::Literal(num));
                    }
                }
                AstNode::Variable(var) => {
                    if must_push.next().unwrap() {
                        variables.push(var);
                        instructions.push(Instr::Push(Source::Variable));
                        args.push(InstrArg::Stack);
                    } else {
                        args.push(InstrArg::Variable(var));
                    }
                }
                AstNode::BinaryOp(opr) => {
                    arg_cons = 2;
                    instructions.push(Instr::Calculate(MarkedFunc::new(
                        CtxFuncPtr::from_ptr_args(opr.into(), nz!(2)),
                        FunctionSource::BinaryOp(opr),
                    )));
                }
                AstNode::UnaryOp(opr) => {
                    arg_cons = 1;
                    instructions.push(Instr::Calculate(MarkedFunc::new(
                        CtxFuncPtr::from_ptr_args(opr.into(), nz!(1)),
                        FunctionSource::UnaryOp(opr),
                    )));
                }
                AstNode::Function(FunctionType::Builtin(bf), argc) => {
                    if bf.is_flex() {
                        stack_exclusive = true;
                    }
                    arg_cons = argc.get();
                    instructions.push(Instr::Calculate(MarkedFunc::new(
                        CtxFuncPtr::from_ptr_args(bf.into(), argc),
                        FunctionSource::BuiltinFunction(bf),
                    )));
                }
                AstNode::Function(FunctionType::Custom(cf), argc) => {
                    let ptr = function_to_pointer(cf);
                    if ptr.is_flex() {
                        stack_exclusive = true;
                    }
                    arg_cons = argc.get();
                    instructions.push(Instr::Calculate(MarkedFunc::new(
                        CtxFuncPtr::from_ptr_args(ptr, argc),
                        FunctionSource::CustomFunction(cf),
                    )));
                }
            }
            if stack_exclusive {
                for _ in 0..arg_cons {
                    assert!(matches!(args.pop(), Some(InstrArg::Stack)))
                }
            } else {
                let range = args.len() - arg_cons as usize..;
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
                            param_sources.push(Source::Stack);
                        }
                    }
                }
            }
            if calculates {
                args.push(InstrArg::Stack);
            }
        }
        assert_eq!(args.len(), 1);
        QuickExpr {
            param_sources,
            literals,
            variables,
            instructions,
        }
    }

    pub fn stack_req_capacity(&self) -> usize {
        let mut p: usize = 0;
        let mut length: usize = 0;
        let mut capacity: usize = 0;
        for instr in &self.instructions {
            macro_rules! asc {
                ($argc: expr) => {{
                    length = length
                        .checked_sub(
                            self.param_sources[p..p + $argc]
                                .iter()
                                .filter(|&&s| s == Source::Stack)
                                .count(),
                        )
                        .unwrap();
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
                        length -= argc.get() as usize;
                        length += 1;
                    }
                },
            }
            if length > capacity {
                capacity = length;
            }
        }
        capacity
    }

    pub fn eval(&self, variable_values: &impl VariableStore<N, V>, stack: &mut Vec<N>) -> N {
        let mut param_sources = self.param_sources.iter().copied();
        let mut literals = self.literals.iter();
        let mut variables = self.variables.iter();
        for instr in &self.instructions {
            let mut removed = 0;
            macro_rules! fetch_args {
                () => {
                    fetch_args(
                        &mut param_sources,
                        &mut literals,
                        &mut variables,
                        variable_values,
                    )
                };
            }
            macro_rules! resolve_arg {
                ($arg: expr, $binding: ident) => {
                    if let Some(arg) = $arg {
                        arg
                    } else {
                        $binding = stack.pop().unwrap();
                        $binding.asarg()
                    }
                };
            }
            macro_rules! resolve_arg_owned {
                ($a: expr) => {
                    if let Some(arg) = $a {
                        <N as Number>::AsArg::to_owned(&arg)
                    } else {
                        stack.pop().unwrap()
                    }
                };
            }
            let result = match instr {
                Instr::Push(src) => {
                    src.fetch_owned(&mut literals, &mut variables, variable_values, stack)
                }
                Instr::Calculate(idfunc) => match idfunc.func {
                    CtxFuncPtr::Single(func) => {
                        let args: [_; 1] = fetch_args!();
                        func(resolve_arg_owned!(args[0]))
                    }
                    CtxFuncPtr::Dual(func) => {
                        let args: [_; 2] = fetch_args!();
                        let a1b;
                        let arg1 = resolve_arg!(args[1], a1b);
                        let arg0 = resolve_arg_owned!(args[0]);
                        func(arg0, arg1)
                    }
                    CtxFuncPtr::Triple(func) => {
                        let args: [_; 3] = fetch_args!();
                        let (a1b, a2b);
                        let arg2 = resolve_arg!(args[2], a2b);
                        let arg1 = resolve_arg!(args[1], a1b);
                        let arg0 = resolve_arg_owned!(args[0]);
                        func(arg0, arg1, arg2)
                    }
                    CtxFuncPtr::Flexible(func, argc) => {
                        removed = argc.get();
                        func(&stack[stack.len() - argc.get() as usize..])
                    }
                    CtxFuncPtr::DynSingle(func) => {
                        let args: [_; 1] = fetch_args!();
                        func(resolve_arg_owned!(args[0]))
                    }
                    CtxFuncPtr::DynDual(func) => {
                        let args: [_; 2] = fetch_args!();
                        let a1b;
                        let arg1 = resolve_arg!(args[1], a1b);
                        let arg0 = resolve_arg_owned!(args[0]);
                        func(arg0, arg1)
                    }
                    CtxFuncPtr::DynTriple(func) => {
                        let args: [_; 3] = fetch_args!();
                        let (a1b, a2b);
                        let arg2 = resolve_arg!(args[2], a2b);
                        let arg1 = resolve_arg!(args[1], a1b);
                        let arg0 = resolve_arg_owned!(args[0]);
                        func(arg0, arg1, arg2)
                    }
                    CtxFuncPtr::DynFlexible(func, argc) => {
                        removed = argc.get();
                        func(&stack[stack.len() - argc.get() as usize..])
                    }
                },
            };
            stack.truncate(stack.len() - removed as usize);
            stack.push(result);
        }
        stack.pop().unwrap()
    }
}

#[cfg(debug_assertions)]
impl<'a, N: Number, V: VarId, F: FuncId> PartialEq for QuickExpr<'a, N, V, F> {
    fn eq(&self, other: &Self) -> bool {
        self.param_sources == other.param_sources
            && self.literals == other.literals
            && self.variables == other.variables
            && self.instructions == other.instructions
    }
}

#[cfg(debug_assertions)]
impl<'a, N: Number, V: VarId, F: FuncId> Eq for QuickExpr<'a, N, V, F> {}

#[cfg(all(debug_assertions, test))]
mod tests {
    use strum::FromRepr;

    use crate::{
        nz,
        syntax::CfInfo,
        tokenizer::{StandardFloatRecognizer as Sfr, TokenStream},
        trie::{EmptyNameTrie, NameTrie, TrieNode},
    };

    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, FromRepr)]
    #[repr(u8)]
    enum TestVar {
        X,
        Y,
        T,
    }

    #[derive(Debug)]
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

    #[derive(Clone, Copy, Debug, PartialEq, Eq, FromRepr)]
    #[repr(u8)]
    enum TestFunc {
        Sigmoid,
        F1,
        Digits,
    }

    impl TestFunc {
        fn with_info(self) -> CfInfo<Self> {
            match self {
                TestFunc::Sigmoid => CfInfo::new(TestFunc::Sigmoid, nz!(1), Some(nz!(1))),
                TestFunc::F1 => CfInfo::new(TestFunc::F1, nz!(3), Some(nz!(3))),
                TestFunc::Digits => CfInfo::new(TestFunc::Digits, nz!(1), None),
            }
        }
        fn as_pointer(self) -> FunctionPointer<'static, f64> {
            match self {
                TestFunc::Sigmoid => FunctionPointer::<f64>::Single(sigmoid),
                TestFunc::F1 => FunctionPointer::<f64>::Triple(func1),
                TestFunc::Digits => FunctionPointer::Flexible(digits),
            }
        }
        fn as_marked(self, argc: NonZeroU8) -> MarkedFunc<'static, f64, Self> {
            MarkedFunc::new(
                CtxFuncPtr::from_ptr_args(self.as_pointer(), argc),
                FunctionSource::CustomFunction(self),
            )
        }
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
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

    struct TestFuncsNameTrie;

    impl NameTrie<CfInfo<TestFunc>> for TestFuncsNameTrie {
        fn nodes(&self) -> &[TrieNode] {
            &[
                TrieNode::Branch('d', 6),
                TrieNode::Branch('i', 5),
                TrieNode::Branch('g', 4),
                TrieNode::Branch('i', 3),
                TrieNode::Branch('t', 2),
                TrieNode::Branch('s', 1),
                TrieNode::Leaf(TestFunc::Digits as u32),
                TrieNode::Branch('f', 5),
                TrieNode::Branch('u', 4),
                TrieNode::Branch('n', 3),
                TrieNode::Branch('c', 2),
                TrieNode::Branch('1', 1),
                TrieNode::Leaf(TestFunc::F1 as u32),
                TrieNode::Branch('s', 7),
                TrieNode::Branch('i', 6),
                TrieNode::Branch('g', 5),
                TrieNode::Branch('m', 4),
                TrieNode::Branch('o', 3),
                TrieNode::Branch('i', 2),
                TrieNode::Branch('d', 1),
                TrieNode::Leaf(TestFunc::Sigmoid as u32),
            ]
        }

        fn leaf_to_value(&self, leaf: u32) -> CfInfo<TestFunc> {
            TestFunc::from_repr(leaf as u8).unwrap().with_info()
        }
    }

    #[test]
    fn convert() {
        fn convert(input: &str) -> QuickExpr<'_, f64, TestVar, TestFunc> {
            let tokens = TokenStream::new::<Sfr>(input)
                .map_err(|e| e.to_general())
                .unwrap()
                .0;
            QuickExpr::new(
                MathAst::new(
                    &tokens,
                    &EmptyNameTrie,
                    &TestFuncsNameTrie,
                    &TestVarsNameTrie,
                )
                .unwrap(),
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
                instructions: vec![Instr::Calculate(BuiltinFunction::Sin.as_markedfunc(nz!(1)))],
            },
        );
        assert_eq!(
            convert("sin(x)+1"),
            QuickExpr {
                param_sources: vec![Source::Variable, Source::Stack, Source::Literal],
                literals: vec![1.0],
                variables: vec![TestVar::X],
                instructions: vec![
                    Instr::Calculate(BuiltinFunction::Sin.as_markedfunc(nz!(1))),
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
                    Source::Stack,
                    Source::Stack,
                    Source::Literal,
                ],
                literals: vec![1.0],
                variables: vec![TestVar::X, TestVar::Y],
                instructions: vec![
                    Instr::Calculate(BuiltinFunction::Sin.as_markedfunc(nz!(1))),
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
                    Source::Stack,
                    Source::Literal,
                ],
                literals: vec![5.0, 1.0],
                variables: vec![TestVar::X, TestVar::Y],
                instructions: vec![
                    Instr::Calculate(BinaryOp::Mul.into()),
                    Instr::Calculate(TestFunc::F1.as_marked(nz!(1)))
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
                    Instr::Calculate(BuiltinFunction::Max.as_markedfunc(nz!(3))),
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
                    Source::Stack,
                    Source::Stack
                ],
                literals: vec![2.0],
                variables: vec![TestVar::X, TestVar::Y],
                instructions: vec![
                    Instr::Calculate(BinaryOp::Pow.into()),
                    Instr::Calculate(BuiltinFunction::Sin.as_markedfunc(nz!(1))),
                    Instr::Calculate(BinaryOp::Add.into())
                ]
            }
        );
        assert_eq!(
            convert("min(5, t, 3) + 2"),
            QuickExpr {
                param_sources: vec![Source::Stack, Source::Literal,],
                literals: vec![5.0, 3.0, 2.0],
                variables: vec![TestVar::T],
                instructions: vec![
                    Instr::Push(Source::Literal),
                    Instr::Push(Source::Variable),
                    Instr::Push(Source::Literal),
                    Instr::Calculate(BuiltinFunction::Min.as_markedfunc(nz!(3))),
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
            1
        );
        assert_eq!(
            QuickExpr::<f64, TestVar, TestFunc> {
                param_sources: vec![Source::Variable, Source::Literal],
                literals: vec![2.0],
                variables: vec![TestVar::X],
                instructions: vec![Instr::Calculate(BinaryOp::Add.into())],
            }
            .stack_req_capacity(),
            1
        );
        assert_eq!(
            QuickExpr {
                param_sources: vec![Source::Stack, Source::Literal,],
                literals: vec![5.0, 3.0, 2.0],
                variables: vec![TestVar::T],
                instructions: vec![
                    Instr::<f64, TestFunc>::Push(Source::Literal),
                    Instr::Push(Source::Variable),
                    Instr::Push(Source::Literal),
                    Instr::Calculate(BuiltinFunction::Min.as_markedfunc(nz!(3))),
                    Instr::Calculate(BinaryOp::Add.into()),
                ]
            }
            .stack_req_capacity(),
            3
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
            .eval(&TestStore, &mut Vec::new())
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
                vec![Source::Literal, Source::Stack, Source::Variable],
                vec![12.0, 5.0],
                vec![TestVar::X],
                vec![
                    Instr::Push(Source::Literal),
                    Instr::Calculate(TestFunc::F1.as_marked(nz!(1)))
                ]
            ),
            55.0
        );
        assert_eq!(
            evaluate(
                vec![Source::Stack, Source::Stack, Source::Stack],
                vec![8.0, 4.0, 2.0],
                vec![],
                vec![
                    Instr::Push(Source::Literal),
                    Instr::Push(Source::Literal),
                    Instr::Push(Source::Literal),
                    Instr::Calculate(TestFunc::Digits.as_marked(nz!(3)))
                ]
            ),
            248.0
        );
    }
}
