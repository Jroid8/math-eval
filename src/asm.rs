use indextree::{Arena, NodeEdge, NodeId};
use smallvec::SmallVec;
use std::{fmt::Debug, usize};

use crate::{
    number::{MathEvalNumber, NativeFunction},
    syntax::{BiOperation, SyntaxNode, UnOperation},
};

type Stack<N> = SmallVec<[N; 16]>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Input<N: MathEvalNumber, V: Clone> {
    Literal(N),
    Variable(V),
    Memory,
}

impl<N, V> Input<N, V>
where
    N: MathEvalNumber,
    V: Clone,
{
    #[inline]
    fn get(&self, variable_evaluator: &impl Fn(&V) -> N, stack: &mut Stack<N>) -> N {
        match self {
            Input::Literal(num) => *num,
            Input::Variable(var) => variable_evaluator(var),
            Input::Memory => stack.pop().unwrap(),
        }
    }
}

#[derive(Copy)]
pub enum Instruction<'a, N: MathEvalNumber, V: Clone, F: Clone> {
    Source(Input<N, V>),
    BiOperation(BiOperation, Input<N, V>, Input<N, V>),
    UnOperation(UnOperation, Input<N, V>),
    NFSingle(fn(N) -> N, Input<N, V>, NativeFunction),
    NFDual(fn(N, N) -> N, Input<N, V>, Input<N, V>, NativeFunction),
    NFFlexible(fn(&[N]) -> N, u8, NativeFunction),
    CFSingle(&'a dyn Fn(N) -> N, Input<N, V>, F),
    CFDual(&'a dyn Fn(N, N) -> N, Input<N, V>, Input<N, V>, F),
    CFTriple(
        &'a dyn Fn(N, N, N) -> N,
        Input<N, V>,
        Input<N, V>,
        Input<N, V>,
        F,
    ),
    CFQuad(
        &'a dyn Fn(N, N, N, N) -> N,
        Input<N, V>,
        Input<N, V>,
        Input<N, V>,
        Input<N, V>,
        F,
    ),
    CFFlexible(&'a dyn Fn(&[N]) -> N, u8, F),
}

impl<N, V, F> Debug for Instruction<'_, N, V, F>
where
    N: MathEvalNumber,
    V: Clone + Debug,
    F: Clone + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Source(arg0) => f.debug_tuple("Source").field(arg0).finish(),
            Self::BiOperation(_, arg1, arg2) => f
                .debug_tuple("BiOperation")
                .field(arg1)
                .field(arg2)
                .finish(),
            Self::UnOperation(_, arg1) => f.debug_tuple("UnOperation").field(arg1).finish(),
            Self::NFSingle(_, arg1, arg2) => {
                f.debug_tuple("NFSingle").field(arg1).field(arg2).finish()
            }
            Self::NFDual(_, arg1, arg2, arg3) => f
                .debug_tuple("NFDual")
                .field(arg1)
                .field(arg2)
                .field(arg3)
                .finish(),
            Self::NFFlexible(_, arg1, arg2) => {
                f.debug_tuple("NFFlexible").field(arg1).field(arg2).finish()
            }
            Self::CFSingle(_, arg1, arg2) => {
                f.debug_tuple("CFSingle").field(arg1).field(arg2).finish()
            }
            Self::CFDual(_, arg1, arg2, arg3) => f
                .debug_tuple("CFDual")
                .field(arg1)
                .field(arg2)
                .field(arg3)
                .finish(),
            Self::CFTriple(_, arg1, arg2, arg3, arg4) => f
                .debug_tuple("CFTriple")
                .field(arg1)
                .field(arg2)
                .field(arg3)
                .field(arg4)
                .finish(),
            Self::CFQuad(_, arg1, arg2, arg3, arg4, arg5) => f
                .debug_tuple("CFQuad")
                .field(arg1)
                .field(arg2)
                .field(arg3)
                .field(arg4)
                .field(arg5)
                .finish(),
            Self::CFFlexible(_, arg1, arg2) => {
                f.debug_tuple("CFFlexible").field(arg1).field(arg2).finish()
            }
        }
    }
}

impl<N, V, F> Clone for Instruction<'_, N, V, F>
where
    N: MathEvalNumber,
    V: Clone,
    F: Clone,
{
    fn clone(&self) -> Self {
        match self {
            Self::Source(arg0) => Self::Source(arg0.clone()),
            Self::BiOperation(arg0, arg1, arg2) => {
                Self::BiOperation(*arg0, arg1.clone(), arg2.clone())
            }
            Self::UnOperation(arg0, arg1) => Self::UnOperation(*arg0, arg1.clone()),
            Self::NFSingle(arg0, arg1, arg2) => Self::NFSingle(*arg0, arg1.clone(), *arg2),
            Self::NFDual(arg0, arg1, arg2, arg3) => {
                Self::NFDual(*arg0, arg1.clone(), arg2.clone(), *arg3)
            }
            Self::NFFlexible(arg0, arg1, arg2) => Self::NFFlexible(*arg0, *arg1, *arg2),
            Self::CFSingle(arg0, arg1, arg2) => Self::CFSingle(*arg0, arg1.clone(), arg2.clone()),
            Self::CFDual(arg0, arg1, arg2, arg3) => {
                Self::CFDual(*arg0, arg1.clone(), arg2.clone(), arg3.clone())
            }
            Self::CFTriple(arg0, arg1, arg2, arg3, arg4) => Self::CFTriple(
                *arg0,
                arg1.clone(),
                arg2.clone(),
                arg3.clone(),
                arg4.clone(),
            ),
            Self::CFQuad(arg0, arg1, arg2, arg3, arg4, arg5) => Self::CFQuad(
                *arg0,
                arg1.clone(),
                arg2.clone(),
                arg3.clone(),
                arg4.clone(),
                arg5.clone(),
            ),
            Self::CFFlexible(arg0, arg1, arg2) => Self::CFFlexible(*arg0, *arg1, arg2.clone()),
        }
    }
}

#[derive(Clone, Default)]
pub struct MathAssembly<'a, N: MathEvalNumber, V: Clone, F: Clone> {
    instructions: Vec<Instruction<'a, N, V, F>>,
    stack: Stack<N>,
}

impl<'a, N, V, F> Debug for MathAssembly<'a, N, V, F>
where
    N: MathEvalNumber,
    V: Clone + Debug,
    F: Clone + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "MathAssembly[")?;
        for line in &self.instructions {
            writeln!(f, "\t{line:?}")?
        }
        writeln!(f, "]")?;
        Ok(())
    }
}

#[derive(Clone, Copy)]
pub enum CFPointer<'a, N>
where
    N: MathEvalNumber,
{
    Single(&'a dyn Fn(N) -> N),
    Dual(&'a dyn Fn(N, N) -> N),
    Triple(&'a dyn Fn(N, N, N) -> N),
    Quad(&'a dyn Fn(N, N, N, N) -> N),
    Flexible(&'a dyn Fn(&[N]) -> N),
}

impl<'a, N, V, F> MathAssembly<'a, N, V, F>
where
    N: MathEvalNumber,
    V: Clone,
    F: Clone,
{
    pub fn new(
        arena: &Arena<SyntaxNode<N, V, F>>,
        root: NodeId,
        function_to_pointer: impl Fn(&F) -> CFPointer<'a, N>,
    ) -> Self {
        let mut result: Vec<Instruction<'a, N, V, F>> = Vec::new();
        let is_fixed_input = |node: Option<NodeId>| match node.map(|id| arena[id].get()) {
            Some(SyntaxNode::BiOperation(_) | SyntaxNode::UnOperation(_)) => true,
            Some(SyntaxNode::NativeFunction(nf)) => !nf.is_fixed(),
            _ => false,
        };

        for current in root.traverse(arena) {
            if let NodeEdge::End(cursor) = current {
                let mut children_as_input = cursor.children(arena).map(|c| match arena[c].get() {
                    SyntaxNode::Number(num) => Input::Literal(*num),
                    SyntaxNode::Variable(var) => Input::Variable(var.clone()),
                    _ => Input::Memory,
                });
                let parent = cursor.ancestors(arena).nth(1);
                let instruction = match arena[cursor].get() {
                    SyntaxNode::Number(num) => {
                        if is_fixed_input(parent) {
                            continue;
                        } else {
                            Instruction::Source(Input::Literal(*num))
                        }
                    }
                    SyntaxNode::Variable(var) => {
                        if is_fixed_input(parent) {
                            continue;
                        } else {
                            Instruction::Source(Input::Variable(var.clone()))
                        }
                    }
                    SyntaxNode::BiOperation(opr) => Instruction::BiOperation(
                        *opr,
                        children_as_input.next().unwrap(),
                        children_as_input.next().unwrap(),
                    ),
                    SyntaxNode::UnOperation(opr) => {
                        Instruction::UnOperation(*opr, children_as_input.next().unwrap())
                    }
                    SyntaxNode::NativeFunction(nf) => match nf.to_pointer() {
                        crate::number::NFPointer::Single(p) => {
                            Instruction::NFSingle(p, children_as_input.next().unwrap(), *nf)
                        }
                        crate::number::NFPointer::Dual(p) => Instruction::NFDual(
                            p,
                            children_as_input.next().unwrap(),
                            children_as_input.next().unwrap(),
                            *nf,
                        ),
                        crate::number::NFPointer::Flexible(p) => {
                            Instruction::NFFlexible(p, cursor.children(arena).count() as u8, *nf)
                        }
                    },
                    SyntaxNode::CustomFunction(cf) => match function_to_pointer(cf) {
                        CFPointer::Single(func) => Instruction::CFSingle(
                            func,
                            children_as_input.next().unwrap(),
                            cf.clone(),
                        ),
                        CFPointer::Dual(func) => Instruction::CFDual(
                            func,
                            children_as_input.next().unwrap(),
                            children_as_input.next().unwrap(),
                            cf.clone(),
                        ),
                        CFPointer::Triple(func) => Instruction::CFTriple(
                            func,
                            children_as_input.next().unwrap(),
                            children_as_input.next().unwrap(),
                            children_as_input.next().unwrap(),
                            cf.clone(),
                        ),
                        CFPointer::Quad(func) => Instruction::CFQuad(
                            func,
                            children_as_input.next().unwrap(),
                            children_as_input.next().unwrap(),
                            children_as_input.next().unwrap(),
                            children_as_input.next().unwrap(),
                            cf.clone(),
                        ),
                        CFPointer::Flexible(func) => Instruction::CFFlexible(
                            func,
                            cursor.children(arena).count() as u8,
                            cf.clone(),
                        ),
                    },
                };
                result.push(instruction);
            }
        }
        let mut stack_capacity = 0;
        let mut stack_len = 0;
        let input_stack_effect = |inp: &Input<N, V>| match inp {
            Input::Memory => -1,
            _ => 0,
        };
        for instr in &result {
            stack_len += match instr {
                Instruction::Source(_) => 0,
                Instruction::BiOperation(_, lhs, rhs)
                | Instruction::NFDual(_, lhs, rhs, _)
                | Instruction::CFDual(_, lhs, rhs, _) => {
                    input_stack_effect(lhs) + input_stack_effect(rhs)
                }
                Instruction::UnOperation(_, val)
                | Instruction::NFSingle(_, val, _)
                | Instruction::CFSingle(_, val, _) => input_stack_effect(val),
                Instruction::CFTriple(_, inp1, inp2, inp3, _) => {
                    input_stack_effect(inp1) + input_stack_effect(inp2) + input_stack_effect(inp3)
                }
                Instruction::CFQuad(_, inp1, inp2, inp3, inp4, _) => {
                    input_stack_effect(inp1)
                        + input_stack_effect(inp2)
                        + input_stack_effect(inp3)
                        + input_stack_effect(inp4)
                }
                Instruction::NFFlexible(_, arg_count, _)
                | Instruction::CFFlexible(_, arg_count, _) => -(*arg_count as i32),
            } + 1;
            if stack_len > stack_capacity {
                stack_capacity = stack_len;
            }
        }
        MathAssembly {
            instructions: result,
            stack: Stack::with_capacity(stack_capacity as usize),
        }
    }

    pub fn eval(&mut self, variable_substituter: impl Fn(&V) -> N) -> N {
        self.stack.clear();
        macro_rules! get {
            ($inp: expr) => {
                $inp.get(&variable_substituter, &mut self.stack)
            };
        }
        for instr in &self.instructions {
            let result = match &instr {
                Instruction::Source(input) => get!(input),
                Instruction::BiOperation(opr, lhs, rhs) => {
                    let rhs = get!(rhs);
                    let lhs = get!(lhs);
                    opr.eval(lhs, rhs)
                }
                Instruction::UnOperation(opr, val) => opr.eval(get!(val)),
                Instruction::NFSingle(func, input, _) => {
                    let input = get!(input);
                    func(input)
                }
                Instruction::NFDual(func, inp1, inp2, _) => func(get!(inp1), get!(inp2)),
                Instruction::NFFlexible(func, arg_count, _) => {
                    let arg_count = *arg_count as usize;
                    let result = func(&self.stack[self.stack.len() - arg_count..]);
                    self.stack.truncate(self.stack.len() - arg_count);
                    result
                }
                Instruction::CFSingle(func, inp, _) => func(get!(inp)),
                Instruction::CFDual(func, inp1, inp2, _) => func(get!(inp1), get!(inp2)),
                Instruction::CFTriple(func, inp1, inp2, inp3, _) => {
                    func(get!(inp1), get!(inp2), get!(inp3))
                }
                Instruction::CFQuad(func, inp1, inp2, inp3, inp4, _) => {
                    func(get!(inp1), get!(inp2), get!(inp3), get!(inp4))
                }
                Instruction::CFFlexible(func, arg_count, _) => {
                    let arg_count = *arg_count as usize;
                    let result = func(&self.stack[self.stack.len() - arg_count..]);
                    self.stack.truncate(self.stack.len() - arg_count);
                    result
                }
            };
            self.stack.push(result);
        }
        self.stack.pop().unwrap()
    }
}

#[cfg(test)]
mod test {
    use std::f64::consts::PI;

    #[test]
    fn test_mathassembly() {
        let parse = crate::EvalBuilder::new()
            .add_variable("x")
            .add_variable("y")
            .add_variable("t")
            .add_constant("c", 299792458.0)
            .add_fn2("dist", &|x: f64, y: f64| (x.powi(2) + y.powi(2)).sqrt())
            .build_as_parser();

        assert_eq!(parse("10", 0.0, 0.0, 0.0), Ok(10.0));
        assert_eq!(parse("-y", 0.0, 13.8, 0.0), Ok(-13.8));
        assert_eq!(parse("abs(-x)", 49.9, 0.0, 0.0), Ok(49.9));
        assert_eq!(parse("4c", 0.0, 0.0, 2.0), Ok(1199169832.0));
        assert_eq!(parse("10(7+x)", 3.0, 0.0, 0.0), Ok(100.0));
        assert_eq!(parse("5sin(pi*3/2)", 0.0, 0.0, 0.0), Ok(-5.0));
        assert_eq!(
            parse("max(cos(pi/2), 1, c)", 0.0, 0.0, 0.0),
            Ok(299792458.0)
        );
        assert_eq!(parse("asin(x-y)", 13.5, 12.5, 0.0), Ok(PI / 2.0));
        assert_eq!(
            parse(
                "sin(x*pi/10*(1.3+sin(t/10))+t*2+sin(y*pi*sin(t/17)+16*sin(t)))+0.05",
                5.5,
                -30.0,
                18.0
            ),
            Ok(0.7967435953555171)
        );
    }
}
