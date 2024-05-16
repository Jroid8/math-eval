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
    CustomFunction(&'a dyn Fn(&[N]) -> N, u8, F),
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
            Self::BiOperation(arg0, arg1, arg2) => f
                .debug_tuple("BiOperation")
                .field(arg0)
                .field(arg1)
                .field(arg2)
                .finish(),
            Self::UnOperation(arg0, arg1) => f
                .debug_tuple("UnOperation")
                .field(arg0)
                .field(arg1)
                .finish(),
            Self::NFSingle(_, arg1, arg2) => {
                f.debug_tuple("NFSingle").field(arg2).field(arg1).finish()
            }
            Self::NFDual(_, arg1, arg2, arg3) => f
                .debug_tuple("NFDual")
                .field(arg3)
                .field(arg1)
                .field(arg2)
                .finish(),
            Self::NFFlexible(_, arg1, arg2) => {
                f.debug_tuple("NFFlexible").field(arg2).field(arg1).finish()
            }
            Self::CustomFunction(_, arg1, arg2) => f
                .debug_tuple("CustomFunction")
                .field(arg2)
                .field(arg1)
                .finish(),
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
            Self::CustomFunction(arg0, arg1, arg2) => {
                Self::CustomFunction(*arg0, *arg1, arg2.clone())
            }
        }
    }
}

#[derive(Clone, Default)]
pub struct MathAssembly<'a, N: MathEvalNumber, V: Clone, F: Clone> {
    instructions: Vec<Instruction<'a, N, V, F>>,
    stack_capacity: usize,
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
impl<'a, N, V, F> MathAssembly<'a, N, V, F>
where
    N: MathEvalNumber,
    V: Clone,
    F: Clone,
{
    pub fn new(
        arena: &Arena<SyntaxNode<N, V, F>>,
        root: NodeId,
        function_to_pointer: impl Fn(&F) -> &'a dyn Fn(&[N]) -> N,
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
                            children_as_input
                                .next()
                                .unwrap_or(if *nf == NativeFunction::Log {
                                    Input::Literal(10.0.into())
                                } else {
                                    panic!()
                                }),
                            *nf,
                        ),
                        crate::number::NFPointer::Flexible(p) => {
                            Instruction::NFFlexible(p, cursor.children(arena).count() as u8, *nf)
                        }
                    },
                    SyntaxNode::CustomFunction(cf) => Instruction::CustomFunction(
                        function_to_pointer(cf),
                        cursor.children(arena).count() as u8,
                        cf.clone(),
                    ),
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
                Instruction::BiOperation(_, lhs, rhs) | Instruction::NFDual(_, lhs, rhs, _) => {
                    input_stack_effect(lhs) + input_stack_effect(rhs)
                }
                Instruction::UnOperation(_, val) | Instruction::NFSingle(_, val, _) => {
                    input_stack_effect(val)
                }
                Instruction::NFFlexible(_, arg_count, _)
                | Instruction::CustomFunction(_, arg_count, _) => -(*arg_count as i32),
            } + 1;
            if stack_len > stack_capacity {
                stack_capacity = stack_len;
            }
        }
        MathAssembly {
            instructions: result,
            stack_capacity: stack_capacity as usize,
        }
    }

    pub fn eval(&self, variable_substituter: impl Fn(&V) -> N) -> N {
        let mut stack = Stack::with_capacity(self.stack_capacity);
        for instr in &self.instructions {
            let result = match &instr {
                Instruction::Source(input) => input.get(&variable_substituter, &mut stack),
                Instruction::BiOperation(opr, lhs, rhs) => {
                    let rhs = rhs.get(&variable_substituter, &mut stack);
                    let lhs = lhs.get(&variable_substituter, &mut stack);
                    opr.eval(lhs, rhs)
                }
                Instruction::UnOperation(opr, val) => {
                    opr.eval(val.get(&variable_substituter, &mut stack))
                }
                Instruction::NFSingle(func, input, _) => {
                    let input = input.get(&variable_substituter, &mut stack);
                    func(input)
                }
                Instruction::NFDual(func, inp1, inp2, _) => func(
                    inp1.get(&variable_substituter, &mut stack),
                    inp2.get(&variable_substituter, &mut stack),
                ),
                Instruction::NFFlexible(func, arg_count, _) => {
                    let arg_count = *arg_count as usize;
                    let result = func(&stack[stack.len() - arg_count..]);
                    stack.truncate(stack.len() - arg_count);
                    result
                }
                Instruction::CustomFunction(func, arg_count, _) => {
                    let arg_count = *arg_count as usize;
                    let result = func(&stack[stack.len() - arg_count..]);
                    stack.truncate(stack.len() - arg_count);
                    result
                }
            };
            stack.push(result);
        }
        stack.pop().unwrap()
    }
}

#[cfg(test)]
mod test {
    use std::f64::consts::PI;

    use crate::{ParsingError, TestVar};

    fn parse(input: &str) -> Result<f64, ParsingError> {
        crate::parse(
            input,
            |_| None,
            crate::parse_test_func,
            crate::parse_test_var,
            crate::test_func_to_pointer,
        )
        .map(|asm| asm.eval(|var| match var {
            TestVar::X => 1.0,
            TestVar::Y => 8.0,
            TestVar::T => 1.5,
        }))
    }

    #[test]
    fn test_mathassembly() {
        assert_eq!(parse("10").unwrap(), 10.0);
        assert_eq!(parse("-y").unwrap(), - 8.0);
        assert_eq!(parse("abs(-x)").unwrap(), 1.0);
        assert_eq!(parse("4t").unwrap(), 6.0);
        assert_eq!(parse("10(7+x)").unwrap(), 80.0);
        assert_eq!(parse("5sin(pi*3/2)").unwrap(), - 5.0);
        assert_eq!(parse("max(cos(pi/2), 1)").unwrap(), 1.0);
        assert_eq!(parse("pi dist(y-5x,4)").unwrap(), PI * 5.0);
        assert_eq!(
            parse("sin(x*pi/10*(1.3+sin(t/10))+t*2+sin(y*pi*sin(t/17)+16*sin(t)))+0.05").unwrap(),
            0.356078696074944
        );
    }
}
