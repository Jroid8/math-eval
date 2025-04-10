use indextree::{Arena, NodeEdge, NodeId};
use smallvec::SmallVec;
use std::{collections::HashMap, fmt::Debug, hash::Hash};

use crate::{
    number::{MathEvalNumber, NativeFunction, Reborrow},
    syntax::{BiOperation, FunctionIdentifier, SyntaxNode, UnOperation, VariableIdentifier},
};

type Stack<N> = SmallVec<[N; 16]>;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Input<N: MathEvalNumber, V: VariableIdentifier> {
    Literal(N),
    Variable(usize, V),
    Memory,
}

#[derive(Copy)]
pub enum Instruction<'a, N: MathEvalNumber, V: VariableIdentifier, F: FunctionIdentifier> {
    Source(Input<N, V>),
    BiOperation(BiOperation, Input<N, V>, Input<N, V>),
    UnOperation(UnOperation, Input<N, V>),
    NFSingle(for<'b> fn(N::AsArg<'b>) -> N, Input<N, V>, NativeFunction),
    NFDual(
        for<'b> fn(N::AsArg<'b>, N::AsArg<'b>) -> N,
        Input<N, V>,
        Input<N, V>,
        NativeFunction,
    ),
    NFFlexible(fn(&[N]) -> N, u8, NativeFunction),
    CFSingle(&'a dyn for<'b> Fn(N::AsArg<'b>) -> N, Input<N, V>, F),
    CFDual(
        &'a dyn for<'b> Fn(N::AsArg<'b>, N::AsArg<'b>) -> N,
        Input<N, V>,
        Input<N, V>,
        F,
    ),
    CFTriple(
        &'a dyn for<'b> Fn(N::AsArg<'b>, N::AsArg<'b>, N::AsArg<'b>) -> N,
        Input<N, V>,
        Input<N, V>,
        Input<N, V>,
        F,
    ),
    CFQuad(
        &'a dyn for<'b> Fn(N::AsArg<'b>, N::AsArg<'b>, N::AsArg<'b>, N::AsArg<'b>) -> N,
        Input<N, V>,
        Input<N, V>,
        Input<N, V>,
        Input<N, V>,
        F,
    ),
    CFFlexible(&'a dyn Fn(&[N]) -> N, u8, F),
}

impl<N, V, F> PartialEq for Instruction<'_, N, V, F>
where
    N: MathEvalNumber,
    V: VariableIdentifier,
    F: FunctionIdentifier + PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Source(l0), Self::Source(r0)) => l0 == r0,
            (Self::BiOperation(l0, l1, l2), Self::BiOperation(r0, r1, r2)) => {
                l0 == r0 && l1 == r1 && l2 == r2
            }
            (Self::UnOperation(l0, l1), Self::UnOperation(r0, r1)) => l0 == r0 && l1 == r1,
            (Self::NFSingle(_, l1, l2), Self::NFSingle(_, r1, r2)) => l1 == r1 && l2 == r2,
            (Self::NFDual(_, l1, l2, l3), Self::NFDual(_, r1, r2, r3)) => {
                l1 == r1 && l2 == r2 && l3 == r3
            }
            (Self::NFFlexible(_, l1, l2), Self::NFFlexible(_, r1, r2)) => l1 == r1 && l2 == r2,
            (Self::CFSingle(_, l1, l2), Self::CFSingle(_, r1, r2)) => l1 == r1 && l2 == r2,
            (Self::CFDual(_, l1, l2, l3), Self::CFDual(_, r1, r2, r3)) => {
                l1 == r1 && l2 == r2 && l3 == r3
            }
            (Self::CFTriple(_, l1, l2, l3, l4), Self::CFTriple(_, r1, r2, r3, r4)) => {
                l1 == r1 && l2 == r2 && l3 == r3 && l4 == r4
            }
            (Self::CFQuad(_, l1, l2, l3, l4, l5), Self::CFQuad(_, r1, r2, r3, r4, r5)) => {
                l1 == r1 && l2 == r2 && l3 == r3 && l4 == r4 && l5 == r5
            }
            (Self::CFFlexible(_, l1, l2), Self::CFFlexible(_, r1, r2)) => l1 == r1 && l2 == r2,
            _ => false,
        }
    }
}

impl<N, V, F> Eq for Instruction<'_, N, V, F>
where
    N: MathEvalNumber,
    V: VariableIdentifier,
    F: FunctionIdentifier + Eq,
{
}

impl<N, V, F> Debug for Instruction<'_, N, V, F>
where
    N: MathEvalNumber,
    V: VariableIdentifier,
    F: FunctionIdentifier,
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
    V: VariableIdentifier,
    F: FunctionIdentifier,
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

#[derive(Clone, Default, PartialEq, Eq)]
pub struct MathAssembly<'a, N: MathEvalNumber, V: VariableIdentifier, F: FunctionIdentifier> {
    instructions: Vec<Instruction<'a, N, V, F>>,
    stack: Stack<N>,
}

impl<N, V, F> Debug for MathAssembly<'_, N, V, F>
where
    N: MathEvalNumber,
    V: VariableIdentifier,
    F: FunctionIdentifier,
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
    Single(&'a dyn for<'b> Fn(N::AsArg<'b>) -> N),
    Dual(&'a dyn for<'b> Fn(N::AsArg<'b>, N::AsArg<'b>) -> N),
    Triple(&'a dyn for<'b> Fn(N::AsArg<'b>, N::AsArg<'b>, N::AsArg<'b>) -> N),
    Quad(&'a dyn for<'b> Fn(N::AsArg<'b>, N::AsArg<'b>, N::AsArg<'b>, N::AsArg<'b>) -> N),
    Flexible(&'a dyn Fn(&[N]) -> N),
}

impl<'a, N, V, F> MathAssembly<'a, N, V, F>
where
    N: MathEvalNumber,
    V: VariableIdentifier,
    F: FunctionIdentifier,
{
    // FIX: panics when variable_order is not exhaustive
    pub fn new(
        arena: &Arena<SyntaxNode<N, V, F>>,
        root: NodeId,
        function_to_pointer: impl Fn(&F) -> CFPointer<'a, N>,
        variable_order: &[V],
    ) -> Self {
        let mut result: Vec<Instruction<'a, N, V, F>> = Vec::new();
        let is_fixed_input = |node: Option<NodeId>| match node.map(|id| arena[id].get()) {
            Some(SyntaxNode::BiOperation(_) | SyntaxNode::UnOperation(_)) => true,
            Some(SyntaxNode::NativeFunction(nf)) => !nf.is_fixed(),
            Some(SyntaxNode::CustomFunction(cf)) => {
                !matches!(function_to_pointer(cf), CFPointer::Flexible(_))
            }
            _ => false,
        };
        let variables = variable_order
            .iter()
            .enumerate()
            .map(|(i, v)| (v.clone(), i))
            .collect::<HashMap<V, usize>>();

        for current in root.traverse(arena) {
            if let NodeEdge::End(cursor) = current {
                let mut children_as_input = cursor.children(arena).map(|c| match arena[c].get() {
                    SyntaxNode::Number(num) => Input::Literal(*num),
                    SyntaxNode::Variable(var) => {
                        Input::Variable(*variables.get(var).unwrap(), var.clone())
                    }
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
                            Instruction::Source(Input::Variable(
                                *variables.get(var).unwrap(),
                                var.clone(),
                            ))
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

    pub fn eval(&mut self, variables: &[N::AsArg<'_>]) -> N {
        self.stack.clear();
        for instr in &self.instructions {
            let mut argnum = self.stack.len();
            macro_rules! get {
                ($inp: expr) => {
                    match $inp {
                        Input::Literal(num) => num.asarg(),
                        Input::Variable(var, _) => variables[*var].reborrow(),
                        Input::Memory => {
                            argnum -= 1;
                            self.stack[argnum].asarg()
                        }
                    }
                };
            }
            let result = match &instr {
                Instruction::Source(input) => match input {
                    Input::Literal(num) => *num,
                    Input::Variable(var, _) => variables[*var].to_owned(),
                    Input::Memory => self.stack.pop().unwrap(),
                },
                Instruction::BiOperation(opr, lhs, rhs) => opr.eval(get!(lhs), get!(rhs)),
                Instruction::UnOperation(opr, val) => opr.eval(get!(val)),
                Instruction::NFSingle(func, input, _) => func(get!(input)),
                Instruction::NFDual(func, inp1, inp2, _) => func(get!(inp1), get!(inp2)),
                Instruction::NFFlexible(func, arg_count, _) => {
                    argnum -= *arg_count as usize;
                    func(&self.stack[argnum..])
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
                    argnum -= *arg_count as usize;
                    func(&self.stack[argnum..])
                }
            };
            self.stack.truncate(argnum);
            self.stack.push(result);
        }
        self.stack.pop().unwrap()
    }
}

impl<N, V, F> MathAssembly<'_, N, V, F>
where
    N: for<'b> MathEvalNumber<AsArg<'b> = N> + Copy,
    V: VariableIdentifier,
    F: FunctionIdentifier,
{
    pub fn eval_copy(&mut self, variables: &[N]) -> N {
        self.stack.clear();
        for instr in &self.instructions {
            macro_rules! get {
                ($inp: expr) => {
                    match $inp {
                        Input::Literal(num) => *num,
                        Input::Variable(var, _) => variables[*var],
                        Input::Memory => self.stack.pop().unwrap(),
                    }
                };
            }
            let result = match &instr {
                Instruction::Source(input) => get!(input),
                Instruction::BiOperation(opr, lhs, rhs) => opr.eval(get!(lhs), get!(rhs)),
                Instruction::UnOperation(opr, val) => opr.eval(get!(val)),
                Instruction::NFSingle(func, input, _) => func(get!(input)),
                Instruction::NFDual(func, inp1, inp2, _) => func(get!(inp1), get!(inp2)),
                Instruction::NFFlexible(func, arg_count, _) => {
                    let argnum = self.stack.len() - *arg_count as usize;
                    let res = func(&self.stack[argnum..]);
                    self.stack.truncate(argnum);
                    res
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
                    let argnum = self.stack.len() - *arg_count as usize;
                    let res = func(&self.stack[argnum..]);
                    self.stack.truncate(argnum);
                    res
                }
            };
            self.stack.push(result);
        }
        self.stack.pop().unwrap()
    }
}

#[cfg(test)]
mod test {
    use crate::{
        asm::{CFPointer, Input, Instruction},
        number::{MathEvalNumber, NativeFunction},
        syntax::{BiOperation, UnOperation},
        ParsingError,
    };

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    enum TestVar {
        X,
        Y,
        T,
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    enum TestFunc {
        Sigmoid,
        ISqrt,
        Dist3,
        Dist2,
        Mean,
    }

    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }
    fn isqrt(x: f64, y: f64) -> f64 {
        (x * x + y * y).sqrt()
    }
    fn dist3(x: f64, y: f64, z: f64) -> f64 {
        (x * x + y * y + z * z).sqrt()
    }
    fn dist2(x1: f64, y1: f64, x2: f64, y2: f64) -> f64 {
        ((x2 - x1).powi(2) + (y2 - y1).powi(2)).sqrt()
    }
    fn mean(nums: &[f64]) -> f64 {
        nums.iter().sum::<f64>() / nums.len() as f64
    }

    fn parse(
        input: &str,
    ) -> Result<Vec<Instruction<'static, f64, TestVar, TestFunc>>, ParsingError> {
        crate::parse(
            input,
            |inp| if inp == "c" { Some(299792458.0) } else { None },
            |inp| match inp {
                "sigmoid" => Some((TestFunc::Sigmoid, 1, Some(1))),
                "isqrt" => Some((TestFunc::ISqrt, 2, Some(2))),
                "dist3" => Some((TestFunc::Dist3, 3, Some(3))),
                "dist2" => Some((TestFunc::Dist2, 4, Some(4))),
                "mean" => Some((TestFunc::Mean, 2, Some(2))),
                _ => None,
            },
            |inp| match inp {
                "x" => Some(TestVar::X),
                "y" => Some(TestVar::Y),
                "t" => Some(TestVar::T),
                _ => None,
            },
            |func| match func {
                TestFunc::Sigmoid => CFPointer::Single(&sigmoid),
                TestFunc::ISqrt => CFPointer::Dual(&isqrt),
                TestFunc::Dist3 => CFPointer::Triple(&dist3),
                TestFunc::Dist2 => CFPointer::Quad(&dist2),
                TestFunc::Mean => CFPointer::Flexible(&mean),
            },
            false,
            &[TestVar::X, TestVar::Y, TestVar::T],
        )
        .map(|ma| ma.instructions)
    }

    #[test]
    fn test_new_mathasm() {
        assert_eq!(
            parse("1"),
            Ok(vec![Instruction::Source(Input::Literal(1.0))])
        );
        assert_eq!(
            parse("x"),
            Ok(vec![Instruction::Source(Input::Variable(0, TestVar::X))])
        );
        assert_eq!(
            parse("y+7"),
            Ok(vec![Instruction::BiOperation(
                BiOperation::Add,
                Input::Variable(1, TestVar::Y),
                Input::Literal(7.0)
            )])
        );
        assert_eq!(
            parse("t!"),
            Ok(vec![Instruction::UnOperation(
                UnOperation::Fac,
                Input::Variable(2, TestVar::T),
            )])
        );
        assert_eq!(
            parse("sin(1)"),
            Ok(vec![Instruction::<'_, f64, TestVar, TestFunc>::NFSingle(
                f64::sin,
                Input::Literal(1.0),
                NativeFunction::Sin,
            )])
        );
        assert_eq!(
            parse("log(x, 5)"),
            Ok(vec![Instruction::<'_, f64, TestVar, TestFunc>::NFDual(
                f64::log,
                Input::Variable(0, TestVar::X),
                Input::Literal(5.0),
                NativeFunction::Log
            )])
        );
        assert_eq!(
            parse("max(x, y, t, 0)"),
            Ok(vec![
                Instruction::Source(Input::Variable(0, TestVar::X)),
                Instruction::Source(Input::Variable(1, TestVar::Y)),
                Instruction::Source(Input::Variable(2, TestVar::T)),
                Instruction::Source(Input::Literal(0.0)),
                Instruction::<'_, f64, TestVar, TestFunc>::NFFlexible(
                    <f64 as MathEvalNumber>::max,
                    4,
                    NativeFunction::Max
                )
            ])
        );
        assert_eq!(
            parse("sigmoid(x)"),
            Ok(vec![Instruction::CFSingle(
                &sigmoid,
                Input::Variable(0, TestVar::X),
                TestFunc::Sigmoid
            )])
        );
        assert_eq!(
            parse("isqrt(x, y)"),
            Ok(vec![Instruction::CFDual(
                &isqrt,
                Input::Variable(0, TestVar::X),
                Input::Variable(1, TestVar::Y),
                TestFunc::ISqrt
            )])
        );
        assert_eq!(
            parse("dist3(x, y, 3)"),
            Ok(vec![Instruction::CFTriple(
                &dist3,
                Input::Variable(0, TestVar::X),
                Input::Variable(1, TestVar::Y),
                Input::Literal(3.0),
                TestFunc::Dist3
            )])
        );
        assert_eq!(
            parse("dist2(x, y, 3, 4)"),
            Ok(vec![Instruction::CFQuad(
                &dist2,
                Input::Variable(0, TestVar::X),
                Input::Variable(1, TestVar::Y),
                Input::Literal(3.0),
                Input::Literal(4.0),
                TestFunc::Dist2
            )])
        );
        assert_eq!(
            parse("mean(x, 2)"),
            Ok(vec![
                Instruction::Source(Input::Variable(0, TestVar::X)),
                Instruction::Source(Input::Literal(2.0)),
                Instruction::CFFlexible(&mean, 2, TestFunc::Mean)
            ])
        );
    }
}
