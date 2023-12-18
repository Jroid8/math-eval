use std::fmt::Debug;

use crate::number::{MathEvalNumber, NativeFunction};
use crate::optimizations::MathAssembly;
use crate::tokenizer::token_tree::{TokenNode, TokenTree};
use crate::tree_utils::{construct, Tree};
use indextree::{NodeEdge, NodeId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaluationError<N, C> {
    DivisionByZero,
    NumberTypeSpecific(N),
    Custom(C),
}

#[derive(Debug, Clone, Copy)]
pub struct DivisionByZero;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnOperation {
    Fac,
    Neg,
}

impl UnOperation {
    pub fn eval<N: MathEvalNumber>(self, value: N) -> Result<N, N::Error> {
        match self {
            UnOperation::Fac => value.factorial(),
            UnOperation::Neg => Ok(-value),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BiOperation {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Mod,
}

const ALL_BIOPERATION_ORDERED: [BiOperation; 6] = [
    BiOperation::Add,
    BiOperation::Sub,
    BiOperation::Mul,
    BiOperation::Div,
    BiOperation::Mod,
    BiOperation::Pow,
];

impl BiOperation {
    pub fn parse(input: char) -> Option<BiOperation> {
        match input {
            '+' => Some(BiOperation::Add),
            '-' => Some(BiOperation::Sub),
            '*' => Some(BiOperation::Mul),
            '/' => Some(BiOperation::Div),
            '^' => Some(BiOperation::Pow),
            '%' => Some(BiOperation::Mod),
            _ => None,
        }
    }
    pub fn eval<N: MathEvalNumber>(self, lhs: N, rhs: N) -> Result<N, DivisionByZero> {
        match self {
            BiOperation::Add => Ok(lhs + rhs),
            BiOperation::Sub => Ok(lhs - rhs),
            BiOperation::Mul => Ok(lhs * rhs),
            BiOperation::Div => {
                if rhs == 0.0.into() {
                    Err(DivisionByZero)
                } else {
                    Ok(lhs / rhs)
                }
            }
            BiOperation::Pow => Ok(lhs.pow(rhs)),
            BiOperation::Mod => Ok(lhs.modulo(rhs)),
        }
    }
    pub fn as_char(self) -> char {
        match self {
            BiOperation::Add => '+',
            BiOperation::Sub => '-',
            BiOperation::Mul => '*',
            BiOperation::Div => '/',
            BiOperation::Pow => '^',
            BiOperation::Mod => '%',
        }
    }
}

pub trait VariableIdentifier: Clone {
    fn parse(input: &str) -> Option<Self>;
}

impl VariableIdentifier for () {
    fn parse(_: &str) -> Option<Self> {
        None
    }
}

pub trait FunctionIdentifier: Clone + Eq {
    type Error;

    fn parse(input: &str) -> Option<Self>;
    fn minimum_arg_count(&self) -> u8;
    fn maximum_arg_count(&self) -> Option<u8>;
}

impl FunctionIdentifier for () {
    type Error = ();

    fn parse(_: &str) -> Option<Self> {
        None
    }

    fn minimum_arg_count(&self) -> u8 {
        0
    }

    fn maximum_arg_count(&self) -> Option<u8> {
        None
    }
}

// 72 bytes in size
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SyntaxNode<N, V, F>
where
    N: MathEvalNumber,
    V: VariableIdentifier,
    F: FunctionIdentifier,
{
    Number(N),
    Variable(V),
    BiOperation(BiOperation),
    UnOperation(UnOperation),
    NativeFunction(NativeFunction),
    CustomFunction(F),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SyntaxError {
    NumberParsingError,
    MisplacedOperator,
    UnknownVariableOrConstant,
    UnknownFunction,
    NotEnoughArguments,
    TooManyArguments,
}

#[derive(Clone, Debug)]
pub struct SyntaxTree<N: MathEvalNumber, V: VariableIdentifier, F: FunctionIdentifier>(
    pub Tree<SyntaxNode<N, V, F>>,
);

impl<V, N, F> SyntaxTree<N, V, F>
where
    N: MathEvalNumber,
    V: VariableIdentifier,
    F: FunctionIdentifier,
{
    pub fn new(
        token_tree: &TokenTree<'_>,
        custom_constant_parser: impl Fn(&str) -> Option<N>,
    ) -> Result<SyntaxTree<N, V, F>, (SyntaxError, NodeId)> {
        let (arena, root) = (&token_tree.0.arena, token_tree.0.root);
        construct::<
            (NodeId, Option<usize>, Option<usize>),
            SyntaxNode<N, V, F>,
            (SyntaxError, NodeId),
        >(
            (root, None, None),
            |(token_node, start, end), call_stack| {
                let children_count = token_node.children(arena).count();
                let start = start.unwrap_or(0);
                let end = end.unwrap_or(children_count - 1);
                let children = || {
                    token_node
                        .reverse_children(arena)
                        .enumerate()
                        .map(|(i, n)| (children_count - 1 - i, n))
                        .skip(children_count - 1 - end)
                        .take(end - start + 1)
                };
                if start == end {
                    let current_node = token_node.children(arena).nth(start).unwrap();
                    match arena[current_node].get() {
                        TokenNode::Number(num) => num
                            .parse::<N>()
                            .map(|n| Some(SyntaxNode::Number(n)))
                            .map_err(|_| SyntaxError::NumberParsingError),
                        TokenNode::Operation(_) => Err(SyntaxError::MisplacedOperator), // operations shouldn't end up here
                        TokenNode::Variable(var) => N::parse_constant(var)
                            .map(|c| SyntaxNode::Number(c))
                            .or_else(|| custom_constant_parser(var).map(|c| SyntaxNode::Number(c)))
                            .or_else(|| V::parse(var).map(|v| SyntaxNode::Variable(v)))
                            .map(Some)
                            .ok_or(SyntaxError::UnknownVariableOrConstant),
                        TokenNode::Parentheses => {
                            call_stack.push((current_node, None, None));
                            Ok(None)
                        }
                        TokenNode::Function(func) => match NativeFunction::parse(func)
                            .map(|nf| (SyntaxNode::NativeFunction(nf), 1, None))
                            .or_else(|| {
                                F::parse(func).map(|cf| {
                                    let (min, max) =
                                        (cf.minimum_arg_count(), cf.maximum_arg_count());
                                    (SyntaxNode::CustomFunction(cf), min, max)
                                })
                            }) {
                            Some((f, min_args, max_args)) => {
                                let arg_count = current_node.children(arena).count();
                                if arg_count < min_args as usize {
                                    Err(SyntaxError::NotEnoughArguments)
                                } else if arg_count > 255
                                    || max_args.is_some_and(|ma| arg_count as u8 > ma)
                                {
                                    Err(SyntaxError::TooManyArguments)
                                } else {
                                    call_stack.extend(
                                        current_node
                                            .children(arena)
                                            .enumerate()
                                            .map(|(_, id)| (id, None, None)),
                                    );
                                    Ok(Some(f))
                                }
                            }
                            None => Err(SyntaxError::UnknownFunction),
                        },
                        TokenNode::Argument => unreachable!(),
                    }
                    .map_err(|e| (e, token_node))
                } else {
                    for opr in ALL_BIOPERATION_ORDERED {
                        // for detecting implied multiplications (e.g. 2pi,3x)
                        if opr == BiOperation::Mul {
                            let mut iter = children();
                            let mut last = iter.next().unwrap().1;
                            for (index, token) in iter {
                                if !matches!(arena[last].get(), TokenNode::Operation(_))
                                    && !matches!(arena[token].get(), TokenNode::Operation(_))
                                {
                                    call_stack.push((token_node, Some(start), Some(index)));
                                    call_stack.push((token_node, Some(index + 1), Some(end)));
                                    return Ok(Some(SyntaxNode::BiOperation(BiOperation::Mul)));
                                }
                                last = token;
                            }
                        }
                        // in a syntax tree, the top item is the evaluated first, so it should be the last in the order of operations.
                        // rev() is used to pick last operation so the constructed tree has the correct order
                        if let Some((index, _)) = children().find(|(_, c)| match arena[*c].get() {
                            TokenNode::Operation(oprchar) => *oprchar == opr.as_char(),
                            _ => false,
                        }) {
                            return if index == 0 {
                                match opr {
                                    BiOperation::Add => {
                                        call_stack.push((token_node, Some(start + 1), Some(end)));
                                        Ok(None)
                                    }
                                    BiOperation::Sub => {
                                        call_stack.push((token_node, Some(start + 1), Some(end)));
                                        Ok(Some(SyntaxNode::UnOperation(UnOperation::Neg)))
                                    }
                                    _ => Err((SyntaxError::MisplacedOperator, token_node)),
                                }
                            } else if index == end {
                                Err((SyntaxError::MisplacedOperator, token_node))
                            } else {
                                call_stack.push((token_node, Some(start), Some(index - 1)));
                                call_stack.push((token_node, Some(index + 1), Some(end)));
                                Ok(Some(SyntaxNode::BiOperation(opr)))
                            };
                        }
                    }
                    if end - start == 1
                        && *arena[token_node.children(arena).nth(end - 1).unwrap()].get()
                            == TokenNode::Operation('!')
                    {
                        call_stack.push((token_node, Some(start), Some(end - 1)));
                        Ok(Some(SyntaxNode::UnOperation(UnOperation::Fac)))
                    } else {
                        Err((SyntaxError::MisplacedOperator, token_node))
                    }
                }
            },
            None,
        )
        .map(|tree| SyntaxTree(tree))
    }

    pub fn to_asm<'a>(
        &self,
        function_to_pointer: impl Fn(&F) -> &'a dyn Fn(&[N]) -> Result<N, F::Error>,
    ) -> MathAssembly<'a, N, V, F> {
        MathAssembly::new(&self.0.arena, self.0.root, function_to_pointer)
    }

    pub fn aot_evaluation<'a>(
        &mut self,
        function_to_pointer: impl Fn(&F) -> &'a dyn Fn(&[N]) -> Result<N, F::Error>,
    ) -> Result<(), EvaluationError<N::Error, F::Error>>
    where
        F::Error: 'a,
    {
        let mut examin: Vec<NodeId> = Vec::new();
        for node in self.0.root.traverse(&self.0.arena) {
            if let NodeEdge::End(node) = node {
                match self.0.arena[node].get() {
                    SyntaxNode::Number(_) | SyntaxNode::Variable(_) => (),
                    _ => examin.push(node),
                }
            }
        }
        for node in examin {
            if node.children(&self.0.arena).all(|c| self.is_number(c)) {
                let answer = MathAssembly::new(&self.0.arena, node, &function_to_pointer)
                    .eval(|_| unreachable!())?;
                *self.0.arena[node].get_mut() = SyntaxNode::Number(answer);
                while let Some(c) = self.0.arena[node].first_child() {
                    c.remove(&mut self.0.arena);
                }
            }
        }
        Ok(())
    }

    fn is_number(&self, node: NodeId) -> bool {
        matches!(self.0.arena[node].get(), SyntaxNode::Number(_))
    }

    pub fn displacing_simplification(&mut self) -> Result<(), DivisionByZero> {
        self._displacing_simplification(BiOperation::Add, BiOperation::Sub, 0.0.into())
            .and_then(|_| {
                self._displacing_simplification(BiOperation::Mul, BiOperation::Div, 1.0.into())
            })
    }

    fn _displacing_simplification(
        &mut self,
        pos: BiOperation,
        neg: BiOperation,
        inital_value: N,
    ) -> Result<(), DivisionByZero> {
        let is_targeting_opr = |node: NodeId| matches!(self.0.arena[node].get(), SyntaxNode::BiOperation(opr) if *opr == pos || *opr == neg);
        let mut found: Vec<NodeId> = Vec::new();
        let mul_opr = |target: BiOperation, side: usize, parent: BiOperation| {
            if side == 0 {
                parent
            } else if target == parent {
                pos
            } else {
                neg
            }
        };
        for node in self.0.root.traverse(&self.0.arena) {
            if let NodeEdge::End(upper) = node {
                if is_targeting_opr(upper)
                    && upper.children(&self.0.arena).all(|lower| {
                        is_targeting_opr(lower)
                            && lower
                                .children(&self.0.arena)
                                .any(|lowest| self.is_number(lowest))
                            || self.is_number(lower)
                    })
                {
                    found.push(upper);
                }
            }
        }
        for upper in found {
            let SyntaxNode::BiOperation(upper_opr) = self.0.arena[upper].get() else {
                panic!();
            };
            let mut symbols: [Option<(NodeId, bool)>; 2] = [None, None];
            let mut lhs = inital_value;
            for (upper_side, lower) in upper.children(&self.0.arena).enumerate() {
                match self.0.arena[lower].get() {
                    SyntaxNode::BiOperation(lower_opr) => {
                        for (lower_side, lowest) in lower.children(&self.0.arena).enumerate() {
                            let opr = mul_opr(
                                *lower_opr,
                                lower_side,
                                if upper_side == 0 { pos } else { *upper_opr },
                            );
                            match self.0.arena[lowest].get() {
                                SyntaxNode::Number(value) => lhs = opr.eval(lhs, *value)?,
                                _ => {
                                    symbols[symbols[0].is_some() as usize] =
                                        Some((lowest, opr == neg))
                                }
                            }
                        }
                    }
                    SyntaxNode::Number(value) => {
                        lhs = (mul_opr(*upper_opr, upper_side, pos)).eval(lhs, *value)?
                    }
                    _ => panic!(),
                }
            }
            let symb1 = symbols[0].unwrap();
            symb1.0.detach(&mut self.0.arena);
            if let Some((sym, _)) = symbols[1] {
                sym.detach(&mut self.0.arena);
            }
            while let Some(child) = upper.children(&self.0.arena).next() {
                child.remove_subtree(&mut self.0.arena);
            }
            upper.append_value(SyntaxNode::Number(lhs), &mut self.0.arena);
            if let Some(symb2) = symbols[1] {
                if symb1.1 == symb2.1 {
                    *self.0.arena[upper].get_mut() =
                        SyntaxNode::BiOperation(if symb1.1 { neg } else { pos });
                    let lower = upper.append_value(SyntaxNode::BiOperation(pos), &mut self.0.arena);
                    lower.append(symb1.0, &mut self.0.arena);
                    lower.append(symb2.0, &mut self.0.arena);
                } else {
                    *self.0.arena[upper].get_mut() = SyntaxNode::BiOperation(pos);
                    let lower = upper.append_value(SyntaxNode::BiOperation(neg), &mut self.0.arena);
                    if symb2.1 {
                        lower.append(symb1.0, &mut self.0.arena);
                        lower.append(symb2.0, &mut self.0.arena);
                    } else {
                        lower.append(symb2.0, &mut self.0.arena);
                        lower.append(symb1.0, &mut self.0.arena);
                    }
                }
            } else {
                *self.0.arena[upper].get_mut() =
                    SyntaxNode::BiOperation(if symb1.1 { neg } else { pos });
                upper.append(symb1.0, &mut self.0.arena);
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::tokenizer::{token_stream::TokenStream, token_tree::TokenTree};
    use crate::tree_utils::VecTree::{self, Leaf};

    #[derive(Debug, PartialEq, Eq, Clone, Copy)]
    enum CustomVar {
        X,
        Y,
        T,
    }

    impl VariableIdentifier for CustomVar {
        fn parse(input: &str) -> Option<Self> {
            match input {
                "x" => Some(Self::X),
                "y" => Some(Self::Y),
                "t" => Some(Self::T),
                _ => None,
            }
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum CustomFunc {
        Dot,
    }

    impl FunctionIdentifier for CustomFunc {
        type Error = ();

        fn parse(input: &str) -> Option<Self> {
            match input {
                "dot" => Some(CustomFunc::Dot),
                _ => None,
            }
        }

        fn minimum_arg_count(&self) -> u8 {
            2
        }

        fn maximum_arg_count(&self) -> Option<u8> {
            Some(2)
        }
    }

    macro_rules! branch {
        ($node:expr, $($children:expr),+) => {
            VecTree::Branch($node,vec![$($children),+])
        };
    }

    #[test]
    fn test_syntaxify() {
        macro_rules! syntaxify {
            ($input:literal) => {
                SyntaxTree::<f64, CustomVar, ()>::new(
                    &TokenTree::new(&TokenStream::new($input).unwrap()).unwrap(),
                    |_| None,
                )
                .map(|syntree| VecTree::new(&syntree.0.arena, syntree.0.root))
            };
        }
        assert_eq!(syntaxify!("0"), Ok(Leaf(SyntaxNode::Number(0.0))));
        assert_eq!(syntaxify!("(0)"), Ok(Leaf(SyntaxNode::Number(0.0))));
        assert_eq!(syntaxify!("((0))"), Ok(Leaf(SyntaxNode::Number(0.0))));
        assert_eq!(
            syntaxify!("1+1"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Add),
                Leaf(SyntaxNode::Number(1.0)),
                Leaf(SyntaxNode::Number(1.0))
            ))
        );
        assert_eq!(
            syntaxify!("-12"),
            Ok(branch!(
                SyntaxNode::UnOperation(UnOperation::Neg),
                Leaf(SyntaxNode::Number(12.0))
            ))
        );
        assert_eq!(
            syntaxify!("8*3+1"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Add),
                branch!(
                    SyntaxNode::BiOperation(BiOperation::Mul),
                    Leaf(SyntaxNode::Number(8.0)),
                    Leaf(SyntaxNode::Number(3.0))
                ),
                Leaf(SyntaxNode::Number(1.0))
            ))
        );
        assert_eq!(
            syntaxify!("12/3/2"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Div),
                branch!(
                    SyntaxNode::BiOperation(BiOperation::Div),
                    Leaf(SyntaxNode::Number(12.0)),
                    Leaf(SyntaxNode::Number(3.0))
                ),
                Leaf(SyntaxNode::Number(2.0))
            ))
        );
        assert_eq!(
            syntaxify!("8*(3+1)"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Mul),
                Leaf(SyntaxNode::Number(8.0)),
                branch!(
                    SyntaxNode::BiOperation(BiOperation::Add),
                    Leaf(SyntaxNode::Number(3.0)),
                    Leaf(SyntaxNode::Number(1.0))
                )
            ))
        );
        assert_eq!(
            syntaxify!("8*3^2+1"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Add),
                branch!(
                    SyntaxNode::BiOperation(BiOperation::Mul),
                    Leaf(SyntaxNode::Number(8.0)),
                    branch!(
                        SyntaxNode::BiOperation(BiOperation::Pow),
                        Leaf(SyntaxNode::Number(3.0)),
                        Leaf(SyntaxNode::Number(2.0))
                    )
                ),
                Leaf(SyntaxNode::Number(1.0))
            ))
        );
        assert_eq!(
            syntaxify!("2x"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Mul),
                Leaf(SyntaxNode::Number(2.0)),
                Leaf(SyntaxNode::Variable(CustomVar::X))
            ))
        );
        assert_eq!(
            syntaxify!("sin(14)"),
            Ok(branch!(
                SyntaxNode::NativeFunction(NativeFunction::Sin),
                Leaf(SyntaxNode::Number(14.0))
            ))
        );
        assert_eq!(
            syntaxify!("max(2, x, 8y, x*y+1)"),
            Ok(branch!(
                SyntaxNode::NativeFunction(NativeFunction::Max),
                Leaf(SyntaxNode::Number(2.0)),
                Leaf(SyntaxNode::Variable(CustomVar::X)),
                branch!(
                    SyntaxNode::BiOperation(BiOperation::Mul),
                    Leaf(SyntaxNode::Number(8.0)),
                    Leaf(SyntaxNode::Variable(CustomVar::Y))
                ),
                branch!(
                    SyntaxNode::BiOperation(BiOperation::Add),
                    branch!(
                        SyntaxNode::BiOperation(BiOperation::Mul),
                        Leaf(SyntaxNode::Variable(CustomVar::X)),
                        Leaf(SyntaxNode::Variable(CustomVar::Y))
                    ),
                    Leaf(SyntaxNode::Number(1.0))
                )
            ))
        );
    }

    #[test]
    fn test_aot_evaluation() {
        macro_rules! compare {
            ($i1:literal, $i2:literal) => {
                let mut syn1 = SyntaxTree::<f64, CustomVar, ()>::new(
                    &TokenTree::new(&TokenStream::new($i1).unwrap()).unwrap(),
                    |_| None,
                )
                .unwrap();
                syn1.aot_evaluation(|_| &|_| Ok(0.0)).unwrap();
                let syn2 = SyntaxTree::<f64, CustomVar, ()>::new(
                    &TokenTree::new(&TokenStream::new($i2).unwrap()).unwrap(),
                    |_| None,
                )
                .unwrap();
                assert_eq!(
                    format!("{:?}", syn1.0.root.debug_pretty_print(&syn1.0.arena)),
                    format!("{:?}", syn2.0.root.debug_pretty_print(&syn2.0.arena))
                );
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
                let mut syn1 = SyntaxTree::<f64, CustomVar, ()>::new(
                    &TokenTree::new(&TokenStream::new($i1).unwrap()).unwrap(),
                    |_| None,
                )
                .unwrap();
                syn1.displacing_simplification().unwrap();
                let syn2 = SyntaxTree::<f64, CustomVar, ()>::new(
                    &TokenTree::new(&TokenStream::new($i2).unwrap()).unwrap(),
                    |_| None,
                )
                .unwrap();
                assert_eq!(
                    format!("{:?}", syn1.0.root.debug_pretty_print(&syn1.0.arena)),
                    format!("{:?}", syn2.0.root.debug_pretty_print(&syn2.0.arena))
                );
            };
        }
        compare!("x/1/8", "0.125*x");
        compare!("(x/16)/(y*4)", "0.015625*(x/y)");
        compare!("(7/x)/(y/2)", "14/(x*y)");
        compare!("(x/4)/(4/y)", "0.0625*(x*y)");
        compare!("10-x+12", "22-x");
        compare!("x*pi*2", "6.283185307179586*x");
    }
}
