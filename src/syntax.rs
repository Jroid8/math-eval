use std::fmt::{Display, Debug};

use crate::number::{MathEvalNumber, NativeFunction};
use crate::tokenizer::token_tree::{TokenNode, TokenTree};
use crate::tree_utils::{construct, Tree};
use indextree::NodeId;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvaluationError<N, C> {
    DivisionByZero,
    NumberTypeSpecific(N),
    Custom(C),
}

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

impl Display for BiOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_char())
    }
}

pub trait VariableIdentifier: Clone + Debug {
    fn parse(input: &str) -> Option<Self>;
}

impl VariableIdentifier for () {
    fn parse(_: &str) -> Option<Self> {
        None
    }
}

pub trait FunctionIdentifier: Copy + Eq {
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
pub struct SyntaxTree<N: MathEvalNumber, V: VariableIdentifier, F: FunctionIdentifier>(pub Tree<SyntaxNode<N,V,F>>);

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
                            call_stack.push(((current_node, None, None), None));
                            Ok(None)
                        }
                        TokenNode::Function(func) => match NativeFunction::parse(func)
                            .map(|nf| (SyntaxNode::NativeFunction(nf), 1, None))
                            .or_else(|| {
                                F::parse(func).map(|cf| {
                                    (
                                        SyntaxNode::CustomFunction(cf),
                                        cf.minimum_arg_count(),
                                        cf.maximum_arg_count(),
                                    )
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
                                            .map(|(_, id)| ((id, None, None), None)),
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
                                    call_stack.push(((token_node, Some(start), Some(index)), None));
                                    call_stack
                                        .push(((token_node, Some(index + 1), Some(end)), None));
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
                                        call_stack
                                            .push(((token_node, Some(start + 1), Some(end)), None));
                                        Ok(None)
                                    }
                                    BiOperation::Sub => {
                                        call_stack
                                            .push(((token_node, Some(start + 1), Some(end)), None));
                                        Ok(Some(SyntaxNode::UnOperation(UnOperation::Neg)))
                                    }
                                    _ => Err((SyntaxError::MisplacedOperator, token_node)),
                                }
                            } else if index == end {
                                Err((SyntaxError::MisplacedOperator, token_node))
                            } else {
                                call_stack.push(((token_node, Some(start), Some(index - 1)), None));
                                call_stack.push(((token_node, Some(index + 1), Some(end)), None));
                                Ok(Some(SyntaxNode::BiOperation(opr)))
                            };
                        }
                    }
                    if end - start == 1
                        && *arena[token_node.children(arena).nth(end - 1).unwrap()].get()
                            == TokenNode::Operation('!')
                    {
                        call_stack.push(((token_node, Some(start), Some(end - 1)), None));
                        Ok(Some(SyntaxNode::UnOperation(UnOperation::Fac)))
                    } else {
                        Err((SyntaxError::MisplacedOperator, token_node))
                    }
                }
            },
            None,
        )
        .map(|(arena, node)| SyntaxTree(Tree { arena, root: node }))
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
}
