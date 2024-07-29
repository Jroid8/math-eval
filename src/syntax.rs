use std::fmt::{Debug, Display};
use std::ops::RangeInclusive;

use crate::asm::{CFPointer, MathAssembly};
use crate::number::{MathEvalNumber, NativeFunction};
use crate::tokenizer::token_tree::{TokenNode, TokenTree};
use crate::tree_utils::{construct, Tree};
use crate::{ParsingError, ParsingErrorKind};
use indextree::{NodeEdge, NodeId};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnOperation {
    Fac,
    Neg,
}

impl UnOperation {
    pub fn parse(input: char) -> Option<Self> {
        match input {
            '!' => Some(UnOperation::Fac),
            '-' => Some(UnOperation::Neg),
            _ => None,
        }
    }

    pub fn eval<N: MathEvalNumber>(self, value: N) -> N {
        match self {
            UnOperation::Fac => value.factorial(),
            UnOperation::Neg => -value,
        }
    }

    pub fn as_char(&self) -> char {
        match self {
            UnOperation::Fac => '!',
            UnOperation::Neg => '-',
        }
    }
}

impl Display for UnOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_char())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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
    pub fn eval<N: MathEvalNumber>(self, lhs: N, rhs: N) -> N {
        match self {
            BiOperation::Add => lhs + rhs,
            BiOperation::Sub => lhs - rhs,
            BiOperation::Mul => lhs * rhs,
            BiOperation::Div => lhs / rhs,
            BiOperation::Pow => lhs.powf(rhs),
            BiOperation::Mod => lhs % rhs,
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

// 72 bytes in size
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SyntaxNode<N, V, F>
where
    N: MathEvalNumber,
    V: Clone,
    F: Clone,
{
    Number(N),
    Variable(V),
    BiOperation(BiOperation),
    UnOperation(UnOperation),
    NativeFunction(NativeFunction),
    CustomFunction(F),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SyntaxErrorKind {
    NumberParsingError,
    MisplacedOperator,
    UnknownVariableOrConstant,
    UnknownFunction,
    NotEnoughArguments,
    TooManyArguments,
    MisplacedToken,
    EmptyParenthesis,
}

fn tokennode2range(
    input: &str,
    token_tree: &TokenTree<'_>,
    target: NodeId,
) -> RangeInclusive<usize> {
    if target.ancestors(&token_tree.0.arena).nth(1).is_none() {
        panic!("Attempted to report the root node as the cause of error, good luck fixing this bug")
    }
    let mut index = 0;
    macro_rules! count_space {
        () => {
            while input.chars().nth(index).unwrap().is_whitespace() {
                index += 1
            }
        };
    }
    for node in token_tree.0.root.traverse(&token_tree.0.arena).skip(1) {
        match node {
            NodeEdge::Start(node) => {
                if *token_tree.0.arena[node].get() != TokenNode::Argument {
                    count_space!();
                }
                let old = index;
                index += match token_tree.0.arena[node].get() {
                    TokenNode::Number(s) | TokenNode::Variable(s) => s.len(),
                    TokenNode::Operation(_) => 1,
                    TokenNode::Parentheses => 1,
                    TokenNode::Function(f) => f.len() + 1,
                    TokenNode::Argument => 0,
                };
                if node == target {
                    return old..=index - 1;
                }
            }
            NodeEdge::End(node) => match token_tree.0.arena[node].get() {
                TokenNode::Argument => {
                    if node
                        .following_siblings(&token_tree.0.arena)
                        .nth(1)
                        .is_some()
                    {
                        count_space!();
                        index += 1;
                    }
                }
                TokenNode::Parentheses | TokenNode::Function(_) => {
                    count_space!();
                    index += 1
                }
                _ => (),
            },
        }
    }
    unreachable!()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SyntaxError(SyntaxErrorKind, NodeId);

impl SyntaxError {
    pub fn to_general(self, input: &str, token_tree: &TokenTree<'_>) -> ParsingError {
        ParsingError {
            at: tokennode2range(input, token_tree, self.1),
            kind: match self.0 {
                SyntaxErrorKind::NumberParsingError => ParsingErrorKind::NumberParsingError,
                SyntaxErrorKind::MisplacedOperator => ParsingErrorKind::MisplacedOperator,
                SyntaxErrorKind::UnknownVariableOrConstant => {
                    ParsingErrorKind::UnknownVariableOrConstant
                }
                SyntaxErrorKind::UnknownFunction => ParsingErrorKind::UnknownFunction,
                SyntaxErrorKind::NotEnoughArguments => ParsingErrorKind::NotEnoughArguments,
                SyntaxErrorKind::TooManyArguments => ParsingErrorKind::TooManyArguments,
                SyntaxErrorKind::MisplacedToken => ParsingErrorKind::MisplacedToken,
                SyntaxErrorKind::EmptyParenthesis => ParsingErrorKind::EmptyParenthesis,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SyntaxTree<N: MathEvalNumber, V: Clone, F: Clone>(pub Tree<SyntaxNode<N, V, F>>);

impl<V, N, F> SyntaxTree<N, V, F>
where
    N: MathEvalNumber,
    V: Clone,
    F: Clone,
{
    pub fn new(
        token_tree: &TokenTree<'_>,
        custom_constant_parser: impl Fn(&str) -> Option<N>,
        custom_function_parser: impl Fn(&str) -> Option<(F, u8, Option<u8>)>,
        custom_variable_parser: impl Fn(&str) -> Option<V>,
    ) -> Result<SyntaxTree<N, V, F>, SyntaxError> {
        let (arena, root) = (&token_tree.0.arena, token_tree.0.root);
        construct::<(NodeId, Option<usize>, Option<usize>), SyntaxNode<N, V, F>, SyntaxError>(
            (root, None, None),
            |(token_node, start, end), call_stack| {
                let children_count = token_node.children(arena).count();
                if children_count == 0 {
                    return Err(SyntaxError(SyntaxErrorKind::EmptyParenthesis, token_node));
                }
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
                            .map_err(|_| SyntaxErrorKind::NumberParsingError),
                        TokenNode::Operation(_) => Err(SyntaxErrorKind::MisplacedOperator), // operations shouldn't end up here
                        TokenNode::Variable(var) => N::parse_constant(var)
                            .map(|c| SyntaxNode::Number(c))
                            .or_else(|| custom_constant_parser(var).map(|c| SyntaxNode::Number(c)))
                            .or_else(|| {
                                custom_variable_parser(var).map(|v| SyntaxNode::Variable(v))
                            })
                            .map(Some)
                            .ok_or(SyntaxErrorKind::UnknownVariableOrConstant),
                        TokenNode::Parentheses => {
                            call_stack.push((current_node, None, None));
                            Ok(None)
                        }
                        TokenNode::Function(func) => match NativeFunction::parse(func)
                            .map(|nf| (SyntaxNode::NativeFunction(nf), 1, None))
                            .or_else(|| {
                                custom_function_parser(func)
                                    .map(|cf| (SyntaxNode::CustomFunction(cf.0), cf.1, cf.2))
                            }) {
                            Some((f, min_args, max_args)) => {
                                let arg_count = current_node.children(arena).count();
                                if arg_count < min_args as usize {
                                    Err(SyntaxErrorKind::NotEnoughArguments)
                                } else if arg_count > 255
                                    || max_args.is_some_and(|ma| arg_count as u8 > ma)
                                {
                                    Err(SyntaxErrorKind::TooManyArguments)
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
                            None => Err(SyntaxErrorKind::UnknownFunction),
                        },
                        TokenNode::Argument => unreachable!(),
                    }
                    .map_err(|e| SyntaxError(e, current_node))
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
                        if let Some((index, node)) =
                            children().find(|(i, c)| match arena[*c].get() {
                                TokenNode::Operation(oprchar) => {
                                    *oprchar == opr.as_char()
                                        && !(*oprchar == '-'
                                            && *i > start
                                            && matches!(
                                                c.preceding_siblings(arena)
                                                    .nth(1)
                                                    .map(|n| arena[n].get()),
                                                Some(TokenNode::Operation(o))
                                                    if UnOperation::parse(*o)
                                                    .filter(|o| *o != UnOperation::Neg).is_none()
                                            ))
                                }
                                _ => false,
                            })
                        {
                            return if index == start {
                                match opr {
                                    BiOperation::Add => {
                                        call_stack.push((token_node, Some(start + 1), Some(end)));
                                        Ok(None)
                                    }
                                    BiOperation::Sub => {
                                        call_stack.push((token_node, Some(start + 1), Some(end)));
                                        Ok(Some(SyntaxNode::UnOperation(UnOperation::Neg)))
                                    }
                                    _ => Err(SyntaxError(SyntaxErrorKind::MisplacedOperator, node)),
                                }
                            } else if index == end {
                                Err(SyntaxError(SyntaxErrorKind::MisplacedOperator, node))
                            } else {
                                call_stack.push((token_node, Some(start), Some(index - 1)));
                                call_stack.push((token_node, Some(index + 1), Some(end)));
                                Ok(Some(SyntaxNode::BiOperation(opr)))
                            };
                        }
                    }
                    if start + 1 == end
                        && *arena[token_node.children(arena).nth(end).unwrap()].get()
                            == TokenNode::Operation('!')
                    {
                        call_stack.push((token_node, Some(start), Some(end - 1)));
                        Ok(Some(SyntaxNode::UnOperation(UnOperation::Fac)))
                    } else {
                        Err(SyntaxError(
                            SyntaxErrorKind::MisplacedToken,
                            token_node.children(arena).nth(start).unwrap(),
                        ))
                    }
                }
            },
            None,
        )
        .map(|tree| SyntaxTree(tree).substitute_log())
    }

    pub fn to_asm<'a>(
        &self,
        function_to_pointer: impl Fn(&F) -> CFPointer<'a, N>,
    ) -> MathAssembly<'a, N, V, F> {
        MathAssembly::new(&self.0.arena, self.0.root, function_to_pointer)
    }

    pub fn aot_evaluation<'a>(&mut self, function_to_pointer: impl Fn(&F) -> CFPointer<'a, N>) {
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
                    .eval(|_| unreachable!());
                *self.0.arena[node].get_mut() = SyntaxNode::Number(answer);
                while let Some(c) = self.0.arena[node].first_child() {
                    c.remove(&mut self.0.arena);
                }
            }
        }
    }

    fn is_number(&self, node: NodeId) -> bool {
        matches!(self.0.arena[node].get(), SyntaxNode::Number(_))
    }

    pub fn displacing_simplification(&mut self) {
        self._displacing_simplification(BiOperation::Add, BiOperation::Sub, N::zero());
        self._displacing_simplification(BiOperation::Mul, BiOperation::Div, N::one());
    }

    fn _displacing_simplification(&mut self, pos: BiOperation, neg: BiOperation, inital_value: N) {
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
                                SyntaxNode::Number(value) => lhs = opr.eval(lhs, *value),
                                _ => {
                                    symbols[symbols[0].is_some() as usize] =
                                        Some((lowest, opr == neg))
                                }
                            }
                        }
                    }
                    SyntaxNode::Number(value) => {
                        lhs = (mul_opr(*upper_opr, upper_side, pos)).eval(lhs, *value)
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
    }

    fn substitute_log(mut self) -> Self {
        let mut matched_logs: Vec<(NodeId, u8)> = Vec::new();
        for node in self.0.root.traverse(&self.0.arena) {
            if let NodeEdge::Start(node) = node {
                if let SyntaxNode::NativeFunction(NativeFunction::Log) = *self.0.arena[node].get() {
                    matched_logs.push((
                        node,
                        match node.children(&self.0.arena).nth(1) {
                            Some(base) => match self.0.arena[base].get() {
                                SyntaxNode::Number(num) => {
                                    if *num == N::from(10).unwrap() {
                                        10
                                    } else if *num == N::from(2).unwrap() {
                                        2
                                    } else {
                                        continue;
                                    }
                                }
                                _ => continue,
                            },
                            None => 10,
                        },
                    ));
                }
            }
        }
        for (node, base) in matched_logs {
            *self.0.arena[node].get_mut() = SyntaxNode::NativeFunction(match base {
                10 => NativeFunction::Log10,
                2 => NativeFunction::Log2,
                _ => unreachable!(),
            });
            if let Some(base) = node.children(&self.0.arena).nth(1) {
                base.remove(&mut self.0.arena)
            }
        }
        self
    }
}

impl<V, N, F> Display for SyntaxTree<N, V, F>
where
    N: MathEvalNumber + Display,
    V: Clone + Display,
    F: Clone + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for edge in self.0.root.traverse(&self.0.arena) {
            match edge {
                NodeEdge::Start(node) => match self.0.arena[node].get() {
                    SyntaxNode::Number(num) => std::fmt::Display::fmt(num, f)?,
                    SyntaxNode::Variable(var) => var.fmt(f)?,
                    SyntaxNode::BiOperation(_) => (),
                    SyntaxNode::UnOperation(opr) => std::fmt::Display::fmt(opr, f)?,
                    SyntaxNode::NativeFunction(nf) => {
                        std::fmt::Display::fmt(nf, f)?;
                        f.write_str("(")?
                    }
                    SyntaxNode::CustomFunction(cf) => {
                        cf.fmt(f)?;
                        f.write_str("(")?
                    }
                },
                NodeEdge::End(node) => {
                    match self.0.arena[node].get() {
                        SyntaxNode::NativeFunction(_) | SyntaxNode::CustomFunction(_) => {
                            f.write_str(")")?
                        }
                        _ => (),
                    };
                    if node.following_siblings(&self.0.arena).nth(1).is_some() {
                        match node
                            .ancestors(&self.0.arena)
                            .nth(1)
                            .map(|p| self.0.arena[p].get())
                        {
                            Some(SyntaxNode::NativeFunction(_) | SyntaxNode::CustomFunction(_)) => {
                                f.write_str(", ")?
                            }
                            Some(SyntaxNode::BiOperation(opr)) => std::fmt::Display::fmt(&opr, f)?,
                            _ => (),
                        }
                    }
                }
            };
        }
        Ok(())
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::tokenizer::{token_stream::TokenStream, token_tree::TokenTree};
    use crate::tree_utils::VecTree::{self, Leaf};

    macro_rules! branch {
        ($node:expr, $($children:expr),+) => {
            VecTree::Branch($node,vec![$($children),+])
        };
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    enum TestVar {
        X,
        Y,
        T,
    }

    impl Display for TestVar {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                TestVar::X => f.write_str("x"),
                TestVar::Y => f.write_str("y"),
                TestVar::T => f.write_str("t"),
            }
        }
    }

    #[derive(Clone, Debug, PartialEq, Eq)]
    enum TestFunc {
        Dist,
        Mean,
    }

    impl Display for TestFunc {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                TestFunc::Dist => f.write_str("dist"),
                TestFunc::Mean => f.write_str("mean"),
            }
        }
    }

    fn parse(input: &str) -> Result<SyntaxTree<f64, TestVar, TestFunc>, ParsingError> {
        let token_stream = TokenStream::new(input).map_err(|e| e.to_general())?;
        let token_tree =
            TokenTree::new(&token_stream).map_err(|e| e.to_general(input, &token_stream))?;
        SyntaxTree::new(
            &token_tree,
            |inp| match inp {
                "c" => Some(299792458.0),
                _ => None,
            },
            |input| match input {
                "dist" => Some((TestFunc::Dist, 2, Some(2))),
                "mean" => Some((TestFunc::Mean, 2, None)),
                _ => None,
            },
            |input| match input {
                "x" => Some(TestVar::X),
                "y" => Some(TestVar::Y),
                "t" => Some(TestVar::T),
                _ => None,
            },
        )
        .map_err(|e| e.to_general(input, &token_tree))
    }

    #[test]
    fn test_syntaxify() {
        fn syntaxify(
            input: &str,
        ) -> Result<VecTree<SyntaxNode<f64, TestVar, TestFunc>>, ParsingError> {
            parse(input).map(|st| VecTree::new(&st.0.arena, st.0.root))
        }
        assert_eq!(syntaxify("0"), Ok(Leaf(SyntaxNode::Number(0.0))));
        assert_eq!(syntaxify("(0)"), Ok(Leaf(SyntaxNode::Number(0.0))));
        assert_eq!(syntaxify("((0))"), Ok(Leaf(SyntaxNode::Number(0.0))));
        assert_eq!(
            syntaxify("pi"),
            Ok(Leaf(SyntaxNode::Number(std::f64::consts::PI)))
        );
        assert_eq!(
            syntaxify("1+1"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Add),
                Leaf(SyntaxNode::Number(1.0)),
                Leaf(SyntaxNode::Number(1.0))
            ))
        );
        assert_eq!(
            syntaxify("-12"),
            Ok(branch!(
                SyntaxNode::UnOperation(UnOperation::Neg),
                Leaf(SyntaxNode::Number(12.0))
            ))
        );
        assert_eq!(
            syntaxify("8*3+1"),
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
            syntaxify("12/3/2"),
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
            syntaxify("8*(3+1)"),
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
            syntaxify("8*3^2+1"),
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
            syntaxify("2x"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Mul),
                Leaf(SyntaxNode::Number(2.0)),
                Leaf(SyntaxNode::Variable(TestVar::X))
            ))
        );
        assert_eq!(
            syntaxify("sin(14)"),
            Ok(branch!(
                SyntaxNode::NativeFunction(NativeFunction::Sin),
                Leaf(SyntaxNode::Number(14.0))
            ))
        );
        assert_eq!(
            syntaxify("dist(c,2344.0)"),
            Ok(branch!(
                SyntaxNode::CustomFunction(TestFunc::Dist),
                Leaf(SyntaxNode::Number(299792458.0)),
                Leaf(SyntaxNode::Number(2344.0))
            ))
        );
        assert_eq!(
            syntaxify("log2(8)"),
            Ok(branch!(
                SyntaxNode::NativeFunction(NativeFunction::Log2),
                Leaf(SyntaxNode::Number(8.0))
            ))
        );
        assert_eq!(
            syntaxify("log(100)"),
            Ok(branch!(
                SyntaxNode::NativeFunction(NativeFunction::Log10),
                Leaf(SyntaxNode::Number(100.0))
            ))
        );
        assert_eq!(
            syntaxify("sin(cos(0))"),
            Ok(branch!(
                SyntaxNode::NativeFunction(NativeFunction::Sin),
                branch!(
                    SyntaxNode::NativeFunction(NativeFunction::Cos),
                    Leaf(SyntaxNode::Number(0.0))
                )
            ))
        );
        assert_eq!(
            syntaxify("x^2 + sin(y)"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Add),
                branch!(
                    SyntaxNode::BiOperation(BiOperation::Pow),
                    Leaf(SyntaxNode::Variable(TestVar::X)),
                    Leaf(SyntaxNode::Number(2.0))
                ),
                branch!(
                    SyntaxNode::NativeFunction(NativeFunction::Sin),
                    Leaf(SyntaxNode::Variable(TestVar::Y))
                )
            ))
        );
        assert_eq!(
            syntaxify("sqrt(max(4, 9))"),
            Ok(branch!(
                SyntaxNode::NativeFunction(NativeFunction::Sqrt),
                branch!(
                    SyntaxNode::NativeFunction(NativeFunction::Max),
                    Leaf(SyntaxNode::Number(4.0)),
                    Leaf(SyntaxNode::Number(9.0))
                )
            ))
        );
        assert_eq!(
            syntaxify("mean(2, 4, 6, 8)"),
            Ok(branch!(
                SyntaxNode::CustomFunction(TestFunc::Mean),
                Leaf(SyntaxNode::Number(2.0)),
                Leaf(SyntaxNode::Number(4.0)),
                Leaf(SyntaxNode::Number(6.0)),
                Leaf(SyntaxNode::Number(8.0))
            ))
        );
        assert_eq!(
            syntaxify("max(2, x, 8y, x*y+1)"),
            Ok(branch!(
                SyntaxNode::NativeFunction(NativeFunction::Max),
                Leaf(SyntaxNode::Number(2.0)),
                Leaf(SyntaxNode::Variable(TestVar::X)),
                branch!(
                    SyntaxNode::BiOperation(BiOperation::Mul),
                    Leaf(SyntaxNode::Number(8.0)),
                    Leaf(SyntaxNode::Variable(TestVar::Y))
                ),
                branch!(
                    SyntaxNode::BiOperation(BiOperation::Add),
                    branch!(
                        SyntaxNode::BiOperation(BiOperation::Mul),
                        Leaf(SyntaxNode::Variable(TestVar::X)),
                        Leaf(SyntaxNode::Variable(TestVar::Y))
                    ),
                    Leaf(SyntaxNode::Number(1.0))
                )
            ))
        );
        assert_eq!(
            syntaxify("-5+3"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Add),
                branch!(
                    SyntaxNode::UnOperation(UnOperation::Neg),
                    Leaf(SyntaxNode::Number(5.0))
                ),
                Leaf(SyntaxNode::Number(3.0))
            ))
        );
        assert_eq!(
            syntaxify("2*x + 3*y"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Add),
                branch!(
                    SyntaxNode::BiOperation(BiOperation::Mul),
                    Leaf(SyntaxNode::Number(2.0)),
                    Leaf(SyntaxNode::Variable(TestVar::X))
                ),
                branch!(
                    SyntaxNode::BiOperation(BiOperation::Mul),
                    Leaf(SyntaxNode::Number(3.0)),
                    Leaf(SyntaxNode::Variable(TestVar::Y))
                )
            ))
        );
        assert_eq!(
            syntaxify("tan(45)"),
            Ok(branch!(
                SyntaxNode::NativeFunction(NativeFunction::Tan),
                Leaf(SyntaxNode::Number(45.0))
            ))
        );
        assert_eq!(
            syntaxify("log10(1000)"),
            Ok(branch!(
                SyntaxNode::NativeFunction(NativeFunction::Log10),
                Leaf(SyntaxNode::Number(1000.0))
            ))
        );
        assert_eq!(
            syntaxify("1/(2+3)"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Div),
                Leaf(SyntaxNode::Number(1.0)),
                branch!(
                    SyntaxNode::BiOperation(BiOperation::Add),
                    Leaf(SyntaxNode::Number(2.0)),
                    Leaf(SyntaxNode::Number(3.0))
                )
            ))
        );
        assert_eq!(
            syntaxify("e^x"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Pow),
                Leaf(SyntaxNode::Number(std::f64::consts::E)),
                Leaf(SyntaxNode::Variable(TestVar::X))
            ))
        );
        assert_eq!(
            syntaxify("x * -2"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Mul),
                Leaf(SyntaxNode::Variable(TestVar::X)),
                branch!(
                    SyntaxNode::UnOperation(UnOperation::Neg),
                    Leaf(SyntaxNode::Number(2.0))
                )
            ))
        );
        assert_eq!(
            syntaxify("4/(-2)"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Div),
                Leaf(SyntaxNode::Number(4.0)),
                branch!(
                    SyntaxNode::UnOperation(UnOperation::Neg),
                    Leaf(SyntaxNode::Number(2.0))
                )
            ))
        );
        assert_eq!(
            syntaxify("sqrt(16)"),
            Ok(branch!(
                SyntaxNode::NativeFunction(NativeFunction::Sqrt),
                Leaf(SyntaxNode::Number(16.0))
            ))
        );
        assert_eq!(
            syntaxify("abs(-5)"),
            Ok(branch!(
                SyntaxNode::NativeFunction(NativeFunction::Abs),
                branch!(
                    SyntaxNode::UnOperation(UnOperation::Neg),
                    Leaf(SyntaxNode::Number(5.0))
                )
            ))
        );
        assert_eq!(
            syntaxify("x/y"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Div),
                Leaf(SyntaxNode::Variable(TestVar::X)),
                Leaf(SyntaxNode::Variable(TestVar::Y))
            ))
        );
        assert_eq!(
            syntaxify("x^2 - y^2"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Sub),
                branch!(
                    SyntaxNode::BiOperation(BiOperation::Pow),
                    Leaf(SyntaxNode::Variable(TestVar::X)),
                    Leaf(SyntaxNode::Number(2.0))
                ),
                branch!(
                    SyntaxNode::BiOperation(BiOperation::Pow),
                    Leaf(SyntaxNode::Variable(TestVar::Y)),
                    Leaf(SyntaxNode::Number(2.0))
                )
            ))
        );
        assert_eq!(
            syntaxify("log2(32)"),
            Ok(branch!(
                SyntaxNode::NativeFunction(NativeFunction::Log2),
                Leaf(SyntaxNode::Number(32.0))
            ))
        );
        assert_eq!(
            syntaxify("4*-x"),
            Ok(branch!(
                SyntaxNode::BiOperation(BiOperation::Mul),
                Leaf(SyntaxNode::Number(4.0)),
                branch!(
                    SyntaxNode::UnOperation(UnOperation::Neg),
                    Leaf(SyntaxNode::Variable(TestVar::X))
                )
            ))
        );
        assert_eq!(
            syntaxify("3!"),
            Ok(branch!(
                SyntaxNode::UnOperation(UnOperation::Fac),
                Leaf(SyntaxNode::Number(3.0))
            ))
        );
        assert_eq!(
            syntaxify("sin(x!-1)"),
            Ok(branch!(
                SyntaxNode::NativeFunction(NativeFunction::Sin),
                branch!(
                    SyntaxNode::BiOperation(BiOperation::Sub),
                    branch!(
                        SyntaxNode::UnOperation(UnOperation::Fac),
                        Leaf(SyntaxNode::Variable(TestVar::X))
                    ),
                    Leaf(SyntaxNode::Number(1.0))
                )
            ))
        );
        // temporary. must make more
        assert_eq!(
            syntaxify("x*()").map_err(|e| *e.kind()),
            Err(ParsingErrorKind::EmptyParenthesis)
        )
    }

    #[test]
    fn test_aot_evaluation() {
        macro_rules! compare {
            ($i1:literal, $i2:literal) => {
                let mut syn1 = parse($i1).unwrap();
                syn1.aot_evaluation(|_| CFPointer::Single(&|_| 0.0));
                let syn2 = parse($i2).unwrap();
                assert_eq!(format!("{}", syn1), format!("{}", syn2));
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
                let mut syn1 = parse($i1).unwrap();
                syn1.displacing_simplification();
                assert_eq!(format!("{}", syn1), $i2);
            };
        }
        compare!("x/1/8", "0.125*x");
        compare!("(x/16)/(y*4)", "0.015625*x/y");
        compare!("(7/x)/(y/2)", "14/x*y");
        compare!("(x/4)/(4/y)", "0.0625*x*y");
        compare!("10-x+12", "22-x");
        compare!("x*pi*2", "6.283185307179586*x");
    }

    #[test]
    fn test_syntax_display() {
        macro_rules! test {
            ($input:literal) => {
                assert_eq!($input, parse($input).unwrap().to_string());
            };
        }
        test!("x");
        test!("1+x");
        test!("sin(x)");
        test!("dist(x, y)");
        test!("min(1, x)");
        test!("min(1, x, y^2)");
        test!("min(1, x, y^2, x*y+1)");
        test!("min(1, x, y^2, x*y+1, sin(x*cos(y)+1))");
    }

    #[test]
    fn test_tokennode2range() {
        let input = " max(1, -18) * sin(pi)";
        let ts = TokenStream::new(input).unwrap();
        let tt = TokenTree::new(&ts).unwrap();
        assert_eq!(
            tokennode2range(
                input,
                &tt,
                tt.0.root.descendants(&tt.0.arena).nth(3).unwrap()
            ),
            5..=5
        );
        assert_eq!(
            tokennode2range(
                input,
                &tt,
                tt.0.root.descendants(&tt.0.arena).nth(6).unwrap()
            ),
            9..=10
        );
        assert_eq!(
            tokennode2range(input, &tt, tt.0.root.children(&tt.0.arena).nth(1).unwrap()),
            13..=13
        );
        assert_eq!(
            tokennode2range(
                input,
                &tt,
                tt.0.root.descendants(&tt.0.arena).nth(10).unwrap()
            ),
            19..=20
        );
    }
}
