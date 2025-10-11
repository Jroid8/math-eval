use std::cmp::Ordering;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::marker::PhantomData;
use std::ops::RangeInclusive;

use crate::asm::{CFPointer, Stack};
use crate::number::{MathEvalNumber, NFPointer, NativeFunction};
use crate::postfix_tree::tree_iterators::NodeEdge;
use crate::postfix_tree::{Node, PostfixTree, subtree_collection::SubtreeCollection};
use crate::tokenizer::Token;
use crate::{
    Associativity, BinaryOp, FunctionIdentifier, ParsingError, ParsingErrorKind, UnaryOp,
    VariableIdentifier, VariableStore,
};
use shunting_yard::{SyAstOutput, SyNumberOutput, parse_or_eval};

mod shunting_yard;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FunctionType<F: FunctionIdentifier> {
    Native(NativeFunction),
    Custom(F),
}

impl<F: FunctionIdentifier> From<NativeFunction> for FunctionType<F> {
    fn from(value: NativeFunction) -> Self {
        Self::Native(value)
    }
}

impl<F> Display for FunctionType<F>
where
    F: FunctionIdentifier + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FunctionType::Native(nf) => <NativeFunction as Display>::fmt(nf, f),
            FunctionType::Custom(cf) => <F as Display>::fmt(cf, f),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AstNode<N, V, F>
where
    N: MathEvalNumber,
    V: VariableIdentifier,
    F: FunctionIdentifier,
{
    Number(N),
    Variable(V),
    BinaryOp(BinaryOp),
    UnaryOp(UnaryOp),
    Function(FunctionType<F>, u8),
}

impl<N, V, F> Node for AstNode<N, V, F>
where
    N: MathEvalNumber,
    V: VariableIdentifier,
    F: FunctionIdentifier,
{
    fn children(&self) -> usize {
        match self {
            AstNode::Number(_) | AstNode::Variable(_) => 0,
            AstNode::BinaryOp(_) => 2,
            AstNode::UnaryOp(_) => 1,
            AstNode::Function(_, argc) => *argc,
        }
        .into()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SyntaxErrorKind {
    NumberParsingError,
    MisplacedOperator,
    UnknownVariableOrConstant,
    UnknownFunction,
    NotEnoughArguments,
    TooManyArguments,
    EmptyParenthesis,
    EmptyArgument,
    EmptyInput,
    MissingOpeningParenthesis,
    MissingClosingParenthesis,
    CommaOutsideFunction,
    PipeAbsNotClosed,
    NameTooLong,
}

fn token_range_to_str_range(
    input: &str,
    tokens: &[Token<'_>],
    token_range: RangeInclusive<usize>,
) -> RangeInclusive<usize> {
    let mut start = 0;
    let mut index = 0;
    for (tk_idx, token) in tokens[..=*token_range.end()].iter().enumerate() {
        while input.chars().nth(index).unwrap().is_whitespace() {
            index += 1
        }
        if tk_idx == *token_range.start() {
            start = index;
        }
        index += token.length();
    }
    start..=index - 1
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SyntaxError(SyntaxErrorKind, RangeInclusive<usize>);

impl SyntaxError {
    pub fn to_general(self, input: &str, tokens: &[Token<'_>]) -> ParsingError {
        ParsingError {
            at: if self.0 == SyntaxErrorKind::EmptyInput {
                0..=0
            } else {
                token_range_to_str_range(input, tokens, self.1)
            },
            kind: match self.0 {
                SyntaxErrorKind::NumberParsingError => ParsingErrorKind::NumberParsingError,
                SyntaxErrorKind::MisplacedOperator => ParsingErrorKind::MisplacedOperator,
                SyntaxErrorKind::UnknownVariableOrConstant => {
                    ParsingErrorKind::UnknownVariableOrConstant
                }
                SyntaxErrorKind::UnknownFunction => ParsingErrorKind::UnknownFunction,
                SyntaxErrorKind::NotEnoughArguments => ParsingErrorKind::NotEnoughArguments,
                SyntaxErrorKind::TooManyArguments => ParsingErrorKind::TooManyArguments,
                SyntaxErrorKind::EmptyParenthesis => ParsingErrorKind::EmptyParenthesis,
                SyntaxErrorKind::EmptyArgument => ParsingErrorKind::EmptyArgument,
                SyntaxErrorKind::MissingOpeningParenthesis => {
                    ParsingErrorKind::MissingOpenParenthesis
                }
                SyntaxErrorKind::MissingClosingParenthesis => {
                    ParsingErrorKind::MissingCloseParenthesis
                }
                SyntaxErrorKind::CommaOutsideFunction => ParsingErrorKind::CommaOutsideFunction,
                SyntaxErrorKind::EmptyInput => ParsingErrorKind::EmptyInput,
                SyntaxErrorKind::PipeAbsNotClosed => ParsingErrorKind::PipeAbsNotClosed,
                SyntaxErrorKind::NameTooLong => ParsingErrorKind::NameTooLong,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MathAst<N: MathEvalNumber, V: VariableIdentifier, F: FunctionIdentifier>(
    PostfixTree<AstNode<N, V, F>>,
);

impl<V, N, F> MathAst<N, V, F>
where
    N: MathEvalNumber,
    V: VariableIdentifier,
    F: FunctionIdentifier,
{
    pub fn new<'a>(
        tokens: &'a [Token<'a>],
        custom_constant_parser: impl Fn(&str) -> Option<N>,
        custom_function_parser: impl Fn(&str) -> Option<(F, u8, Option<u8>)>,
        custom_variable_parser: impl Fn(&str) -> Option<V>,
    ) -> Result<MathAst<N, V, F>, SyntaxError> {
        parse_or_eval(
            SyAstOutput(SubtreeCollection::new()),
            tokens,
            custom_constant_parser,
            custom_function_parser,
            custom_variable_parser,
        )
    }

    pub fn parse_and_eval<'a, 'b, 'c, S: VariableStore<N, V>>(
        tokens: &'a [Token<'a>],
        custom_constant_parser: impl Fn(&str) -> Option<N>,
        custom_function_parser: impl Fn(&str) -> Option<(F, u8, Option<u8>)>,
        custom_variable_parser: impl Fn(&str) -> Option<V>,
        variable_values: &'b S,
        function_to_pointer: impl Fn(F) -> CFPointer<'c, N>,
    ) -> Result<N, SyntaxError> {
        parse_or_eval(
            SyNumberOutput {
                args: Vec::new(),
                variable_store: variable_values,
                cf2pointer: function_to_pointer,
                var_ident: PhantomData,
                func_ident: PhantomData,
            },
            tokens,
            custom_constant_parser,
            custom_function_parser,
            custom_variable_parser,
        )
    }

    pub fn from_nodes(nodes: impl IntoIterator<Item = AstNode<N, V, F>>) -> Self {
        MathAst(PostfixTree::from_nodes(nodes))
    }

    fn eval_stack_capacity<'a>(tree: impl IntoIterator<Item = &'a AstNode<N, V, F>>) -> usize {
        let mut stack_capacity = 0usize;
        let mut stack_len = 0i64;
        for node in tree {
            stack_len += match node {
                AstNode::Number(_) | AstNode::Variable(_) => 1,
                AstNode::BinaryOp(_) => -1,
                AstNode::UnaryOp(_) => 0,
                AstNode::Function(_, args) => -(*args as i64) + 1,
            };
            if stack_len as usize > stack_capacity {
                stack_capacity = stack_len as usize;
            }
        }
        stack_capacity
    }

    pub fn eval<'a>(
        &self,
        function_to_pointer: impl Fn(F) -> CFPointer<'a, N>,
        variable_values: &impl crate::VariableStore<N, V>,
    ) -> N {
        Self::_eval(
            self.0.postorder_iter(),
            function_to_pointer,
            variable_values,
            &mut Stack::with_capacity(Self::eval_stack_capacity(self.0.postorder_iter())),
        )
        .unwrap()
    }

    fn _eval<'a, 'b>(
        tree: impl IntoIterator<Item = &'a AstNode<N, V, F>>,
        functibn_to_pointer: impl Fn(F) -> CFPointer<'b, N>,
        variable_values: &impl crate::VariableStore<N, V>,
        stack: &mut Stack<N>,
    ) -> Result<N, usize> {
        for (idx, node) in tree.into_iter().enumerate() {
            let mut pop = || stack.pop().ok_or(idx);
            let result: N = match node {
                AstNode::Number(num) => num.clone(),
                AstNode::Variable(var) => variable_values.get(*var).to_owned(),
                AstNode::BinaryOp(opr) => {
                    let rhs = pop()?;
                    opr.eval(pop()?.asarg(), rhs.asarg())
                }
                AstNode::UnaryOp(opr) => opr.eval(pop()?.asarg()),
                AstNode::Function(FunctionType::Native(nf), argc) => match nf.to_pointer() {
                    NFPointer::Single(func) => func(pop()?.asarg()),
                    NFPointer::Dual(func) => {
                        let arg2 = pop()?;
                        func(pop()?.asarg(), arg2.asarg())
                    }
                    NFPointer::Flexible(func) => {
                        let new_len = stack.len() - *argc as usize;
                        let res = func(&stack[new_len..]);
                        stack.truncate(new_len);
                        res
                    }
                },
                AstNode::Function(FunctionType::Custom(cf), argc) => match functibn_to_pointer(*cf)
                {
                    CFPointer::Single(func) => func(pop()?.asarg()),
                    CFPointer::Dual(func) => {
                        let arg2 = pop()?;
                        func(pop()?.asarg(), arg2.asarg())
                    }
                    CFPointer::Triple(func) => {
                        let arg3 = pop()?;
                        let arg2 = pop()?;
                        func(pop()?.asarg(), arg2.asarg(), arg3.asarg())
                    }
                    CFPointer::Flexible(func) => {
                        let new_len = stack.len() - *argc as usize;
                        let res = func(&stack[new_len..]);
                        stack.truncate(new_len);
                        res
                    }
                },
            };
            stack.push(result);
        }
        stack.pop().ok_or(0)
    }

    pub fn aot_evaluation<'a>(&mut self, function_to_pointer: impl Fn(F) -> CFPointer<'a, N>) {
        let mut varless: Vec<bool> = Vec::with_capacity(self.0.len());
        for (idx, node) in self.0.postorder_iter().enumerate() {
            let res = match node {
                AstNode::Number(_) => true,
                AstNode::Variable(_) => false,
                _ => self.0.children_iter(idx).all(|(_, i)| varless[i]),
            };
            varless.push(res);
        }
        let mut idx = self.0.len() - 1;
        let mut stack: Stack<N> = Stack::with_capacity(0);
        loop {
            if varless[idx] && !matches!(self.0[idx], AstNode::Number(_)) {
                let required_capacity =
                    Self::eval_stack_capacity(self.0.postorder_subtree_iter(idx));
                if required_capacity > stack.capacity() {
                    stack.reserve(required_capacity - stack.capacity());
                }
                self.0.replace(
                    idx,
                    [AstNode::Number(
                        Self::_eval(
                            self.0.postorder_subtree_iter(idx),
                            &function_to_pointer,
                            &(),
                            &mut stack,
                        )
                        .unwrap(),
                    )],
                );
                if let Some(i) = self.0.subtree_start(idx).checked_sub(1) {
                    idx = i;
                } else {
                    break;
                }
            } else if idx > 0 {
                idx -= 1;
            } else {
                break;
            }
        }
    }

    pub fn displacing_simplification(&mut self) {
        fn can_simplify(head: BinaryOp, arm: BinaryOp) -> bool {
            matches!(
                (head, arm),
                (BinaryOp::Add | BinaryOp::Sub, BinaryOp::Add | BinaryOp::Sub)
                    | (BinaryOp::Mul | BinaryOp::Div, BinaryOp::Mul | BinaryOp::Div)
            )
        }
        let mut symbol_space: SubtreeCollection<AstNode<N, V, F>> =
            SubtreeCollection::from_alloc(Vec::with_capacity(0));
        for idx in 4..self.0.len() {
            if let AstNode::BinaryOp(head) = self.0[idx]
                && self
                    .0
                    .children_iter(idx)
                    .any(|(arm, _)| matches!(arm, AstNode::BinaryOp(_)))
                && self
                    .0
                    .children_iter(idx)
                    .map(|(node, idx)| match node {
                        AstNode::Number(_) => 1,
                        AstNode::BinaryOp(_) => self
                            .0
                            .children_iter(idx)
                            .map(|(n, _)| matches!(n, AstNode::Number(_)) as u8)
                            .sum(),
                        _ => 0,
                    })
                    .sum::<u8>()
                    >= 2
                && !self.0.children_iter(idx).any(
                    |(arm, _)| matches!(arm, AstNode::BinaryOp(arm) if !can_simplify(head, *arm)),
                )
            {
                self._displacing_simplification(idx, head, &mut symbol_space);
            }
        }
    }

    fn _displacing_simplification(
        &mut self,
        head_idx: usize,
        head_opr: BinaryOp,
        symbol_space: &mut SubtreeCollection<AstNode<N, V, F>>,
    ) {
        let (positive, negative, mut lhs) = match head_opr {
            BinaryOp::Add | BinaryOp::Sub => (BinaryOp::Add, BinaryOp::Sub, N::from(0)),
            BinaryOp::Mul | BinaryOp::Div => (BinaryOp::Mul, BinaryOp::Div, N::from(1)),
            _ => return,
        };
        let apply_pos = |parent: BinaryOp, pos: usize| -> BinaryOp {
            if pos == 0 && parent == negative {
                negative
            } else {
                positive
            }
        };
        let multiply_oprs = |head: BinaryOp, arm: BinaryOp| -> BinaryOp {
            if head == arm { positive } else { negative }
        };
        let mut symbols: [Option<(usize, BinaryOp)>; 2] = [None, None];
        for (arm_pos, (arm, arm_idx)) in self.0.children_iter(head_idx).enumerate() {
            let sign = apply_pos(head_opr, arm_pos);
            match arm {
                AstNode::Number(num) => lhs = sign.eval(lhs.asarg(), num.asarg()),
                AstNode::BinaryOp(arm_opr) => {
                    for (tail_pos, (tail, tail_idx)) in self.0.children_iter(arm_idx).enumerate() {
                        let sign = multiply_oprs(
                            apply_pos(head_opr, arm_pos),
                            apply_pos(*arm_opr, tail_pos),
                        );
                        match tail {
                            AstNode::Number(num) => {
                                lhs = sign.eval(lhs.asarg(), num.asarg());
                            }
                            _ => symbols[symbols[0].is_some() as usize] = Some((tail_idx, sign)),
                        }
                    }
                }
                _ => symbols[symbols[0].is_some() as usize] = Some((arm_idx, sign)),
            }
        }
        let (s0_idx, s0_sign) = symbols[0].unwrap();
        debug_assert!(symbol_space.is_empty());
        symbol_space.push(AstNode::Number(lhs));
        if let Some((s1_idx, s1_sign)) = symbols[1] {
            let mut symbol_indices = [s0_idx, s1_idx];
            if s0_sign == negative && s1_sign == positive {
                symbol_indices.reverse();
            }
            for idx in symbol_indices {
                symbol_space.extend_from_tree(&self.0, idx);
            }
            symbol_space.push(AstNode::BinaryOp(if s0_sign == s1_sign {
                positive
            } else {
                negative
            }));
            symbol_space.push(AstNode::BinaryOp(
                if s0_sign == negative && s1_sign == negative {
                    negative
                } else {
                    positive
                },
            ));
        } else {
            symbol_space.extend_from_tree(&self.0, s0_idx);
            symbol_space.push(AstNode::BinaryOp(s0_sign));
        }
        self.0
            .replace_from_sc_move(head_idx, symbol_space, symbol_space.len() - 1);
    }
}

fn parenthesis_required<N, V, F>(
    tree: &PostfixTree<AstNode<N, V, F>>,
    parent: usize,
    target: usize,
) -> bool
where
    N: MathEvalNumber,
    V: VariableIdentifier,
    F: FunctionIdentifier,
{
    match (&tree[parent], &tree[target]) {
        (AstNode::BinaryOp(head), AstNode::BinaryOp(arm)) => {
            match head.precedence().cmp(&arm.precedence()) {
                // Without the parenthesis around y-z in x-(y-z) the order of operations
                // would be the opposite of what's intended.
                Ordering::Equal => match head.associativity() {
                    Associativity::Both => false,
                    // parent is left associative and child is on the right
                    Associativity::Left => tree.nth_child(parent, 1) == Some(target),
                    // parent is right associative and child is on the left
                    Associativity::Right => tree.nth_child(parent, 0) == Some(target),
                },
                Ordering::Less => false,
                Ordering::Greater => true,
            }
        }
        // Not only does multiplication, division and exponentiation take precedence over negation,
        // wrapping parenthesis around negation removes ambiguity when paired with addition and subtraction.
        (AstNode::BinaryOp(_), AstNode::UnaryOp(UnaryOp::Neg)) => true,
        // Negation takes precedence over addition and subtraction.
        (AstNode::UnaryOp(UnaryOp::Neg), AstNode::BinaryOp(opr)) => opr.precedence() == 0,
        // Factorial always takes precedence over binary operations.
        (AstNode::UnaryOp(UnaryOp::Fac | UnaryOp::DoubleFac), AstNode::BinaryOp(_)) => true,
        (AstNode::BinaryOp(_), AstNode::UnaryOp(UnaryOp::Fac | UnaryOp::DoubleFac)) => false,
        // Parenthesis is required for -(-x) and (x!)!, not just when the child's precedence is lower.
        (AstNode::UnaryOp(parent), AstNode::UnaryOp(current)) => {
            current.precedence() <= parent.precedence()
        }
        // Functions, variables and numbers take precedence over operators.
        (
            AstNode::BinaryOp(_) | AstNode::UnaryOp(_),
            AstNode::Function(_, _) | AstNode::Number(_) | AstNode::Variable(_),
        ) => false,
        // Function arguments are clearly separated with commas.
        (AstNode::Function(_, _), _) => false,
        // Numbers and variables shouldn't have a child.
        (AstNode::Number(_) | AstNode::Variable(_), _) => unreachable!(),
    }
}

impl<V, N, F> Display for MathAst<N, V, F>
where
    N: MathEvalNumber + Display,
    V: VariableIdentifier + Display,
    F: FunctionIdentifier + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for edge in self.0.euler_tour() {
            if let Some(p) = self.0.parent(edge.index())
                && parenthesis_required(&self.0, p, edge.index())
            {
                match edge {
                    NodeEdge::Start(_, _) => f.write_str("(")?,
                    NodeEdge::End(_, _) => f.write_str(")")?,
                }
            }
            match edge {
                NodeEdge::Start(AstNode::Number(num), _) => <N as Display>::fmt(num, f)?,
                NodeEdge::Start(AstNode::Variable(var), _) => <V as Display>::fmt(var, f)?,
                NodeEdge::Start(AstNode::UnaryOp(UnaryOp::Neg), _) => f.write_str("-")?,
                NodeEdge::Start(AstNode::Function(func, _), _) => write!(f, "{func}(")?,
                NodeEdge::End(AstNode::UnaryOp(UnaryOp::Fac), idx) => f.write_str("!")?,
                NodeEdge::End(AstNode::Function(_, _), _) => f.write_str(")")?,
                _ => (),
            }
            if let NodeEdge::End(_, idx) = edge
                && let Some((AstNode::BinaryOp(opr), p)) =
                    self.0.parent(idx).map(|p| (&self.0[p], p))
                && self.0.nth_child(p, 0) == Some(idx)
            {
                if matches!(opr, BinaryOp::Add | BinaryOp::Sub) {
                    write!(f, " {opr} ")?;
                } else {
                    write!(f, "{opr}")?;
                }
            }
            if let NodeEdge::End(_, idx) = edge
                && let Some((AstNode::Function(func, argc), p)) =
                    self.0.parent(idx).map(|p| (&self.0[p], p))
                && self.0.nth_child(p, *argc as usize - 1).unwrap() != idx
            {
                f.write_str(", ")?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::*;

    use super::*;
    use crate::VariableStore;
    use crate::tokenizer::TokenStream;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
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

    impl Display for TestVar {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                TestVar::X => f.write_str("x"),
                TestVar::Y => f.write_str("y"),
                TestVar::T => f.write_str("t"),
            }
        }
    }

    struct TestVarStore;

    impl VariableStore<f64, TestVar> for TestVarStore {
        fn get(&self, var: TestVar) -> f64 {
            match var {
                TestVar::X => 1.0,
                TestVar::Y => 5.0,
                TestVar::T => 0.1,
            }
        }
    }

    fn parse_constant(input: &str) -> Option<f64> {
        match input {
            "c" => Some(299792458.0),
            _ => None,
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum TestFunc {
        Deg2Rad,
        ExpD,
        Clamp,
        Digits,
    }

    impl TestFunc {
        fn parse(input: &str) -> Option<(Self, u8, Option<u8>)> {
            match input {
                "deg2rad" => Some((TestFunc::Deg2Rad, 1, Some(1))),
                "expd" => Some((TestFunc::ExpD, 2, Some(2))),
                "clamp" => Some((TestFunc::Clamp, 3, Some(3))),
                "digits" => Some((TestFunc::Digits, 1, None)),
                _ => None,
            }
        }
        fn to_pointer(self) -> CFPointer<'static, f64> {
            match self {
                TestFunc::Deg2Rad => CFPointer::Single(&|x: f64| x.to_radians()),
                TestFunc::ExpD => CFPointer::Dual(&|l: f64, x: f64| l.powf(-l * x)),
                TestFunc::Clamp => {
                    CFPointer::Triple(&|x: f64, min: f64, max: f64| x.min(max).max(min))
                }
                TestFunc::Digits => CFPointer::Flexible(&|values: &[f64]| {
                    values
                        .iter()
                        .enumerate()
                        .map(|(i, &v)| 10f64.powi(i as i32) * v)
                        .sum()
                }),
            }
        }
    }

    impl Display for TestFunc {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                TestFunc::Deg2Rad => f.write_str("deg2rad"),
                TestFunc::ExpD => f.write_str("expd"),
                TestFunc::Clamp => f.write_str("clamp"),
                TestFunc::Digits => f.write_str("digits"),
            }
        }
    }

    fn parse(input: &str) -> Result<MathAst<f64, TestVar, TestFunc>, ParsingError> {
        let tokens = TokenStream::new(input).map_err(|e| e.to_general())?.0;
        MathAst::new(&tokens, parse_constant, TestFunc::parse, TestVar::parse)
            .map_err(|e| e.to_general(input, &tokens))
    }

    #[test]
    fn parse_to_ast() {
        fn syntaxify(input: &str) -> Result<Vec<AstNode<f64, TestVar, TestFunc>>, ParsingError> {
            parse(input).map(|st| {
                st.0.into_inner()
                    .into_iter()
                    .map(|e| e.into_inner())
                    .collect::<Vec<_>>()
            })
        }
        assert_eq!(syntaxify("0"), Ok(vec![AstNode::Number(0.0)]));
        assert_eq!(syntaxify("(0)"), Ok(vec![AstNode::Number(0.0)]));
        assert_eq!(syntaxify("((0))"), Ok(vec![AstNode::Number(0.0)]));
        assert_eq!(syntaxify("pi"), Ok(vec![AstNode::Number(PI)]));
        assert_eq!(
            syntaxify("1+1"),
            Ok(vec![
                AstNode::Number(1.0),
                AstNode::Number(1.0),
                AstNode::BinaryOp(BinaryOp::Add),
            ])
        );
        assert_eq!(syntaxify("-0.5"), Ok(vec![AstNode::Number(-0.5)]));
        assert_eq!(
            syntaxify("5-3"),
            Ok(vec![
                AstNode::Number(5.0),
                AstNode::Number(3.0),
                AstNode::BinaryOp(BinaryOp::Sub),
            ])
        );
        assert_eq!(
            syntaxify("-y!"),
            Ok(vec![
                AstNode::Variable(TestVar::Y),
                AstNode::UnaryOp(UnaryOp::Fac),
                AstNode::UnaryOp(UnaryOp::Neg),
            ])
        );
        assert_eq!(
            syntaxify("t!!"),
            Ok(vec![
                AstNode::Variable(TestVar::T),
                AstNode::UnaryOp(UnaryOp::DoubleFac)
            ])
        );
        assert_eq!(
            syntaxify("8*3+1"),
            Ok(vec![
                AstNode::Number(8.0),
                AstNode::Number(3.0),
                AstNode::BinaryOp(BinaryOp::Mul),
                AstNode::Number(1.0),
                AstNode::BinaryOp(BinaryOp::Add),
            ])
        );
        assert_eq!(
            syntaxify("12/3/2"),
            Ok(vec![
                AstNode::Number(12.0),
                AstNode::Number(3.0),
                AstNode::BinaryOp(BinaryOp::Div),
                AstNode::Number(2.0),
                AstNode::BinaryOp(BinaryOp::Div),
            ])
        );
        assert_eq!(
            syntaxify("8*(3+1)"),
            Ok(vec![
                AstNode::Number(8.0),
                AstNode::Number(3.0),
                AstNode::Number(1.0),
                AstNode::BinaryOp(BinaryOp::Add),
                AstNode::BinaryOp(BinaryOp::Mul),
            ])
        );
        assert_eq!(
            syntaxify("8*3^2-1"),
            Ok(vec![
                AstNode::Number(8.0),
                AstNode::Number(3.0),
                AstNode::Number(2.0),
                AstNode::BinaryOp(BinaryOp::Pow),
                AstNode::BinaryOp(BinaryOp::Mul),
                AstNode::Number(1.0),
                AstNode::BinaryOp(BinaryOp::Sub),
            ])
        );
        assert_eq!(
            syntaxify("2x"),
            Ok(vec![
                AstNode::Number(2.0),
                AstNode::Variable(TestVar::X),
                AstNode::BinaryOp(BinaryOp::Mul),
            ])
        );
        assert_eq!(
            syntaxify("sin(14)"),
            Ok(vec![
                AstNode::Number(14.0),
                AstNode::Function(NativeFunction::Sin.into(), 1),
            ])
        );
        assert_eq!(
            syntaxify("deg2rad(80)"),
            Ok(vec![
                AstNode::Number(80.0),
                AstNode::Function(FunctionType::Custom(TestFunc::Deg2Rad), 1),
            ])
        );
        assert_eq!(
            syntaxify("expd(0.2, x)"),
            Ok(vec![
                AstNode::Number(0.2),
                AstNode::Variable(TestVar::X),
                AstNode::Function(FunctionType::Custom(TestFunc::ExpD), 2),
            ])
        );
        assert_eq!(
            syntaxify("clamp(t, y, x)"),
            Ok(vec![
                AstNode::Variable(TestVar::T),
                AstNode::Variable(TestVar::Y),
                AstNode::Variable(TestVar::X),
                AstNode::Function(FunctionType::Custom(TestFunc::Clamp), 3),
            ])
        );
        assert_eq!(
            syntaxify("digits(3, 1, 5, 7, 2, x)"),
            Ok(vec![
                AstNode::Number(3.0),
                AstNode::Number(1.0),
                AstNode::Number(5.0),
                AstNode::Number(7.0),
                AstNode::Number(2.0),
                AstNode::Variable(TestVar::X),
                AstNode::Function(FunctionType::Custom(TestFunc::Digits), 6),
            ])
        );
        assert_eq!(
            syntaxify("lb(8)"),
            Ok(vec![
                AstNode::Number(8.0),
                AstNode::Function(NativeFunction::Log2.into(), 1),
            ])
        );
        assert_eq!(
            syntaxify("log(100)"),
            Ok(vec![
                AstNode::Number(100.0),
                AstNode::Function(NativeFunction::Log10.into(), 1),
            ])
        );
        assert_eq!(
            syntaxify("sin(cos(0))"),
            Ok(vec![
                AstNode::Number(0.0),
                AstNode::Function(NativeFunction::Cos.into(), 1),
                AstNode::Function(NativeFunction::Sin.into(), 1),
            ])
        );
        assert_eq!(
            syntaxify("x^2 + sin(y)"),
            Ok(vec![
                AstNode::Variable(TestVar::X),
                AstNode::Number(2.0),
                AstNode::BinaryOp(BinaryOp::Pow),
                AstNode::Variable(TestVar::Y),
                AstNode::Function(NativeFunction::Sin.into(), 1),
                AstNode::BinaryOp(BinaryOp::Add),
            ])
        );
        assert_eq!(
            syntaxify("sqrt(max(4, 9))"),
            Ok(vec![
                AstNode::Number(4.0),
                AstNode::Number(9.0),
                AstNode::Function(NativeFunction::Max.into(), 2),
                AstNode::Function(NativeFunction::Sqrt.into(), 1),
            ])
        );
        assert_eq!(
            syntaxify("max(2, x, 8y, xy+1)"),
            Ok(vec![
                AstNode::Number(2.0),
                AstNode::Variable(TestVar::X),
                AstNode::Number(8.0),
                AstNode::Variable(TestVar::Y),
                AstNode::BinaryOp(BinaryOp::Mul),
                AstNode::Variable(TestVar::X),
                AstNode::Variable(TestVar::Y),
                AstNode::BinaryOp(BinaryOp::Mul),
                AstNode::Number(1.0),
                AstNode::BinaryOp(BinaryOp::Add),
                AstNode::Function(NativeFunction::Max.into(), 4),
            ])
        );
        assert_eq!(
            syntaxify("2*x + 3*y"),
            Ok(vec![
                AstNode::Number(2.0),
                AstNode::Variable(TestVar::X),
                AstNode::BinaryOp(BinaryOp::Mul),
                AstNode::Number(3.0),
                AstNode::Variable(TestVar::Y),
                AstNode::BinaryOp(BinaryOp::Mul),
                AstNode::BinaryOp(BinaryOp::Add),
            ])
        );
        assert_eq!(
            syntaxify("log10(1000)"),
            Ok(vec![
                AstNode::Number(1000.0),
                AstNode::Function(NativeFunction::Log10.into(), 1),
            ])
        );
        assert_eq!(
            syntaxify("1/(2+3)"),
            Ok(vec![
                AstNode::Number(1.0),
                AstNode::Number(2.0),
                AstNode::Number(3.0),
                AstNode::BinaryOp(BinaryOp::Add),
                AstNode::BinaryOp(BinaryOp::Div),
            ])
        );
        assert_eq!(
            syntaxify("e^x"),
            Ok(vec![
                AstNode::Number(E),
                AstNode::Variable(TestVar::X),
                AstNode::BinaryOp(BinaryOp::Pow),
            ])
        );
        assert_eq!(
            syntaxify("x * -2"),
            Ok(vec![
                AstNode::Variable(TestVar::X),
                AstNode::Number(-2.0),
                AstNode::BinaryOp(BinaryOp::Mul),
            ])
        );
        assert_eq!(
            syntaxify("4/-1.33"),
            Ok(vec![
                AstNode::Number(4.0),
                AstNode::Number(-1.33),
                AstNode::BinaryOp(BinaryOp::Div),
            ])
        );
        assert_eq!(
            syntaxify("sqrt(16)"),
            Ok(vec![
                AstNode::Number(16.0),
                AstNode::Function(NativeFunction::Sqrt.into(), 1),
            ])
        );
        assert_eq!(
            syntaxify("abs(-5)"),
            Ok(vec![
                AstNode::Number(-5.0),
                AstNode::Function(NativeFunction::Abs.into(), 1),
            ])
        );
        assert_eq!(
            syntaxify("x^2 - y^2"),
            Ok(vec![
                AstNode::Variable(TestVar::X),
                AstNode::Number(2.0),
                AstNode::BinaryOp(BinaryOp::Pow),
                AstNode::Variable(TestVar::Y),
                AstNode::Number(2.0),
                AstNode::BinaryOp(BinaryOp::Pow),
                AstNode::BinaryOp(BinaryOp::Sub),
            ])
        );
        assert_eq!(
            syntaxify("|x|"),
            Ok(vec![
                AstNode::Variable(TestVar::X),
                AstNode::Function(NativeFunction::Abs.into(), 1),
            ])
        );
        assert_eq!(
            syntaxify("|-x|"),
            Ok(vec![
                AstNode::Variable(TestVar::X),
                AstNode::UnaryOp(UnaryOp::Neg),
                AstNode::Function(NativeFunction::Abs.into(), 1),
            ])
        );
        assert_eq!(
            syntaxify("4*-x"),
            Ok(vec![
                AstNode::Number(4.0),
                AstNode::Variable(TestVar::X),
                AstNode::UnaryOp(UnaryOp::Neg),
                AstNode::BinaryOp(BinaryOp::Mul),
            ])
        );
        assert_eq!(
            syntaxify("8.3 + -1"),
            Ok(vec![
                AstNode::Number(8.3),
                AstNode::Number(-1.0),
                AstNode::BinaryOp(BinaryOp::Add),
            ])
        );
        assert_eq!(syntaxify("++1"), Ok(vec![AstNode::Number(1.0)]));
        assert_eq!(syntaxify("+-1"), Ok(vec![AstNode::Number(-1.0)]));
        assert_eq!(syntaxify("--1"), Ok(vec![AstNode::Number(1.0)]));
        assert_eq!(syntaxify("---1"), Ok(vec![AstNode::Number(-1.0)]));
        assert_eq!(syntaxify("----1"), Ok(vec![AstNode::Number(1.0)]));
        assert_eq!(
            syntaxify("x + +1"),
            Ok(vec![
                AstNode::Variable(TestVar::X),
                AstNode::Number(1.0),
                AstNode::BinaryOp(BinaryOp::Add),
            ])
        );
        assert_eq!(
            syntaxify("-1^2"),
            Ok(vec![
                AstNode::Number(1.0),
                AstNode::Number(2.0),
                AstNode::BinaryOp(BinaryOp::Pow),
                AstNode::UnaryOp(UnaryOp::Neg),
            ])
        );
        assert_eq!(
            syntaxify("x!y"),
            Ok(vec![
                AstNode::Variable(TestVar::X),
                AstNode::UnaryOp(UnaryOp::Fac),
                AstNode::Variable(TestVar::Y),
                AstNode::BinaryOp(BinaryOp::Mul),
            ])
        );
        assert_eq!(
            syntaxify("sin(x!-1)"),
            Ok(vec![
                AstNode::Variable(TestVar::X),
                AstNode::UnaryOp(UnaryOp::Fac),
                AstNode::Number(1.0),
                AstNode::BinaryOp(BinaryOp::Sub),
                AstNode::Function(NativeFunction::Sin.into(), 1),
            ])
        );
        assert_eq!(
            syntaxify("3|x-1|/2 + 1"),
            Ok(vec![
                AstNode::Number(3.0),
                AstNode::Variable(TestVar::X),
                AstNode::Number(1.0),
                AstNode::BinaryOp(BinaryOp::Sub),
                AstNode::Function(NativeFunction::Abs.into(), 1),
                AstNode::BinaryOp(BinaryOp::Mul),
                AstNode::Number(2.0),
                AstNode::BinaryOp(BinaryOp::Div),
                AstNode::Number(1.0),
                AstNode::BinaryOp(BinaryOp::Add),
            ])
        );
        assert_eq!(
            syntaxify(""),
            Err(ParsingError {
                kind: ParsingErrorKind::EmptyInput,
                at: 0..=0
            })
        );
        assert_eq!(
            syntaxify("sin(x)+ja"),
            Err(ParsingError {
                kind: ParsingErrorKind::UnknownVariableOrConstant,
                at: 7..=8
            })
        );
        assert_eq!(
            syntaxify("x*()"),
            Err(ParsingError {
                kind: ParsingErrorKind::EmptyParenthesis,
                at: 2..=3
            })
        );
        assert_eq!(
            syntaxify("5+pi*sinj(x)"),
            Err(ParsingError {
                kind: ParsingErrorKind::UnknownFunction,
                at: 5..=9
            })
        );
        assert_eq!(
            syntaxify("1+expd(2y)"),
            Err(ParsingError {
                kind: ParsingErrorKind::NotEnoughArguments,
                at: 2..=9
            })
        );
        assert_eq!(
            syntaxify("5(1+clamp(2y, 1))"),
            Err(ParsingError {
                kind: ParsingErrorKind::NotEnoughArguments,
                at: 4..=15
            })
        );
        assert_eq!(
            syntaxify("deg2rad(1, pi)"),
            Err(ParsingError {
                kind: ParsingErrorKind::TooManyArguments,
                at: 0..=13
            })
        );
        assert_eq!(
            syntaxify("-expd(1, pi, sin(x))"),
            Err(ParsingError {
                kind: ParsingErrorKind::TooManyArguments,
                at: 1..=19
            })
        );
        assert_eq!(
            syntaxify("9t-clamp(1, pi, sin(x), 15tan(y))"),
            Err(ParsingError {
                kind: ParsingErrorKind::TooManyArguments,
                at: 3..=32
            })
        );
        assert_eq!(
            syntaxify("x*sin()"),
            Err(ParsingError {
                kind: ParsingErrorKind::EmptyArgument,
                at: 2..=6
            })
        );
        assert_eq!(
            syntaxify("x*log(y,)"),
            Err(ParsingError {
                kind: ParsingErrorKind::EmptyArgument,
                at: 7..=8
            })
        );
        assert_eq!(
            syntaxify("x*expd(,5y)"),
            Err(ParsingError {
                kind: ParsingErrorKind::EmptyArgument,
                at: 2..=7
            })
        );
        assert_eq!(
            syntaxify("x*clamp(x,,5y)"),
            Err(ParsingError {
                kind: ParsingErrorKind::EmptyArgument,
                at: 9..=10
            })
        );
        assert_eq!(
            syntaxify("x)"),
            Err(ParsingError {
                kind: ParsingErrorKind::MissingOpenParenthesis,
                at: 1..=1
            })
        );
        assert_eq!(
            syntaxify("(x"),
            Err(ParsingError {
                kind: ParsingErrorKind::MissingCloseParenthesis,
                at: 0..=0
            })
        );
        assert_eq!(
            syntaxify("(x + 1)*sin(y"),
            Err(ParsingError {
                kind: ParsingErrorKind::MissingCloseParenthesis,
                at: 8..=11
            })
        );
        assert_eq!(
            syntaxify("(x + (y)"),
            Err(ParsingError {
                kind: ParsingErrorKind::MissingCloseParenthesis,
                at: 0..=0
            })
        );
        assert_eq!(
            syntaxify("(10) * y) + 4x"),
            Err(ParsingError {
                kind: ParsingErrorKind::MissingOpenParenthesis,
                at: 8..=8
            })
        );
        assert_eq!(
            syntaxify("*10 2"),
            Err(ParsingError {
                kind: ParsingErrorKind::MisplacedOperator,
                at: 0..=0
            })
        );
        assert_eq!(
            syntaxify("(*x 2)"),
            Err(ParsingError {
                kind: ParsingErrorKind::MisplacedOperator,
                at: 1..=1
            })
        );
        assert_eq!(
            syntaxify("1+(x 2-)"),
            Err(ParsingError {
                kind: ParsingErrorKind::MisplacedOperator,
                at: 6..=6
            })
        );
        assert_eq!(
            syntaxify("(+)"),
            Err(ParsingError {
                kind: ParsingErrorKind::MisplacedOperator,
                at: 1..=1
            })
        );
        assert_eq!(
            syntaxify("(!)"),
            Err(ParsingError {
                kind: ParsingErrorKind::MisplacedOperator,
                at: 1..=1
            })
        );
        assert_eq!(
            syntaxify("|x"),
            Err(ParsingError {
                kind: ParsingErrorKind::PipeAbsNotClosed,
                at: 0..=0
            })
        );
        assert_eq!(
            syntaxify("3+|(|y|+1)/2"),
            Err(ParsingError {
                kind: ParsingErrorKind::PipeAbsNotClosed,
                at: 2..=2
            })
        );
        assert_eq!(
            syntaxify("|sin(|x)|"),
            Err(ParsingError {
                kind: ParsingErrorKind::PipeAbsNotClosed,
                at: 5..=5
            })
        );
    }

    #[test]
    fn parse_to_number() {
        fn evaluate(input: &str) -> f64 {
            MathAst::parse_and_eval(
                &TokenStream::new(input).unwrap().0,
                parse_constant,
                TestFunc::parse,
                TestVar::parse,
                &TestVarStore,
                TestFunc::to_pointer,
            )
            .unwrap()
        }
        assert_eq!(evaluate("1"), 1.0);
        assert_eq!(evaluate("-0.5"), -0.5);
        assert_eq!(evaluate("857-999"), -142.0);
        assert_eq!(evaluate("8*19"), 152.0);
        assert_eq!(evaluate("-4!"), -24.0);
        assert_eq!(evaluate("y!!"), 15.0);
        assert_eq!(evaluate("1-27/9"), -2.0);
        assert_eq!(evaluate("121/11/t"), 110.0);
        assert_eq!(evaluate("y*(x+11)"), 60.0);
        assert_eq!(evaluate("8*2^y-t"), 255.9);
        assert_eq!(evaluate("18x"), 18.0);
        assert_eq!(evaluate("sin(pi/2)"), 1.0);
        assert_eq!(evaluate("log(243, 3)"), 5.0 - 1e-15);
        assert_eq!(evaluate("min(-13,x,y,0)"), -13.0);
        assert_eq!(evaluate("deg2rad(90)"), FRAC_PI_2);
        assert_eq!(evaluate("expd(0.5, 2)"), 2.0);
        assert_eq!(evaluate("clamp(x, 0, 2)"), 1.0);
        assert_eq!(evaluate("digits(x, y, t, 9)/1000"), 9.061);
        assert_eq!(evaluate("lb(2048)"), 11.0);
        assert_eq!(evaluate("log(1000000)"), 6.0);
        assert_eq!(evaluate("lg(0.0001)"), -4.0);
        assert_eq!(evaluate("ln(e^23)"), 23.0);
        assert_eq!(evaluate("ln(cos(0))"), 0.0);
        assert_eq!(evaluate("y * -2"), -10.0);
        assert_eq!(evaluate("sin(pi/-2)"), -1.0);
        assert_eq!(evaluate("+-+75"), -75.0);
        assert_eq!(evaluate("----7"), 7.0);
        assert_eq!(evaluate("|x|"), 1.0);
        assert_eq!(evaluate("|-x|"), 1.0);
        assert_eq!(evaluate("x^2 + y^2"), 26.0);
        assert_eq!(evaluate("y*-0.5"), -2.5);
        assert_eq!(evaluate("sin(pix/2)*t+cos(pix)*t"), 0.2);
    }

    #[test]
    fn aot_evaluation() {
        let simplify = |nodes: &[AstNode<f64, TestVar, TestFunc>]| {
            let mut ast = MathAst::from_nodes(nodes.iter().copied());
            ast.aot_evaluation(TestFunc::to_pointer);
            ast.0.postorder_iter().cloned().collect::<Vec<_>>()
        };
        assert_eq!(
            simplify(&[
                AstNode::Number(16.0),
                AstNode::Number(8.0),
                AstNode::BinaryOp(BinaryOp::Div),
                AstNode::Number(11.0),
                AstNode::BinaryOp(BinaryOp::Add),
            ]),
            vec![AstNode::Number(13.0)]
        );
        assert_eq!(
            simplify(&[
                AstNode::Number(9.0),
                AstNode::Function(NativeFunction::Sqrt.into(), 1),
            ]),
            vec![AstNode::Number(3.0)]
        );
        assert_eq!(
            simplify(&[
                AstNode::Number(1.0),
                AstNode::Number(8.0),
                AstNode::BinaryOp(BinaryOp::Div),
                AstNode::Variable(TestVar::T),
                AstNode::BinaryOp(BinaryOp::Add),
                AstNode::Function(NativeFunction::Sin.into(), 1),
            ]),
            vec![
                AstNode::Number(0.125),
                AstNode::Variable(TestVar::T),
                AstNode::BinaryOp(BinaryOp::Add),
                AstNode::Function(NativeFunction::Sin.into(), 1),
            ]
        );
        assert_eq!(
            simplify(&[
                AstNode::Number(80.0),
                AstNode::Number(5.0),
                AstNode::BinaryOp(BinaryOp::Div),
                AstNode::Variable(TestVar::X),
                AstNode::Number(2.0),
                AstNode::BinaryOp(BinaryOp::Pow),
                AstNode::Number(1.0),
                AstNode::Number(0.0),
                AstNode::Function(NativeFunction::Sin.into(), 1),
                AstNode::Function(NativeFunction::Min.into(), 2),
                AstNode::Function(NativeFunction::Max.into(), 3),
                AstNode::Number(121.0),
                AstNode::Function(NativeFunction::Sqrt.into(), 1),
                AstNode::BinaryOp(BinaryOp::Add)
            ]),
            vec![
                AstNode::Number(16.0),
                AstNode::Variable(TestVar::X),
                AstNode::Number(2.0),
                AstNode::BinaryOp(BinaryOp::Pow),
                AstNode::Number(0.0),
                AstNode::Function(NativeFunction::Max.into(), 3),
                AstNode::Number(11.0),
                AstNode::BinaryOp(BinaryOp::Add)
            ]
        );
    }

    #[test]
    fn displacing_simplification() {
        let simplify = |nodes: &[AstNode<f64, TestVar, TestFunc>]| {
            let mut ast = MathAst::from_nodes(nodes.iter().copied());
            ast.displacing_simplification();
            ast.0.postorder_iter().cloned().collect::<Vec<_>>()
        };
        assert_eq!(
            simplify(&[
                AstNode::Variable(TestVar::X),
                AstNode::Number(1.0),
                AstNode::Number(8.0),
                AstNode::BinaryOp(BinaryOp::Div),
                AstNode::BinaryOp(BinaryOp::Div),
            ]),
            vec![
                AstNode::Number(8.0),
                AstNode::Variable(TestVar::X),
                AstNode::BinaryOp(BinaryOp::Mul),
            ]
        );
        assert_eq!(
            simplify(&[
                AstNode::Number(16.0),
                AstNode::Variable(TestVar::Y),
                AstNode::BinaryOp(BinaryOp::Div),
                AstNode::Number(8.0),
                AstNode::Variable(TestVar::X),
                AstNode::BinaryOp(BinaryOp::Div),
                AstNode::BinaryOp(BinaryOp::Div),
            ]),
            vec![
                AstNode::Number(2.0),
                AstNode::Variable(TestVar::X),
                AstNode::Variable(TestVar::Y),
                AstNode::BinaryOp(BinaryOp::Div),
                AstNode::BinaryOp(BinaryOp::Mul),
            ]
        );
        assert_eq!(
            simplify(&[
                AstNode::Variable(TestVar::X),
                AstNode::Number(-2.0),
                AstNode::BinaryOp(BinaryOp::Add),
                AstNode::Variable(TestVar::Y),
                AstNode::Number(17.0),
                AstNode::BinaryOp(BinaryOp::Add),
                AstNode::BinaryOp(BinaryOp::Add),
            ]),
            vec![
                AstNode::Number(15.0),
                AstNode::Variable(TestVar::Y),
                AstNode::Variable(TestVar::X),
                AstNode::BinaryOp(BinaryOp::Add),
                AstNode::BinaryOp(BinaryOp::Add),
            ]
        );
        assert_eq!(
            simplify(&[
                AstNode::Number(1.0),
                AstNode::Number(2.0),
                AstNode::Variable(TestVar::X),
                AstNode::BinaryOp(BinaryOp::Mul),
                AstNode::BinaryOp(BinaryOp::Div),
            ]),
            vec![
                AstNode::Number(0.5),
                AstNode::Variable(TestVar::X),
                AstNode::BinaryOp(BinaryOp::Div),
            ]
        );
        assert_eq!(
            simplify(&[
                AstNode::Number(125.0),
                AstNode::Variable(TestVar::T),
                AstNode::Number(5.0),
                AstNode::BinaryOp(BinaryOp::Add),
                AstNode::BinaryOp(BinaryOp::Sub)
            ]),
            vec![
                AstNode::Number(120.0),
                AstNode::Variable(TestVar::T),
                AstNode::BinaryOp(BinaryOp::Sub),
            ]
        );
        assert_eq!(
            simplify(&[
                AstNode::Number(17.0),
                AstNode::Variable(TestVar::Y),
                AstNode::Function(NativeFunction::Log.into(), 1),
                AstNode::BinaryOp(BinaryOp::Sub),
                AstNode::Number(10.0),
                AstNode::Number(1.0),
                AstNode::BinaryOp(BinaryOp::Sub),
                AstNode::BinaryOp(BinaryOp::Sub),
            ]),
            vec![
                AstNode::Number(8.0),
                AstNode::Variable(TestVar::Y),
                AstNode::Function(NativeFunction::Log.into(), 1),
                AstNode::BinaryOp(BinaryOp::Sub),
            ]
        );
        assert_eq!(
            simplify(&[
                AstNode::Number(7.0),
                AstNode::Variable(TestVar::T),
                AstNode::BinaryOp(BinaryOp::Mul),
                AstNode::Number(2.0),
                AstNode::Variable(TestVar::X),
                AstNode::Function(NativeFunction::Sin.into(), 1),
                AstNode::BinaryOp(BinaryOp::Mul),
                AstNode::BinaryOp(BinaryOp::Div),
            ]),
            vec![
                AstNode::Number(3.5),
                AstNode::Variable(TestVar::T),
                AstNode::Variable(TestVar::X),
                AstNode::Function(NativeFunction::Sin.into(), 1),
                AstNode::BinaryOp(BinaryOp::Div),
                AstNode::BinaryOp(BinaryOp::Mul),
            ]
        );
        assert_eq!(
            simplify(&[
                AstNode::Number(5.0),
                AstNode::Variable(TestVar::X),
                AstNode::BinaryOp(BinaryOp::Div),
                AstNode::Number(7.0),
                AstNode::Variable(TestVar::Y),
                AstNode::BinaryOp(BinaryOp::Div),
                AstNode::BinaryOp(BinaryOp::Mul),
            ]),
            vec![
                AstNode::Number(35.0),
                AstNode::Variable(TestVar::Y),
                AstNode::Variable(TestVar::X),
                AstNode::BinaryOp(BinaryOp::Mul),
                AstNode::BinaryOp(BinaryOp::Div),
            ]
        );
        assert_eq!(
            simplify(&[
                AstNode::Number(81.0),
                AstNode::Number(3.0),
                AstNode::Function(NativeFunction::Log.into(), 2),
                AstNode::Number(8.9),
                AstNode::BinaryOp(BinaryOp::Sub),
                AstNode::Number(1.4),
                AstNode::Number(5.993),
                AstNode::Variable(TestVar::X),
                AstNode::Function(NativeFunction::Max.into(), 3),
                AstNode::Number(3.9),
                AstNode::BinaryOp(BinaryOp::Sub),
                AstNode::BinaryOp(BinaryOp::Sub),
            ]),
            vec![
                AstNode::Number(-5.0),
                AstNode::Number(81.0),
                AstNode::Number(3.0),
                AstNode::Function(NativeFunction::Log.into(), 2),
                AstNode::Number(1.4),
                AstNode::Number(5.993),
                AstNode::Variable(TestVar::X),
                AstNode::Function(NativeFunction::Max.into(), 3),
                AstNode::BinaryOp(BinaryOp::Sub),
                AstNode::BinaryOp(BinaryOp::Add),
            ]
        )
    }

    #[test]
    fn syntax_display() {
        let cases = [
            "x",
            "-y!",
            "1 + x",
            "10 - t + 12",
            "3*x - 2*y",
            "y^3 - 4*y^2 + y - 7",
            "x*5*y/4*3",
            "(x + 1)*(y - 1)",
            "1/(y - 1)",
            "1/(2*sqrt(y))",
            "y - (x + 1)",
            "x - y!",
            "x/(-y)",
            "x/y!",
            "(x^2)^y",
            "7/x/(y/2)",
            "(t^2 + 3*t + 2)/(t + 1)",
            "sin(x)",
            "x/sin(x + 1)",
            "clamp(x, 0, 1)",
            "min(1, x, y^2, x*y + 1)",
            "max(1, x, y^2, x*y + 1, sin(x*cos(y) + 1))",
            "digits(1, deg2rad(t), y^2, x*y + 1, sin(x*cos(y) + 1))",
        ];

        for c in cases {
            assert_eq!(parse(c).unwrap().to_string(), c);
        }
    }

    #[test]
    fn ast_eval() {
        fn eval(input: &str) -> f64 {
            parse(input)
                .unwrap()
                .eval(TestFunc::to_pointer, &TestVarStore)
        }

        assert_eq!(eval("1"), 1.0);
        assert_eq!(eval("x*3"), 3.0);
        assert_eq!(eval("3!"), 6.0);
        assert_eq!(eval("-t"), -0.1);
        assert_eq!(eval("y+100*t"), 15.0);
        assert_eq!(eval("sin(pi*t)"), 0.3090169943749474);
        assert_eq!(eval("log(6561, 3)"), 8.0);
        assert_eq!(eval("max(x, y, -18)*t"), 0.5);
        assert_eq!(eval("clamp(x + y, -273.15, t)"), 0.1);
        assert_eq!(eval("digits(y, 1)"), 15.0);
        assert_eq!(eval("digits(5, 4, 9)*t"), 94.5);
    }

    #[test]
    fn token2range() {
        let input = " max(pi, 1, -4)*3";
        let ts = TokenStream::new(input).unwrap().0;
        assert_eq!(token_range_to_str_range(input, &ts, 0..=0), 1..=4);
        assert_eq!(token_range_to_str_range(input, &ts, 1..=1), 5..=6);
        assert_eq!(token_range_to_str_range(input, &ts, 2..=2), 7..=7);
        assert_eq!(token_range_to_str_range(input, &ts, 3..=3), 9..=9);
    }
}
