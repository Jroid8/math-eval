use std::cmp::Ordering;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::marker::PhantomData;
use std::mem;
use std::ops::RangeInclusive;

use crate::number::{NFPointer, NativeFunction, Number};
use crate::postfix_tree::subtree_collection::{MultipleRoots, NotEnoughOrphans};
use crate::postfix_tree::tree_iterators::NodeEdge;
use crate::postfix_tree::{Node, PostfixTree, subtree_collection::SubtreeCollection};
use crate::tokenizer::Token;
use crate::trie::NameTrie;
use crate::{
    Associativity, BinaryOp, FunctionIdentifier, FunctionPointer, ParsingError, ParsingErrorKind,
    UnaryOp, VariableIdentifier, VariableStore,
};
use shunting_yard::{SyAstOutput, SyNumberOutput, parse_or_eval};

mod shunting_yard;
mod token_fragmentation;

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
    N: Number,
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
    N: Number,
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
    EmptyPipeAbs,
    MissingOpeningParenthesis,
    MissingClosingParenthesis,
    CommaOutsideFunction,
    PipeAbsNotClosed,
    NameTooLong,
    UnexpectedToken,
    UnknownError,
}

fn token_range_to_str_range<S: AsRef<str>>(
    input: &str,
    tokens: &[Token<S>],
    token_range: RangeInclusive<usize>,
) -> RangeInclusive<usize> {
    let mut start = 0;
    let mut index = 0;
    for (tk_idx, token) in tokens[..=*token_range.end()].iter().enumerate() {
        while let Some(ws) = Some(input.chars().nth(index).unwrap()).filter(|ch| ch.is_whitespace())
        {
            index += ws.len_utf8();
        }
        if tk_idx == *token_range.start() {
            start = index;
        }
        index += token.byte_len();
    }
    start..=index - 1
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SyntaxError(SyntaxErrorKind, RangeInclusive<usize>);

impl SyntaxError {
    pub fn to_general<S: AsRef<str>>(
        self,
        input: &str,
        tokens: impl AsRef<[Token<S>]>,
    ) -> ParsingError {
        ParsingError {
            at: if self.0 == SyntaxErrorKind::EmptyInput {
                0..=0
            } else {
                token_range_to_str_range(input, tokens.as_ref(), self.1)
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
                SyntaxErrorKind::UnexpectedToken => ParsingErrorKind::UnexpectedCharacter,
                SyntaxErrorKind::EmptyPipeAbs => ParsingErrorKind::EmptyPipeAbs,
                SyntaxErrorKind::UnknownError => ParsingErrorKind::UnknownError,
            },
        }
    }
}

impl From<NotEnoughOrphans> for SyntaxErrorKind {
    fn from(_: NotEnoughOrphans) -> Self {
        SyntaxErrorKind::UnknownError
    }
}

impl From<MultipleRoots> for SyntaxErrorKind {
    fn from(_: MultipleRoots) -> Self {
        SyntaxErrorKind::UnknownError
    }
}

impl From<NotEnoughOrphans> for SyntaxError {
    fn from(_: NotEnoughOrphans) -> Self {
        SyntaxError(SyntaxErrorKind::UnknownError, 0..=0)
    }
}

impl From<MultipleRoots> for SyntaxError {
    fn from(_: MultipleRoots) -> Self {
        SyntaxError(SyntaxErrorKind::UnknownError, 0..=0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MathAst<N: Number, V: VariableIdentifier, F: FunctionIdentifier>(
    PostfixTree<AstNode<N, V, F>>,
);

impl<V, N, F> MathAst<N, V, F>
where
    N: Number,
    V: VariableIdentifier,
    F: FunctionIdentifier,
{
    pub fn new<'a, S: AsRef<str>>(
        tokens: impl AsRef<[Token<S>]>,
        custom_constants: &impl NameTrie<&'a N>,
        custom_functions: &impl NameTrie<(F, u8, Option<u8>)>,
        custom_variables: &impl NameTrie<V>,
    ) -> Result<MathAst<N, V, F>, SyntaxError> {
        parse_or_eval(
            SyAstOutput(SubtreeCollection::new()),
            tokens,
            custom_constants,
            custom_functions,
            custom_variables,
        )
    }

    pub fn parse_and_eval<'a, 'b, 'c, S: VariableStore<N, V>, A: AsRef<str>>(
        tokens: impl AsRef<[Token<A>]>,
        custom_constants: &impl NameTrie<&'c N>,
        custom_functions: &impl NameTrie<(F, u8, Option<u8>)>,
        custom_variables: &impl NameTrie<V>,
        variable_values: &'a S,
        function_to_pointer: impl Fn(F) -> FunctionPointer<'b, N>,
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
            custom_constants,
            custom_functions,
            custom_variables,
        )
    }

    pub fn from_nodes(nodes: impl IntoIterator<Item = AstNode<N, V, F>>) -> Self {
        MathAst(PostfixTree::from_nodes(nodes))
    }

    pub fn as_tree(&self) -> &PostfixTree<AstNode<N, V, F>> {
        &self.0
    }

    pub fn into_tree(self) -> PostfixTree<AstNode<N, V, F>> {
        self.0
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
        function_to_pointer: impl Fn(F) -> FunctionPointer<'a, N>,
        variable_values: &impl crate::VariableStore<N, V>,
    ) -> N {
        Self::_eval(
            self.0.postorder_iter(),
            function_to_pointer,
            variable_values,
            &mut Vec::with_capacity(Self::eval_stack_capacity(self.0.postorder_iter())),
        )
        .unwrap()
    }

    fn _eval<'a, 'b>(
        tree: impl IntoIterator<Item = &'a AstNode<N, V, F>>,
        functibn_to_pointer: impl Fn(F) -> FunctionPointer<'b, N>,
        variable_values: &impl crate::VariableStore<N, V>,
        stack: &mut Vec<N>,
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
                AstNode::Function(FunctionType::Native(nf), argc) => match nf.as_pointer() {
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
                    FunctionPointer::Single(func) => func(pop()?.asarg()),
                    FunctionPointer::Dual(func) => {
                        let arg2 = pop()?;
                        func(pop()?.asarg(), arg2.asarg())
                    }
                    FunctionPointer::Triple(func) => {
                        let arg3 = pop()?;
                        let arg2 = pop()?;
                        func(pop()?.asarg(), arg2.asarg(), arg3.asarg())
                    }
                    FunctionPointer::Flexible(func) => {
                        let new_len = stack.len() - *argc as usize;
                        let res = func(&stack[new_len..]);
                        stack.truncate(new_len);
                        res
                    }
                    FunctionPointer::DynSingle(func) => func(pop()?.asarg()),
                    FunctionPointer::DynDual(func) => {
                        let arg2 = pop()?;
                        func(pop()?.asarg(), arg2.asarg())
                    }
                    FunctionPointer::DynTriple(func) => {
                        let arg3 = pop()?;
                        let arg2 = pop()?;
                        func(pop()?.asarg(), arg2.asarg(), arg3.asarg())
                    }
                    FunctionPointer::DynFlexible(func) => {
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

    pub fn aot_evaluation<'a>(
        &mut self,
        function_to_pointer: impl Fn(F) -> FunctionPointer<'a, N>,
    ) {
        let mut varless: Vec<bool> = Vec::with_capacity(self.0.len());
        for (idx, node) in self.0.postorder_iter().enumerate() {
            let res = match node {
                AstNode::Number(_) => true,
                AstNode::Variable(_) => false,
                _ => self.0.children_iter(idx).all(|(_, i)| varless[i]),
            };
            varless.push(res);
        }
        let mut idx = self.0.len();
        let mut stack: Vec<N> = Vec::with_capacity(0);
        while idx > 0 {
            idx -= 1;
            if varless[idx] && !matches!(self.0[idx], AstNode::Number(_)) {
                let required_capacity =
                    Self::eval_stack_capacity(self.0.postorder_subtree_iter(idx));
                if required_capacity > stack.capacity() {
                    stack.reserve(required_capacity - stack.capacity());
                }
                let result = Self::_eval(
                    self.0.postorder_subtree_iter(idx),
                    &function_to_pointer,
                    &(),
                    &mut stack,
                )
                .unwrap();
                let start = self.0.subtree_start(idx);
                self.0
                    .replace(mem::replace(&mut idx, start), [AstNode::Number(result)]);
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
        let mut idx = 4;
        while idx < self.0.len() {
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
                && self.0.children_iter(idx).any(|(node, idx)| match node {
                    AstNode::Number(_) => false,
                    AstNode::BinaryOp(_) => self
                        .0
                        .children_iter(idx)
                        .any(|(n, _)| !matches!(n, AstNode::Number(_))),
                    _ => true,
                })
                && !self.0.children_iter(idx).any(
                    |(arm, _)| matches!(arm, AstNode::BinaryOp(arm) if !can_simplify(head, *arm)),
                )
            {
                self._displacing_simplification(idx, head, &mut symbol_space);
            }
            idx += 1;
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
        symbol_space.push(AstNode::Number(lhs)).unwrap();
        if let Some((s1_idx, s1_sign)) = symbols[1] {
            let mut symbol_indices = [s0_idx, s1_idx];
            if s0_sign == negative && s1_sign == positive {
                symbol_indices.reverse();
            }
            for idx in symbol_indices {
                symbol_space.extend_from_tree(&self.0, idx);
            }
            symbol_space
                .push(AstNode::BinaryOp(if s0_sign == s1_sign {
                    positive
                } else {
                    negative
                }))
                .unwrap();
            symbol_space
                .push(AstNode::BinaryOp(
                    if s0_sign == negative && s1_sign == negative {
                        negative
                    } else {
                        positive
                    },
                ))
                .unwrap();
        } else {
            symbol_space.extend_from_tree(&self.0, s0_idx);
            symbol_space.push(AstNode::BinaryOp(s0_sign)).unwrap();
        }
        self.0
            .replace_from_sc_move(head_idx, symbol_space, symbol_space.len() - 1);
    }
}

pub fn parenthesis_required<N, V, F>(
    tree: &PostfixTree<AstNode<N, V, F>>,
    parent: usize,
    target: usize,
) -> bool
where
    N: Number,
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
    N: Number + Display,
    V: VariableIdentifier + Display,
    F: FunctionIdentifier + Display,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (node, edge, idx) in self.0.euler_tour() {
            if let Some(p) = self.0.parent(idx)
                && parenthesis_required(&self.0, p, idx)
            {
                match edge {
                    NodeEdge::Start => f.write_str("(")?,
                    NodeEdge::End => f.write_str(")")?,
                }
            }
            match (node, edge) {
                (AstNode::Number(num), NodeEdge::Start) => <N as Display>::fmt(num, f)?,
                (AstNode::Variable(var), NodeEdge::Start) => <V as Display>::fmt(var, f)?,
                (AstNode::UnaryOp(UnaryOp::Neg), NodeEdge::Start) => f.write_str("-")?,
                (AstNode::Function(func, _), NodeEdge::Start) => write!(f, "{func}(")?,
                (AstNode::UnaryOp(UnaryOp::Fac), NodeEdge::End) => f.write_str("!")?,
                (AstNode::Function(_, _), NodeEdge::End) => f.write_str(")")?,
                _ => (),
            }
            if let Some((AstNode::BinaryOp(opr), p)) = self.0.parent(idx).map(|p| (&self.0[p], p))
                && edge == NodeEdge::End
                && self.0.nth_child(p, 0) == Some(idx)
            {
                if matches!(opr, BinaryOp::Add | BinaryOp::Sub) {
                    write!(f, " {opr} ")?;
                } else {
                    write!(f, "{opr}")?;
                }
            }
            if let Some((AstNode::Function(_, argc), p)) =
                self.0.parent(idx).map(|p| (&self.0[p], p))
                && self.0.nth_child(p, *argc as usize - 1).unwrap() != idx
                && edge == NodeEdge::End
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

    use strum::FromRepr;

    use super::*;
    use crate::VariableStore;
    use crate::tokenizer::{StandardFloatRecognizer as Sfr, TokenStream};
    use crate::trie::TrieNode;

    #[derive(Clone, Copy, Debug, PartialEq, Eq, FromRepr)]
    #[repr(u8)]
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

    #[derive(Debug)]
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

    #[derive(Clone, Copy, Debug, PartialEq, Eq, FromRepr)]
    #[repr(u8)]
    enum TestFunc {
        Deg2Rad,
        ExpD,
        Clamp,
        Digits,
    }

    impl TestFunc {
        const fn with_mmargs(self) -> (Self, u8, Option<u8>) {
            match self {
                TestFunc::Deg2Rad => (TestFunc::Deg2Rad, 1, Some(1)),
                TestFunc::ExpD => (TestFunc::ExpD, 2, Some(2)),
                TestFunc::Clamp => (TestFunc::Clamp, 3, Some(3)),
                TestFunc::Digits => (TestFunc::Digits, 1, None),
            }
        }
        fn as_pointer(self) -> FunctionPointer<'static, f64> {
            match self {
                TestFunc::Deg2Rad => FunctionPointer::Single(|x: f64| x.to_radians()),
                TestFunc::ExpD => FunctionPointer::Dual(|l: f64, x: f64| l.powf(-l * x)),
                TestFunc::Clamp => {
                    FunctionPointer::Triple(|x: f64, min: f64, max: f64| x.min(max).max(min))
                }
                TestFunc::Digits => FunctionPointer::Flexible(|values: &[f64]| {
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

    struct TestConstsNameTrie;

    impl NameTrie<&'static f64> for TestConstsNameTrie {
        fn nodes(&self) -> &[TrieNode] {
            &[TrieNode::Branch('c', 1), TrieNode::Leaf(0)]
        }

        fn leaf_to_value(&self, _leaf: u32) -> &'static f64 {
            &299792458.0
        }
    }

    struct TestFuncsNameTrie;

    impl NameTrie<(TestFunc, u8, Option<u8>)> for TestFuncsNameTrie {
        fn nodes(&self) -> &[TrieNode] {
            &[
                TrieNode::Branch('c', 5),
                TrieNode::Branch('l', 4),
                TrieNode::Branch('a', 3),
                TrieNode::Branch('m', 2),
                TrieNode::Branch('p', 1),
                TrieNode::Leaf(TestFunc::Clamp as u32),
                TrieNode::Branch('d', 13),
                TrieNode::Branch('e', 6),
                TrieNode::Branch('g', 5),
                TrieNode::Branch('2', 4),
                TrieNode::Branch('r', 3),
                TrieNode::Branch('a', 2),
                TrieNode::Branch('d', 1),
                TrieNode::Leaf(TestFunc::Deg2Rad as u32),
                TrieNode::Branch('i', 5),
                TrieNode::Branch('g', 4),
                TrieNode::Branch('i', 3),
                TrieNode::Branch('t', 2),
                TrieNode::Branch('s', 1),
                TrieNode::Leaf(TestFunc::Digits as u32),
                TrieNode::Branch('e', 4),
                TrieNode::Branch('x', 3),
                TrieNode::Branch('p', 2),
                TrieNode::Branch('d', 1),
                TrieNode::Leaf(TestFunc::ExpD as u32),
            ]
        }
        fn leaf_to_value(&self, leaf: u32) -> (TestFunc, u8, Option<u8>) {
            TestFunc::from_repr(leaf as u8).unwrap().with_mmargs()
        }
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

    fn parse(input: &str) -> Result<MathAst<f64, TestVar, TestFunc>, ParsingError> {
        let tokens = TokenStream::new::<Sfr>(input)
            .map_err(|e| e.to_general())?
            .0;
        MathAst::new(
            &tokens,
            &TestConstsNameTrie,
            &TestFuncsNameTrie,
            &TestVarsNameTrie,
        )
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
            syntaxify("xy+1"),
            Ok(vec![
                AstNode::Variable(TestVar::X),
                AstNode::Variable(TestVar::Y),
                AstNode::BinaryOp(BinaryOp::Mul),
                AstNode::Number(1.0),
                AstNode::BinaryOp(BinaryOp::Add),
            ])
        );
        assert_eq!(
            syntaxify("ysin(x)+1"),
            Ok(vec![
                AstNode::Variable(TestVar::Y),
                AstNode::Variable(TestVar::X),
                AstNode::Function(NativeFunction::Sin.into(), 1),
                AstNode::BinaryOp(BinaryOp::Mul),
                AstNode::Number(1.0),
                AstNode::BinaryOp(BinaryOp::Add),
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
            syntaxify("2^-1"),
            Ok(vec![
                AstNode::Number(2.0),
                AstNode::Number(-1.0),
                AstNode::BinaryOp(BinaryOp::Pow),
            ])
        );
        assert_eq!(
            syntaxify("2^-(x+2)"),
            Ok(vec![
                AstNode::Number(2.0),
                AstNode::Variable(TestVar::X),
                AstNode::Number(2.0),
                AstNode::BinaryOp(BinaryOp::Add),
                AstNode::UnaryOp(UnaryOp::Neg),
                AstNode::BinaryOp(BinaryOp::Pow),
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
            syntaxify("sinx"),
            Ok(vec![
                AstNode::Variable(TestVar::X),
                AstNode::Function(NativeFunction::Sin.into(), 1),
            ])
        );
        assert_eq!(
            syntaxify("xcosy"),
            Ok(vec![
                AstNode::Variable(TestVar::X),
                AstNode::Variable(TestVar::Y),
                AstNode::Function(NativeFunction::Cos.into(), 1),
                AstNode::BinaryOp(BinaryOp::Mul),
            ])
        );
        assert_eq!(
            syntaxify("ln2x"),
            Ok(vec![
                AstNode::Number(2.0),
                AstNode::Variable(TestVar::X),
                AstNode::BinaryOp(BinaryOp::Mul),
                AstNode::Function(FunctionType::Native(NativeFunction::Ln), 1),
            ])
        );
        assert_eq!(
            syntaxify("sin-x"),
            Ok(vec![
                AstNode::Variable(TestVar::X),
                AstNode::UnaryOp(UnaryOp::Neg),
                AstNode::Function(NativeFunction::Sin.into(), 1),
            ])
        );
        assert_eq!(
            syntaxify("sinpix"),
            Ok(vec![
                AstNode::Number(PI),
                AstNode::Variable(TestVar::X),
                AstNode::BinaryOp(BinaryOp::Mul),
                AstNode::Function(NativeFunction::Sin.into(), 1),
            ])
        );
        assert_eq!(
            syntaxify("lnsqrtx"),
            Ok(vec![
                AstNode::Variable(TestVar::X),
                AstNode::Function(NativeFunction::Sqrt.into(), 1),
                AstNode::Function(NativeFunction::Ln.into(), 1),
            ])
        );
        assert_eq!(
            syntaxify("lnsqrt(x)"),
            Ok(vec![
                AstNode::Variable(TestVar::X),
                AstNode::Function(NativeFunction::Sqrt.into(), 1),
                AstNode::Function(NativeFunction::Ln.into(), 1),
            ])
        );
        assert_eq!(
            syntaxify("sqrt|x-y|"),
            Ok(vec![
                AstNode::Variable(TestVar::X),
                AstNode::Variable(TestVar::Y),
                AstNode::BinaryOp(BinaryOp::Sub),
                AstNode::Function(NativeFunction::Abs.into(), 1),
                AstNode::Function(NativeFunction::Sqrt.into(), 1),
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
            syntaxify("x(-1)"),
            Ok(vec![
                AstNode::Variable(TestVar::X),
                AstNode::Number(-1.0),
                AstNode::BinaryOp(BinaryOp::Mul)
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
        assert_eq!(
            syntaxify("x*"),
            Err(ParsingError {
                kind: ParsingErrorKind::MisplacedOperator,
                at: 1..=1
            })
        );
        assert_eq!(
            syntaxify("lnx)"),
            Err(ParsingError {
                kind: ParsingErrorKind::MissingOpenParenthesis,
                at: 3..=3
            })
        );
        assert_eq!(
            syntaxify("ln/x"),
            Err(ParsingError {
                kind: ParsingErrorKind::MisplacedOperator,
                at: 2..=2
            })
        );
    }

    #[test]
    fn parse_to_number() {
        fn evaluate(input: &str) -> f64 {
            MathAst::parse_and_eval(
                &TokenStream::new::<Sfr>(input).unwrap().0,
                &TestConstsNameTrie,
                &TestFuncsNameTrie,
                &TestVarsNameTrie,
                &TestVarStore,
                TestFunc::as_pointer,
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
        assert_eq!(evaluate("sin(pix/2)*t-cos(pix)*t"), 0.2);
    }

    #[test]
    fn aot_evaluation() {
        let simplify = |nodes: &[AstNode<f64, TestVar, TestFunc>]| {
            let mut ast = MathAst::from_nodes(nodes.iter().copied());
            ast.aot_evaluation(TestFunc::as_pointer);
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
        );
        assert_eq!(
            simplify(&[
                AstNode::Number(0.6),
                AstNode::Number(0.4),
                AstNode::BinaryOp(BinaryOp::Sub),
                AstNode::Number(0.6),
                AstNode::BinaryOp(BinaryOp::Sub),
            ]),
            vec![
                AstNode::Number(0.6),
                AstNode::Number(0.4),
                AstNode::BinaryOp(BinaryOp::Sub),
                AstNode::Number(0.6),
                AstNode::BinaryOp(BinaryOp::Sub),
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
                .eval(TestFunc::as_pointer, &TestVarStore)
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
        let ts = TokenStream::new::<Sfr>(input).unwrap().0;
        assert_eq!(token_range_to_str_range(input, &ts, 0..=0), 1..=4);
        assert_eq!(token_range_to_str_range(input, &ts, 1..=1), 5..=6);
        assert_eq!(token_range_to_str_range(input, &ts, 2..=2), 7..=7);
        assert_eq!(token_range_to_str_range(input, &ts, 3..=3), 9..=9);
    }
}
