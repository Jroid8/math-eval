use std::ops::RangeInclusive;

use asm::MathAssembly;
use number::MathEvalNumber;
use syntax::SyntaxTree;
use tokenizer::{
    token_stream::TokenStream,
    token_tree::TokenTree,
};

pub mod asm;
pub mod number;
pub mod syntax;
pub mod tokenizer;
pub mod tree_utils;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsingError {
    kind: ParsingErrorKind,
    at: RangeInclusive<usize>,
}

impl ParsingError {
    pub fn kind(&self) -> &ParsingErrorKind {
        &self.kind
    }

    pub fn at(&self) -> &RangeInclusive<usize> {
        &self.at
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParsingErrorKind {
    UnexpectedCharacter,
    CommaOutsideFunction,
    MissingOpenParenthesis,
    MissingCloseParenthesis,
    NumberParsingError,
    MisplacedOperator,
    UnknownVariableOrConstant,
    UnknownFunction,
    NotEnoughArguments,
    TooManyArguments,
}


pub fn parse<'a, N: MathEvalNumber, V: Clone, F: Clone>(
    input: &str,
    custom_constant_parser: impl Fn(&str) -> Option<N>,
    custom_function_parser: impl Fn(&str) -> Option<(F, u8, Option<u8>)>,
    custom_variable_parser: impl Fn(&str) -> Option<V>,
    function_to_pointer: impl Fn(&F) -> &'a dyn Fn(&[N]) -> N,
) -> Result<MathAssembly<'a, N, V, F>, ParsingError> {
    let token_stream = TokenStream::new(input).map_err(|e| e.to_general())?;
    let token_tree = TokenTree::new(&token_stream).map_err(|e| e.to_general(input, &token_stream))?;
    let mut syntax_tree = SyntaxTree::new(
        &token_tree,
        custom_constant_parser,
        custom_function_parser,
        custom_variable_parser,
    )
    .map_err(|e| e.to_general(input, &token_tree))?;
    syntax_tree.aot_evaluation(&function_to_pointer);
    syntax_tree.displacing_simplification();
    Ok(MathAssembly::new(
        &syntax_tree.0.arena,
        syntax_tree.0.root,
        function_to_pointer,
    ))
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TestVar {
    X,
    Y,
    T
}

#[cfg(test)]
impl std::fmt::Display for TestVar {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TestVar::X => f.write_str("x"),
            TestVar::Y => f.write_str("y"),
            TestVar::T => f.write_str("t"),
        }
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TestFunc {
    Gcd,
    Lcm,
    Mean,
    Dist,
}

#[cfg(test)]
impl std::fmt::Display for TestFunc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TestFunc::Gcd => f.write_str("gcd"),
            TestFunc::Lcm => f.write_str("lcm"),
            TestFunc::Mean => f.write_str("mean"),
            TestFunc::Dist => f.write_str("dist"),
        }
    }
}

#[cfg(test)]
pub(crate) fn parse_test_func(input: &str) -> Option<(TestFunc, u8, Option<u8>)> {
    match input {
        "gcd" => Some((TestFunc::Gcd, 2, Some(2))),
        "lcm" => Some((TestFunc::Lcm, 2, Some(2))),
        "mean" => Some((TestFunc::Mean, 2, None)),
        "dist" => Some((TestFunc::Dist, 2, Some(2))),
        _ => None
    }
}

#[cfg(test)]
pub(crate) fn parse_test_var(input: &str) -> Option<TestVar> {
    match input {
        "x" => Some(TestVar::X),
        "y" => Some(TestVar::Y),
        "t" => Some(TestVar::T),
        _ => None
    }
}

#[cfg(test)]
pub(crate) fn test_func_to_pointer(func: &TestFunc) -> &'static dyn Fn(&[f64]) -> f64 {
    match func {
        TestFunc::Gcd => todo!(),
        TestFunc::Lcm => todo!(),
        TestFunc::Mean => todo!(),
        TestFunc::Dist => todo!(),
    }
}
