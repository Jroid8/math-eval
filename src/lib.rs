use std::{collections::HashMap, ops::RangeInclusive};

use asm::MathAssembly;
use number::MathEvalNumber;
use syntax::SyntaxTree;
use tokenizer::{token_stream::TokenStream, token_tree::TokenTree};

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
    let token_tree =
        TokenTree::new(&token_stream).map_err(|e| e.to_general(input, &token_stream))?;
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

#[derive(Clone, Debug)]
struct NoVariable;

#[derive(Clone, Debug)]
struct OneVariable(String);

#[derive(Clone, Debug)]
struct TwoVariables(String, String);

#[derive(Clone, Debug)]
struct ThreeVariables(String, String, String);

#[derive(Clone, Debug)]
struct FourVariables(String, String, String, String);

#[derive(Clone, Debug)]
struct ManyVariables(Vec<String>);

#[derive(Clone)]
pub struct EvalBuilder<'a, N, V>
where
    N: MathEvalNumber,
{
    constants: HashMap<String, N>,
    function_identifier: HashMap<String, (usize, u8, Option<u8>)>,
    functions: Vec<&'a dyn Fn(&[N]) -> N>,
    variables: V,
}

impl<'a, N> Default for EvalBuilder<'a, N, NoVariable>
where
    N: MathEvalNumber,
{
    fn default() -> Self {
        Self {
            constants: HashMap::default(),
            function_identifier: HashMap::default(),
            functions: Vec::default(),
            variables: NoVariable,
        }
    }
}

impl<'a, N, V> EvalBuilder<'a, N, V>
where
    N: MathEvalNumber,
{
    pub fn add_constant(mut self, name: impl Into<String>, value: N) -> Self {
        self.constants.insert(name.into(), value);
        self
    }
    pub fn add_function(
        mut self,
        name: impl Into<String>,
        mininum_argument_count: u8,
        maximum_argument_count: Option<u8>,
        function: &'a dyn Fn(&[N]) -> N,
    ) -> Self {
        self.function_identifier.insert(
            name.into(),
            (
                self.functions.len(),
                mininum_argument_count,
                maximum_argument_count,
            ),
        );
        self.functions.push(function);
        self
    }
}

impl<'a, N> EvalBuilder<'a, N, NoVariable>
where
    N: MathEvalNumber,
{
    pub fn new() -> Self {
        Self::default()
    }
    pub fn add_variable(self, name: impl Into<String>) -> EvalBuilder<'a, N, OneVariable> {
        EvalBuilder {
            constants: self.constants,
            function_identifier: self.function_identifier,
            functions: self.functions,
            variables: OneVariable(name.into()),
        }
    }
    pub fn build(self, input: &str) -> Result<impl FnMut() -> N + 'a, ParsingError> {
        let mut expr = parse(
            input,
            |inp| self.constants.get(inp).copied(),
            |inp| self.function_identifier.get(inp).copied(),
            |_| None,
            |index| self.functions[*index],
        )?;
        Ok(move || expr.eval(|_: &()| 0.0.into()))
    }
}

impl<'a, N> EvalBuilder<'a, N, OneVariable>
where
    N: MathEvalNumber,
{
    pub fn add_variable(self, name: impl Into<String>) -> EvalBuilder<'a, N, TwoVariables> {
        EvalBuilder {
            constants: self.constants,
            function_identifier: self.function_identifier,
            functions: self.functions,
            variables: TwoVariables(self.variables.0, name.into()),
        }
    }
    pub fn build(self, input: &str) -> Result<impl FnMut(N) -> N + 'a, ParsingError> {
        let mut expr = parse(
            input,
            |inp| self.constants.get(inp).copied(),
            |inp| self.function_identifier.get(inp).copied(),
            |inp| {
                if self.variables.0 == inp {
                    Some(())
                } else {
                    None
                }
            },
            |index| self.functions[*index],
        )?;
        Ok(move |var| expr.eval(|_: &()| var))
    }
}

impl<'a, N> EvalBuilder<'a, N, TwoVariables>
where
    N: MathEvalNumber,
{
    pub fn add_variable(self, name: impl Into<String>) -> EvalBuilder<'a, N, ThreeVariables> {
        EvalBuilder {
            constants: self.constants,
            function_identifier: self.function_identifier,
            functions: self.functions,
            variables: ThreeVariables(self.variables.0, self.variables.1, name.into()),
        }
    }
    pub fn build(self, input: &str) -> Result<impl FnMut(N, N) -> N + 'a, ParsingError> {
        let mut expr = parse(
            input,
            |inp| self.constants.get(inp).copied(),
            |inp| self.function_identifier.get(inp).copied(),
            |inp| {
                [&self.variables.0, &self.variables.1]
                    .into_iter()
                    .position(|var| *var == inp)
                    .map(|i| i == 0)
            },
            |index| self.functions[*index],
        )?;
        Ok(move |v1, v2| expr.eval(|var: &bool| if *var { v1 } else { v2 }))
    }
}

impl<'a, N> EvalBuilder<'a, N, ThreeVariables>
where
    N: MathEvalNumber,
{
    pub fn add_variable(self, name: impl Into<String>) -> EvalBuilder<'a, N, FourVariables> {
        EvalBuilder {
            constants: self.constants,
            function_identifier: self.function_identifier,
            functions: self.functions,
            variables: FourVariables(
                self.variables.0,
                self.variables.1,
                self.variables.2,
                name.into(),
            ),
        }
    }
    pub fn build(self, input: &str) -> Result<impl FnMut(N, N, N) -> N + 'a, ParsingError> {
        let mut expr = parse(
            input,
            |inp| self.constants.get(inp).copied(),
            |inp| self.function_identifier.get(inp).copied(),
            |inp| {
                [&self.variables.0, &self.variables.1, &self.variables.2]
                    .into_iter()
                    .position(|var| *var == inp)
            },
            |index| self.functions[*index],
        )?;
        Ok(move |v1, v2, v3| {
            expr.eval(|var: &usize| match var {
                0 => v1,
                1 => v2,
                3 => v3,
                _ => unreachable!(),
            })
        })
    }
}

impl<'a, N> EvalBuilder<'a, N, FourVariables>
where
    N: MathEvalNumber,
{
    pub fn add_variable(self, name: impl Into<String>) -> EvalBuilder<'a, N, ManyVariables> {
        EvalBuilder {
            constants: self.constants,
            function_identifier: self.function_identifier,
            functions: self.functions,
            variables: ManyVariables(vec![
                self.variables.0,
                self.variables.1,
                self.variables.2,
                self.variables.3,
                name.into(),
            ]),
        }
    }
    pub fn build(self, input: &str) -> Result<impl FnMut(N, N, N, N) -> N + 'a, ParsingError> {
        let mut expr = parse(
            input,
            |inp| self.constants.get(inp).copied(),
            |inp| self.function_identifier.get(inp).copied(),
            |inp| {
                [
                    &self.variables.0,
                    &self.variables.1,
                    &self.variables.2,
                    &self.variables.3,
                ]
                .into_iter()
                .position(|var| *var == inp)
            },
            |index| self.functions[*index],
        )?;
        Ok(move |v1, v2, v3, v4| {
            expr.eval(|var: &usize| match var {
                0 => v1,
                1 => v2,
                3 => v3,
                4 => v4,
                _ => unreachable!(),
            })
        })
    }
}

impl<'a, N> EvalBuilder<'a, N, ManyVariables>
where
    N: MathEvalNumber,
{
    pub fn build(self, input: &str) -> Result<impl FnMut(&[N]) -> N + 'a, ParsingError> {
        let mut expr = parse(
            input,
            |inp| self.constants.get(inp).copied(),
            |inp| self.function_identifier.get(inp).copied(),
            |inp| self.variables.0.iter().position(|var| *var == inp),
            |index| self.functions[*index],
        )?;
        Ok(move |vars: &[N]| expr.eval(|v: &usize| vars[*v]))
    }
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TestVar {
    X,
    Y,
    T,
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
        _ => None,
    }
}

#[cfg(test)]
pub(crate) fn parse_test_var(input: &str) -> Option<TestVar> {
    match input {
        "x" => Some(TestVar::X),
        "y" => Some(TestVar::Y),
        "t" => Some(TestVar::T),
        _ => None,
    }
}

#[cfg(test)]
pub(crate) fn test_func_to_pointer(func: &TestFunc) -> &'static dyn Fn(&[f64]) -> f64 {
    fn gcd(inputs: &[f64]) -> f64 {
        // not fool proof, but good for tests
        let (mut num, mut den) = (inputs[0], inputs[1]);
        loop {
            let rem = num.rem_euclid(inputs[1]);
            if rem == 0.0 {
                return den;
            } else {
                num = den;
                den = rem;
            }
        }
    }
    fn lcm(inputs: &[f64]) -> f64 {
        inputs[0] * inputs[1] / gcd(inputs)
    }
    fn mean(inputs: &[f64]) -> f64 {
        inputs.iter().sum::<f64>() / inputs.len() as f64
    }
    fn dist(inputs: &[f64]) -> f64 {
        (inputs[0] * inputs[0] + inputs[1] * inputs[1]).sqrt()
    }

    match func {
        TestFunc::Gcd => &gcd,
        TestFunc::Lcm => &lcm,
        TestFunc::Mean => &mean,
        TestFunc::Dist => &dist,
    }
}

#[cfg(test)]
pub(crate) fn test_parse(input: &str) -> Result<f64, ParsingError> {
    parse(
        input,
        |_| None,
        parse_test_func,
        parse_test_var,
        test_func_to_pointer,
    )
    .map(|mut asm| {
        asm.eval(|var| match var {
            TestVar::X => 1.0,
            TestVar::Y => 8.0,
            TestVar::T => 1.5,
        })
    })
}

#[test]
fn test_eval_parser() {
    assert_eq!(
        EvalBuilder::new()
            .add_constant("g", 10.0)
            .add_function("inv", 1, Some(1), &|inputs: &[f64]| 1.0 / inputs[0],)
            .build("inv(g)")
            .unwrap()(),
        0.1
    );
    assert_eq!(
        EvalBuilder::new()
            .add_constant("c", 299792.0)
            .add_function("tometers", 1, Some(1), &|inputs: &[f64]| inputs[0] * 1000.0)
            .add_function("tomilimeters", 1, Some(1), &|inputs: &[f64]| inputs[0]
                * 1000.0)
            .build("tomilimeters(tometers(c))")
            .unwrap()(),
        299792000000.0
    );
    let mut func1 = EvalBuilder::new()
        .add_constant("g", 9.8)
        .add_variable("t")
        .build("0.5*g*t^2")
        .unwrap();
    assert_eq!(func1(12.0), 705.6);
    assert_eq!(func1(3.0), 44.1);
    let mut func2 = EvalBuilder::new()
        .add_variable("t")
        .add_variable("v")
        .add_constant("g", 9.8)
        .build("0.5*g*t^2 + v*t")
        .unwrap();
    assert_eq!(func2(12.0, 0.0), 705.6);
    assert_eq!(func2(3.0, 46.9), 44.1+140.7);
}
