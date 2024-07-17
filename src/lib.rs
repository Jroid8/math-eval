use std::{collections::HashMap, fmt::Display, ops::RangeInclusive};

use asm::{CFPointer, MathAssembly};
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

    pub fn print_colored(&self, input: &str) {
        for (i, c) in input.chars().enumerate() {
            if i == *self.at.start() {
                print!("\x1b[0;31m")
            }
            print!("{c}");
            if i == *self.at.end() {
                print!("\x1b[0m")
            }
        }
    }
}

impl Display for ParsingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!(
            "{} at [{}, {}]",
            self.kind,
            self.at.start(),
            self.at.end()
        ))
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
    MisplacedToken,
    EmptyParenthesis,
}

impl Display for ParsingErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            ParsingErrorKind::UnexpectedCharacter => "Unexpected character encountered",
            ParsingErrorKind::CommaOutsideFunction => "Comma found outside of function",
            ParsingErrorKind::MissingOpenParenthesis => "Opening parenthesis is missing",
            ParsingErrorKind::MissingCloseParenthesis => "Closing parenthesis is missing",
            ParsingErrorKind::NumberParsingError => "Unable to parse number",
            ParsingErrorKind::MisplacedOperator => "Misplaced operator",
            ParsingErrorKind::UnknownVariableOrConstant => "Unrecognized variable or constant",
            ParsingErrorKind::UnknownFunction => "Unrecognized function",
            ParsingErrorKind::NotEnoughArguments => "Insufficient arguments for function",
            ParsingErrorKind::TooManyArguments => "Too many arguments for function",
            ParsingErrorKind::MisplacedToken => "Misplaced token",
            ParsingErrorKind::EmptyParenthesis => "Parentheses should not be empty",
        })
    }
}

pub fn parse<'a, N: MathEvalNumber, V: Clone, F: Clone>(
    input: &str,
    custom_constant_parser: impl Fn(&str) -> Option<N>,
    custom_function_parser: impl Fn(&str) -> Option<(F, u8, Option<u8>)>,
    custom_variable_parser: impl Fn(&str) -> Option<V>,
    function_to_pointer: impl Fn(&F) -> CFPointer<'a, N>,
    optimize: bool,
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
    if optimize {
        syntax_tree.aot_evaluation(&function_to_pointer);
        syntax_tree.displacing_simplification();
    }
    Ok(MathAssembly::new(
        &syntax_tree.0.arena,
        syntax_tree.0.root,
        function_to_pointer,
    ))
}

pub fn evaluate<N: MathEvalNumber>(input: &str) -> Result<N, ParsingError> {
    parse(
        input,
        |_| None,
        |_| None,
        |_| None,
        |_: &()| unreachable!(),
        false,
    )
    .map(|mut asm| asm.eval(|_: &()| unreachable!()))
}

#[derive(Clone, Debug)]
pub struct NoVariable;

#[derive(Clone, Debug)]
pub struct OneVariable(String);

#[derive(Clone, Debug)]
pub struct TwoVariables(String, String);

#[derive(Clone, Debug)]
pub struct ThreeVariables([String; 3]);

#[derive(Clone, Debug)]
pub struct FourVariables([String; 4]);

#[derive(Clone, Debug)]
pub struct ManyVariables(Vec<String>);

#[derive(Clone)]
pub struct EvalBuilder<'a, N, V>
where
    N: MathEvalNumber,
{
    constants: HashMap<String, N>,
    function_identifier: HashMap<String, (usize, u8, Option<u8>)>,
    functions: Vec<CFPointer<'a, N>>,
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
    pub fn add_fn1(mut self, name: impl Into<String>, function: &'a dyn Fn(N) -> N) -> Self {
        self.function_identifier
            .insert(name.into(), (self.functions.len(), 1, Some(1)));
        self.functions.push(CFPointer::Single(function));
        self
    }
    pub fn add_fn2(mut self, name: impl Into<String>, function: &'a dyn Fn(N, N) -> N) -> Self {
        self.function_identifier
            .insert(name.into(), (self.functions.len(), 2, Some(2)));
        self.functions.push(CFPointer::Dual(function));
        self
    }
    pub fn add_fn3(mut self, name: impl Into<String>, function: &'a dyn Fn(N, N, N) -> N) -> Self {
        self.function_identifier
            .insert(name.into(), (self.functions.len(), 3, Some(3)));
        self.functions.push(CFPointer::Triple(function));
        self
    }
    pub fn add_fn4(
        mut self,
        name: impl Into<String>,
        function: &'a dyn Fn(N, N, N, N) -> N,
    ) -> Self {
        self.function_identifier
            .insert(name.into(), (self.functions.len(), 4, Some(4)));
        self.functions.push(CFPointer::Quad(function));
        self
    }
    pub fn add_fn_flex(
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
        self.functions.push(CFPointer::Flexible(function));
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
    fn parse(
        &self,
        input: &str,
        optimize: bool,
    ) -> Result<MathAssembly<'a, N, (), usize>, ParsingError> {
        parse(
            input,
            |inp| self.constants.get(inp).copied(),
            |inp| self.function_identifier.get(inp).copied(),
            |_| None,
            |index| self.functions[*index],
            optimize,
        )
    }
    pub fn build_as_parser(self) -> impl Fn(&str) -> Result<N, ParsingError> + 'a {
        move |input: &str| {
            self.parse(input, false)
                .map(|mut asm| asm.eval(|_: &()| 0.0.into()))
        }
    }

    pub fn build_as_function(self, input: &str) -> Result<impl FnMut() -> N + 'a, ParsingError> {
        let mut expr = self.parse(input, true)?;
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
    fn parse(
        &self,
        input: &str,
        optimize: bool,
    ) -> Result<MathAssembly<'a, N, (), usize>, ParsingError> {
        parse(
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
            optimize,
        )
    }
    pub fn build_as_parser(self) -> impl Fn(&str, N) -> Result<N, ParsingError> + 'a {
        move |input: &str, var| {
            self.parse(input, false)
                .map(|mut asm| asm.eval(|_: &()| var))
        }
    }
    pub fn build_as_function(self, input: &str) -> Result<impl FnMut(N) -> N + 'a, ParsingError> {
        let mut expr = self.parse(input, true)?;
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
            variables: ThreeVariables([self.variables.0, self.variables.1, name.into()]),
        }
    }
    fn parse(
        &self,
        input: &str,
        optimize: bool,
    ) -> Result<MathAssembly<'a, N, bool, usize>, ParsingError> {
        parse(
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
            optimize,
        )
    }
    pub fn build_as_parser(self) -> impl Fn(&str, N, N) -> Result<N, ParsingError> + 'a {
        move |input, v1, v2| {
            self.parse(input, false)
                .map(|mut asm| asm.eval(|var: &bool| if *var { v1 } else { v2 }))
        }
    }
    pub fn build_as_function(
        self,
        input: &str,
    ) -> Result<impl FnMut(N, N) -> N + 'a, ParsingError> {
        let mut expr = self.parse(input, true)?;
        Ok(move |v1, v2| expr.eval(|var: &bool| if *var { v1 } else { v2 }))
    }
}

impl<'a, N> EvalBuilder<'a, N, ThreeVariables>
where
    N: MathEvalNumber,
{
    pub fn add_variable(self, name: impl Into<String>) -> EvalBuilder<'a, N, FourVariables> {
        let mut iter = self.variables.0.into_iter();
        EvalBuilder {
            constants: self.constants,
            function_identifier: self.function_identifier,
            functions: self.functions,
            variables: FourVariables([
                iter.next().unwrap(),
                iter.next().unwrap(),
                iter.next().unwrap(),
                name.into(),
            ]),
        }
    }
    fn parse(
        &self,
        input: &str,
        optimize: bool,
    ) -> Result<MathAssembly<'a, N, u8, usize>, ParsingError> {
        parse(
            input,
            |inp| self.constants.get(inp).copied(),
            |inp| self.function_identifier.get(inp).copied(),
            |inp| {
                self.variables
                    .0
                    .iter()
                    .position(|var| *var == inp)
                    .map(|i| i as u8)
            },
            |index| self.functions[*index],
            optimize,
        )
    }
    fn select_variable(&self, i: u8, v1: N, v2: N, v3: N) -> N {
        match i {
            0 => v1,
            1 => v2,
            2 => v3,
            _ => unreachable!(),
        }
    }
    pub fn build_as_parser(self) -> impl Fn(&str, N, N, N) -> Result<N, ParsingError> + 'a {
        move |input, v1, v2, v3| {
            self.parse(input, false)
                .map(|mut asm| asm.eval(|var: &u8| self.select_variable(*var, v1, v2, v3)))
        }
    }
    pub fn build_as_function(
        self,
        input: &str,
    ) -> Result<impl FnMut(N, N, N) -> N + 'a, ParsingError> {
        let mut expr = self.parse(input, true)?;
        Ok(move |v1, v2, v3| expr.eval(|var: &u8| self.select_variable(*var, v1, v2, v3)))
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
            variables: ManyVariables(self.variables.0.into_iter().chain([name.into()]).collect()),
        }
    }
    fn select_variable(&self, i: u8, v1: N, v2: N, v3: N, v4: N) -> N {
        match i {
            0 => v1,
            1 => v2,
            2 => v3,
            3 => v4,
            _ => unreachable!(),
        }
    }
    fn parse(
        &self,
        input: &str,
        optimize: bool,
    ) -> Result<MathAssembly<'a, N, u8, usize>, ParsingError> {
        parse(
            input,
            |inp| self.constants.get(inp).copied(),
            |inp| self.function_identifier.get(inp).copied(),
            |inp| {
                self.variables
                    .0
                    .iter()
                    .position(|var| var == inp)
                    .map(|i| i as u8)
            },
            |index| self.functions[*index],
            optimize,
        )
    }
    pub fn build_as_parser(self) -> impl Fn(&str, N, N, N, N) -> Result<N, ParsingError> + 'a {
        move |input, v1, v2, v3, v4| {
            self.parse(input, false)
                .map(|mut asm| asm.eval(|var: &u8| self.select_variable(*var, v1, v2, v3, v4)))
        }
    }
    pub fn build_as_function(
        self,
        input: &str,
    ) -> Result<impl FnMut(N, N, N, N) -> N + 'a, ParsingError> {
        let mut expr = self.parse(input, true)?;
        Ok(move |v1, v2, v3, v4| expr.eval(|var: &u8| self.select_variable(*var, v1, v2, v3, v4)))
    }
}

impl<'a, N> EvalBuilder<'a, N, ManyVariables>
where
    N: MathEvalNumber,
{
    fn parse(
        &self,
        input: &str,
        optimize: bool,
    ) -> Result<MathAssembly<'a, N, usize, usize>, ParsingError> {
        parse(
            input,
            |inp| self.constants.get(inp).copied(),
            |inp| self.function_identifier.get(inp).copied(),
            |inp| self.variables.0.iter().position(|var| *var == inp),
            |index| self.functions[*index],
            optimize,
        )
    }
    pub fn build_as_parser(self) -> impl Fn(&str, &[N]) -> Result<N, ParsingError> + 'a {
        move |input, vars: &[N]| {
            self.parse(input, false)
                .map(|mut asm| asm.eval(|v: &usize| vars[*v]))
        }
    }
    pub fn build_as_function(
        self,
        input: &str,
    ) -> Result<impl FnMut(&[N]) -> N + 'a, ParsingError> {
        let mut expr = self.parse(input, true)?;
        Ok(move |vars: &[N]| expr.eval(|v: &usize| vars[*v]))
    }
}

#[cfg(test)]
#[test]
fn test_eval_parser() {
    assert_eq!(
        EvalBuilder::new()
            .add_constant("g", 10.0)
            .add_fn1("inv", &|v| 1.0 / v)
            .build_as_function("inv(g)")
            .unwrap()(),
        0.1
    );
    assert_eq!(
        EvalBuilder::new()
            .add_constant("c", 299792.0)
            .add_fn1("tometers", &|v| v * 1000.0)
            .add_fn1("tomilimeters", &|v| v * 1000.0)
            .build_as_parser()("tomilimeters(tometers(c))")
        .unwrap(),
        299792000000.0
    );
    let mut func1 = EvalBuilder::new()
        .add_constant("g", 9.8)
        .add_variable("t")
        .build_as_function("0.5*g*t^2")
        .unwrap();
    assert_eq!(func1(12.0), 705.6);
    assert_eq!(func1(3.0), 44.1);
    let mut func2 = EvalBuilder::new()
        .add_variable("t")
        .add_variable("v")
        .add_constant("g", 9.8)
        .build_as_function("0.5*g*t^2 + v*t")
        .unwrap();
    assert_eq!(func2(12.0, 0.0), 705.6);
    assert_eq!(func2(3.0, 46.9), 44.1 + 140.7);
}
