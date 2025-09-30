#![allow(unused_variables)]
use std::{
    collections::HashMap,
    fmt::{Debug, Display},
    hash::Hash,
    ops::RangeInclusive,
};

use asm::{CFPointer, MathAssembly, Stack};
use number::MathEvalNumber;
use seq_macro::seq;
use syntax::PostfixMathAst;
use tokenizer::TokenStream;

pub mod asm;
pub mod number;
pub mod syntax;
pub mod tokenizer;

const NAME_LIMIT: u8 = 32;
const NAME_LIMIT_ERROR_MSG: &str = "Identifier exceeds maximum length of 32 characters.";

pub trait VariableIdentifier: Clone + Copy + Debug + Eq + 'static {}

impl<T> VariableIdentifier for T where T: Clone + Copy + Debug + Hash + Eq + 'static {}

pub trait FunctionIdentifier: Clone + Copy + Eq + Debug + 'static {}

impl<T> FunctionIdentifier for T where T: Clone + Copy + Debug + Eq + 'static {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Fac,
    Neg,
    DoubleFac,
}

impl UnaryOp {
    pub fn parse(input: char) -> Option<Self> {
        match input {
            '!' => Some(UnaryOp::Fac),
            '-' => Some(UnaryOp::Neg),
            _ => None,
        }
    }

    pub fn eval<N: MathEvalNumber>(self, value: N::AsArg<'_>) -> N {
        match self {
            UnaryOp::Fac => N::factorial(value),
            UnaryOp::Neg => -value,
            UnaryOp::DoubleFac => N::double_factorial(value),
        }
    }
}

impl Display for UnaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            UnaryOp::Fac => "!",
            UnaryOp::Neg => "-",
            UnaryOp::DoubleFac => "!!",
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Mod,
}

impl BinaryOp {
    pub fn parse(input: char) -> Option<BinaryOp> {
        match input {
            '+' => Some(BinaryOp::Add),
            '-' => Some(BinaryOp::Sub),
            '*' => Some(BinaryOp::Mul),
            '/' => Some(BinaryOp::Div),
            '^' => Some(BinaryOp::Pow),
            '%' => Some(BinaryOp::Mod),
            _ => None,
        }
    }
    pub fn eval<N: MathEvalNumber>(self, lhs: N::AsArg<'_>, rhs: N::AsArg<'_>) -> N {
        match self {
            BinaryOp::Add => lhs + rhs,
            BinaryOp::Sub => lhs - rhs,
            BinaryOp::Mul => lhs * rhs,
            BinaryOp::Div => lhs / rhs,
            BinaryOp::Pow => N::pow(lhs, rhs),
            BinaryOp::Mod => N::modulo(lhs, rhs),
        }
    }
    pub fn as_char(self) -> char {
        match self {
            BinaryOp::Add => '+',
            BinaryOp::Sub => '-',
            BinaryOp::Mul => '*',
            BinaryOp::Div => '/',
            BinaryOp::Pow => '^',
            BinaryOp::Mod => '%',
        }
    }
    pub fn is_commutative(self) -> bool {
        matches!(self, BinaryOp::Add | BinaryOp::Mul)
    }
    pub fn precedence(self) -> u8 {
        match self {
            BinaryOp::Add => 0,
            BinaryOp::Sub => 0,
            BinaryOp::Mul => 1,
            BinaryOp::Div => 1,
            BinaryOp::Mod => 1,
            BinaryOp::Pow => 2,
        }
    }
    pub fn is_left_associative(self) -> bool {
        !matches!(self, BinaryOp::Pow)
    }
}

impl Display for BinaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_char())
    }
}

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
    PipeAbsNotClosed,
    NumberParsingError,
    MisplacedOperator,
    UnknownVariableOrConstant,
    UnknownFunction,
    NotEnoughArguments,
    TooManyArguments,
    EmptyParenthesis,
    EmptyArgument,
    EmptyInput,
    NameTooLong,
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
            ParsingErrorKind::EmptyParenthesis => "Parentheses should not be empty",
            ParsingErrorKind::EmptyArgument => "Function arguments shouldn't be empty",
            ParsingErrorKind::EmptyInput => "Input shouldn't be empty",
            ParsingErrorKind::PipeAbsNotClosed => "Unmatched '|' in absolute value expression",
            ParsingErrorKind::NameTooLong => NAME_LIMIT_ERROR_MSG,
        })
    }
}

pub trait VariableStore<N: MathEvalNumber, V: VariableIdentifier> {
    fn get<'a>(&'a self, var: V) -> N::AsArg<'a>;
}

impl<N, V> VariableStore<N, V> for ()
where
    N: MathEvalNumber,
    V: VariableIdentifier,
{
    #[inline]
    fn get<'a>(&'a self, var: V) -> N::AsArg<'a> {
        panic!("Tried to get \"{var:?}\" variable from an empty variable store")
    }
}

impl<N> VariableStore<N, ()> for (N,)
where
    N: MathEvalNumber,
{
    #[inline]
    fn get<'a>(&'a self, _var: ()) -> N::AsArg<'a> {
        self.0.asarg()
    }
}

pub fn compile<'a, N: MathEvalNumber, V: VariableIdentifier, F: FunctionIdentifier>(
    input: &str,
    custom_constant_parser: impl Fn(&str) -> Option<N>,
    custom_function_parser: impl Fn(&str) -> Option<(F, u8, Option<u8>)>,
    custom_variable_parser: impl Fn(&str) -> Option<V>,
    function_to_pointer: impl Fn(F) -> CFPointer<'a, N>,
) -> Result<MathAssembly<'a, N, F>, ParsingError> {
    let token_stream = TokenStream::new(input).map_err(|e| e.to_general())?;
    let mut syntax_tree = PostfixMathAst::new(
        &token_stream.0,
        custom_constant_parser,
        custom_function_parser,
        custom_variable_parser,
    )
    .map_err(|e| e.to_general(input, &token_stream.0))?;
    syntax_tree.aot_evaluation(&function_to_pointer);
    syntax_tree.displacing_simplification();
    todo!()
}

pub fn evaluate<'a, 'b, N: MathEvalNumber, V: VariableIdentifier, F: FunctionIdentifier>(
    input: &str,
    custom_constant_parser: impl Fn(&str) -> Option<N>,
    custom_function_parser: impl Fn(&str) -> Option<(F, u8, Option<u8>)>,
    custom_variable_parser: impl Fn(&str) -> Option<V>,
    function_to_pointer: impl Fn(F) -> CFPointer<'a, N>,
    variable_values: &impl VariableStore<N, V>,
) -> Result<N, ParsingError> {
    let token_stream = TokenStream::new(input).map_err(|e| e.to_general())?;
    match PostfixMathAst::new(
        &token_stream.0,
        custom_constant_parser,
        custom_function_parser,
        custom_variable_parser,
    ) {
        Ok(s) => Ok(s.eval(function_to_pointer, variable_values)),
        Err(e) => Err(e.to_general(input, &token_stream.0)),
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NoVariable;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OneVariable(String);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TwoVariables([String; 2]);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThreeVariables([String; 3]);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FourVariables([String; 4]);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManyVariables(Vec<String>);

#[derive(Clone, Debug)]
pub struct EvalRef;

#[derive(Clone, Debug)]
pub struct EvalCopy;

#[derive(Clone, Debug)]
pub struct EvalBuilder<'a, N, V, E>
where
    N: MathEvalNumber,
{
    constants: HashMap<String, N>,
    function_identifier: HashMap<String, (usize, u8, Option<u8>)>,
    functions: Vec<CFPointer<'a, N>>,
    variables: V,
    evalmethod: E,
}

impl<N> Default for EvalBuilder<'_, N, NoVariable, EvalCopy>
where
    N: MathEvalNumber,
{
    fn default() -> Self {
        Self {
            constants: HashMap::default(),
            function_identifier: HashMap::default(),
            functions: Vec::default(),
            variables: NoVariable,
            evalmethod: EvalCopy,
        }
    }
}

impl<'a, N, V, E> EvalBuilder<'a, N, V, E>
where
    N: MathEvalNumber,
{
    pub fn add_constant(mut self, name: impl Into<String>, value: N) -> Self {
        self.constants.insert(name.into(), value);
        self
    }
    pub fn add_fn1(
        mut self,
        name: impl Into<String>,
        function: &'a dyn Fn(N::AsArg<'_>) -> N,
    ) -> Self {
        self.function_identifier
            .insert(name.into(), (self.functions.len(), 1, Some(1)));
        self.functions.push(CFPointer::Single(function));
        self
    }
    pub fn add_fn2(
        mut self,
        name: impl Into<String>,
        function: &'a dyn Fn(N::AsArg<'_>, N::AsArg<'_>) -> N,
    ) -> Self {
        self.function_identifier
            .insert(name.into(), (self.functions.len(), 2, Some(2)));
        self.functions.push(CFPointer::Dual(function));
        self
    }
    pub fn add_fn3(
        mut self,
        name: impl Into<String>,
        function: &'a dyn Fn(N::AsArg<'_>, N::AsArg<'_>, N::AsArg<'_>) -> N,
    ) -> Self {
        self.function_identifier
            .insert(name.into(), (self.functions.len(), 3, Some(3)));
        self.functions.push(CFPointer::Triple(function));
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

impl<'a, N, V> EvalBuilder<'a, N, V, EvalCopy>
where
    N: MathEvalNumber,
{
    pub fn use_ref(self) -> EvalBuilder<'a, N, V, EvalRef> {
        EvalBuilder {
            constants: self.constants,
            function_identifier: self.function_identifier,
            functions: self.functions,
            variables: self.variables,
            evalmethod: EvalRef,
        }
    }
}

impl<'a, N, E> EvalBuilder<'a, N, NoVariable, E>
where
    N: MathEvalNumber,
    E: 'static,
{
    pub fn add_variable(self, name: impl Into<String>) -> EvalBuilder<'a, N, OneVariable, E> {
        EvalBuilder {
            variables: OneVariable(name.into()),
            constants: self.constants,
            function_identifier: self.function_identifier,
            functions: self.functions,
            evalmethod: self.evalmethod,
        }
    }
    pub fn build_as_evaluator(self) -> impl Fn(&str) -> Result<N, ParsingError> + 'a {
        move |input: &str| {
            evaluate(
                input,
                |inp| self.constants.get(inp).cloned(),
                |inp| self.function_identifier.get(inp).copied(),
                |_| None::<()>,
                |idx| self.functions[idx],
                &(),
            )
        }
    }
}

impl<'a, N> EvalBuilder<'a, N, NoVariable, EvalRef>
where
    N: MathEvalNumber,
{
    pub fn build_as_function(self, input: &str) -> Result<impl FnMut() -> N + 'a, ParsingError> {
        let expr = compile(
            input,
            |inp| self.constants.get(inp).cloned(),
            |inp| self.function_identifier.get(inp).copied(),
            |_| None::<()>,
            |idx| self.functions[idx],
        )?;
        let mut stack = Stack::with_capacity(expr.stack_alloc_size());
        Ok(move || expr.eval(&[], &mut stack))
    }
}

impl<'a, N> EvalBuilder<'a, N, NoVariable, EvalCopy>
where
    N: for<'b> MathEvalNumber<AsArg<'b> = N> + Copy,
{
    pub fn new() -> Self {
        Self::default()
    }
    pub fn build_as_function(self, input: &str) -> Result<impl FnMut() -> N + 'a, ParsingError> {
        let expr = compile(
            input,
            |inp| self.constants.get(inp).copied(),
            |inp| self.function_identifier.get(inp).copied(),
            |_| None::<()>,
            |idx| self.functions[idx],
        )?;
        let mut stack = Stack::with_capacity(expr.stack_alloc_size());
        Ok(move || expr.eval_copy(&[], &mut stack))
    }
}

impl<'a, N, E> EvalBuilder<'a, N, OneVariable, E>
where
    N: MathEvalNumber,
    E: 'static,
{
    pub fn add_variable(self, name: impl Into<String>) -> EvalBuilder<'a, N, TwoVariables, E> {
        EvalBuilder {
            constants: self.constants,
            function_identifier: self.function_identifier,
            functions: self.functions,
            variables: TwoVariables([self.variables.0, name.into()]),
            evalmethod: self.evalmethod,
        }
    }
    pub fn build_as_evaluator(self) -> impl Fn(&str, N) -> Result<N, ParsingError> + 'a {
        move |input: &str, v0: N| {
            evaluate(
                input,
                |inp| self.constants.get(inp).cloned(),
                |inp| self.function_identifier.get(inp).copied(),
                |inp| (self.variables.0 == inp).then_some(()),
                |idx| self.functions[idx],
                &(v0,),
            )
        }
    }
}

impl<'a, N> EvalBuilder<'a, N, OneVariable, EvalRef>
where
    N: MathEvalNumber,
{
    pub fn build_as_function<'b>(
        self,
        input: &str,
    ) -> Result<impl FnMut(N::AsArg<'b>) -> N + 'a, ParsingError> {
        let expr = compile(
            input,
            |inp| self.constants.get(inp).cloned(),
            |inp| self.function_identifier.get(inp).copied(),
            |inp| (self.variables.0 == inp).then_some(()),
            |idx| self.functions[idx],
        )?;
        let mut stack = Stack::with_capacity(expr.stack_alloc_size());
        Ok(move |v0| expr.eval(&[v0], &mut stack))
    }
}

impl<'a, N> EvalBuilder<'a, N, OneVariable, EvalCopy>
where
    N: for<'b> MathEvalNumber<AsArg<'b> = N> + Copy,
{
    pub fn build_as_function<'b>(
        self,
        input: &str,
    ) -> Result<impl FnMut(N::AsArg<'b>) -> N + 'a, ParsingError> {
        let expr = compile(
            input,
            |inp| self.constants.get(inp).copied(),
            |inp| self.function_identifier.get(inp).copied(),
            |inp| (self.variables.0 == inp).then_some(()),
            |idx| self.functions[idx],
        )?;
        let mut stack = Stack::with_capacity(expr.stack_alloc_size());
        Ok(move |v0| expr.eval_copy(&[v0], &mut stack))
    }
}

seq!(I in 2..10 {
    impl<N> VariableStore<N, u8> for [N; I]
    where
        N: MathEvalNumber,
    {
        #[inline]
        fn get<'a>(&'a self, var: u8) -> N::AsArg<'a> {
            self[var as usize].asarg()
        }
    }
});

macro_rules! fn_build_as_evaluator {
    ($n: expr) => {
        seq!(I in 0..$n {
            pub fn build_as_evaluator(self)
                -> impl Fn(&str, #(N,)*) -> Result<N, ParsingError> + 'a {
                move |input, #(v~I,)*| {
                    evaluate(
                        input,
                        |inp| self.constants.get(inp).cloned(),
                        |inp| self.function_identifier.get(inp).copied(),
                        |inp| self.variables.0
                            .iter()
                            .position(|var| var == inp)
                            .map(|i| i as u8),
                        |idx| self.functions[idx],
                        &[#(v~I,)*],
                    )
                }
            }
        });
    };
}

macro_rules! fn_build_as_function {
    ($n: expr, $f: ident) => {
        seq!(I in 0..$n {
            pub fn build_as_function<'b>(
                self,
                input: &str,
            ) -> Result<impl FnMut(#(N::AsArg<'b>,)*) -> N + 'a, ParsingError> {
                let expr = compile(
                    input,
                    |inp| self.constants.get(inp).cloned(),
                    |inp| self.function_identifier.get(inp).copied(),
                    |inp| self.variables.0.iter().position(|var| var == inp),
                    |idx| self.functions[idx],
                )?;
                let mut stack = Stack::with_capacity(expr.stack_alloc_size());
                Ok(move |#(v~I,)*| expr.$f(&[#(v~I,)*], &mut stack))
            }
        });
    };
}

macro_rules! fn_add_variable {
    ($n: expr, $next: ident) => {
        seq!(I in 0..$n {
            pub fn add_variable(self, name: impl Into<String>) -> EvalBuilder<'a, N, $next, E> {
                let mut iter = self.variables.0.into_iter();
                EvalBuilder {
                    constants: self.constants,
                    function_identifier: self.function_identifier,
                    functions: self.functions,
                    variables: $next([#(iter.next().unwrap(),)* name.into()]),
                    evalmethod: self.evalmethod
                }
            }
        });
    };
}

impl<'a, N, E> EvalBuilder<'a, N, TwoVariables, E>
where
    N: MathEvalNumber,
    E: 'static,
{
    fn_add_variable!(2, ThreeVariables);
    fn_build_as_evaluator!(2);
}

impl<'a, N> EvalBuilder<'a, N, TwoVariables, EvalCopy>
where
    N: for<'b> MathEvalNumber<AsArg<'b> = N> + Copy,
{
    fn_build_as_function!(2, eval_copy);
}

impl<'a, N> EvalBuilder<'a, N, TwoVariables, EvalRef>
where
    N: MathEvalNumber,
{
    fn_build_as_function!(2, eval);
}

impl<'a, N, E> EvalBuilder<'a, N, ThreeVariables, E>
where
    N: MathEvalNumber,
    E: 'static,
{
    fn_add_variable!(3, FourVariables);
    fn_build_as_evaluator!(3);
}

impl<'a, N> EvalBuilder<'a, N, ThreeVariables, EvalRef>
where
    N: MathEvalNumber,
{
    fn_build_as_function!(3, eval);
}

impl<'a, N> EvalBuilder<'a, N, ThreeVariables, EvalCopy>
where
    N: for<'b> MathEvalNumber<AsArg<'b> = N> + Copy,
{
    fn_build_as_function!(3, eval_copy);
}

impl<'a, N, E> EvalBuilder<'a, N, FourVariables, E>
where
    N: MathEvalNumber,
    E: 'static,
{
    pub fn add_variable(self, name: impl Into<String>) -> EvalBuilder<'a, N, ManyVariables, E> {
        EvalBuilder {
            constants: self.constants,
            function_identifier: self.function_identifier,
            functions: self.functions,
            variables: ManyVariables(self.variables.0.into_iter().chain([name.into()]).collect()),
            evalmethod: self.evalmethod,
        }
    }
    fn_build_as_evaluator!(4);
}

impl<'a, N> EvalBuilder<'a, N, FourVariables, EvalRef>
where
    N: MathEvalNumber,
{
    fn_build_as_function!(4, eval);
}

impl<'a, N> EvalBuilder<'a, N, FourVariables, EvalCopy>
where
    N: for<'b> MathEvalNumber<AsArg<'b> = N> + Copy,
{
    fn_build_as_function!(4, eval_copy);
}

impl<N> VariableStore<N, usize> for &'_ [N]
where
    N: MathEvalNumber,
{
    #[inline]
    fn get<'a>(&'a self, var: usize) -> <N as MathEvalNumber>::AsArg<'a> {
        self[var].asarg()
    }
}

impl<'a, N, E> EvalBuilder<'a, N, ManyVariables, E>
where
    N: MathEvalNumber,
    E: 'static,
{
    pub fn build_as_evaluator(self) -> impl Fn(&str, &[N]) -> Result<N, ParsingError> + 'a {
        move |input, vars| {
            evaluate(
                input,
                |inp| self.constants.get(inp).cloned(),
                |inp| self.function_identifier.get(inp).copied(),
                |inp| self.variables.0.iter().position(|var| var == inp),
                |idx| self.functions[idx],
                &vars,
            )
        }
    }
}

impl<'a, N> EvalBuilder<'a, N, ManyVariables, EvalRef>
where
    N: MathEvalNumber,
{
    pub fn build_as_function<'b>(
        self,
        input: &str,
    ) -> Result<impl FnMut(&'b [N::AsArg<'b>]) -> N + 'a, ParsingError> {
        let expr = compile(
            input,
            |inp| self.constants.get(inp).cloned(),
            |inp| self.function_identifier.get(inp).copied(),
            |inp| self.variables.0.iter().position(|var| var == inp),
            |idx| self.functions[idx],
        )?;
        let mut stack = Stack::with_capacity(expr.stack_alloc_size());
        Ok(move |vars| expr.eval(vars, &mut stack))
    }
}

impl<'a, N> EvalBuilder<'a, N, ManyVariables, EvalCopy>
where
    N: for<'b> MathEvalNumber<AsArg<'b> = N> + Copy,
{
    pub fn build_as_function<'b>(
        self,
        input: &str,
    ) -> Result<impl FnMut(&'b [N::AsArg<'b>]) -> N + 'a, ParsingError> {
        let expr = compile(
            input,
            |inp| self.constants.get(inp).copied(),
            |inp| self.function_identifier.get(inp).copied(),
            |inp| self.variables.0.iter().position(|var| var == inp),
            |idx| self.functions[idx],
        )?;
        let mut stack = Stack::with_capacity(expr.stack_alloc_size());
        Ok(move |vars| expr.eval_copy(vars, &mut stack))
    }
}

#[cfg(test)]
mod test {
    use std::collections::HashMap;

    use super::*;
    use crate::asm::CFPointer;

    #[derive(PartialEq, Debug)]
    struct UnexpectedBuilderFieldValues<V> {
        constants: Option<(HashMap<String, f64>, HashMap<String, f64>)>,
        function_identifier: Option<(
            HashMap<String, (usize, u8, Option<u8>)>,
            HashMap<String, (usize, u8, Option<u8>)>,
        )>,
        functions: Vec<(usize, f64)>,
        variables: Option<(V, V)>,
    }

    impl<V> Display for UnexpectedBuilderFieldValues<V>
    where
        V: Debug,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            if let Some(consts) = &self.constants {
                writeln!(f, "constants: {:?} != {:?}", consts.0, consts.1)?;
            }
            if let Some(fi) = &self.function_identifier {
                writeln!(f, "function_identifier: {:?} != {:?}", fi.0, fi.1)?;
            }
            if !self.functions.is_empty() {
                writeln!(f, "functions: {:?}", self.functions)?;
            }
            if let Some(vars) = &self.variables {
                writeln!(f, "variables: {:?} != {:?}", vars.0, vars.1)?;
            }
            Ok(())
        }
    }

    #[allow(clippy::result_large_err)]
    fn compare<V: PartialEq, E>(
        builder: EvalBuilder<'_, f64, V, E>,
        constants: impl Iterator<Item = (&'static str, f64)>,
        function_identifier: impl Iterator<Item = (&'static str, usize, u8, Option<u8>)>,
        variables: V,
        _evalmethod: E,
    ) -> Option<UnexpectedBuilderFieldValues<V>> {
        let constants = constants.map(|(n, v)| (n.to_string(), v)).collect();
        let function_identifier = function_identifier
            .map(|(n, id, min, max)| (n.to_string(), (id, min, max)))
            .collect();
        let res = UnexpectedBuilderFieldValues {
            constants: (builder.constants != constants).then_some((builder.constants, constants)),
            function_identifier: (builder.function_identifier != function_identifier)
                .then_some((builder.function_identifier, function_identifier)),
            functions: builder
                .functions
                .iter()
                .enumerate()
                .filter_map(|(i, cfp)| match cfp {
                    CFPointer::Single(f) => Some((i, f(0.0))).filter(|(i, v)| *i != *v as usize),
                    CFPointer::Dual(f) => Some((i, f(0.0, 0.0))).filter(|(i, v)| *i != *v as usize),
                    CFPointer::Triple(f) => {
                        Some((i, f(0.0, 0.0, 0.0))).filter(|(i, v)| *i != *v as usize)
                    }
                    CFPointer::Flexible(f) => Some((i, f(&[]))).filter(|(i, v)| *i != *v as usize),
                })
                .collect(),
            variables: (builder.variables != variables).then_some((builder.variables, variables)),
        };

        (res.constants.is_some()
            || res.function_identifier.is_some()
            || !res.functions.is_empty()
            || res.variables.is_some())
        .then_some(res)
    }

    #[test]
    fn test_eval_builder() {
        macro_rules! test {
            ($cmp: expr) => {
                if let Some(ubfv) = $cmp {
                    panic!("Difference in parameters found:\n{ubfv}");
                }
            };
        }

        test!(compare(
            EvalBuilder::new(),
            [].into_iter(),
            [].into_iter(),
            NoVariable,
            EvalCopy
        ));

        test!(compare(
            EvalBuilder::new().add_constant("c", 3.57),
            [("c", 3.57)].into_iter(),
            [].into_iter(),
            NoVariable,
            EvalCopy
        ));

        test!(compare(
            EvalBuilder::new().add_variable("x"),
            [].into_iter(),
            [].into_iter(),
            OneVariable(String::from("x")),
            EvalCopy
        ));

        test!(compare(
            EvalBuilder::new().use_ref(),
            [].into_iter(),
            [].into_iter(),
            NoVariable,
            EvalRef
        ));

        test!(compare(
            EvalBuilder::new().add_fn1("f1", &|_| 0.0),
            [].into_iter(),
            [("f1", 0, 1, Some(1))].into_iter(),
            NoVariable,
            EvalCopy
        ));

        test!(compare(
            EvalBuilder::new()
                .add_fn1("f2", &|_| 0.0)
                .add_constant("c1", 7.319),
            [("c1", 7.319)].into_iter(),
            [("f2", 0, 1, Some(1))].into_iter(),
            NoVariable,
            EvalCopy
        ));

        test!(compare(
            EvalBuilder::new()
                .add_fn1("f1", &|_| 0.0)
                .add_fn2("f2", &|_, _| 1.0),
            [].into_iter(),
            [("f1", 0, 1, Some(1)), ("f2", 1, 2, Some(2))].into_iter(),
            NoVariable,
            EvalCopy
        ));

        test!(compare(
            EvalBuilder::new()
                .add_fn2("f2", &|_, _| 0.0)
                .add_fn3("f3", &|_, _, _| 1.0),
            [].into_iter(),
            [("f2", 0, 2, Some(2)), ("f3", 1, 3, Some(3))].into_iter(),
            NoVariable,
            EvalCopy
        ));

        test!(compare(
            EvalBuilder::new()
                .add_fn2("f2", &|_, _| 0.0)
                .add_fn3("f3", &|_, _, _| 1.0)
                .add_fn_flex("ff", 3, None, &|_| 2.0),
            [].into_iter(),
            [
                ("f2", 0, 2, Some(2)),
                ("f3", 1, 3, Some(3)),
                ("ff", 2, 3, None)
            ]
            .into_iter(),
            NoVariable,
            EvalCopy
        ));

        test!(compare(
            EvalBuilder::new()
                .add_variable("t")
                .add_fn2("f2", &|_, _| 0.0)
                .add_fn3("f3", &|_, _, _| 1.0)
                .add_fn_flex("ff", 3, None, &|_| 2.0),
            [].into_iter(),
            [
                ("f2", 0, 2, Some(2)),
                ("f3", 1, 3, Some(3)),
                ("ff", 2, 3, None)
            ]
            .into_iter(),
            OneVariable(String::from("t")),
            EvalCopy
        ));

        test!(compare(
            EvalBuilder::new().add_variable("y").add_variable("x"),
            [].into_iter(),
            [].into_iter(),
            TwoVariables([String::from("y"), String::from("x")]),
            EvalCopy
        ));

        test!(compare(
            EvalBuilder::new()
                .add_variable("y")
                .use_ref()
                .add_variable("x"),
            [].into_iter(),
            [].into_iter(),
            TwoVariables([String::from("y"), String::from("x")]),
            EvalRef
        ));

        test!(compare(
            EvalBuilder::new()
                .add_variable("y")
                .add_fn2("f2", &|_, _| 0.0)
                .add_variable("x")
                .add_fn3("f3", &|_, _, _| 1.0)
                .add_fn_flex("ff", 3, None, &|_| 2.0),
            [].into_iter(),
            [
                ("f2", 0, 2, Some(2)),
                ("f3", 1, 3, Some(3)),
                ("ff", 2, 3, None)
            ]
            .into_iter(),
            TwoVariables([String::from("y"), String::from("x")]),
            EvalCopy
        ));

        test!(compare(
            EvalBuilder::new()
                .add_variable("y")
                .add_variable("x")
                .add_variable("z"),
            [].into_iter(),
            [].into_iter(),
            ThreeVariables([String::from("y"), String::from("x"), String::from("z")]),
            EvalCopy
        ));

        test!(compare(
            EvalBuilder::new()
                .add_fn3("f0", &|_, _, _| 0.0)
                .add_variable("y")
                .add_fn3("f1", &|_, _, _| 1.0)
                .add_fn3("f2", &|_, _, _| 2.0)
                .add_variable("x")
                .add_fn3("f3", &|_, _, _| 3.0)
                .add_variable("z")
                .use_ref(),
            [].into_iter(),
            [
                ("f0", 0, 3, Some(3)),
                ("f1", 1, 3, Some(3)),
                ("f2", 2, 3, Some(3)),
                ("f3", 3, 3, Some(3))
            ]
            .into_iter(),
            ThreeVariables([String::from("y"), String::from("x"), String::from("z")]),
            EvalRef
        ));

        test!(compare(
            EvalBuilder::new()
                .add_variable("y")
                .add_variable("x")
                .add_variable("z")
                .add_variable("w"),
            [].into_iter(),
            [].into_iter(),
            FourVariables([
                String::from("y"),
                String::from("x"),
                String::from("z"),
                String::from("w")
            ]),
            EvalCopy
        ));

        test!(compare(
            EvalBuilder::new()
                .add_fn1("f0", &|_| 0.0)
                .add_variable("y")
                .add_constant("c", 9.999999)
                .add_fn2("f1", &|_, _| 1.0)
                .add_variable("x")
                .add_fn3("f2", &|_, _, _| 2.0)
                .add_variable("z")
                .add_fn_flex("f3", 1, Some(5), &|_| 3.0)
                .add_variable("w")
                .add_fn1("f4", &|_| 4.0),
            [("c", 9.999999)].into_iter(),
            [
                ("f0", 0, 1, Some(1)),
                ("f1", 1, 2, Some(2)),
                ("f2", 2, 3, Some(3)),
                ("f3", 3, 1, Some(5)),
                ("f4", 4, 1, Some(1))
            ]
            .into_iter(),
            FourVariables([
                String::from("y"),
                String::from("x"),
                String::from("z"),
                String::from("w")
            ]),
            EvalCopy
        ));

        test!(compare(
            EvalBuilder::new()
                .add_variable("y")
                .add_variable("x")
                .add_variable("z")
                .add_variable("v")
                .add_variable("w"),
            [].into_iter(),
            [].into_iter(),
            ManyVariables(vec![
                String::from("y"),
                String::from("x"),
                String::from("z"),
                String::from("v"),
                String::from("w")
            ]),
            EvalCopy
        ));

        test!(compare(
            EvalBuilder::new()
                .add_fn1("f0", &|_| 0.0)
                .add_variable("y")
                .use_ref()
                .add_constant("c", 9.999999)
                .add_constant("ce", 2.222222)
                .add_fn2("f1", &|_, _| 1.0)
                .add_variable("x")
                .add_fn3("f2", &|_, _, _| 2.0)
                .add_variable("z")
                .add_fn_flex("f3", 1, Some(5), &|_| 3.0)
                .add_variable("w")
                .add_fn1("f4", &|_| 4.0)
                .add_variable("t")
                .add_fn_flex("f5", 5, None, &|_| 5.0),
            [("c", 9.999999), ("ce", 2.222222)].into_iter(),
            [
                ("f0", 0, 1, Some(1)),
                ("f1", 1, 2, Some(2)),
                ("f2", 2, 3, Some(3)),
                ("f3", 3, 1, Some(5)),
                ("f4", 4, 1, Some(1)),
                ("f5", 5, 5, None)
            ]
            .into_iter(),
            ManyVariables(vec![
                String::from("y"),
                String::from("x"),
                String::from("z"),
                String::from("w"),
                String::from("t")
            ]),
            EvalRef
        ));
    }
}
