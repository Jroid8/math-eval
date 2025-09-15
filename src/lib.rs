use std::{
    collections::HashMap,
    fmt::{Debug, Display},
    hash::Hash,
    ops::RangeInclusive,
};

use asm::{CFPointer, MathAssembly, Stack};
use number::MathEvalNumber;
use seq_macro::seq;
use syntax::SyntaxTree;
use tokenizer::{token_stream::TokenStream, token_tree::TokenTree};

pub mod asm;
pub mod number;
pub mod syntax;
pub mod tokenizer;
pub mod tree_utils;

pub trait VariableIdentifier: Clone + Copy + Eq + 'static {}

impl<T> VariableIdentifier for T where T: Clone + Copy + Debug + Hash + Eq + 'static {}

pub trait FunctionIdentifier: Clone + Copy + 'static {}

impl<T> FunctionIdentifier for T where T: Clone + Copy + Debug + 'static {}

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

pub trait VariableStore<N: MathEvalNumber, V: VariableIdentifier> {
    fn get<'a>(&'a self, var: V) -> N::AsArg<'a>;
}

impl<N, V> VariableStore<N, V> for ()
where
    N: MathEvalNumber,
    V: VariableIdentifier + Debug,
{
    fn get<'a>(&'a self, var: V) -> N::AsArg<'a> {
        panic!("Tried to get \"{var:?}\" variable from an empty variable store")
    }
}

impl<N> VariableStore<N, ()> for (N,)
where
    N: MathEvalNumber,
{
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
    variable_order: &[V],
) -> Result<MathAssembly<'a, N, F>, ParsingError> {
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
        variable_order,
    ))
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
    let token_tree =
        TokenTree::new(&token_stream).map_err(|e| e.to_general(input, &token_stream))?;
    match SyntaxTree::new(
        &token_tree,
        custom_constant_parser,
        custom_function_parser,
        custom_variable_parser,
    ) {
        Ok(s) => Ok(s.eval(function_to_pointer, variable_values)),
        Err(e) => Err(e.to_general(input, &token_tree)),
    }
}

#[derive(Clone, Debug)]
pub struct NoVariable;

#[derive(Clone, Debug)]
pub struct OneVariable(String);

#[derive(Clone, Debug)]
pub struct TwoVariables([String; 2]);

#[derive(Clone, Debug)]
pub struct ThreeVariables([String; 3]);

#[derive(Clone, Debug)]
pub struct FourVariables([String; 4]);

#[derive(Clone, Debug)]
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
    pub fn add_fn4(
        mut self,
        name: impl Into<String>,
        function: &'a dyn Fn(N::AsArg<'_>, N::AsArg<'_>, N::AsArg<'_>, N::AsArg<'_>) -> N,
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
            &[],
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
            &[],
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
                |inp| Some(()).filter(|_| self.variables.0 == inp),
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
            |inp| Some(()).filter(|_| self.variables.0 == inp),
            |idx| self.functions[idx],
            &[()],
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
            |inp| Some(()).filter(|_| self.variables.0 == inp),
            |idx| self.functions[idx],
            &[()],
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
                    &[#(I,)*],
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

struct EBManyVarStore<'a, N: MathEvalNumber>(&'a [N]);

impl<N> VariableStore<N, usize> for EBManyVarStore<'_, N>
where
    N: MathEvalNumber,
{
    fn get<'a>(&'a self, var: usize) -> <N as MathEvalNumber>::AsArg<'a> {
        self.0[var].asarg()
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
                &EBManyVarStore(vars),
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
            &(0..self.variables.0.len()).collect::<Vec<_>>(),
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
            &(0..self.variables.0.len()).collect::<Vec<_>>(),
        )?;
        let mut stack = Stack::with_capacity(expr.stack_alloc_size());
        Ok(move |vars| expr.eval_copy(vars, &mut stack))
    }
}

#[cfg(test)]
mod test {
    use std::f64::consts::*;
    use crate::EvalBuilder;

    fn sinh(x: f64) -> f64 {
        (E.powf(x) - E.powf(-x)) / 2.0
    }
    fn hypot(x: f64, y: f64) -> f64 {
        (x * x + y * y).sqrt()
    }
    fn clamp(x: f64, min: f64, max: f64) -> f64 {
        x.min(max).max(min)
    }
    fn lerp(a: f64, b: f64, t: f64) -> f64 {
        a + t * (b - a)
    }
    fn deg2rad(x: f64) -> f64 {
        x * PI / 180.0
    }
    fn atan2(x: f64, y: f64) -> f64 {
        x.atan2(y)
    }
    fn rad2deg(x: f64) -> f64 {
        x * 180.0 / PI
    }
    fn mean(vals: &[f64]) -> f64 {
        vals.iter().sum::<f64>() / vals.len() as f64
    }
}
