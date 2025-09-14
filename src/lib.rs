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

pub trait VariableIdentifier: Clone + Copy + Debug + Hash + Eq + 'static {}

impl<T> VariableIdentifier for T where T: Clone + Copy + Debug + Hash + Eq + 'static {}

pub trait FunctionIdentifier: Clone + Copy + Debug + 'static {}

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
    V: VariableIdentifier,
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
    if optimize {
        syntax_tree.aot_evaluation(&function_to_pointer);
        syntax_tree.displacing_simplification();
    }
    Ok(MathAssembly::new(
        &syntax_tree.0.arena,
        syntax_tree.0.root,
        function_to_pointer,
        variable_order,
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
        &[],
    )
    .map(|asm: MathAssembly<'_, N, (), ()>| {
        asm.eval(&[], &mut Stack::with_capacity(asm.stack_alloc_size()))
    })
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
            &[],
        )
    }
    pub fn build_as_parser(self) -> impl Fn(&str) -> Result<N, ParsingError> + 'a {
        move |input: &str| {
            self.parse(input, false)
                .map(|asm| asm.eval(&[], &mut Stack::with_capacity(asm.stack_alloc_size())))
        }
    }
}

impl<'a, N> EvalBuilder<'a, N, NoVariable, EvalRef>
where
    N: MathEvalNumber,
{
    pub fn build_as_function(self, input: &str) -> Result<impl FnMut() -> N + 'a, ParsingError> {
        let expr = self.parse(input, true)?;
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
        let expr = self.parse(input, true)?;
        let mut stack = Stack::with_capacity(expr.stack_alloc_size());
        Ok(move || expr.eval_copy(&[], &mut stack))
    }
}

macro_rules! fn_build_as_parser {
    ($n: expr) => {
        seq!(I in 0..$n {
            pub fn build_as_parser<'b>(
                self,
            ) -> impl Fn(&str, #(N::AsArg<'b>,)*) -> Result<N, ParsingError> + 'a {
                move |input, #(v~I,)*| {
                    self.parse(input, false)
                        .map(|asm| asm.eval(&[#(v~I,)*],
                            &mut Stack::with_capacity(asm.stack_alloc_size())))
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
                let expr = self.parse(input, true)?;
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

macro_rules! fn_parse {
    ($n: expr) => {
        seq!(I in 0..$n {
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
                    &[#(I,)*],
                )
            }
        });
    };
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
            &[()],
        )
    }
    fn_build_as_parser!(1);
}

impl<'a, N> EvalBuilder<'a, N, OneVariable, EvalRef>
where
    N: MathEvalNumber,
{
    fn_build_as_function!(1, eval);
}

impl<'a, N> EvalBuilder<'a, N, OneVariable, EvalCopy>
where
    N: for<'b> MathEvalNumber<AsArg<'b> = N> + Copy,
{
    fn_build_as_function!(1, eval_copy);
}

impl<'a, N, E> EvalBuilder<'a, N, TwoVariables, E>
where
    N: MathEvalNumber,
    E: 'static,
{
    fn_add_variable!(2, ThreeVariables);
    fn_parse!(2);
    fn_build_as_parser!(2);
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
    fn_parse!(3);
    fn_build_as_parser!(3);
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
    fn_parse!(4);
    fn_build_as_parser!(4);
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
            (0..self.variables.0.len()).collect::<Vec<_>>().as_slice(),
        )
    }
    pub fn build_as_parser<'b>(
        self,
    ) -> impl Fn(&str, &[N::AsArg<'b>]) -> Result<N, ParsingError> + 'a {
        move |input, vars: &[N::AsArg<'b>]| {
            self.parse(input, false)
                .map(|asm| asm.eval(vars, &mut Stack::with_capacity(asm.stack_alloc_size())))
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
    ) -> Result<impl FnMut(&[N::AsArg<'b>]) -> N + 'a, ParsingError> {
        let expr = self.parse(input, true)?;
        let mut stack = Stack::with_capacity(expr.stack_alloc_size());
        Ok(move |vars: &[N::AsArg<'b>]| expr.eval(vars, &mut stack))
    }
}

impl<'a, N> EvalBuilder<'a, N, ManyVariables, EvalCopy>
where
    N: for<'b> MathEvalNumber<AsArg<'b> = N> + Copy,
{
    pub fn build_as_function<'b>(
        self,
        input: &str,
    ) -> Result<impl FnMut(&[N::AsArg<'b>]) -> N + 'a, ParsingError> {
        let expr = self.parse(input, true)?;
        let mut stack = Stack::with_capacity(expr.stack_alloc_size());
        Ok(move |vars: &[N::AsArg<'b>]| expr.eval_copy(vars, &mut stack))
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

    #[test]
    fn test_eval_builder() {
        let threshold = f64::EPSILON * 10.0;
        macro_rules! test {
            ($bldr: expr, $exp: expr, ($($vars: tt)*), $res: expr) => {
                let func_res = $bldr.clone().build_as_function($exp).unwrap()($($vars)*);
                let parser_res = $bldr.clone().build_as_parser()($exp, $($vars)*).unwrap();
                let func_ref_res = $bldr.clone().use_ref().build_as_parser()($exp, $($vars)*).unwrap();
                let parser_ref_res = $bldr.use_ref().build_as_parser()($exp, $($vars)*).unwrap();
                assert!((func_res - $res).abs() < threshold, "build_as_function result returned {func_res}, expected {}", $res);
                assert!((parser_res - $res).abs() < threshold, "build_as_parser result returned {parser_res}, expected {}", $res);
                assert!((func_ref_res - $res).abs() < threshold, "build_as_function with use_ref result returned {func_res}, expected {}", $res);
                assert!((parser_ref_res - $res).abs() < threshold, "build_as_parser with use_ref result returned {parser_res}, expected {}", $res);
            };
        }
        test!(EvalBuilder::<'_, f64, _, _>::new(), "2+2", (), 4.0);
        test!(
            EvalBuilder::new().add_constant("two", 2.0),
            "11*two",
            (),
            22.0
        );
        test!(
            EvalBuilder::new().add_fn1("sinh", &sinh),
            "sinh(1)",
            (),
            1.175201193643801
        );
        test!(
            EvalBuilder::new().add_fn2("hypot", &hypot),
            "hypot(3,4)",
            (),
            5.0
        );
        test!(
            EvalBuilder::new().add_fn3("clamp", &clamp),
            "clamp(10, -3, 3)",
            (),
            3.0
        );
        test!(
            EvalBuilder::new().add_fn4("slope", &|x1: f64, y1: f64, x2: f64, y2: f64| (y2 - y1)
                / (x2 - x1)),
            "slope(18, 11, 21, 20)",
            (),
            3.0
        );
        test!(
            EvalBuilder::new().add_fn_flex("mean", 1, None, &mean),
            "mean(1,1,1,1,1,1,1,1,1)",
            (),
            1.0
        );
        test!(
            EvalBuilder::<'_, f64, _, _>::new().add_variable("x"),
            "-x",
            (0.11),
            -0.11
        );
        test!(
            EvalBuilder::new()
                .add_fn1("deg2rad", &deg2rad)
                .add_variable("x"),
            "deg2rad(x)",
            (30.0),
            FRAC_PI_6
        );
        test!(
            EvalBuilder::new()
                .add_fn1("rad2deg", &rad2deg)
                .add_fn2("atan2", &atan2)
                .add_variable("x")
                .add_constant("height", 10.0),
            "rad2deg(atan2(x, height))",
            (10.0),
            45.0
        );
        test!(
            EvalBuilder::new()
                .add_fn3("lerp", &lerp)
                .add_variable("t")
                .add_constant("low", -10.0)
                .add_constant("high", 10.0),
            "lerp(low, high, t)",
            (0.5),
            0.0
        );
        test!(
            EvalBuilder::new()
                .add_variable("x")
                .add_variable("y")
                .add_fn2("atan2", &atan2),
            "atan2(y, x)",
            (1.0, (3f64).sqrt()),
            FRAC_PI_3
        );
        test!(
            EvalBuilder::new()
                .add_variable("d")
                .add_variable("m")
                .add_fn1("deg2rad", &deg2rad),
            "sin(deg2rad(d)) * m",
            (30.0, 2.0),
            1.0
        );
        test!(
            EvalBuilder::<'_, f64, _, _>::new()
                .add_variable("base")
                .add_variable("height"),
            "base * height / 2",
            (10.0, 4.0),
            20.0
        );
        test!(
            EvalBuilder::<'_, f64, _, _>::new()
                .add_variable("a")
                .add_variable("b")
                .add_variable("w"),
            "(a * w + b * (1 - w))",
            (90.0, 60.0, 0.75),
            82.5
        );
        test!(
            EvalBuilder::new()
                .add_variable("temp")
                .add_variable("bias")
                .add_variable("maxLimit")
                .add_fn3("clamp", &clamp),
            "clamp(temp + bias, -100, maxLimit)",
            (38.0, 10.0, 45.0),
            45.0
        );
        test!(
            EvalBuilder::new()
                .add_variable("x1")
                .add_variable("y1")
                .add_variable("x2")
                .add_variable("y2")
                .add_fn2("hypot", &hypot),
            "hypot(x2 - x1, y2 - y1)",
            (1.0, 2.0, 4.0, 6.0),
            5.0
        );
        test!(
            EvalBuilder::new()
                .add_variable("a1")
                .add_variable("a2")
                .add_variable("w1")
                .add_variable("w2")
                .add_variable("t")
                .add_fn3("lerp", &lerp),
            "lerp(a1 * w1 + a2 * (1 - w1), a2 * w2 + a1 * (1 - w2), t)",
            (&[80.0, 60.0, 0.6, 0.3, 0.5]),
            69.5
        );
        test!(
            EvalBuilder::<'_, f64, _, _>::new()
                .add_variable("a")
                .add_variable("b")
                .add_variable("c")
                .add_variable("d")
                .add_variable("e"),
            "exp((ln(a) + ln(b) + ln(c) + ln(d) + ln(e)) / 5)",
            (&[1.0, 10.0, 100.0, 1000.0, 10000.0]),
            100.0
        );
    }
}
