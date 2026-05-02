use std::{
    fmt::{Debug, Display},
    hash::Hash,
    ops::RangeInclusive,
};

use number::Number;
use seq_macro::seq;
use syntax::MathAst;
use tokenizer::TokenStream;

use crate::{number::BfPointer, quick_expr::QuickExpr, syntax::CfInfo, trie::NameTrie};

pub mod builder;
pub mod number;
pub mod postfix_tree;
pub mod quick_expr;
pub mod syntax;
pub mod tokenizer;
pub mod trie;

pub use builder::EvalBuilder;

pub trait VariableIdentifier: Clone + Copy + Debug + Eq + 'static {}

impl<T> VariableIdentifier for T where T: Clone + Copy + Debug + Eq + 'static {}

pub trait FunctionIdentifier: Clone + Copy + Eq + Debug + 'static {}

impl<T> FunctionIdentifier for T where T: Clone + Copy + Debug + Eq + 'static {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum UnaryOp {
    Fac,
    Neg,
    DoubleFac,
}

impl UnaryOp {
    pub fn eval<N: Number>(self, value: N) -> N {
        match self {
            UnaryOp::Fac => N::factorial(value),
            UnaryOp::Neg => -value,
            UnaryOp::DoubleFac => N::double_factorial(value),
        }
    }

    pub fn precedence(self) -> u8 {
        match self {
            UnaryOp::Neg => 0,
            UnaryOp::Fac | UnaryOp::DoubleFac => 1,
        }
    }

    pub fn as_pointer<N: Number>(self) -> fn(N) -> N {
        match self {
            UnaryOp::Fac => N::factorial,
            UnaryOp::Neg => N::neg,
            UnaryOp::DoubleFac => N::double_factorial,
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
    pub fn eval<N: Number>(self, lhs: N, rhs: N::AsArg<'_>) -> N {
        match self {
            BinaryOp::Add => lhs + rhs,
            BinaryOp::Sub => lhs - rhs,
            BinaryOp::Mul => lhs * rhs,
            BinaryOp::Div => lhs / rhs,
            BinaryOp::Pow => N::pow(lhs, rhs),
            BinaryOp::Mod => N::modulo(lhs, rhs),
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

    pub fn associativity(self) -> Associativity {
        match self {
            BinaryOp::Add | BinaryOp::Mul => Associativity::Both,
            BinaryOp::Sub | BinaryOp::Div | BinaryOp::Mod => Associativity::Left,
            BinaryOp::Pow => Associativity::Right,
        }
    }

    pub fn as_pointer<N: Number>(self) -> for<'a> fn(N, N::AsArg<'a>) -> N {
        match self {
            BinaryOp::Add => |x, y| x + y,
            BinaryOp::Sub => |x, y| x - y,
            BinaryOp::Mul => |x, y| x * y,
            BinaryOp::Div => |x, y| x / y,
            BinaryOp::Pow => N::pow,
            BinaryOp::Mod => N::modulo,
        }
    }
}

pub enum Associativity {
    Both,
    Left,
    Right,
}

impl Display for BinaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryOp::Add => f.write_str("+"),
            BinaryOp::Sub => f.write_str("-"),
            BinaryOp::Mul => f.write_str("*"),
            BinaryOp::Div => f.write_str("/"),
            BinaryOp::Pow => f.write_str("^"),
            BinaryOp::Mod => f.write_str("%"),
        }
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
    MissingOpeningParenthesis,
    MissingClosingParenthesis,
    MissingOpeningBrackets,
    MissingClosingBrackets,
    MissingOpeningBraces,
    MissingClosingBraces,
    MissingOpeningPipe,
    MissingClosingPipe,
    NumberParsingError,
    MisplacedOperator,
    UnknownVariableOrConstant,
    UnknownFunction,
    NotEnoughArguments,
    TooManyArguments,
    EmptyParenthesis,
    EmptyPipePair,
    EmptyBrackets,
    EmptyBraces,
    EmptyArgument,
    EmptyInput,
    NameTooLong,
    UnknownError,
}

impl ParsingErrorKind {
    const fn message(self) -> &'static str {
        match self {
            ParsingErrorKind::UnexpectedCharacter => "Unexpected character encountered",
            ParsingErrorKind::CommaOutsideFunction => "Comma found outside of function",
            ParsingErrorKind::MissingOpeningParenthesis => "Missing opening parenthesis",
            ParsingErrorKind::MissingClosingParenthesis => "Missing closing parenthesis",
            ParsingErrorKind::MissingOpeningBrackets => "Missing opening brackets",
            ParsingErrorKind::MissingClosingBrackets => "Missing closing brackets",
            ParsingErrorKind::MissingOpeningBraces => "Missing opening braces",
            ParsingErrorKind::MissingClosingBraces => "Missing closing braces",
            ParsingErrorKind::MissingOpeningPipe => {
                "Missing opening vertical bar symbol for absolute value"
            }
            ParsingErrorKind::MissingClosingPipe => {
                "Missing closing vertical bar symbol for absolute value"
            }
            ParsingErrorKind::NumberParsingError => "Unable to parse number",
            ParsingErrorKind::MisplacedOperator => "Misplaced operator",
            ParsingErrorKind::UnknownVariableOrConstant => "Unrecognized variable or constant",
            ParsingErrorKind::UnknownFunction => "Unrecognized function",
            ParsingErrorKind::NotEnoughArguments => "Insufficient arguments for function",
            ParsingErrorKind::TooManyArguments => "Too many arguments for function",
            ParsingErrorKind::EmptyParenthesis => "Parentheses should not be empty",
            ParsingErrorKind::EmptyBrackets => "Brackets should not be empty",
            ParsingErrorKind::EmptyBraces => "Braces should not be empty",
            ParsingErrorKind::EmptyArgument => "Function arguments shouldn't be empty",
            ParsingErrorKind::EmptyInput => "Input shouldn't be empty",
            ParsingErrorKind::NameTooLong => "Identifier exceeds maximum length of 32 characters.",
            ParsingErrorKind::EmptyPipePair => {
                "Pairs of vertical bars for the absolute value shouldn not be empty"
            }
            Self::UnknownError => "Unknown error",
        }
    }
}

impl Display for ParsingErrorKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.message())
    }
}

pub trait VariableStore<N: Number, V: VariableIdentifier>: Debug {
    fn get<'a>(&'a self, var: V) -> N::AsArg<'a>;
}

impl<N, V> VariableStore<N, V> for ()
where
    N: Number,
    V: VariableIdentifier,
{
    #[inline]
    fn get<'a>(&'a self, var: V) -> N::AsArg<'a> {
        panic!("Tried to get \"{var:?}\" variable from an empty variable store")
    }
}

impl<N> VariableStore<N, ()> for (N,)
where
    N: Number,
{
    #[inline]
    fn get<'a>(&'a self, _var: ()) -> N::AsArg<'a> {
        self.0.asarg()
    }
}

#[derive(Clone)]
pub enum FunctionPointer<'a, N: Number> {
    Single(fn(N) -> N),
    Dual(for<'b> fn(N, N::AsArg<'b>) -> N),
    Triple(for<'b, 'c> fn(N, N::AsArg<'b>, N::AsArg<'c>) -> N),
    Flexible(fn(&[N]) -> N),
    DynSingle(&'a dyn Fn(N) -> N),
    DynDual(&'a dyn for<'b> Fn(N, N::AsArg<'b>) -> N),
    DynTriple(&'a dyn for<'b, 'c> Fn(N, N::AsArg<'b>, N::AsArg<'c>) -> N),
    DynFlexible(&'a dyn Fn(&[N]) -> N),
}

impl<'a, N> Copy for FunctionPointer<'a, N> where N: Number {}

impl<N: Number> Debug for FunctionPointer<'_, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Single(_) => f.write_str("Single"),
            Self::Dual(_) => f.write_str("Dual"),
            Self::Triple(_) => f.write_str("Triple"),
            Self::Flexible(_) => f.write_str("Flexible"),
            Self::DynSingle(_) => f.write_str("DynSingle"),
            Self::DynDual(_) => f.write_str("DynDual"),
            Self::DynTriple(_) => f.write_str("DynTriple"),
            Self::DynFlexible(_) => f.write_str("DynFlexible"),
        }
    }
}

impl<N: Number> From<BfPointer<N>> for FunctionPointer<'static, N> {
    fn from(value: BfPointer<N>) -> Self {
        match value {
            BfPointer::Single(func) => FunctionPointer::Single(func),
            BfPointer::Dual(func) => FunctionPointer::Dual(func),
            BfPointer::Flexible(func) => FunctionPointer::Flexible(func),
        }
    }
}

impl<N: Number> From<BinaryOp> for FunctionPointer<'static, N> {
    fn from(value: BinaryOp) -> Self {
        Self::Dual(value.as_pointer())
    }
}

impl<N: Number> From<UnaryOp> for FunctionPointer<'static, N> {
    fn from(value: UnaryOp) -> Self {
        Self::Single(value.as_pointer())
    }
}

impl<N: Number> FunctionPointer<'_, N> {
    fn is_flex(&self) -> bool {
        matches!(
            self,
            FunctionPointer::Flexible(_) | FunctionPointer::DynFlexible(_)
        )
    }
}

pub fn compile<'f, N: Number, V: VariableIdentifier, F: FunctionIdentifier>(
    input: &str,
    custom_constants: &impl NameTrie<N>,
    custom_functions: &impl NameTrie<CfInfo<F>>,
    custom_variables: &impl NameTrie<V>,
    function_to_pointer: impl Fn(F) -> FunctionPointer<'f, N>,
) -> Result<QuickExpr<'f, N, V, F>, ParsingError> {
    let token_stream = TokenStream::new::<N::Recognizer>(input).map_err(|e| e.to_general())?;
    let mut syntax_tree = MathAst::new(
        &token_stream.0,
        custom_constants,
        custom_functions,
        custom_variables,
    )
    .map_err(|e| e.to_general(input, &token_stream.0))?;
    syntax_tree.substitute_spec_funcs_equivalents();
    syntax_tree.aot_evaluation(&function_to_pointer);
    syntax_tree.displacing_simplification();
    Ok(QuickExpr::new(syntax_tree, function_to_pointer))
}

pub fn evaluate<'c, 'f, N: Number, V: VariableIdentifier, F: FunctionIdentifier>(
    input: &str,
    custom_constants: &impl NameTrie<N>,
    custom_functions: &impl NameTrie<CfInfo<F>>,
    custom_variables: &impl NameTrie<V>,
    function_to_pointer: impl Fn(F) -> FunctionPointer<'f, N>,
    variable_values: &impl VariableStore<N, V>,
) -> Result<N, ParsingError> {
    let token_stream = TokenStream::new::<N::Recognizer>(input).map_err(|e| e.to_general())?;
    MathAst::parse_and_eval(
        &token_stream.0,
        custom_constants,
        custom_functions,
        custom_variables,
        variable_values,
        function_to_pointer,
    )
    .map_err(|e| e.to_general(input, &token_stream.0))
}

seq!(I in 2..10 {
    impl<N: Number> VariableStore<N, u8> for [N; I]
    where
        N: Number,
    {
        #[inline]
        fn get<'a>(&'a self, var: u8) -> N::AsArg<'a> {
            self[var as usize].asarg()
        }
    }
});

impl<N: Number> VariableStore<N, usize> for &'_ [N] {
    #[inline]
    fn get<'a>(&'a self, var: usize) -> <N as Number>::AsArg<'a> {
        self[var].asarg()
    }
}

macro_rules! nz {
    ($v: literal) => {
        const { std::num::NonZero::new($v).unwrap() }
    };
}

pub(crate) use nz;
