use std::collections::HashMap;

use seq_macro::seq;

use crate::{FunctionPointer, ParsingError, compile, evaluate, number::Number};

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
pub struct EvalBuilder<'a, N, V = NoVariable>
where
    N: Number,
{
    constants: HashMap<String, N>,
    function_identifier: HashMap<String, (usize, u8, Option<u8>)>,
    functions: Vec<FunctionPointer<'a, N>>,
    variables: V,
}

impl<N> Default for EvalBuilder<'_, N>
where
    N: Number,
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

impl<'a, N> EvalBuilder<'a, N>
where
    N: Number,
{
    pub fn new() -> Self {
        Self::default()
    }
}

impl<'a, N, V> EvalBuilder<'a, N, V>
where
    N: Number,
{
    pub fn add_constant(mut self, name: impl Into<String>, value: N) -> Self {
        self.constants.insert(name.into(), value);
        self
    }

    pub fn add_fn1(
        mut self,
        name: impl Into<String>,
        function: for<'b> fn(N::AsArg<'b>) -> N,
    ) -> Self {
        self.function_identifier
            .insert(name.into(), (self.functions.len(), 1, Some(1)));
        self.functions.push(FunctionPointer::Single(function));
        self
    }

    pub fn add_fn2(
        mut self,
        name: impl Into<String>,
        function: for<'b, 'c> fn(N::AsArg<'b>, N::AsArg<'c>) -> N,
    ) -> Self {
        self.function_identifier
            .insert(name.into(), (self.functions.len(), 2, Some(2)));
        self.functions.push(FunctionPointer::Dual(function));
        self
    }

    pub fn add_fn3(
        mut self,
        name: impl Into<String>,
        function: for<'b, 'c, 'd> fn(N::AsArg<'b>, N::AsArg<'c>, N::AsArg<'d>) -> N,
    ) -> Self {
        self.function_identifier
            .insert(name.into(), (self.functions.len(), 3, Some(3)));
        self.functions.push(FunctionPointer::Triple(function));
        self
    }

    pub fn add_fn_flex(
        mut self,
        name: impl Into<String>,
        mininum_argument_count: u8,
        maximum_argument_count: Option<u8>,
        function: fn(&[N]) -> N,
    ) -> Self {
        self.function_identifier.insert(
            name.into(),
            (
                self.functions.len(),
                mininum_argument_count,
                maximum_argument_count,
            ),
        );
        self.functions.push(FunctionPointer::Flexible(function));
        self
    }

    pub fn add_dyn_fn1(
        mut self,
        name: impl Into<String>,
        function: &'a dyn for<'b> Fn(N::AsArg<'b>) -> N,
    ) -> Self {
        self.function_identifier
            .insert(name.into(), (self.functions.len(), 1, Some(1)));
        self.functions.push(FunctionPointer::DynSingle(function));
        self
    }

    pub fn add_dyn_fn2(
        mut self,
        name: impl Into<String>,
        function: &'a dyn for<'b, 'c> Fn(N::AsArg<'b>, N::AsArg<'c>) -> N,
    ) -> Self {
        self.function_identifier
            .insert(name.into(), (self.functions.len(), 2, Some(2)));
        self.functions.push(FunctionPointer::DynDual(function));
        self
    }

    pub fn add_dyn_fn3(
        mut self,
        name: impl Into<String>,
        function: &'a dyn for<'b, 'c, 'd> Fn(N::AsArg<'b>, N::AsArg<'c>, N::AsArg<'d>) -> N,
    ) -> Self {
        self.function_identifier
            .insert(name.into(), (self.functions.len(), 3, Some(3)));
        self.functions.push(FunctionPointer::DynTriple(function));
        self
    }

    pub fn add_dyn_fn_flex(
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
        self.functions.push(FunctionPointer::DynFlexible(function));
        self
    }
}

impl<'a, N> EvalBuilder<'a, N, NoVariable>
where
    N: Number,
{
    pub fn add_variable(self, name: impl Into<String>) -> EvalBuilder<'a, N, OneVariable> {
        EvalBuilder {
            variables: OneVariable(name.into()),
            constants: self.constants,
            function_identifier: self.function_identifier,
            functions: self.functions,
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

impl<'a, N> EvalBuilder<'a, N, NoVariable>
where
    N: for<'b> Number<AsArg<'b> = N> + Copy,
{
    pub fn build_as_function(self, input: &str) -> Result<impl FnMut() -> N + 'a, ParsingError> {
        let expr = compile(
            input,
            |inp| self.constants.get(inp).copied(),
            |inp| self.function_identifier.get(inp).copied(),
            |_| None::<()>,
            |idx| self.functions[idx],
        )?;
        let mut stack = Vec::with_capacity(expr.stack_req_capacity().unwrap());
        Ok(move || expr.eval(&(), &mut stack).unwrap())
    }
}

impl<'a, N> EvalBuilder<'a, N, OneVariable>
where
    N: Number,
{
    pub fn add_variable(self, name: impl Into<String>) -> EvalBuilder<'a, N, TwoVariables> {
        EvalBuilder {
            constants: self.constants,
            function_identifier: self.function_identifier,
            functions: self.functions,
            variables: TwoVariables([self.variables.0, name.into()]),
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

impl<'a, N> EvalBuilder<'a, N, OneVariable>
where
    N: Number,
{
    pub fn build_as_function(self, input: &str) -> Result<impl FnMut(N) -> N + 'a, ParsingError> {
        let expr = compile(
            input,
            |inp| self.constants.get(inp).cloned(),
            |inp| self.function_identifier.get(inp).copied(),
            |inp| (self.variables.0 == inp).then_some(()),
            |idx| self.functions[idx],
        )?;
        let mut stack = Vec::with_capacity(expr.stack_req_capacity().unwrap());
        Ok(move |v0| expr.eval(&(v0,), &mut stack).unwrap())
    }
}

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
    ($n: expr) => {
        seq!(I in 0..$n {
            pub fn build_as_function(
                self,
                input: &str,
            ) -> Result<impl FnMut(#(N,)*) -> N + 'a, ParsingError> {
                let expr = compile(
                    input,
                    |inp| self.constants.get(inp).cloned(),
                    |inp| self.function_identifier.get(inp).copied(),
                    |inp| self.variables.0.iter().position(|var| var == inp).map(|i| i as u8),
                    |idx| self.functions[idx],
                )?;
                let mut stack = Vec::with_capacity(expr.stack_req_capacity().unwrap());
                Ok(move |#(v~I,)*| expr.eval(&[#(v~I,)*], &mut stack).unwrap())
            }
        });
    };
}

macro_rules! fn_add_variable {
    ($n: expr, $next: ident) => {
        seq!(I in 0..$n {
            pub fn add_variable(self, name: impl Into<String>) -> EvalBuilder<'a, N, $next> {
                let mut iter = self.variables.0.into_iter();
                EvalBuilder {
                    constants: self.constants,
                    function_identifier: self.function_identifier,
                    functions: self.functions,
                    variables: $next([#(iter.next().unwrap(),)* name.into()]),
                }
            }
        });
    };
}

impl<'a, N> EvalBuilder<'a, N, TwoVariables>
where
    N: Number,
{
    fn_add_variable!(2, ThreeVariables);
    fn_build_as_evaluator!(2);
}

impl<'a, N> EvalBuilder<'a, N, TwoVariables>
where
    N: Number,
{
    fn_build_as_function!(2);
}

impl<'a, N> EvalBuilder<'a, N, ThreeVariables>
where
    N: Number,
{
    fn_add_variable!(3, FourVariables);
    fn_build_as_evaluator!(3);
}

impl<'a, N> EvalBuilder<'a, N, ThreeVariables>
where
    N: Number,
{
    fn_build_as_function!(3);
}

impl<'a, N> EvalBuilder<'a, N, FourVariables>
where
    N: Number,
{
    pub fn add_variable(self, name: impl Into<String>) -> EvalBuilder<'a, N, ManyVariables> {
        EvalBuilder {
            constants: self.constants,
            function_identifier: self.function_identifier,
            functions: self.functions,
            variables: ManyVariables(self.variables.0.into_iter().chain([name.into()]).collect()),
        }
    }
    fn_build_as_evaluator!(4);
}

impl<'a, N> EvalBuilder<'a, N, FourVariables>
where
    N: Number,
{
    fn_build_as_function!(4);
}

impl<'a, N> EvalBuilder<'a, N, ManyVariables>
where
    N: Number,
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

impl<'a, N> EvalBuilder<'a, N, ManyVariables>
where
    N: Number,
{
    pub fn build_as_function<'b>(
        self,
        input: &str,
    ) -> Result<impl FnMut(&'b [N]) -> N + 'a, ParsingError> {
        let expr = compile(
            input,
            |inp| self.constants.get(inp).cloned(),
            |inp| self.function_identifier.get(inp).copied(),
            |inp| self.variables.0.iter().position(|var| var == inp),
            |idx| self.functions[idx],
        )?;
        let mut stack = Vec::with_capacity(expr.stack_req_capacity().unwrap());
        Ok(move |vars| expr.eval(&vars, &mut stack).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::HashMap, fmt::{Debug, Display}};

    use super::*;
    use crate::FunctionPointer;

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
    fn compare<V: PartialEq>(
        builder: EvalBuilder<'_, f64, V>,
        constants: impl Iterator<Item = (&'static str, f64)>,
        function_identifier: impl Iterator<Item = (&'static str, usize, u8, Option<u8>)>,
        variables: V,
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
                    FunctionPointer::Single(f) => {
                        Some((i, f(0.0))).filter(|(i, v)| *i != *v as usize)
                    }
                    FunctionPointer::Dual(f) => {
                        Some((i, f(0.0, 0.0))).filter(|(i, v)| *i != *v as usize)
                    }
                    FunctionPointer::Triple(f) => {
                        Some((i, f(0.0, 0.0, 0.0))).filter(|(i, v)| *i != *v as usize)
                    }
                    FunctionPointer::Flexible(f) => {
                        Some((i, f(&[]))).filter(|(i, v)| *i != *v as usize)
                    }
                    FunctionPointer::DynSingle(f) => {
                        Some((i, f(0.0))).filter(|(i, v)| *i != *v as usize)
                    }
                    FunctionPointer::DynDual(f) => {
                        Some((i, f(0.0, 0.0))).filter(|(i, v)| *i != *v as usize)
                    }
                    FunctionPointer::DynTriple(f) => {
                        Some((i, f(0.0, 0.0, 0.0))).filter(|(i, v)| *i != *v as usize)
                    }
                    FunctionPointer::DynFlexible(f) => {
                        Some((i, f(&[]))).filter(|(i, v)| *i != *v as usize)
                    }
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
    fn eval_builder() {
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
        ));

        test!(compare(
            EvalBuilder::new().add_constant("c", 3.57),
            [("c", 3.57)].into_iter(),
            [].into_iter(),
            NoVariable,
        ));

        test!(compare(
            EvalBuilder::new().add_variable("x"),
            [].into_iter(),
            [].into_iter(),
            OneVariable(String::from("x")),
        ));

        test!(compare(
            EvalBuilder::new(),
            [].into_iter(),
            [].into_iter(),
            NoVariable,
        ));

        test!(compare(
            EvalBuilder::new().add_fn1("f1", |_| 0.0),
            [].into_iter(),
            [("f1", 0, 1, Some(1))].into_iter(),
            NoVariable,
        ));

        test!(compare(
            EvalBuilder::new()
                .add_fn1("f2", |_| 0.0)
                .add_constant("c1", 7.319),
            [("c1", 7.319)].into_iter(),
            [("f2", 0, 1, Some(1))].into_iter(),
            NoVariable,
        ));

        let zero = String::from("0");
        test!(compare(
            EvalBuilder::new()
                .add_constant("c1", 7.319)
                .add_dyn_fn1("f2", &|_| zero.parse().unwrap()),
            [("c1", 7.319)].into_iter(),
            [("f2", 0, 1, Some(1))].into_iter(),
            NoVariable,
        ));

        test!(compare(
            EvalBuilder::new()
                .add_fn1("f1", |_| 0.0)
                .add_fn2("f2", |_, _| 1.0),
            [].into_iter(),
            [("f1", 0, 1, Some(1)), ("f2", 1, 2, Some(2))].into_iter(),
            NoVariable,
        ));

        let one = String::from("1");
        test!(compare(
            EvalBuilder::new()
                .add_dyn_fn1("f1", &|_| zero.parse().unwrap())
                .add_dyn_fn2("f2", &|_, _| one.parse().unwrap()),
            [].into_iter(),
            [("f1", 0, 1, Some(1)), ("f2", 1, 2, Some(2))].into_iter(),
            NoVariable,
        ));

        test!(compare(
            EvalBuilder::new()
                .add_dyn_fn2("f2", &|_, _| zero.parse().unwrap())
                .add_fn3("f3", |_, _, _| 1.0),
            [].into_iter(),
            [("f2", 0, 2, Some(2)), ("f3", 1, 3, Some(3))].into_iter(),
            NoVariable,
        ));

        test!(compare(
            EvalBuilder::new()
                .add_fn2("f2", |_, _| 0.0)
                .add_dyn_fn3("f3", &|_, _, _| one.parse().unwrap())
                .add_fn_flex("ff", 3, None, |_| 2.0),
            [].into_iter(),
            [
                ("f2", 0, 2, Some(2)),
                ("f3", 1, 3, Some(3)),
                ("ff", 2, 3, None)
            ]
            .into_iter(),
            NoVariable,
        ));

        let two = String::from("2");
        test!(compare(
            EvalBuilder::new()
                .add_variable("t")
                .add_fn2("f2", |_, _| 0.0)
                .add_dyn_fn3("f3", &|_, _, _| one.parse().unwrap())
                .add_dyn_fn_flex("ff", 3, None, &|_| two.parse().unwrap()),
            [].into_iter(),
            [
                ("f2", 0, 2, Some(2)),
                ("f3", 1, 3, Some(3)),
                ("ff", 2, 3, None)
            ]
            .into_iter(),
            OneVariable(String::from("t")),
        ));

        test!(compare(
            EvalBuilder::new().add_variable("y").add_variable("x"),
            [].into_iter(),
            [].into_iter(),
            TwoVariables([String::from("y"), String::from("x")]),
        ));

        test!(compare(
            EvalBuilder::new().add_variable("y").add_variable("x"),
            [].into_iter(),
            [].into_iter(),
            TwoVariables([String::from("y"), String::from("x")]),
        ));

        test!(compare(
            EvalBuilder::new()
                .add_variable("y")
                .add_dyn_fn2("f2", &|_, _| zero.parse().unwrap())
                .add_variable("x")
                .add_dyn_fn3("f3", &|_, _, _| one.parse().unwrap())
                .add_dyn_fn_flex("ff", 3, None, &|_| two.parse().unwrap()),
            [].into_iter(),
            [
                ("f2", 0, 2, Some(2)),
                ("f3", 1, 3, Some(3)),
                ("ff", 2, 3, None)
            ]
            .into_iter(),
            TwoVariables([String::from("y"), String::from("x")]),
        ));

        test!(compare(
            EvalBuilder::new()
                .add_variable("y")
                .add_variable("x")
                .add_variable("z"),
            [].into_iter(),
            [].into_iter(),
            ThreeVariables([String::from("y"), String::from("x"), String::from("z")]),
        ));

        test!(compare(
            EvalBuilder::new()
                .add_fn3("f0", |_, _, _| 0.0)
                .add_variable("y")
                .add_fn3("f1", |_, _, _| 1.0)
                .add_fn3("f2", |_, _, _| 2.0)
                .add_variable("x")
                .add_fn3("f3", |_, _, _| 3.0)
                .add_variable("z"),
            [].into_iter(),
            [
                ("f0", 0, 3, Some(3)),
                ("f1", 1, 3, Some(3)),
                ("f2", 2, 3, Some(3)),
                ("f3", 3, 3, Some(3))
            ]
            .into_iter(),
            ThreeVariables([String::from("y"), String::from("x"), String::from("z")]),
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
        ));

        let three = String::from("3");
        test!(compare(
            EvalBuilder::new()
                .add_dyn_fn1("f0", &|_| zero.parse().unwrap())
                .add_variable("y")
                .add_constant("c", 9.999999)
                .add_fn2("f1", |_, _| 1.0)
                .add_variable("x")
                .add_fn3("f2", |_, _, _| 2.0)
                .add_variable("z")
                .add_dyn_fn_flex("f3", 1, Some(5), &|_| three.parse().unwrap())
                .add_variable("w")
                .add_fn1("f4", |_| 4.0),
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
        ));

        test!(compare(
            EvalBuilder::new()
                .add_fn1("f0", |_| 0.0)
                .add_variable("y")
                .add_constant("c", 9.999999)
                .add_constant("ce", 2.222222)
                .add_fn2("f1", |_, _| 1.0)
                .add_variable("x")
                .add_fn3("f2", |_, _, _| 2.0)
                .add_variable("z")
                .add_fn_flex("f3", 1, Some(5), |_| 3.0)
                .add_variable("w")
                .add_fn1("f4", |_| 4.0)
                .add_variable("t")
                .add_fn_flex("f5", 5, None, |_| 5.0),
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
        ));

        let four = String::from("4");
        test!(compare(
            EvalBuilder::new()
                .add_dyn_fn1("f0", &|_| zero.parse().unwrap())
                .add_variable("y")
                .add_constant("c", 9.999999)
                .add_constant("ce", 2.222222)
                .add_dyn_fn2("f1", &|_, _| one.parse().unwrap())
                .add_variable("x")
                .add_dyn_fn3("f2", &|_, _, _| two.parse().unwrap())
                .add_variable("z")
                .add_dyn_fn_flex("f3", 1, Some(5), &|_| three.parse().unwrap())
                .add_variable("w")
                .add_dyn_fn1("f4", &|_| four.parse().unwrap())
                .add_variable("t")
                .add_dyn_fn_flex("f5", 5, None, &|_| three.parse::<f64>().unwrap()
                    + two.parse::<f64>().unwrap()),
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
        ))
    }
}
