use seq_macro::seq;

use crate::{
    FunctionPointer, ParsingError, compile, evaluate,
    number::Number,
    trie::{EmptyNameTrie, VecNameTrie},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NoVariable;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OneVariable<'a>(&'a str);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TwoVariables<'a>([(&'a str, u8); 2]);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThreeVariables<'a>([(&'a str, u8); 3]);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FourVariables<'a>([(&'a str, u8); 4]);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ManyVariables<'a>(Vec<(&'a str, usize)>);

#[derive(Clone, Debug)]
pub struct EvalBuilder<'c, 'n, 'f, N: Number, V = NoVariable> {
    constants: Vec<(&'n str, &'c N)>,
    function_identifier: Vec<(&'n str, (usize, u8, Option<u8>))>,
    functions: Vec<FunctionPointer<'f, N>>,
    variables: V,
}

impl<N: Number> Default for EvalBuilder<'_, '_, '_, N> {
    fn default() -> Self {
        Self {
            constants: Vec::default(),
            function_identifier: Vec::default(),
            functions: Vec::default(),
            variables: NoVariable,
        }
    }
}

impl<N: Number> EvalBuilder<'_, '_, '_, N> {
    pub fn new() -> Self {
        Self::default()
    }
}

impl<'c, 'n, 'f, N: Number, V> EvalBuilder<'c, 'n, 'f, N, V> {
    pub fn add_constant(mut self, name: &'n str, value: &'c N) -> Self {
        self.constants.push((name, value));
        self
    }

    pub fn add_fn1(mut self, name: &'n str, function: for<'a> fn(N::AsArg<'a>) -> N) -> Self {
        self.function_identifier
            .push((name, (self.functions.len(), 1, Some(1))));
        self.functions.push(FunctionPointer::Single(function));
        self
    }

    pub fn add_fn2(
        mut self,
        name: &'n str,
        function: for<'a, 'b> fn(N::AsArg<'a>, N::AsArg<'b>) -> N,
    ) -> Self {
        self.function_identifier
            .push((name, (self.functions.len(), 2, Some(2))));
        self.functions.push(FunctionPointer::Dual(function));
        self
    }

    pub fn add_fn3(
        mut self,
        name: &'n str,
        function: for<'i, 'j, 'k> fn(N::AsArg<'i>, N::AsArg<'j>, N::AsArg<'k>) -> N,
    ) -> Self {
        self.function_identifier
            .push((name, (self.functions.len(), 3, Some(3))));
        self.functions.push(FunctionPointer::Triple(function));
        self
    }

    pub fn add_fn_flex(
        mut self,
        name: &'n str,
        mininum_argument_count: u8,
        maximum_argument_count: Option<u8>,
        function: fn(&[N]) -> N,
    ) -> Self {
        self.function_identifier.push((
            name,
            (
                self.functions.len(),
                mininum_argument_count,
                maximum_argument_count,
            ),
        ));
        self.functions.push(FunctionPointer::Flexible(function));
        self
    }

    pub fn add_dyn_fn1(
        mut self,
        name: &'n str,
        function: &'f dyn for<'b> Fn(N::AsArg<'b>) -> N,
    ) -> Self {
        self.function_identifier
            .push((name, (self.functions.len(), 1, Some(1))));
        self.functions.push(FunctionPointer::DynSingle(function));
        self
    }

    pub fn add_dyn_fn2(
        mut self,
        name: &'n str,
        function: &'f dyn for<'a, 'b> Fn(N::AsArg<'a>, N::AsArg<'b>) -> N,
    ) -> Self {
        self.function_identifier
            .push((name, (self.functions.len(), 2, Some(2))));
        self.functions.push(FunctionPointer::DynDual(function));
        self
    }

    pub fn add_dyn_fn3(
        mut self,
        name: &'n str,
        function: &'f dyn for<'i, 'j, 'k> Fn(N::AsArg<'i>, N::AsArg<'j>, N::AsArg<'k>) -> N,
    ) -> Self {
        self.function_identifier
            .push((name, (self.functions.len(), 3, Some(3))));
        self.functions.push(FunctionPointer::DynTriple(function));
        self
    }

    pub fn add_dyn_fn_flex(
        mut self,
        name: &'n str,
        mininum_argument_count: u8,
        maximum_argument_count: Option<u8>,
        function: &'f dyn Fn(&[N]) -> N,
    ) -> Self {
        self.function_identifier.push((
            name,
            (
                self.functions.len(),
                mininum_argument_count,
                maximum_argument_count,
            ),
        ));
        self.functions.push(FunctionPointer::DynFlexible(function));
        self
    }
}

impl<'c, 'n, 'f, N: Number> EvalBuilder<'c, 'n, 'f, N, NoVariable> {
    pub fn add_variable(self, name: &'n str) -> EvalBuilder<'c, 'n, 'f, N, OneVariable<'n>> {
        EvalBuilder {
            variables: OneVariable(name),
            constants: self.constants,
            function_identifier: self.function_identifier,
            functions: self.functions,
        }
    }
    pub fn build_as_evaluator(mut self) -> impl Fn(&str) -> Result<N, ParsingError> {
        let constant_parser = VecNameTrie::new(self.constants.as_mut_slice());
        let function_parser = VecNameTrie::new(self.function_identifier.as_mut_slice());
        move |input: &str| {
            evaluate::<_, (), _>(
                input,
                &constant_parser,
                &function_parser,
                &EmptyNameTrie,
                |idx| self.functions[idx],
                &(),
            )
        }
    }
}

impl<'c, 'n, 'f, N> EvalBuilder<'c, 'n, 'f, N, NoVariable>
where
    N: for<'b> Number<AsArg<'b> = N> + Copy,
{
    pub fn build_as_function(mut self, input: &str) -> Result<impl FnMut() -> N, ParsingError> {
        let constant_parser = VecNameTrie::new(self.constants.as_mut_slice());
        let function_parser = VecNameTrie::new(self.function_identifier.as_mut_slice());
        let expr = compile::<_, (), _>(
            input,
            &constant_parser,
            &function_parser,
            &EmptyNameTrie,
            |idx| self.functions[idx],
        )?;
        let mut stack = Vec::with_capacity(expr.stack_req_capacity().unwrap());
        Ok(move || expr.eval(&(), &mut stack).unwrap())
    }
}

impl<'c, 'n, 'f, N: Number> EvalBuilder<'c, 'n, 'f, N, OneVariable<'n>> {
    pub fn add_variable(self, name: &'n str) -> EvalBuilder<'c, 'n, 'f, N, TwoVariables<'n>> {
        EvalBuilder {
            constants: self.constants,
            function_identifier: self.function_identifier,
            functions: self.functions,
            variables: TwoVariables([(self.variables.0, 0), (name, 1)]),
        }
    }
    pub fn build_as_evaluator(mut self) -> impl Fn(&str, N) -> Result<N, ParsingError> {
        let constant_parser = VecNameTrie::new(self.constants.as_mut_slice());
        let function_parser = VecNameTrie::new(self.function_identifier.as_mut_slice());
        let variable_parser = VecNameTrie::new(&mut [(&self.variables.0, ())]);
        move |input: &str, v0: N| {
            evaluate(
                input,
                &constant_parser,
                &function_parser,
                &variable_parser,
                |idx| self.functions[idx],
                &(v0,),
            )
        }
    }
}

impl<'c, 'n, 'f, N: Number> EvalBuilder<'c, 'n, 'f, N, OneVariable<'n>> {
    pub fn build_as_function(mut self, input: &str) -> Result<impl FnMut(N) -> N, ParsingError> {
        let constant_parser = VecNameTrie::new(self.constants.as_mut_slice());
        let function_parser = VecNameTrie::new(self.function_identifier.as_mut_slice());
        let variable_parser = VecNameTrie::new(&mut [(&self.variables.0, ())]);
        let expr = compile(
            input,
            &constant_parser,
            &function_parser,
            &variable_parser,
            |idx| self.functions[idx],
        )?;
        let mut stack = Vec::with_capacity(expr.stack_req_capacity().unwrap());
        Ok(move |v0| expr.eval(&(v0,), &mut stack).unwrap())
    }
}

macro_rules! fn_build_as_evaluator {
    ($n: expr) => {
        seq!(I in 0..$n {
            pub fn build_as_evaluator(mut self)
                -> impl Fn(&str, #(N,)*) -> Result<N, ParsingError> {
                let constant_parser = VecNameTrie::new(self.constants.as_mut_slice());
                let function_parser = VecNameTrie::new(self.function_identifier.as_mut_slice());
                let variable_parser = VecNameTrie::new(self.variables.0.as_mut_slice());
                move |input, #(v~I,)*| {
                    evaluate(
                        input,
                        &constant_parser,
                        &function_parser,
                        &variable_parser,
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
                mut self,
                input: &str,
            ) -> Result<impl FnMut(#(N,)*) -> N, ParsingError> {
                let constant_parser = VecNameTrie::new(self.constants.as_mut_slice());
                let function_parser = VecNameTrie::new(self.function_identifier.as_mut_slice());
                let variable_parser = VecNameTrie::new(self.variables.0.as_mut_slice());
                let expr = compile(
                    input,
                    &constant_parser,
                    &function_parser,
                    &variable_parser,
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
            pub fn add_variable(self, name: &'n str) -> EvalBuilder<'c, 'n, 'f, N, $next<'n>> {
                let mut iter = self.variables.0.into_iter();
                EvalBuilder {
                    constants: self.constants,
                    function_identifier: self.function_identifier,
                    functions: self.functions,
                    variables: $next([#(iter.next().unwrap(),)* (name, $n)]),
                }
            }
        });
    };
}

impl<'c, 'n, 'f, N: Number> EvalBuilder<'c, 'n, 'f, N, TwoVariables<'n>> {
    fn_add_variable!(2, ThreeVariables);
    fn_build_as_evaluator!(2);
}

impl<'c, 'n, 'f, N: Number> EvalBuilder<'c, 'n, 'f, N, TwoVariables<'n>> {
    fn_build_as_function!(2);
}

impl<'c, 'n, 'f, N: Number> EvalBuilder<'c, 'n, 'f, N, ThreeVariables<'n>> {
    fn_add_variable!(3, FourVariables);
    fn_build_as_evaluator!(3);
}

impl<'c, 'n, 'f, N: Number> EvalBuilder<'c, 'n, 'f, N, ThreeVariables<'n>> {
    fn_build_as_function!(3);
}

impl<'c, 'n, 'f, N: Number> EvalBuilder<'c, 'n, 'f, N, FourVariables<'n>> {
    pub fn add_variable(self, name: &'n str) -> EvalBuilder<'c, 'n, 'f, N, ManyVariables<'n>> {
        EvalBuilder {
            constants: self.constants,
            function_identifier: self.function_identifier,
            functions: self.functions,
            variables: ManyVariables(
                self.variables
                    .0
                    .into_iter()
                    .map(|(n, i)| (n, i as usize))
                    .chain(Some((name, self.variables.0.len())))
                    .collect(),
            ),
        }
    }
    fn_build_as_evaluator!(4);
}

impl<'c, 'n, 'f, N: Number> EvalBuilder<'c, 'n, 'f, N, FourVariables<'n>> {
    fn_build_as_function!(4);
}

impl<'c, 'n, 'f, N: Number> EvalBuilder<'c, 'n, 'f, N, ManyVariables<'n>> {
    pub fn build_as_evaluator(mut self) -> impl Fn(&str, &[N]) -> Result<N, ParsingError> {
        let constant_parser = VecNameTrie::new(self.constants.as_mut_slice());
        let function_parser = VecNameTrie::new(self.function_identifier.as_mut_slice());
        let variable_parser = VecNameTrie::new(self.variables.0.as_mut_slice());
        move |input, vars| {
            evaluate(
                input,
                &constant_parser,
                &function_parser,
                &variable_parser,
                |idx| self.functions[idx],
                &vars,
            )
        }
    }
}

impl<'c, 'n, 'f, N: Number> EvalBuilder<'c, 'n, 'f, N, ManyVariables<'n>> {
    pub fn build_as_function<'b>(
        mut self,
        input: &str,
    ) -> Result<impl FnMut(&'b [N]) -> N, ParsingError> {
        let constant_parser = VecNameTrie::new(self.constants.as_mut_slice());
        let function_parser = VecNameTrie::new(self.function_identifier.as_mut_slice());
        let variable_parser = VecNameTrie::new(self.variables.0.as_mut_slice());
        let expr = compile(
            input,
            &constant_parser,
            &function_parser,
            &variable_parser,
            |idx| self.functions[idx],
        )?;
        let mut stack = Vec::with_capacity(expr.stack_req_capacity().unwrap());
        Ok(move |vars| expr.eval(&vars, &mut stack).unwrap())
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::{Debug, Display};

    use super::*;
    use crate::FunctionPointer;

    #[derive(PartialEq, Debug)]
    struct UnexpectedBuilderFieldValues<'n, 'c, V> {
        constants: Option<(Vec<(&'n str, &'c f64)>, Vec<(&'n str, &'c f64)>)>,
        function_identifier: Option<(
            Vec<(&'n str, (usize, u8, Option<u8>))>,
            Vec<(&'n str, (usize, u8, Option<u8>))>,
        )>,
        functions: Vec<(usize, f64)>,
        variables: Option<(V, V)>,
    }

    impl<V: Debug> Display for UnexpectedBuilderFieldValues<'_, '_, V> {
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
    fn compare<'n, 'c, V: PartialEq>(
        builder: EvalBuilder<'c, 'n, '_, f64, V>,
        constants: impl Iterator<Item = (&'n str, &'c f64)>,
        function_identifier: impl Iterator<Item = (&'n str, usize, u8, Option<u8>)>,
        variables: V,
    ) -> Option<UnexpectedBuilderFieldValues<'n, 'c, V>> {
        let constants: Vec<_> = constants.collect();
        let function_identifier = function_identifier
            .map(|(n, id, min, max)| (n, (id, min, max)))
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
            EvalBuilder::new().add_constant("c", &3.57),
            [("c", &3.57)].into_iter(),
            [].into_iter(),
            NoVariable,
        ));

        test!(compare(
            EvalBuilder::new().add_variable("x"),
            [].into_iter(),
            [].into_iter(),
            OneVariable("x"),
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
                .add_constant("c1", &7.319),
            [("c1", &7.319)].into_iter(),
            [("f2", 0, 1, Some(1))].into_iter(),
            NoVariable,
        ));

        let zero = String::from("0");
        test!(compare(
            EvalBuilder::new()
                .add_constant("c1", &7.319)
                .add_dyn_fn1("f2", &|_| zero.parse().unwrap()),
            [("c1", &7.319)].into_iter(),
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
            OneVariable("t"),
        ));

        test!(compare(
            EvalBuilder::new().add_variable("y").add_variable("x"),
            [].into_iter(),
            [].into_iter(),
            TwoVariables([("y", 0), ("x", 1)]),
        ));

        test!(compare(
            EvalBuilder::new().add_variable("y").add_variable("x"),
            [].into_iter(),
            [].into_iter(),
            TwoVariables([("y", 0), ("x", 1)]),
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
            TwoVariables([("y", 0), ("x", 1)]),
        ));

        test!(compare(
            EvalBuilder::new()
                .add_variable("y")
                .add_variable("x")
                .add_variable("z"),
            [].into_iter(),
            [].into_iter(),
            ThreeVariables([("y", 0), ("x", 1), ("z", 2)]),
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
            ThreeVariables([("y", 0), ("x", 1), ("z", 2)]),
        ));

        test!(compare(
            EvalBuilder::new()
                .add_variable("y")
                .add_variable("x")
                .add_variable("z")
                .add_variable("w"),
            [].into_iter(),
            [].into_iter(),
            FourVariables([("y", 0), ("x", 1), ("z", 2), ("w", 3)]),
        ));

        let three = String::from("3");
        test!(compare(
            EvalBuilder::new()
                .add_dyn_fn1("f0", &|_| zero.parse().unwrap())
                .add_variable("y")
                .add_constant("c", &9.999999)
                .add_fn2("f1", |_, _| 1.0)
                .add_variable("x")
                .add_fn3("f2", |_, _, _| 2.0)
                .add_variable("z")
                .add_dyn_fn_flex("f3", 1, Some(5), &|_| three.parse().unwrap())
                .add_variable("w")
                .add_fn1("f4", |_| 4.0),
            [("c", &9.999999)].into_iter(),
            [
                ("f0", 0, 1, Some(1)),
                ("f1", 1, 2, Some(2)),
                ("f2", 2, 3, Some(3)),
                ("f3", 3, 1, Some(5)),
                ("f4", 4, 1, Some(1))
            ]
            .into_iter(),
            FourVariables([("y", 0), ("x", 1), ("z", 2), ("w", 3)]),
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
            ManyVariables(vec![("y", 0), ("x", 1), ("z", 2), ("v", 3), ("w", 4)]),
        ));

        test!(compare(
            EvalBuilder::new()
                .add_fn1("f0", |_| 0.0)
                .add_variable("y")
                .add_constant("c", &9.999999)
                .add_constant("ce", &2.222222)
                .add_fn2("f1", |_, _| 1.0)
                .add_variable("x")
                .add_fn3("f2", |_, _, _| 2.0)
                .add_variable("z")
                .add_fn_flex("f3", 1, Some(5), |_| 3.0)
                .add_variable("w")
                .add_fn1("f4", |_| 4.0)
                .add_variable("t")
                .add_fn_flex("f5", 5, None, |_| 5.0),
            [("c", &9.999999), ("ce", &2.222222)].into_iter(),
            [
                ("f0", 0, 1, Some(1)),
                ("f1", 1, 2, Some(2)),
                ("f2", 2, 3, Some(3)),
                ("f3", 3, 1, Some(5)),
                ("f4", 4, 1, Some(1)),
                ("f5", 5, 5, None)
            ]
            .into_iter(),
            ManyVariables(vec![("y", 0), ("x", 1), ("z", 2), ("w", 3), ("t", 4)]),
        ));

        let four = String::from("4");
        test!(compare(
            EvalBuilder::new()
                .add_dyn_fn1("f0", &|_| zero.parse().unwrap())
                .add_variable("y")
                .add_constant("c", &9.999999)
                .add_constant("ce", &2.222222)
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
            [("c", &9.999999), ("ce", &2.222222)].into_iter(),
            [
                ("f0", 0, 1, Some(1)),
                ("f1", 1, 2, Some(2)),
                ("f2", 2, 3, Some(3)),
                ("f3", 3, 1, Some(5)),
                ("f4", 4, 1, Some(1)),
                ("f5", 5, 5, None)
            ]
            .into_iter(),
            ManyVariables(vec![("y", 0), ("x", 1), ("z", 2), ("w", 3), ("t", 4)]),
        ))
    }
}
