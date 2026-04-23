use crate::{
    FunctionIdentifier as FuncId, VariableIdentifier as VarId,
    number::{BuiltinFuncsNameTrie, Number},
    syntax::FunctionType,
    tokenizer::NumberRecognizer,
    trie::NameTrie,
};

pub const NAME_LIMIT: u8 = 32;

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) enum FragKind<'c, N: Number, V: VarId, F: FuncId> {
    Literal(N),
    Constant(N::AsArg<'c>),
    Variable(V),
    Function(FunctionType<F>, u8, Option<u8>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct ParsedFragment<'c, N: Number, V: VarId, F: FuncId> {
    pub(super) kind: FragKind<'c, N, V, F>,
    pub(super) used_bytes: u8,
}

impl<'c, N: Number, V: VarId, F: FuncId> ParsedFragment<'c, N, V, F> {
    pub(super) fn new(kind: FragKind<'c, N, V, F>, used_bytes: u8) -> Self {
        Self { kind, used_bytes }
    }
}

fn longest_number<N: Number>(input: &str) -> Option<(N, u8)> {
    let mut chars = input.char_indices().map(|(i, c)| (i + c.len_utf8(), c));
    let (mut end, first_char) = chars.next()?;
    let mut nr = N::Recognizer::new(first_char)?;
    while let Some((right_edge, _)) = chars.next().filter(|&(_, c)| nr.recognize(c)) {
        end = right_edge;
    }
    Some((input[..end].parse().ok()?, end as u8))
}

pub(super) fn fragment_token<'c, N: Number, V: VarId, F: FuncId>(
    input: &str,
    dest: &mut Vec<ParsedFragment<'c, N, V, F>>,
    constants: &impl NameTrie<&'c N>,
    variables: &impl NameTrie<V>,
    functions: &impl NameTrie<(F, u8, Option<u8>)>,
) -> bool {
    dest.clear();
    dest.reserve(NAME_LIMIT as usize);
    let mut offset = 0;
    let mut cap = input.len();
    loop {
        let slice = &input[offset..cap];
        let Some(first_char) = slice.chars().next() else {
            return false;
        };
        let longest_frag = if first_char.is_numeric() {
            longest_number(slice).map(|(num, end)| ParsedFragment::new(FragKind::Literal(num), end))
        } else {
            N::CONSTS_NAME_TRIE
                .longest_match(slice)
                .into_iter()
                .chain(constants.longest_match(slice).into_iter())
                .map(|(c, i)| ParsedFragment::new(FragKind::Constant(c.asarg()), i as u8))
                .chain(
                    variables
                        .longest_match(slice)
                        .map(|(var, i)| ParsedFragment::new(FragKind::Variable(var), i as u8)),
                )
                .chain(BuiltinFuncsNameTrie.longest_match(slice).map(|(func, i)| {
                    ParsedFragment::new(
                        FragKind::Function(func.into(), func.min_args(), func.max_args()),
                        i as u8,
                    )
                }))
                .chain(functions.longest_match(slice).map(|((func, min, max), i)| {
                    ParsedFragment::new(
                        FragKind::Function(FunctionType::Custom(func), min, max),
                        i as u8,
                    )
                }))
                .max_by_key(|frag| frag.used_bytes)
        };
        if let Some(frag) = longest_frag {
            offset += frag.used_bytes as usize;
            dest.push(frag);
            cap = input.len();
            if offset == input.len() {
                return true;
            }
            continue;
        }
        // backtracking
        loop {
            let Some(last) = dest.pop() else { return false };
            if !matches!(last.kind, FragKind::Literal(_)) {
                let lo = offset;
                offset -= last.used_bytes as usize;
                cap = input[offset..lo].char_indices().last().unwrap().0;
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{number::BuiltinFunction, trie::VecNameTrie};

    use super::*;
    use std::f64::consts::*;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum TestVar {
        T,
        Sigma,
        Angle,
        Var5,
        ShallNotBeNamed,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum TestFunc {
        Func1,
        F2,
        VeryLongFunction,
    }

    #[test]
    fn fragment_token() {
        const C: f64 = 299792458.0;
        let myvars = VecNameTrie::new(&mut [
            ("t", TestVar::T),
            ("angle", TestVar::Angle),
            ("σ", TestVar::Sigma),
            ("var5", TestVar::Var5),
            ("shallnotbenamed", TestVar::ShallNotBeNamed),
        ]);
        let myfuncs = VecNameTrie::new(&mut [
            ("func1", (TestFunc::Func1, 1, None)),
            ("f2", (TestFunc::F2, 1, None)),
            ("verylongfunction", (TestFunc::VeryLongFunction, 1, None)),
        ]);
        let myconsts = VecNameTrie::new(&mut [("c", &C), ("pi2", &FRAC_2_PI)]);
        let mut alloc = Vec::new();
        let mut fragment = |input: &str| {
            if super::fragment_token(input, &mut alloc, &myconsts, &myvars, &myfuncs) {
                Some(alloc.iter().map(|f| f.kind).collect::<Vec<_>>())
            } else {
                None
            }
        };
        assert_eq!(
            fragment("tσ"),
            Some(vec![
                FragKind::Variable(TestVar::T),
                FragKind::Variable(TestVar::Sigma)
            ])
        );
        assert_eq!(
            fragment("ct"),
            Some(vec![FragKind::Constant(C), FragKind::Variable(TestVar::T)])
        );
        assert_eq!(
            fragment("tangle"),
            Some(vec![
                FragKind::Variable(TestVar::T),
                FragKind::Variable(TestVar::Angle),
            ])
        );
        assert_eq!(
            fragment("tcost"),
            Some(vec![
                FragKind::Variable(TestVar::T),
                FragKind::Function(BuiltinFunction::Cos.into(), 1, Some(1)),
                FragKind::Variable(TestVar::T),
            ])
        );
        assert_eq!(
            fragment("pit"),
            Some(vec![FragKind::Constant(PI), FragKind::Variable(TestVar::T)])
        );
        assert_eq!(
            fragment("t2var5"),
            Some(vec![
                FragKind::Variable(TestVar::T),
                FragKind::Literal(2.0),
                FragKind::Variable(TestVar::Var5)
            ])
        );
        assert_eq!(
            fragment("t2σshallnotbenamedt"),
            Some(vec![
                FragKind::Variable(TestVar::T),
                FragKind::Literal(2.0),
                FragKind::Variable(TestVar::Sigma),
                FragKind::Variable(TestVar::ShallNotBeNamed),
                FragKind::Variable(TestVar::T),
            ])
        );
        assert_eq!(
            fragment("pi2t"),
            Some(vec![
                FragKind::Constant(FRAC_2_PI),
                FragKind::Variable(TestVar::T)
            ])
        );
        assert_eq!(
            fragment("σf2"),
            Some(vec![
                FragKind::Variable(TestVar::Sigma),
                FragKind::Function(FunctionType::Custom(TestFunc::F2), 1, None)
            ])
        );
        assert_eq!(
            fragment("cmin"),
            Some(vec![
                FragKind::Constant(299792458.0),
                FragKind::Function(BuiltinFunction::Min.into(), 2, None)
            ])
        );
        assert_eq!(
            fragment("sinsinangle"),
            Some(vec![
                FragKind::Function(BuiltinFunction::Sin.into(), 1, Some(1)),
                FragKind::Function(BuiltinFunction::Sin.into(), 1, Some(1)),
                FragKind::Variable(TestVar::Angle),
            ])
        );
        assert_eq!(
            fragment("σ55sint"),
            Some(vec![
                FragKind::Variable(TestVar::Sigma),
                FragKind::Literal(55.0),
                FragKind::Function(BuiltinFunction::Sin.into(), 1, Some(1)),
                FragKind::Variable(TestVar::T),
            ])
        );
        assert_eq!(
            fragment("anglesinvar5func1"),
            Some(vec![
                FragKind::Variable(TestVar::Angle),
                FragKind::Function(BuiltinFunction::Sin.into(), 1, Some(1)),
                FragKind::Variable(TestVar::Var5),
                FragKind::Function(FunctionType::Custom(TestFunc::Func1), 1, None),
            ])
        );
        assert_eq!(
            fragment("ttlnttvar5verylongfunction"),
            Some(vec![
                FragKind::Variable(TestVar::T),
                FragKind::Variable(TestVar::T),
                FragKind::Function(BuiltinFunction::Ln.into(), 1, Some(1)),
                FragKind::Variable(TestVar::T),
                FragKind::Variable(TestVar::T),
                FragKind::Variable(TestVar::Var5),
                FragKind::Function(FunctionType::Custom(TestFunc::VeryLongFunction), 1, None),
            ])
        );
        assert_eq!(fragment("a"), None);
        assert_eq!(fragment("asdf"), None);
        assert_eq!(fragment("xxlnxxvar5verylo"), None);
        assert_eq!(fragment("tangl"), None);
    }
}
