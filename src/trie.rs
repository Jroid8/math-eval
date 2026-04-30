use std::{cmp::Ordering, fmt::Debug, marker::PhantomData};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrieNode {
    Branch(char, u32),
    Leaf(u32),
}

pub trait NameTrie<T>: Sized {
    fn nodes(&self) -> &[TrieNode];
    fn leaf_to_value(&self, leaf: u32) -> T;

    fn start_search(&self) -> TrieSearch<'_, T, Self> {
        TrieSearch::new(self)
    }

    fn exact_match(&self, name: &str) -> Option<T> {
        let mut search = self.start_search();
        let mut chars = name.chars().peekable();
        while let Some(c) = chars.next() {
            let (ended, value) = search.advance(c);
            if chars.peek().is_none() {
                return value;
            } else if ended {
                return None;
            }
        }
        None
    }

    fn longest_match(&self, name: &str) -> Option<(T, usize)> {
        let mut search = self.start_search();
        let mut result = None;
        for (right_edge, target) in name.char_indices().map(|(i, c)| (i + c.len_utf8(), c)) {
            let (ended, value) = search.advance(target);
            if let Some(value) = value {
                result = Some((value, right_edge));
            }
            if ended {
                return result;
            }
        }
        result
    }
}

pub struct TrieSearch<'a, T, R: NameTrie<T>> {
    trie: &'a R,
    idx: usize,
    end: usize,
    val_type: PhantomData<T>,
}

impl<'a, T, R: NameTrie<T>> TrieSearch<'a, T, R> {
    pub fn new(trie: &'a R) -> Self {
        Self {
            trie,
            idx: 0,
            end: trie.nodes().len(),
            val_type: PhantomData,
        }
    }

    pub fn advance(&mut self, target: char) -> (bool, Option<T>) {
        let trie = self.trie.nodes();
        while self.idx < self.end {
            match trie[self.idx] {
                TrieNode::Branch(ch, dc) => {
                    if ch == target {
                        self.idx += 1;
                        self.end = self.idx + dc as usize;
                        let value = if let Some(&TrieNode::Leaf(val)) = trie.get(self.idx) {
                            Some(self.trie.leaf_to_value(val))
                        } else {
                            None
                        };
                        return (dc == 1, value);
                    } else {
                        self.idx += dc as usize + 1;
                    }
                }
                TrieNode::Leaf(_) => self.idx += 1,
            }
        }
        (true, None)
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct VecNameTrie<T: Clone>(Vec<TrieNode>, Vec<T>);

impl<T: Clone> VecNameTrie<T> {
    pub fn new(pairs: &mut [(&str, T)]) -> Self {
        pairs.sort_unstable_by_key(|x| x.0);
        Self::from_sorted_pairs(pairs)
    }

    pub fn from_sorted_pairs(pairs: &[(&str, T)]) -> Self {
        // FIX: dedup values
        let mut trie = Vec::with_capacity(pairs.iter().map(|(n, _)| n.len()).sum());
        let mut values = Vec::with_capacity(pairs.len());
        if let Some(first) = pairs.first() {
            assert!(!first.0.is_empty());
            let first_cc = first.0.chars().count();
            for (i, ch) in first.0.chars().enumerate() {
                trie.push(TrieNode::Branch(ch, (first_cc - i) as u32));
            }
            trie.push(TrieNode::Leaf(0));
            values.push(first.1.clone());
            let mut leaf_idx = 0;
            let mut last = first;
            let mut last_cc = first_cc;
            let mut root = 0;
            for cur in &pairs[1..] {
                assert!(!cur.0.is_empty());
                let cur_cc = cur.0.chars().count();
                let diff_point = if let Some(pos) =
                    last.0.chars().zip(cur.0.chars()).position(|(l, c)| l != c)
                {
                    pos
                } else {
                    match cur.0.len().cmp(&last.0.len()) {
                        Ordering::Greater => last_cc,
                        Ordering::Less => panic!("Pairs not sorted"),
                        Ordering::Equal => panic!("Duplicate names"),
                    }
                };
                // adjust ancestors' descendent counts
                if diff_point > 0 {
                    let mut idx = root;
                    macro_rules! adjust_dc {
                        () => {
                            match &mut trie[idx] {
                                TrieNode::Branch(_, dc) => *dc += (cur_cc - diff_point + 1) as u32,
                                TrieNode::Leaf(_) => unreachable!(),
                            }
                        };
                    }
                    adjust_dc!();
                    for cur_char in cur.0.chars().take(diff_point).skip(1) {
                        idx += 1;
                        loop {
                            match trie[idx] {
                                TrieNode::Branch(ch, dc) => {
                                    if ch == cur_char {
                                        break;
                                    } else {
                                        idx += dc as usize + 1
                                    }
                                }
                                TrieNode::Leaf(_) => idx += 1,
                            }
                        }
                        adjust_dc!();
                    }
                } else {
                    root = trie.len();
                }
                // insert new characters
                for (i, ch) in cur.0.chars().enumerate().skip(diff_point) {
                    trie.push(TrieNode::Branch(ch, (cur_cc - i) as u32));
                }
                leaf_idx += 1;
                trie.push(TrieNode::Leaf(leaf_idx));
                values.push(cur.1.clone());

                last = cur;
                last_cc = cur_cc;
            }
        }
        VecNameTrie(trie, values)
    }
}

impl<T: Clone + Debug> Debug for VecNameTrie<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("trie: [\n")?;
        for node in &self.0 {
            match node {
                TrieNode::Branch(ch, dc) => writeln!(f, "\tTrieNode::Branch({ch:?}, {dc}),")?,
                TrieNode::Leaf(i) => writeln!(f, "\tTrieNode::Leaf({i}),")?,
            }
        }
        write!(f, "]\nvalues: {:?}", self.1)
    }
}

impl<T: Clone> NameTrie<T> for VecNameTrie<T> {
    fn nodes(&self) -> &[TrieNode] {
        &self.0
    }
    fn leaf_to_value(&self, leaf: u32) -> T {
        self.1[leaf as usize].clone()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EmptyNameTrie;

impl<T> NameTrie<T> for EmptyNameTrie {
    fn nodes(&self) -> &[TrieNode] {
        &[]
    }

    fn leaf_to_value(&self, _leaf: u32) -> T {
        unreachable!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use strum::FromRepr;

    #[derive(Debug, Clone, Copy, PartialEq, Eq, FromRepr)]
    #[repr(u8)]
    enum MyVar {
        Var,
        Var3,
        Variable1,
        Variable2,
        X,
        Y,
        Sigma,
    }

    #[test]
    fn vec_name_trie_new() {
        use MyVar::*;
        assert_eq!(
            VecNameTrie::new(&mut [
                ("var", Var),
                ("variable1", Variable1),
                ("variable2", Variable2),
                ("var3", Var3),
                ("x", X),
                ("y", Y),
                ("σ", Sigma),
            ]),
            VecNameTrie(
                vec![
                    TrieNode::Branch('v', 14),
                    TrieNode::Branch('a', 13),
                    TrieNode::Branch('r', 12),
                    TrieNode::Leaf(0),
                    TrieNode::Branch('3', 1),
                    TrieNode::Leaf(1),
                    TrieNode::Branch('i', 8),
                    TrieNode::Branch('a', 7),
                    TrieNode::Branch('b', 6),
                    TrieNode::Branch('l', 5),
                    TrieNode::Branch('e', 4),
                    TrieNode::Branch('1', 1),
                    TrieNode::Leaf(2),
                    TrieNode::Branch('2', 1),
                    TrieNode::Leaf(3),
                    TrieNode::Branch('x', 1),
                    TrieNode::Leaf(4),
                    TrieNode::Branch('y', 1),
                    TrieNode::Leaf(5),
                    TrieNode::Branch('σ', 1),
                    TrieNode::Leaf(6),
                ],
                vec![Var, Var3, Variable1, Variable2, X, Y, Sigma]
            )
        );
        assert_eq!(
            VecNameTrie::new(&mut [
                ("jamshid", 0),
                ("jamvid", 1),
                ("jelly", 2),
                ("joker", 3),
                ("balatro", 4)
            ]),
            VecNameTrie::<u8>(
                vec![
                    TrieNode::Branch('b', 7),
                    TrieNode::Branch('a', 6),
                    TrieNode::Branch('l', 5),
                    TrieNode::Branch('a', 4),
                    TrieNode::Branch('t', 3),
                    TrieNode::Branch('r', 2),
                    TrieNode::Branch('o', 1),
                    TrieNode::Leaf(0),
                    TrieNode::Branch('j', 21),
                    TrieNode::Branch('a', 10),
                    TrieNode::Branch('m', 9),
                    TrieNode::Branch('s', 4),
                    TrieNode::Branch('h', 3),
                    TrieNode::Branch('i', 2),
                    TrieNode::Branch('d', 1),
                    TrieNode::Leaf(1),
                    TrieNode::Branch('v', 3),
                    TrieNode::Branch('i', 2),
                    TrieNode::Branch('d', 1),
                    TrieNode::Leaf(2),
                    TrieNode::Branch('e', 4),
                    TrieNode::Branch('l', 3),
                    TrieNode::Branch('l', 2),
                    TrieNode::Branch('y', 1),
                    TrieNode::Leaf(3),
                    TrieNode::Branch('o', 4),
                    TrieNode::Branch('k', 3),
                    TrieNode::Branch('e', 2),
                    TrieNode::Branch('r', 1),
                    TrieNode::Leaf(4)
                ],
                vec![4, 0, 1, 2, 3]
            )
        )
    }

    struct TestTrie;

    impl NameTrie<MyVar> for TestTrie {
        fn nodes(&self) -> &[TrieNode] {
            &[
                TrieNode::Branch('v', 14),
                TrieNode::Branch('a', 13),
                TrieNode::Branch('r', 12),
                TrieNode::Leaf(MyVar::Var as u32),
                TrieNode::Branch('3', 1),
                TrieNode::Leaf(MyVar::Var3 as u32),
                TrieNode::Branch('i', 8),
                TrieNode::Branch('a', 7),
                TrieNode::Branch('b', 6),
                TrieNode::Branch('l', 5),
                TrieNode::Branch('e', 4),
                TrieNode::Branch('1', 1),
                TrieNode::Leaf(MyVar::Variable1 as u32),
                TrieNode::Branch('2', 1),
                TrieNode::Leaf(MyVar::Variable2 as u32),
                TrieNode::Branch('x', 1),
                TrieNode::Leaf(MyVar::X as u32),
                TrieNode::Branch('y', 1),
                TrieNode::Leaf(MyVar::Y as u32),
                TrieNode::Branch('σ', 1),
                TrieNode::Leaf(MyVar::Sigma as u32),
            ]
        }

        fn leaf_to_value(&self, leaf: u32) -> MyVar {
            MyVar::from_repr(leaf as u8).unwrap()
        }
    }

    #[test]
    fn trie_search() {
        let mut s1 = TestTrie.start_search();
        assert_eq!(s1.advance('v'), (false, None));
        assert_eq!(s1.advance('a'), (false, None));
        assert_eq!(s1.advance('r'), (false, Some(MyVar::Var)));
        assert_eq!(s1.advance('3'), (true, Some(MyVar::Var3)));
        assert_eq!(s1.advance('i'), (true, None));
        let mut s2 = TestTrie.start_search();
        assert_eq!(s2.advance('v'), (false, None));
        assert_eq!(s2.advance('a'), (false, None));
        assert_eq!(s2.advance('r'), (false, Some(MyVar::Var)));
        assert_eq!(s2.advance('i'), (false, None));
        assert_eq!(s2.advance('a'), (false, None));
        assert_eq!(s2.advance('b'), (false, None));
        assert_eq!(s2.advance('l'), (false, None));
        assert_eq!(s2.advance('e'), (false, None));
        assert_eq!(s2.advance('1'), (true, Some(MyVar::Variable1)));
        assert_eq!(s2.advance('2'), (true, None));
        let mut s3 = TestTrie.start_search();
        assert_eq!(s3.advance('x'), (true, Some(MyVar::X)));
        let mut s4 = TestTrie.start_search();
        assert_eq!(s4.advance('σ'), (true, Some(MyVar::Sigma)));
    }

    #[test]
    fn exact_match() {
        assert_eq!(TestTrie.exact_match("var"), Some(MyVar::Var));
        assert_eq!(TestTrie.exact_match("var3"), Some(MyVar::Var3));
        assert_eq!(TestTrie.exact_match("variable1"), Some(MyVar::Variable1));
        assert_eq!(TestTrie.exact_match("variable2"), Some(MyVar::Variable2));
        assert_eq!(TestTrie.exact_match("x"), Some(MyVar::X));
        assert_eq!(TestTrie.exact_match("y"), Some(MyVar::Y));
        assert_eq!(TestTrie.exact_match("σ"), Some(MyVar::Sigma));
        assert_eq!(TestTrie.exact_match("variabl"), None);
        assert_eq!(TestTrie.exact_match("x2"), None);
        assert_eq!(TestTrie.exact_match("variable3"), None);
        assert_eq!(TestTrie.exact_match("variaf"), None);
        assert_eq!(TestTrie.exact_match("xx"), None);
    }
}
