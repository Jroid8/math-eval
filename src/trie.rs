use std::{cmp::Ordering, fmt::Debug};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrieNode {
    Branch(char, u32),
    Leaf(u32),
}

pub fn exact_match(trie: &[TrieNode], name: &str) -> Option<u32> {
    if trie.is_empty() {
        return None;
    }
    let mut idx = 0;
    let mut cap = trie.len() - 1;
    let mut chars = name.chars();
    let mut target = chars.next()?;
    while idx <= cap {
        match trie.get(idx)? {
            &TrieNode::Branch(ch, dc) => {
                if ch == target {
                    if let Some(next) = chars.next() {
                        target = next;
                        cap = idx + dc as usize;
                        idx += 1;
                    } else if let TrieNode::Leaf(num) = trie[idx + 1] {
                        return Some(num);
                    } else {
                        return None;
                    }
                } else {
                    idx += dc as usize + 1;
                }
            }
            TrieNode::Leaf(_) => idx += 1,
        }
    }
    None
}

pub fn longest_prefix<'a>(trie: &[TrieNode], text: &'a str) -> Option<(u32, &'a str)> {
    if trie.is_empty() {
        return None;
    }
    let mut idx = 0;
    let mut cap = trie.len() - 1;
    let mut chars = text.char_indices();
    let (mut tar_idx, mut target) = chars.next()?;
    let mut res = None;
    while idx <= cap {
        match trie.get(idx)? {
            &TrieNode::Branch(ch, dc) => {
                if ch == target {
                    if let TrieNode::Leaf(val) = trie[idx + 1] {
                        res = Some((val, &text[tar_idx + target.len_utf8()..]));
                    }
                    if let Some(next) = chars.next() {
                        tar_idx = next.0;
                        target = next.1;
                    } else {
                        break;
                    }
                    cap = idx + dc as usize;
                    idx += 1;
                } else {
                    idx += dc as usize + 1;
                }
            }
            TrieNode::Leaf(_) => idx += 1,
        }
    }
    res
}

pub trait NameTrie<T: Copy> {
    fn exact_match(&self, name: &str) -> Option<T>;
    fn longest_prefix<'a>(&self, text: &'a str) -> Option<(T, &'a str)>;
}

#[derive(Clone, PartialEq, Eq)]
pub struct VecNameTrie<T: Copy>(Vec<TrieNode>, Vec<T>);

impl<T: Copy> VecNameTrie<T> {
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
            values.push(first.1);
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
                values.push(cur.1);

                last = cur;
                last_cc = cur_cc;
            }
        }
        VecNameTrie(trie, values)
    }
}

impl<T: Copy + Debug> Debug for VecNameTrie<T> {
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

impl<T: Copy> NameTrie<T> for VecNameTrie<T> {
    fn exact_match(&self, name: &str) -> Option<T> {
        exact_match(&self.0, name).map(|idx| self.1[idx as usize])
    }

    fn longest_prefix<'a>(&self, text: &'a str) -> Option<(T, &'a str)> {
        longest_prefix(&self.0, text).map(|(idx, sl)| (self.1[idx as usize], sl))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct ConstNameTrie<T: 'static + Copy> {
    pub trie: &'static [TrieNode],
    pub leaf_to_value: fn(u32) -> T,
}

impl<T: Copy> NameTrie<T> for ConstNameTrie<T> {
    fn exact_match(&self, name: &str) -> Option<T> {
        exact_match(&self.trie, name).map(self.leaf_to_value)
    }

    fn longest_prefix<'a>(&self, text: &'a str) -> Option<(T, &'a str)> {
        longest_prefix(&self.trie, text).map(|(l, s)| ((self.leaf_to_value)(l), s))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EmptyNameTrie;

impl<T: Copy> NameTrie<T> for EmptyNameTrie {
    fn exact_match(&self, _name: &str) -> Option<T> {
        None
    }

    fn longest_prefix<'a>(&self, _text: &'a str) -> Option<(T, &'a str)> {
        None
    }
}

#[derive(Debug, Clone, Copy)]
pub struct UnitNameTrie<'a, T: Copy> {
    pub name: &'a str,
    pub value: T,
}

impl<T: Copy> NameTrie<T> for UnitNameTrie<'_, T> {
    fn exact_match(&self, name: &str) -> Option<T> {
        if name == self.name {
            Some(self.value)
        } else {
            None
        }
    }

    fn longest_prefix<'a>(&self, text: &'a str) -> Option<(T, &'a str)> {
        if text.starts_with(self.name) {
            Some((self.value, &text[self.name.len()..]))
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use strum::{FromRepr};

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

    #[test]
    fn exact_match() {
        const TRIE: [TrieNode; 21] = [
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
        ];
        fn exact_match(name: &str) -> Option<u32> {
            super::exact_match(&TRIE, name)
        }

        assert_eq!(exact_match("var"), Some(0));
        assert_eq!(exact_match("var3"), Some(1));
        assert_eq!(exact_match("variable1"), Some(2));
        assert_eq!(exact_match("variable2"), Some(3));
        assert_eq!(exact_match("x"), Some(4));
        assert_eq!(exact_match("y"), Some(5));
        assert_eq!(exact_match("σ"), Some(6));
        assert_eq!(exact_match("variabl"), None);
        assert_eq!(exact_match("x2"), None);
        assert_eq!(exact_match("variable3"), None);
        assert_eq!(exact_match("variaf"), None);
        assert_eq!(exact_match("xx"), None);
    }

    #[test]
    fn longest_prefix() {
        const TRIE: [TrieNode; 21] = [
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
        ];
        fn longest_prefix(name: &str) -> Option<(u32, &str)> {
            super::longest_prefix(&TRIE, name)
        }

        assert_eq!(longest_prefix("var"), Some((0, "")));
        assert_eq!(longest_prefix("var5"), Some((0, "5")));
        assert_eq!(longest_prefix("variable1xy"), Some((2, "xy")));
        assert_eq!(longest_prefix("ysdfja;k"), Some((5, "sdfja;k")));
        assert_eq!(longest_prefix("vax"), None);
        assert_eq!(longest_prefix("vaxiable"), None);
    }
}
