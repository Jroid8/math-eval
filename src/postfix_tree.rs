use std::{iter::FusedIterator, ops::Index};

pub trait Arity {
    fn arity(&self) -> u8;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node<T: Arity> {
    item: T,
    descendants_count: usize,
}

impl<T: Arity> Node<T> {
    pub fn new(item: T, descendants_count: usize) -> Self {
        Node {
            item,
            descendants_count,
        }
    }

    pub fn get(&self) -> &T {
        &self.item
    }

    pub fn get_mut(&mut self) -> &mut T {
        &mut self.item
    }

    pub fn into_inner(self) -> T {
        self.item
    }

    pub fn descendants_count(&self) -> usize {
        self.descendants_count
    }

    pub fn calc_descendents(previous: &[Node<T>], new: T) -> Self {
        let mut dc = 0;
        for ac in 0..new.arity() {
            dc += previous[previous.len() - dc - 1].descendants_count + 1;
        }
        Node::new(new, dc)
    }

    pub fn from_items(items: impl IntoIterator<Item = T>) -> Vec<Node<T>> {
        let items = items.into_iter();
        let mut res = Vec::with_capacity(items.size_hint().1.unwrap_or(64));
        for k in items {
            res.push(Node::calc_descendents(&res, k));
        }
        res
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct IterativeTreeBuilder<T: Arity>(Vec<Node<T>>);

impl<T: Arity> IterativeTreeBuilder<T> {
    pub fn new() -> Self {
        IterativeTreeBuilder(Vec::new())
    }

    pub fn push(&mut self, item: T) {
        self.0.push(Node::calc_descendents(&self.0, item))
    }

    pub fn pop(&mut self) -> Option<T> {
        self.0.pop().map(|n| n.item)
    }

    pub fn build(self) -> PostfixTree<T> {
        PostfixTree(self.0)
    }

    pub fn last(&self) -> Option<&T> {
        self.0.last().map(|n| &n.item)
    }

    pub fn last_mut(&mut self) -> Option<&mut T> {
        self.0.last_mut().map(|n| &mut n.item)
    }

    pub fn inner(&self) -> &[Node<T>] {
        &self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChildIter<'a, T: Arity> {
    tree: &'a PostfixTree<T>,
    pos: usize,
    arity: u8,
}

impl<'a, T: Arity> Iterator for ChildIter<'a, T> {
    type Item = (&'a T, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.arity == 0 {
            return None;
        }
        self.arity -= 1;
        let idx = self.pos;
        let cur = &self.tree.0[self.pos];
        if cur.descendants_count < self.pos {
            self.pos -= cur.descendants_count + 1;
        } else {
            debug_assert_eq!(cur.descendants_count, self.pos);
            debug_assert_eq!(self.arity, 0);
        }
        Some((&cur.item, idx))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.arity as usize;
        (len, Some(len))
    }
}

impl<'a, T: Arity> ExactSizeIterator for ChildIter<'a, T> {
    fn len(&self) -> usize {
        self.arity as usize
    }
}

impl<'a, T: Arity> FusedIterator for ChildIter<'a, T> {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PostOrderIter<'a, T: Arity> {
    slice: &'a [Node<T>],
    index: usize,
}

impl<'a, T: Arity> Iterator for PostOrderIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.slice.len() {
            return None;
        }
        self.index += 1;
        self.slice.get(self.index - 1).map(|n| &n.item)
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.slice.len() - self.index;
        (len, Some(len))
    }
}

impl<'a, T: Arity> ExactSizeIterator for PostOrderIter<'a, T> {
    fn len(&self) -> usize {
        self.slice.len() - self.index
    }
}

impl<'a, T: Arity> FusedIterator for PostOrderIter<'a, T> {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PostfixTree<T: Arity>(Vec<Node<T>>);

impl<T: Arity> PostfixTree<T> {
    pub fn from_inner_unchecked(vector: Vec<Node<T>>) -> Self {
        PostfixTree(vector)
    }

    pub fn from_items(items: impl IntoIterator<Item = T>) -> Self {
        // FIX: needs some verification
        PostfixTree(Node::from_items(items))
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.0.get(index).map(|n| &n.item)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.0.get_mut(index).map(|n| &mut n.item)
    }

    pub fn iter_children<'a>(&'a self, head: usize) -> ChildIter<'a, T> {
        let arity = self.0[head].item.arity();
        ChildIter {
            tree: self,
            pos: head.wrapping_sub(1),
            arity,
        }
    }

    pub fn postorder_subtree_iter<'a>(&'a self, head: usize) -> PostOrderIter<'a, T> {
        PostOrderIter {
            slice: &self.0[head - self.0[head].descendants_count..=head],
            index: 0,
        }
    }

    pub fn postorder_iter<'a>(&'a self) -> PostOrderIter<'a, T> {
        PostOrderIter {
            slice: &self.0,
            index: 0,
        }
    }

    pub fn as_slice(&self) -> &[Node<T>] {
        &self.0
    }

    pub fn replace_with_unit(&mut self, unit: T, head: usize) {
        let start = head - self.0[head].descendants_count;
        self.0[start] = Node::new(unit, 0);
        self.0.drain(start + 1..=head);
    }

    pub fn subtree_start(&self, head: usize) -> usize {
        head - self.0[head].descendants_count
    }
}

impl<T: Arity> Index<usize> for PostfixTree<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index].item
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl Arity for u8 {
        fn arity(&self) -> u8 {
            *self
        }
    }

    #[test]
    fn test_child_node_iter() {
        let tree = PostfixTree(Node::from_items([0, 0, 0, 2, 0, 1, 0, 0, 2, 0, 4, 2]));
        let collect_children = |idx| {
            tree.iter_children(idx)
                .map(|(c, i)| (*c, i))
                .collect::<Vec<_>>()
        };
        assert_eq!(collect_children(0), vec![]);
        assert_eq!(collect_children(1), vec![]);
        assert_eq!(collect_children(3), vec![(0, 2), (0, 1)]);
        assert_eq!(collect_children(5), vec![(0, 4)]);
        assert_eq!(collect_children(8), vec![(0, 7), (0, 6)]);
        assert_eq!(collect_children(10), vec![(0, 9), (2, 8), (1, 5), (2, 3)]);
    }

    #[test]
    fn test_iterative_tree_builder() {
        macro_rules! test {
            ($items: expr, $res: expr) => {{
                let mut builder = IterativeTreeBuilder::new();
                for it in $items {
                    builder.push(it);
                }
                assert_eq!(
                    builder
                        .build()
                        .0
                        .into_iter()
                        .map(|n| n.descendants_count)
                        .collect::<Vec<_>>(),
                    $res
                );
            }};
        }

        test!([0, 0, 2], [0, 0, 2]);
        test!([0, 1, 1], [0, 1, 2]);
        test!([0, 1, 0, 2, 0, 2], [0, 1, 0, 3, 0, 5]);
        test!([0, 0, 0, 2, 2, 0, 2], [0, 0, 0, 2, 4, 0, 6]);
        test!([0, 0, 0, 3, 1], [0, 0, 0, 3, 4]);
        test!([0, 0, 0, 0, 0, 4, 0, 3], [0, 0, 0, 0, 0, 4, 0, 7]);
    }
}
