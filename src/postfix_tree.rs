use std::{
    fmt::Debug,
    ops::{Index, Range},
};

use subtree_collection::SubtreeCollection;
use tree_iterators::{ChildIter, ParentIter, PostOrderIter};

use crate::postfix_tree::tree_iterators::EulerTour;

pub mod subtree_collection;
pub mod tree_iterators;

pub trait Node: Clone + std::fmt::Debug {
    fn children(&self) -> usize;
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct OptUsize(usize);

impl OptUsize {
    #[expect(dead_code)]
    fn is_some(&self) -> bool {
        self.0 != usize::MAX
    }

    fn is_none(&self) -> bool {
        self.0 == usize::MAX
    }

    fn as_opt(&self) -> Option<usize> {
        if self.0 == usize::MAX {
            None
        } else {
            Some(self.0)
        }
    }

    fn as_mut(&mut self) -> Option<&mut usize> {
        if self.0 == usize::MAX {
            None
        } else {
            Some(&mut self.0)
        }
    }

    #[expect(dead_code)]
    fn map<T>(self, f: impl FnOnce(usize) -> T) -> Option<T> {
        self.as_opt().map(f)
    }

    fn unwrap(self) -> usize {
        if self.0 == usize::MAX {
            panic!("called `unwrap()` on a `None` value")
        } else {
            self.0
        }
    }
}

impl std::fmt::Debug for OptUsize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.0 == usize::MAX {
            write!(f, "None")
        } else {
            f.debug_tuple("Some").field(&self.0).finish()
        }
    }
}

impl From<usize> for OptUsize {
    fn from(value: usize) -> Self {
        OptUsize(value)
    }
}

impl From<Option<usize>> for OptUsize {
    fn from(value: Option<usize>) -> Self {
        match value {
            None => OptUsize(usize::MAX),
            Some(v) => OptUsize(v),
        }
    }
}

/// Holds info crucial for optimal traversal of the tree
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry<T: Node> {
    node: T,
    descendants_count: usize,
    parent_distance: OptUsize,
}

impl<T: Node> Entry<T> {
    pub fn new(node: T, descendants_count: usize, parent: Option<usize>) -> Self {
        Entry {
            node,
            descendants_count,
            parent_distance: parent.into(),
        }
    }

    pub fn get(&self) -> &T {
        &self.node
    }

    pub fn get_mut(&mut self) -> &mut T {
        &mut self.node
    }

    pub fn into_inner(self) -> T {
        self.node
    }

    pub fn descendants_count(&self) -> usize {
        self.descendants_count
    }

    /// Calculates by summing the descendent count of the nodes `new` takes as children.
    /// Panics if the number of orphan nodes present in `previous` is smaller than what
    /// `new` expects.
    pub fn calc_descendents(previous: &[Entry<T>], new: &T) -> usize {
        let mut dc = 0;
        for ac in 0..new.children() {
            dc += previous[previous
                .len()
                .checked_sub(dc + 1)
                .expect("Not enough orphan nodes")]
            .descendants_count
                + 1;
        }
        dc
    }

    fn configure_children_for_head(tree: &mut [Entry<T>]) {
        let head = tree.len() - 1;
        let mut cc = tree[head].node.children();
        let mut idx = head;
        while cc > 0 {
            idx -= 1;
            tree[idx].parent_distance = Some(head - idx).into();
            cc -= 1;
            idx -= tree[idx].descendants_count;
        }
    }

    fn from_nodes(nodes: impl IntoIterator<Item = T>) -> Vec<Entry<T>> {
        let nodes = nodes.into_iter();
        let mut res = Vec::with_capacity(nodes.size_hint().1.unwrap_or(64));
        for node in nodes {
            let dc = Self::calc_descendents(&res, &node);
            res.push(Self::new(node, dc, None));
            Self::configure_children_for_head(&mut res);
        }
        res
    }

    fn find_nth_child(tree: &[Entry<T>], parent: usize, n: usize) -> Option<usize> {
        let mut cc = tree.get(parent)?.node.children();
        if cc <= n {
            return None;
        }
        let mut idx = parent - 1;
        cc -= n + 1;
        while cc > 0 {
            idx -= tree[idx].descendants_count + 1;
            cc -= 1;
        }
        Some(idx)
    }

    fn find_next_sibling(tree: &[Entry<T>], child: usize) -> Option<usize> {
        let pd = tree.get(child)?.parent_distance.unwrap();
        if pd == 1 {
            return None;
        }
        let mut idx = pd + child - 1;
        loop {
            let prev = idx - tree[idx].descendants_count - 1;
            if prev > child {
                idx = prev;
            } else {
                break Some(idx);
            }
        }
    }
}

fn offset_idx(idx: usize, offset: i64) -> usize {
    // unchecked casting of idx to i64 when usize could be 64 bits may raise some eyebrows, but
    // consider that a tree of at least 9223372036854775807 nodes must exist for this cast to fail
    // and I don't think my library would ever be used in this way.
    (idx as i64 + offset).try_into().unwrap()
}

fn sew_replacement<T: Node>(tree: &mut [Entry<T>], head: usize, offset: i64) {
    let mut current = head;
    while let Some(po) = tree[current].parent_distance.as_opt() {
        let parent = current + po;
        // fix the descendant counts of the parent
        tree[parent].descendants_count = offset_idx(tree[parent].descendants_count, offset);
        // fix the parent offsets of current node's previous siblings
        let mut edge = current - tree[current].descendants_count;
        while edge > parent - tree[parent].descendants_count {
            edge -= 1;
            let po = tree[edge].parent_distance.as_mut().unwrap();
            *po = offset_idx(*po, offset);
            edge -= tree[edge].descendants_count;
        }
        current = parent;
    }
}

#[derive(Clone, PartialEq, Eq)]
pub struct PostfixTree<T: Node>(Vec<Entry<T>>);

impl<T: Node> PostfixTree<T> {
    pub fn from_inner_unchecked(vector: Vec<Entry<T>>) -> Self {
        PostfixTree(vector)
    }

    pub fn from_nodes(nodes: impl IntoIterator<Item = T>) -> Self {
        let entries = Entry::from_nodes(nodes);
        let Some(root) = entries.last() else {
            panic!("nodes can't be empty");
        };
        assert_eq!(root.descendants_count + 1, entries.len());
        PostfixTree(entries)
    }

    pub fn into_inner(self) -> Vec<Entry<T>> {
        self.0
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.0.get(index).map(|n| &n.node)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.0.get_mut(index).map(|n| &mut n.node)
    }

    pub fn children_iter<'a>(&'a self, head: usize) -> ChildIter<'a, T> {
        ChildIter::new(self, head)
    }

    pub fn parent_iter<'a>(&'a self, head: usize) -> ParentIter<'a, T> {
        ParentIter::new(&self.0, head)
    }

    pub fn postorder_iter<'a>(&'a self) -> PostOrderIter<'a, T> {
        PostOrderIter::new(&self.0)
    }

    pub fn postorder_subtree_iter<'a>(&'a self, head: usize) -> PostOrderIter<'a, T> {
        PostOrderIter::new(self.subtree_slice(head))
    }

    pub fn euler_tour<'a>(&'a self) -> EulerTour<'a, T> {
        EulerTour::new(self, self.len() - 1)
    }

    pub fn euler_tour_subtree<'a>(&'a self, head: usize) -> EulerTour<'a, T> {
        EulerTour::new(self, head)
    }

    pub fn as_slice(&self) -> &[Entry<T>] {
        &self.0
    }

    pub fn replace<I>(&mut self, head: usize, replace_with: I)
    where
        I: IntoIterator<Item = T>,
    {
        let before_len = self.len();
        let start = self.subtree_start(head);
        let parent_dist = self.0[head].parent_distance;
        self.0.splice(
            start..=head,
            replace_with.into_iter().map(|n| Entry::new(n, 0, None)),
        );
        let delta = self.len() as i64 - before_len as i64;
        let new_head = offset_idx(head, delta);
        self.0[new_head].parent_distance = parent_dist;
        for i in start..=new_head {
            self.0[i].descendants_count =
                Entry::calc_descendents(&self.0[start..i], &self.0[i].node);
            Entry::configure_children_for_head(&mut self.0[start..=i]);
        }
        assert_eq!(
            start,
            new_head - self.0[new_head].descendants_count,
            "replace_with produced more than one branch"
        );
        sew_replacement(&mut self.0, new_head, delta);
    }

    pub fn replace_from_sc_clone(
        &mut self,
        head: usize,
        src: &SubtreeCollection<T>,
        subtree_head: usize,
    ) {
        let delta =
            src.vec[subtree_head].descendants_count as i64 - self.0[head].descendants_count as i64;
        let parent_dist = self.0[head].parent_distance;
        self.0.splice(
            self.subtree_range(head),
            src.subtree_slice(subtree_head).iter().cloned(),
        );
        let new_head = offset_idx(head, delta);
        self.0[new_head].parent_distance = parent_dist;
        sew_replacement(&mut self.0, new_head, delta);
    }

    pub fn replace_from_sc_move(
        &mut self,
        head: usize,
        src: &mut SubtreeCollection<T>,
        subtree_head: usize,
    ) {
        let delta =
            src.vec[subtree_head].descendants_count as i64 - self.0[head].descendants_count as i64;
        let parent_dist = self.0[head].parent_distance;
        self.0
            .splice(self.subtree_range(head), src.drain(subtree_head));
        let new_head = offset_idx(head, delta);
        self.0[new_head].parent_distance = parent_dist;
        sew_replacement(&mut self.0, new_head, delta);
    }

    pub fn subtree_start(&self, head: usize) -> usize {
        head - self.0[head].descendants_count
    }

    pub fn subtree_range(&self, head: usize) -> Range<usize> {
        head - self.0[head].descendants_count..head + 1
    }

    pub fn subtree_slice(&self, head: usize) -> &[Entry<T>] {
        &self.0[head - self.0[head].descendants_count..=head]
    }

    pub fn parent(&self, idx: usize) -> Option<usize> {
        self.0
            .get(idx)
            .and_then(|e| e.parent_distance.as_opt())
            .map(|pd| pd + idx)
    }

    pub fn nth_child(&self, idx: usize, n: usize) -> Option<usize> {
        Entry::find_nth_child(&self.0, idx, n)
    }

    pub fn next_sibling(&self, idx: usize) -> Option<usize> {
        Entry::find_next_sibling(&self.0, idx)
    }
}

impl<T> Debug for PostfixTree<T>
where
    T: Node + Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PostfixTree[")?;
        for e in &self.0 {
            f.write_str("\n\t")?;
            <Entry<T> as Debug>::fmt(e, f)?;
        }
        f.write_str("\n]")
    }
}

impl<T: Node> Index<usize> for PostfixTree<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.0[index].node
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl Node for usize {
        fn children(&self) -> usize {
            *self
        }
    }

    #[test]
    fn entry_from_nodes() {
        fn convert(nodes: impl IntoIterator<Item = usize>) -> Vec<(usize, usize, Option<usize>)> {
            Entry::from_nodes(nodes)
                .into_iter()
                .map(|e| (e.node, e.descendants_count, e.parent_distance.as_opt()))
                .collect()
        }
        assert_eq!(convert([0]), vec![(0, 0, None)]);
        assert_eq!(
            convert([0, 0, 2]),
            vec![(0, 0, Some(2)), (0, 0, Some(1)), (2, 2, None)]
        );
        assert_eq!(
            convert([0, 0, 2, 0, 0, 2]),
            vec![
                (0, 0, Some(2)),
                (0, 0, Some(1)),
                (2, 2, None),
                (0, 0, Some(2)),
                (0, 0, Some(1)),
                (2, 2, None),
            ]
        );
        assert_eq!(
            convert([0, 0, 2, 0, 0, 2, 0, 3]),
            vec![
                (0, 0, Some(2)),
                (0, 0, Some(1)),
                (2, 2, Some(5)),
                (0, 0, Some(2)),
                (0, 0, Some(1)),
                (2, 2, Some(2)),
                (0, 0, Some(1)),
                (3, 7, None),
            ]
        );
    }

    #[test]
    fn replace() {
        let mut tree = PostfixTree::from_nodes([0, 0, 2, 0, 0, 2, 2]);
        let mut replace = |head: usize, nodes: &[usize]| {
            tree.replace(head, nodes.iter().copied());
            tree.0
                .iter()
                .map(|e| (e.node, e.descendants_count, e.parent_distance.as_opt()))
                .collect::<Vec<_>>()
        };
        assert_eq!(
            replace(0, &[0, 1]),
            vec![
                (0, 0, Some(1)),
                (1, 1, Some(2)),
                (0, 0, Some(1)),
                (2, 3, Some(4)),
                (0, 0, Some(2)),
                (0, 0, Some(1)),
                (2, 2, Some(1)),
                (2, 7, None)
            ]
        );
        assert_eq!(
            replace(3, &[0]),
            vec![
                (0, 0, Some(4)),
                (0, 0, Some(2)),
                (0, 0, Some(1)),
                (2, 2, Some(1)),
                (2, 4, None)
            ]
        );
        assert_eq!(
            replace(3, &[0, 0, 0, 0, 4, 0, 2]),
            vec![
                (0, 0, Some(8)),
                (0, 0, Some(4)),
                (0, 0, Some(3)),
                (0, 0, Some(2)),
                (0, 0, Some(1)),
                (4, 4, Some(2)),
                (0, 0, Some(1)),
                (2, 6, Some(1)),
                (2, 8, None)
            ]
        );
        assert_eq!(replace(8, &[0]), vec![(0, 0, None)]);
    }

    #[test]
    fn replace_from_sc() {
        let mut tree_m = PostfixTree::from_nodes([0, 0, 2]);
        let mut tree_c = PostfixTree::from_nodes([0, 0, 2]);
        let mut replace = |head: usize, src: &[usize], subtree_head: usize| {
            let mut sc = SubtreeCollection::from_nodes(src.iter().cloned());
            tree_c.replace_from_sc_clone(head, &sc, subtree_head);
            tree_m.replace_from_sc_move(head, &mut sc, subtree_head);
            (
                tree_c.0.iter().map(|e| e.node).collect::<Vec<_>>(),
                tree_m.0.iter().map(|e| e.node).collect::<Vec<_>>(),
            )
        };
        let dup = |vec: Vec<usize>| (vec.clone(), vec);
        assert_eq!(replace(0, &[0, 1], 1), dup(vec![0, 1, 0, 2]));
        assert_eq!(replace(2, &[0, 0, 2], 2), dup(vec![0, 1, 0, 0, 2, 2]));
        assert_eq!(
            replace(3, &[0, 0, 0, 3, 0, 1, 0, 2], 3),
            dup(vec![0, 1, 0, 0, 0, 0, 3, 2, 2])
        );
    }
}
