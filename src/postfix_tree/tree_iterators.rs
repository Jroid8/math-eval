use std::iter::FusedIterator;

use crate::postfix_tree::PostfixTree;

use super::{Entry, Node};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChildIter<'a, T: Node> {
    tree: &'a PostfixTree<T>,
    pos: usize,
    children: usize,
}

impl<'a, T: Node> ChildIter<'a, T> {
    pub fn new(tree: &'a PostfixTree<T>, target: usize) -> Self {
        Self {
            tree,
            pos: target.wrapping_sub(1),
            children: tree[target].children(),
        }
    }
}

impl<'a, T: Node> Iterator for ChildIter<'a, T> {
    type Item = (&'a T, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.children == 0 {
            return None;
        }
        self.children -= 1;
        let idx = self.pos;
        let cur = &self.tree.0[self.pos];
        if self.pos > cur.descendants_count {
            self.pos -= cur.descendants_count + 1;
        } else {
            debug_assert_eq!(cur.descendants_count, self.pos);
            debug_assert_eq!(self.children, 0);
        }
        Some((&cur.node, idx))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.children, Some(self.children))
    }
}

impl<'a, T: Node> ExactSizeIterator for ChildIter<'a, T> {
    #[inline]
    fn len(&self) -> usize {
        self.children
    }
}

impl<'a, T: Node> FusedIterator for ChildIter<'a, T> {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParentIter<'a, T: Node> {
    tree: &'a [Entry<T>],
    pos: usize,
}

impl<'a, T: Node> ParentIter<'a, T> {
    pub fn new(tree: &'a [Entry<T>], start: usize) -> Self {
        Self { tree, pos: start }
    }
}

impl<'a, T: Node> Iterator for ParentIter<'a, T> {
    type Item = (&'a T, usize);

    fn next(&mut self) -> Option<Self::Item> {
        let parent = self.tree[self.pos].parent_distance.as_opt()? + self.pos;
        self.pos = parent;
        Some((&self.tree[parent].node, parent))
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.tree.len() - self.pos - 1))
    }
}

impl<'a, T: Node> FusedIterator for ParentIter<'a, T> {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PostOrderIter<'a, T: Node> {
    slice: &'a [Entry<T>],
    index: usize,
}

impl<'a, T: Node> PostOrderIter<'a, T> {
    pub fn new(slice: &'a [Entry<T>]) -> Self {
        PostOrderIter { slice, index: 0 }
    }
}

impl<'a, T: Node> Iterator for PostOrderIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.slice.len() {
            return None;
        }
        self.index += 1;
        self.slice.get(self.index - 1).map(|n| &n.node)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl<'a, T: Node> ExactSizeIterator for PostOrderIter<'a, T> {
    #[inline]
    fn len(&self) -> usize {
        self.slice.len() - self.index
    }
}

impl<'a, T: Node> FusedIterator for PostOrderIter<'a, T> {}

#[derive(Debug, Clone, Copy)]
enum CurrentNodeEdge {
    Start(usize),
    End(usize),
    Done,
}

// Inspired by indextree's traverse
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeEdge<'a, T: Node> {
    Start(&'a T, usize),
    End(&'a T, usize),
}

impl<'a, T: Node> NodeEdge<'a, T> {
    #[inline]
    pub fn node(&self) -> &'a T {
        match self {
            NodeEdge::Start(node, _) => node,
            NodeEdge::End(node, _) => node
        }
    }

    #[inline]
    pub fn index(&self) -> usize {
        match self {
            NodeEdge::Start(_, idx) => *idx,
            NodeEdge::End(_, idx) => *idx
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct EulerTour<'a, T: Node> {
    tree: &'a PostfixTree<T>,
    root: usize,
    current: CurrentNodeEdge,
    progress: usize,
}

impl<'a, T: Node> EulerTour<'a, T> {
    pub fn new(tree: &'a PostfixTree<T>, root: usize) -> Self {
        EulerTour {
            tree,
            root,
            current: CurrentNodeEdge::Start(root),
            progress: 0,
        }
    }
}

impl<'a, T: Node> Iterator for EulerTour<'a, T> {
    type Item = NodeEdge<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        let next = match self.current {
            CurrentNodeEdge::Start(idx) => {
                if let Some(fc) = Entry::find_nth_child(&self.tree.0, idx, 0) {
                    CurrentNodeEdge::Start(fc)
                } else {
                    CurrentNodeEdge::End(idx)
                }
            }
            CurrentNodeEdge::End(idx) => {
                if let Some(pd) = self.tree.0[idx].parent_distance.as_opt()
                    && idx < self.root
                {
                    if let Some(sib) = Entry::find_next_sibling(&self.tree.0, idx) {
                        CurrentNodeEdge::Start(sib)
                    } else {
                        CurrentNodeEdge::End(idx + pd)
                    }
                } else {
                    CurrentNodeEdge::Done
                }
            }
            CurrentNodeEdge::Done => return None,
        };
        self.progress += 1;
        match std::mem::replace(&mut self.current, next) {
            CurrentNodeEdge::Start(idx) => Some(NodeEdge::Start(&self.tree[idx], idx)),
            CurrentNodeEdge::End(idx) => Some(NodeEdge::End(&self.tree[idx], idx)),
            CurrentNodeEdge::Done => None,
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl<'a, T: Node> ExactSizeIterator for EulerTour<'a, T> {
    #[inline]
    fn len(&self) -> usize {
        self.tree.len() * 2 - self.progress
    }
}

impl<'a, T: Node> FusedIterator for EulerTour<'a, T> {}

#[cfg(test)]
mod tests {
    use super::NodeEdge;
    use crate::postfix_tree::PostfixTree;

    #[test]
    fn child_node_iter() {
        let tree = PostfixTree::from_nodes([0, 0, 0, 2, 0, 1, 0, 0, 2, 0, 4, 2]);
        let collect_children = |head| {
            tree.children_iter(head)
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
    fn euler_tour() {
        let tree = PostfixTree::from_nodes([0, 0, 0, 2, 0, 1, 0, 1, 2, 0, 0, 0, 3, 4]);
        let collect_tour = |head| tree.euler_tour_subtree(head).collect::<Vec<_>>();
        assert_eq!(
            collect_tour(0),
            vec![NodeEdge::Start(&0, 0), NodeEdge::End(&0, 0)]
        );
        assert_eq!(
            collect_tour(3),
            vec![
                NodeEdge::Start(&2, 3),
                NodeEdge::Start(&0, 1),
                NodeEdge::End(&0, 1),
                NodeEdge::Start(&0, 2),
                NodeEdge::End(&0, 2),
                NodeEdge::End(&2, 3),
            ]
        );
        assert_eq!(
            collect_tour(8),
            vec![
                NodeEdge::Start(&2, 8),
                NodeEdge::Start(&1, 5),
                NodeEdge::Start(&0, 4),
                NodeEdge::End(&0, 4),
                NodeEdge::End(&1, 5),
                NodeEdge::Start(&1, 7),
                NodeEdge::Start(&0, 6),
                NodeEdge::End(&0, 6),
                NodeEdge::End(&1, 7),
                NodeEdge::End(&2, 8),
            ]
        );
        assert_eq!(
            collect_tour(13),
            vec![
                NodeEdge::Start(&4, 13),
                NodeEdge::Start(&0, 0),
                NodeEdge::End(&0, 0),
                NodeEdge::Start(&2, 3),
                NodeEdge::Start(&0, 1),
                NodeEdge::End(&0, 1),
                NodeEdge::Start(&0, 2),
                NodeEdge::End(&0, 2),
                NodeEdge::End(&2, 3),
                NodeEdge::Start(&2, 8),
                NodeEdge::Start(&1, 5),
                NodeEdge::Start(&0, 4),
                NodeEdge::End(&0, 4),
                NodeEdge::End(&1, 5),
                NodeEdge::Start(&1, 7),
                NodeEdge::Start(&0, 6),
                NodeEdge::End(&0, 6),
                NodeEdge::End(&1, 7),
                NodeEdge::End(&2, 8),
                NodeEdge::Start(&3, 12),
                NodeEdge::Start(&0, 9),
                NodeEdge::End(&0, 9),
                NodeEdge::Start(&0, 10),
                NodeEdge::End(&0, 10),
                NodeEdge::Start(&0, 11),
                NodeEdge::End(&0, 11),
                NodeEdge::End(&3, 12),
                NodeEdge::End(&4, 13),
            ]
        );
    }
}
