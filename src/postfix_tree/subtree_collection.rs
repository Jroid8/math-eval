use super::{Entry, Node, PostfixTree};
use std::{ops::Range, vec::Drain};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubtreeCollection<T: Node> {
    pub(super) vec: Vec<Entry<T>>,
    pub(super) orphan_subtrees: usize,
}

impl<T: Node> Default for SubtreeCollection<T> {
    fn default() -> Self {
        SubtreeCollection {
            vec: Vec::default(),
            orphan_subtrees: 0,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NotEnoughOrphans;

#[derive(Debug, Clone)]
pub struct MultipleRoots;

/// The subtrees are kept valid except they don't need to have a parent
/// until it is converted to a PostfixTree, where a quick O(1) check is performed
/// so only a single root exists at the end.
impl<T: Node> SubtreeCollection<T> {
    pub fn new() -> Self {
        SubtreeCollection::default()
    }

    pub fn from_alloc(mut vec: Vec<Entry<T>>) -> Self {
        vec.clear();
        SubtreeCollection {
            vec,
            orphan_subtrees: 0,
        }
    }

    pub fn from_nodes(nodes: impl IntoIterator<Item = T>) -> Self {
        let entries = Entry::from_nodes(nodes);
        let mut orphans = 0;
        let mut idx = entries.len();
        while idx > 0 {
            idx -= 1;
            idx -= entries[idx].descendants_count;
            orphans += 1;
        }
        SubtreeCollection {
            vec: entries,
            orphan_subtrees: orphans,
        }
    }

    pub fn push(&mut self, node: T) -> Result<(), NotEnoughOrphans> {
        if node.children() > self.orphan_subtrees {
            Err(NotEnoughOrphans)
        } else {
            self.orphan_subtrees = self.orphan_subtrees + 1 - node.children();
            let dc = Entry::calc_descendents(&self.vec, &node);
            self.vec.push(Entry::new(node, dc, None));
            Entry::configure_children_for_head(&mut self.vec);
            Ok(())
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        if let Some(entry) = self.vec.pop() {
            self.orphan_subtrees += entry.node.children();
            Some(entry.node)
        } else {
            None
        }
    }

    pub fn into_tree(self) -> Result<PostfixTree<T>, MultipleRoots> {
        if self.orphan_subtrees > 1 {
            Err(MultipleRoots)
        } else {
            Ok(PostfixTree(self.vec))
        }
    }

    pub fn into_inner(self) -> Vec<Entry<T>> {
        self.vec
    }

    pub fn len(&self) -> usize {
        self.vec.len()
    }

    pub fn is_empty(&self) -> bool {
        self.vec.is_empty()
    }

    pub fn last(&self) -> Option<&T> {
        self.vec.last().map(|n| &n.node)
    }

    pub fn last_mut(&mut self) -> Option<&mut T> {
        self.vec.last_mut().map(|n| &mut n.node)
    }

    pub fn inner(&self) -> &[Entry<T>] {
        &self.vec
    }

    pub fn orphan_subtrees(&self) -> usize {
        self.orphan_subtrees
    }

    pub fn extend_from_tree(&mut self, tree: &PostfixTree<T>, target_subtree: usize) {
        self.orphan_subtrees += 1;
        self.vec
            .extend_from_slice(tree.subtree_slice(target_subtree));
    }

    pub fn clear(&mut self) {
        self.orphan_subtrees = 0;
        self.vec.clear();
    }

    pub(super) fn drain(&mut self, head: usize) -> Drain<'_, Entry<T>> {
        assert!(self.vec[head].parent_distance.is_none());
        self.orphan_subtrees -= 1;
        self.vec.drain(self.subtree_start(head)..=head)
    }

    /// the node being removed *must* be an orphan
    pub fn remove(&mut self, head: usize) {
        self.drain(head);
    }

    pub fn subtree_start(&self, head: usize) -> usize {
        head - self.vec[head].descendants_count
    }

    pub fn subtree_slice(&self, head: usize) -> &[Entry<T>] {
        &self.vec[head - self.vec[head].descendants_count..=head]
    }

    pub fn subtree_range(&self, head: usize) -> Range<usize> {
        head - self.vec[head].descendants_count..head + 1
    }

    pub fn parent(&self, idx: usize) -> Option<usize> {
        self.vec
            .get(idx)
            .and_then(|e| e.parent_distance.as_opt())
            .map(|pd| pd + idx)
    }

    pub fn nth_child(&self, idx: usize, n: usize) -> Option<usize> {
        Entry::find_nth_child(&self.vec, idx, n)
    }

    pub fn next_sibling(&self, idx: usize) -> Option<usize> {
        Entry::find_next_sibling(&self.vec, idx)
    }
}

#[cfg(test)]
mod tests {
    use std::panic::catch_unwind;

    use super::*;

    #[test]
    fn push() {
        let mass_push = |nodes: &[usize]| {
            let mut builder = SubtreeCollection::new();
            for it in nodes {
                builder.push(*it).unwrap();
            }
            builder
                .into_tree()
                .unwrap()
                .0
                .into_iter()
                .map(|n| n.descendants_count)
                .collect::<Vec<_>>()
        };
        assert_eq!(mass_push(&[0, 0, 2]), [0, 0, 2]);
        assert_eq!(mass_push(&[0, 1, 1]), [0, 1, 2]);
        assert_eq!(mass_push(&[0, 1, 0, 2, 0, 2]), [0, 1, 0, 3, 0, 5]);
        assert_eq!(mass_push(&[0, 0, 0, 2, 2, 0, 2]), [0, 0, 0, 2, 4, 0, 6]);
        assert_eq!(mass_push(&[0, 0, 0, 3, 1]), [0, 0, 0, 3, 4]);
        assert_eq!(
            mass_push(&[0, 0, 0, 0, 0, 4, 0, 3]),
            [0, 0, 0, 0, 0, 4, 0, 7]
        );

        let catch_push_fail = |nodes: &[usize]| {
            catch_unwind(|| {
                let mut builder = SubtreeCollection::new();
                for it in nodes {
                    builder.push(*it).unwrap();
                }
                builder
            })
        };

        catch_push_fail(&[1]).unwrap_err();
        catch_push_fail(&[0, 2]).unwrap_err();
        catch_push_fail(&[0, 0, 0, 3, 2]).unwrap_err();
        catch_push_fail(&[0, 0, 0, 3, 0, 4]).unwrap_err();
    }
}
