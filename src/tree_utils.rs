use indextree::{Arena, NodeId};
use std::fmt::Debug;

#[derive(Clone)]
pub struct Tree<T> {
    pub arena: Arena<T>,
    pub root: NodeId,
}

impl<T> Debug for Tree<T>
where
    T: Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.root.debug_pretty_print(&self.arena))
    }
}

impl<T> PartialEq for Tree<T>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.root
            .traverse(&self.arena)
            .eq(other.root.traverse(&self.arena))
    }
}

impl<T> Eq for Tree<T> where T: Eq {}

#[cfg(test)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum VecTree<T: Clone> {
    Leaf(T),
    Branch(T, Vec<VecTree<T>>),
}

#[cfg(test)]
impl<T> VecTree<T>
where
    T: Clone,
{
    pub(crate) fn new(arena: &Arena<T>, node: NodeId) -> VecTree<T> {
        let t = arena[node].get().clone();
        if node.children(arena).count() > 0 {
            Self::Branch(
                t,
                node.children(arena).map(|c| Self::new(arena, c)).collect(),
            )
        } else {
            Self::Leaf(t)
        }
    }
}

#[cfg(test)]
mod test {
    use super::VecTree::{self, Leaf};
    use indextree::Arena;

    macro_rules! branch {
        ($node:expr, $($children:expr),+) => {
            VecTree::Branch($node,vec![$($children),+])
        };
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Tree {
        A,
        B,
        End(usize),
    }

    #[test]
    fn test_new_vectree() {
        use Tree::*;

        // A
        // |-- End(1)
        // |-- End(2)
        // |-- End(3)
        // `-- B
        //     |-- End(4)
        //     `-- A
        //         `-- End(5)
        let mut arena: Arena<Tree> = Arena::new();
        let a1 = arena.new_node(A);
        a1.append_value(End(1), &mut arena);
        a1.append_value(End(2), &mut arena);
        a1.append_value(End(3), &mut arena);
        let b1 = a1.append_value(B, &mut arena);
        b1.append_value(End(4), &mut arena);
        let a2 = b1.append_value(A, &mut arena);
        a2.append_value(End(5), &mut arena);

        // A
        // |-- End(1)
        // `-- A
        //     |-- End(2)
        //     |-- End(3)
        //     |-- End(4)
        //     `-- B
        //         |-- End(5)
        //         `-- B
        //             |-- End(6)
        //             `-- End(7)
        let a3 = arena.new_node(A);
        a3.append_value(End(1), &mut arena);
        let a4 = a3.append_value(A, &mut arena);
        a4.append_value(End(2), &mut arena);
        a4.append_value(End(3), &mut arena);
        a4.append_value(End(4), &mut arena);
        let b2 = a4.append_value(B, &mut arena);
        b2.append_value(End(5), &mut arena);
        let b3 = b2.append_value(B, &mut arena);
        b3.append_value(End(6), &mut arena);
        b3.append_value(End(7), &mut arena);

        let vb3 = branch!(B, Leaf(End(6)), Leaf(End(7)));
        let vb2 = branch!(B, Leaf(End(5)), vb3.clone());
        let va1 = branch!(
            A,
            Leaf(End(1)),
            Leaf(End(2)),
            Leaf(End(3)),
            branch!(B, Leaf(End(4)), branch!(A, Leaf(End(5))))
        );
        let va3 = branch!(
            A,
            Leaf(End(1)),
            branch!(A, Leaf(End(2)), Leaf(End(3)), Leaf(End(4)), vb2.clone())
        );

        assert_eq!(VecTree::new(&arena, b3), vb3);
        assert_eq!(VecTree::new(&arena, b2), vb2);
        assert_eq!(VecTree::new(&arena, a1), va1);
        assert_eq!(VecTree::new(&arena, a3), va3);
    }
}
