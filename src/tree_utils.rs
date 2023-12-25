use indextree::{Arena, NodeId};
use std::fmt::Debug;

pub(crate) fn construct<A, R, E>(
    inital_args: A,
    recur_func: impl Fn(A, &mut Vec<A>) -> Result<Option<R>, E>,
    arena: Option<Arena<R>>,
) -> Result<Tree<R>, E> {
    // anyone can convert a recursive algorithm to an iterative one by knowing why and where they need recursion
    // In my case I needed recursion to prepare the children of every node before constructing the parent.
    // local variables where only used to pass the right arguments to itself, and the result was never transformed.
    // this function is a recursion simulator that gives me these features without actual recursion
    let mut arena = arena.unwrap_or_default();
    let mut call_stack = vec![inital_args];
    // stores nodes that are waiting for their children to be prepared and attached, and also the number of children it needs
    // each node is the parent of every node that comes after it
    let mut result_stack: Vec<(NodeId, usize)> = Vec::new();
    let mut header = None;
    while let Some(current_call) = call_stack.pop() {
        let prev_len = call_stack.len();
        match recur_func(current_call, &mut call_stack) {
            Ok(Some(result)) => {
                let result_node = arena.new_node(result);
                if let Some((node, req)) = result_stack.last() {
                    node.prepend(result_node, &mut arena);
                    if node.children(&arena).count() == *req {
                        result_stack.pop();
                    }
                } else {
                    header = Some(result_node);
                }
                let req_children = call_stack.len() - prev_len;
                // if the size of the call stack is increased then this node must be waiting for it's children
                if req_children > 0 {
                    result_stack.push((result_node, req_children));
                }
            }
            // in some cases like (x) or +x, the function just calls itself again and directly returns it's result
            Ok(None) => (),
            Err(e) => {
                return Err(e);
            }
        }
    }
    Ok(Tree {
        arena,
        root: header.unwrap(),
    })
}

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

impl<T> PartialEq for Tree<T> where T: PartialEq {
    fn eq(&self, other: &Self) -> bool {
        self.root.traverse(&self.arena).eq(other.root.traverse(&self.arena))
    }
}

impl<T> Eq for Tree<T> where T: Eq {}

#[cfg(test)]
impl<T> Tree<T>
where
    T: Clone,
{
    fn to_vectree(&self) -> VecTree<T> {
        VecTree::new(&self.arena, self.root)
    }
}

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

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum Input {
        A(Vec<Input>),
        B(Box<Input>, Box<Input>),
        C(Box<Input>),
        End(usize),
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

    #[test]
    fn test_construct() {
        macro_rules! create {
            ($input:expr) => {{
                super::construct::<&Input, Tree, ()>(
                    &$input,
                    |input, cs| match input {
                        Input::A(v) => {
                            cs.extend(v);
                            Ok(Some(Tree::A))
                        }
                        Input::B(a, b) => {
                            cs.push(a);
                            cs.push(b);
                            Ok(Some(Tree::B))
                        }
                        Input::C(c) => {
                            cs.push(c);
                            Ok(None)
                        }
                        Input::End(i) => {
                            if *i == 999 {
                                Err(())
                            } else {
                                Ok(Some(Tree::End(*i)))
                            }
                        }
                    },
                    None,
                )
                .map(|tree| tree.to_vectree())
            }};
        }

        assert_eq!(create!(Input::End(123)).unwrap(), Leaf(Tree::End(123)));
        assert_eq!(
            create!(Input::B(Box::new(Input::End(1)), Box::new(Input::End(2)))).unwrap(),
            branch!(Tree::B, Leaf(Tree::End(1)), Leaf(Tree::End(2)))
        );
        assert_eq!(
            create!(Input::A(vec![
                Input::End(1),
                Input::End(2),
                Input::End(3),
                Input::End(4)
            ]))
            .unwrap(),
            branch!(
                Tree::A,
                Leaf(Tree::End(1)),
                Leaf(Tree::End(2)),
                Leaf(Tree::End(3)),
                Leaf(Tree::End(4))
            )
        );
        assert_eq!(
            create!(Input::C(Box::new(Input::End(12)))).unwrap(),
            Leaf(Tree::End(12))
        );
        assert_eq!(
            create!(Input::B(
                Box::new(Input::B(Box::new(Input::End(12)), Box::new(Input::End(13)))),
                Box::new(Input::B(Box::new(Input::End(11)), Box::new(Input::End(19))))
            ))
            .unwrap(),
            branch!(
                Tree::B,
                branch!(Tree::B, Leaf(Tree::End(12)), Leaf(Tree::End(13))),
                branch!(Tree::B, Leaf(Tree::End(11)), Leaf(Tree::End(19)))
            )
        );
        assert_eq!(
            create!(Input::A(vec![
                Input::B(Box::new(Input::End(22)), Box::new(Input::End(82))),
                Input::B(
                    Box::new(Input::C(Box::new(Input::B(
                        Box::new(Input::End(11)),
                        Box::new(Input::End(22))
                    )))),
                    Box::new(Input::End(33))
                ),
                Input::C(Box::new(Input::End(1193))),
                Input::A(vec![
                    Input::B(Box::new(Input::End(22)), Box::new(Input::End(82))),
                    Input::B(
                        Box::new(Input::B(Box::new(Input::End(11)), Box::new(Input::End(22)))),
                        Box::new(Input::End(33))
                    ),
                    Input::C(Box::new(Input::End(302))),
                    Input::End(291)
                ])
            ]))
            .unwrap(),
            branch!(
                Tree::A,
                branch!(Tree::B, Leaf(Tree::End(22)), Leaf(Tree::End(82))),
                branch!(
                    Tree::B,
                    branch!(Tree::B, Leaf(Tree::End(11)), Leaf(Tree::End(22))),
                    Leaf(Tree::End(33))
                ),
                Leaf(Tree::End(1193)),
                branch!(
                    Tree::A,
                    branch!(Tree::B, Leaf(Tree::End(22)), Leaf(Tree::End(82))),
                    branch!(
                        Tree::B,
                        branch!(Tree::B, Leaf(Tree::End(11)), Leaf(Tree::End(22))),
                        Leaf(Tree::End(33))
                    ),
                    Leaf(Tree::End(302)),
                    Leaf(Tree::End(291))
                )
            )
        );
        assert_eq!(create!(Input::End(999)), Err(()));
    }
}
