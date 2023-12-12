use indextree::{Arena, NodeId};
use std::fmt::Debug;

pub(crate) fn construct<A, R, E>(
    inital_args: A,
    recur_func: impl Fn(A, &mut Vec<(A, Option<Box<dyn Fn(E) -> E>>)>) -> Result<Option<R>, E>,
    arena: Option<Arena<R>>,
) -> Result<(Arena<R>, NodeId), E> {
    let mut arena = arena.unwrap_or_default();
    let mut call_stack = vec![(inital_args, None)];
    let mut result_stack: Vec<(NodeId, usize)> = Vec::new();
    let mut header = None;
    let mut stack_trace: Vec<Box<dyn Fn(E) -> E>> = Vec::new();
    while let Some((current_call, error_modifier)) = call_stack.pop() {
        let prev_len = call_stack.len();
        match recur_func(current_call, &mut call_stack) {
            Ok(Some(result)) => {
                let result_node = arena.new_node(result);
                if let Some((node, req)) = result_stack.last() {
                    node.prepend(result_node, &mut arena);
                    if node.children(&arena).count() == *req {
                        result_stack.pop();
                        stack_trace.pop();
                    }
                } else {
                    header = Some(result_node);
                }
                let req_children = call_stack.len() - prev_len;
                if req_children > 0 {
                    result_stack.push((result_node, req_children));
                    if let Some(error_modifier) = error_modifier {
                        stack_trace.push(error_modifier);
                    }
                }
            }
            Ok(None) => (),
            Err(e) => {
                return Err(stack_trace.iter().fold(
                    match error_modifier {
                        Some(em) => em(e),
                        None => e,
                    },
                    |acc, f| f(acc),
                ));
            }
        }
    }
    Ok((arena, header.unwrap()))
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
                super::construct::<&Input, Tree, usize>(
                    &$input,
                    |input, cs| match input {
                        Input::A(v) => {
                            cs.extend(v.iter().enumerate().map(|(i, a)| {
                                (
                                    a,
                                    Some(Box::new(move |e| e + i) as Box<dyn Fn(usize) -> usize>),
                                )
                            }));
                            Ok(Some(Tree::A))
                        }
                        Input::B(a, b) => {
                            cs.push((a, None));
                            cs.push((b, Some(Box::new(|e| e + 1))));
                            Ok(Some(Tree::B))
                        }
                        Input::C(c) => {
                            cs.push((c, None));
                            Ok(None)
                        }
                        Input::End(i) => {
                            if *i == 999 {
                                Err(0)
                            } else {
                                Ok(Some(Tree::End(*i)))
                            }
                        }
                    },
                    None,
                )
                .map(|(arena, node)| VecTree::new(&arena, node))
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

        assert_eq!(create!(Input::End(999)), Err(0));
        assert_eq!(
            create!(Input::B(Box::new(Input::End(1)), Box::new(Input::End(999)))),
            Err(1)
        );
        assert_eq!(
            create!(Input::B(
                Box::new(Input::End(1)),
                Box::new(Input::B(
                    Box::new(Input::End(111)),
                    Box::new(Input::End(999)),
                ))
            )),
            Err(2)
        );
        assert_eq!(
            create!(Input::A(vec![
                Input::End(1),
                Input::End(2),
                Input::End(999)
            ])),
            Err(2)
        );
        assert_eq!(
            create!(Input::A(vec![
                Input::End(1),
                Input::End(3),
                Input::B(Box::new(Input::End(999)), Box::new(Input::End(1))),
                Input::End(6),
            ])),
            Err(2)
        );
        assert_eq!(
            create!(Input::A(vec![
                Input::End(1),
                Input::End(7),
                Input::B(Box::new(Input::End(1)), Box::new(Input::End(999))),
                Input::End(34),
                Input::End(0),
            ])),
            Err(3)
        );
        assert_eq!(
            create!(Input::A(vec![
                Input::End(1),
                Input::End(7),
                Input::End(34),
                Input::A(vec![
                    Input::End(321),
                    Input::End(31),
                    Input::A(vec![Input::End(448), Input::End(999)])
                ]),
                Input::End(0),
            ])),
            Err(6)
        );
        assert_eq!(
            create!(Input::A(vec![
                Input::B(Box::new(Input::End(1)), Box::new(Input::End(9))),
                Input::B(Box::new(Input::End(1)), Box::new(Input::End(9))),
                Input::B(Box::new(Input::End(1)), Box::new(Input::End(9))),
                Input::A(vec![
                    Input::End(321),
                    Input::A(vec![Input::End(448), Input::End(999)]),
                    Input::End(31),
                ]),
                Input::B(Box::new(Input::End(1)), Box::new(Input::End(9))),
            ])),
            Err(5)
        );
    }
}
