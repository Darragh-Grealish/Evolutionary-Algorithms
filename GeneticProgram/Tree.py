from __future__ import annotations
from collections import deque
from typing import Any, Callable, Generator, Iterable, List, Optional, TypeVar

# Tree.py

T = TypeVar("T")


class TreeNode:
    def __init__(self, value: T, children: Optional[Iterable["TreeNode"]] = None) -> None:
        self.value: T = value
        self.parent: Optional["TreeNode"] = None
        self.children: List["TreeNode"] = []
        if children:
            for c in children:
                self.add_child(c)

    def __repr__(self) -> str:
        return f"TreeNode({self.value!r})"

    # Relationship management
    def add_child(self, node: "TreeNode") -> "TreeNode":
        """Attach node as a child. Reparents node if it had a previous parent."""
        if node is self:
            raise ValueError("Cannot add node as a child of itself")
        if node.parent is not None:
            node.parent.remove_child(node)
        node.parent = self
        self.children.append(node)
        return node

    def remove_child(self, node: "TreeNode") -> None:
        """Remove a child node (if present) and clear its parent."""
        try:
            self.children.remove(node)
            node.parent = None
        except ValueError:
            pass

    def is_root(self) -> bool:
        return self.parent is None

    def is_leaf(self) -> bool:
        return not self.children

    # Depth & Height
    def depth(self) -> int:
        """Distance from this node to the root (root.depth() == 0)."""
        d = 0
        node = self
        while node.parent is not None:
            d += 1
            node = node.parent
        return d

    def height(self) -> int:
        """Max distance from this node down to any leaf (leaf.height() == 0)."""
        if self.is_leaf():
            return 0
        return 1 + max(child.height() for child in self.children)

    # Traversals
    def dfs_iter(self) -> Generator["TreeNode", None, None]:
        """Depth-first traversal (pre-order), iterative."""
        stack = [self]
        while stack:
            node = stack.pop()
            yield node
            # push children in reverse to visit left-to-right
            for child in reversed(node.children):
                stack.append(child)

    def dfs_recursive(self) -> Generator["TreeNode", None, None]:
        """Depth-first traversal (pre-order), recursive."""
        yield self
        for child in self.children:
            yield from child.dfs_recursive()

    def bfs(self) -> Generator["TreeNode", None, None]:
        """Breadth-first traversal (level order)."""
        q = deque([self])
        while q:
            node = q.popleft()
            yield node
            for child in node.children:
                q.append(child)

    # Convenience value iterators
    def dfs_values(self) -> Generator[T, None, None]:
        for n in self.dfs_iter():
            yield n.value

    def bfs_values(self) -> Generator[T, None, None]:
        for n in self.bfs():
            yield n.value

    # Find helpers
    def find_dfs(self, predicate: Callable[["TreeNode"], bool]) -> Optional["TreeNode"]:
        """Return first node matching predicate using DFS, or None."""
        for n in self.dfs_iter():
            if predicate(n):
                return n
        return None

    def find_bfs(self, predicate: Callable[["TreeNode"], bool]) -> Optional["TreeNode"]:
        """Return first node matching predicate using BFS, or None."""
        for n in self.bfs():
            if predicate(n):
                return n
        return None

    # Utilities
    def size(self) -> int:
        """Number of nodes in the subtree rooted at this node."""
        return sum(1 for _ in self.dfs_iter())

    def to_tuple(self) -> tuple:
        """Represent subtree as nested tuples: (value, [children...])"""
        return (self.value, [c.to_tuple() for c in self.children])
