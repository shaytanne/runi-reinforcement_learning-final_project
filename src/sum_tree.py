import numpy as np


class SumTree:
    """
    Binary sum tree for O(log N) prioritized buffer sampling in DQN PER

    Leaf nodes store priorities, internal nodes store sums
    Tree array layout: [internal nodes | leaf nodes]
        - internal nodes: indices [0, capacity-1]
        - leaf nodes:     indices [capacity, 2*capacity-1]
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity, dtype=np.float32)
        self.write_index = 0

    def _propagate(self, tree_idx: int) -> None:
        """Update parent sums from leaf to root"""
        parent = tree_idx // 2
        while parent >= 1:
            self.tree[parent] = self.tree[2 * parent] + self.tree[2 * parent + 1]
            parent //= 2

    def update(self, data_idx: int, priority: float) -> None:
        """Set priority for specific data/leaf index"""
        tree_idx = data_idx + self.capacity
        self.tree[tree_idx] = priority
        self._propagate(tree_idx)

    def add(self, priority: float) -> int:
        """Add priority at next position (circular), return data index"""
        data_idx = self.write_index
        self.update(data_idx, priority)
        self.write_index = (self.write_index + 1) % self.capacity
        return data_idx

    def sample(self, value: float) -> int:
        """Sample a leaf index proportional to priority. value in [0, total)"""
        idx = 1  # start at root
        while idx < self.capacity:
            left = 2 * idx
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        return idx - self.capacity

    @property
    def total(self) -> float:
        return self.tree[1]

    @property
    def max(self) -> float:
        return self.tree[self.capacity:].max()