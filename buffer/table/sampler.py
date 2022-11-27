import numpy as np
import time


class Sampler:
    def __init__(self, table=None):
        self.table = table

    def sample(self):
        raise NotImplementedError


class UniformSampler(Sampler):
    def __init__(self, table):
        super().__init__(table)

    def sample(self, indices, n):
        assert len(indices) >= n
        indices = np.random.choice(indices, size=n, replace=False)
        return indices


class LUMRFSampler(Sampler):
    "Less Usage Most Recent (inserted) First Sampler"

    def __init__(self, table=None):
        super().__init__(table)
        assert hasattr(table, "usage_ctrs")
        assert hasattr(table, "insert_timestamps")

    def sample(self, indices, n):
        assert len(indices) >= n
        usage_ctrs = self.table.usage_ctrs[indices]
        timestamps = self.table.insert_timestamps[indices]
        _indices = np.lexsort([-timestamps, usage_ctrs])[:n]
        indices = indices[_indices]
        assert len(indices) == n
        return indices


class LULRFSampler(Sampler):
    "Less Usage Least Recent (inserted) First Sampler"

    def __init__(self, table=None):
        super().__init__(table)
        assert hasattr(table, "usage_ctrs")
        assert hasattr(table, "insert_timestamps")

    def sample(self, indices, n):
        assert len(indices) >= n
        usage_ctrs = self.table.usage_ctrs[indices]
        timestamps = self.table.insert_timestamps[indices]
        _indices = np.lexsort([timestamps, usage_ctrs])[:n]
        indices = indices[_indices]
        assert len(indices) == n
        return indices
