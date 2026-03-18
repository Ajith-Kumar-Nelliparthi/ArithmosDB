"""
Unit tests for FlatIndex.

Run with:
    pytest tests/ -v

Requirements:
    - libvectordb.so must be built and on LD_LIBRARY_PATH
    - CUDA GPU must be available
"""
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Skip entire module if CUDA / the library is unavailable
# ---------------------------------------------------------------------------
try:
    from arithmosdb import FlatIndex
    HAS_GPU = True
except (FileNotFoundError, OSError):
    HAS_GPU = False

pytestmark = pytest.mark.skipif(not HAS_GPU, reason="libvectordb.so not found or no GPU")

RNG = np.random.default_rng(42)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def brute_force_topk(vectors, queries, k):
    """CPU reference implementation."""
    diff  = queries[:, None, :] - vectors[None, :, :]   # [nq, n, d]
    dists = (diff ** 2).sum(axis=-1)                    # [nq, n]
    idx   = np.argsort(dists, axis=1)[:, :k]
    return dists[np.arange(len(queries))[:, None], idx], idx


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestFlatIndexBasic:

    def test_add_and_len(self):
        idx = FlatIndex(d=4)
        vecs = RNG.random((100, 4), dtype=np.float32)
        idx.add(vecs)
        assert len(idx) == 100

    def test_wrong_dimension_raises(self):
        idx = FlatIndex(d=8)
        with pytest.raises(ValueError):
            idx.add(np.ones((10, 4), dtype=np.float32))

    def test_search_returns_correct_shape(self):
        idx = FlatIndex(d=16)
        idx.add(RNG.random((200, 16), dtype=np.float32))
        dists, ids = idx.search(RNG.random((5, 16), dtype=np.float32), k=3)
        assert dists.shape == (5, 3)
        assert ids.shape   == (5, 3)

    def test_search_empty_raises(self):
        idx = FlatIndex(d=4)
        with pytest.raises(RuntimeError):
            idx.search(np.ones((1, 4), dtype=np.float32), k=1)

    def test_single_query_broadcast(self):
        """1-D query should be accepted."""
        idx = FlatIndex(d=8)
        idx.add(RNG.random((50, 8), dtype=np.float32))
        dists, ids = idx.search(RNG.random(8, dtype=np.float32), k=5)
        assert dists.shape == (1, 5)


class TestFlatIndexCorrectness:

    @pytest.mark.parametrize("n,d,nq,k", [
        (500,   16,  10,  5),
        (1000,  32,  20,  10),
        (200,   128,  5,   3),
    ])
    def test_top1_matches_cpu(self, n, d, nq, k):
        """GPU top-1 result must match brute-force CPU."""
        vecs    = RNG.random((n, d), dtype=np.float32)
        queries = RNG.random((nq, d), dtype=np.float32)

        idx = FlatIndex(d=d)
        idx.add(vecs)
        gpu_dists, gpu_ids = idx.search(queries, k=k)

        cpu_dists, cpu_ids = brute_force_topk(vecs, queries, k)

        # Nearest neighbour must be identical
        assert np.array_equal(gpu_ids[:, 0], cpu_ids[:, 0]), \
            "Top-1 index mismatch between GPU and CPU"

        # Distances should be close
        np.testing.assert_allclose(gpu_dists[:, 0], cpu_dists[:, 0], rtol=1e-4)

    def test_distances_are_sorted_ascending(self):
        idx = FlatIndex(d=32)
        idx.add(RNG.random((1000, 32), dtype=np.float32))
        dists, _ = idx.search(RNG.random((10, 32), dtype=np.float32), k=10)
        for row in dists:
            assert np.all(row[:-1] <= row[1:]), "Distances not sorted ascending"

    def test_indices_are_valid(self):
        n, d, nq, k = 300, 64, 8, 10
        idx = FlatIndex(d=d)
        idx.add(RNG.random((n, d), dtype=np.float32))
        _, ids = idx.search(RNG.random((nq, d), dtype=np.float32), k=k)
        assert ids.min() >= 0
        assert ids.max() < n

    def test_exact_nearest_neighbour(self):
        """Query == one of the vectors → that vector must be the top result."""
        vecs = RNG.random((100, 16), dtype=np.float32)
        idx  = FlatIndex(d=16)
        idx.add(vecs)

        target = 42
        query  = vecs[target:target+1].copy()
        _, ids = idx.search(query, k=1)
        assert ids[0, 0] == target, f"Expected index {target}, got {ids[0,0]}"


class TestFlatIndexEdgeCases:

    def test_k_equals_n(self):
        n, d = 50, 8
        idx = FlatIndex(d=d)
        idx.add(RNG.random((n, d), dtype=np.float32))
        dists, ids = idx.search(RNG.random((3, d), dtype=np.float32), k=n)
        assert ids.shape == (3, n)

    def test_replace_vectors(self):
        """Calling add() twice should replace the index contents."""
        idx = FlatIndex(d=4)
        idx.add(RNG.random((100, 4), dtype=np.float32))
        assert len(idx) == 100
        idx.add(RNG.random((50, 4), dtype=np.float32))
        assert len(idx) == 50

    def test_float64_input_is_cast(self):
        """float64 input should be silently cast to float32."""
        idx  = FlatIndex(d=8)
        vecs = RNG.random((30, 8))           # float64
        idx.add(vecs)
        dists, ids = idx.search(RNG.random((2, 8)), k=3)
        assert dists.dtype == np.float32
