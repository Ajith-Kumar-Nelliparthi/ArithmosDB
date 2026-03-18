"""
Correctness validation against FAISS (optional dependency).

Run:
    pip install faiss-cpu
    pytest tests/test_correctness.py -v
"""
import numpy as np
import pytest

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from arithmosdb import FlatIndex
    HAS_GPU = True
except (FileNotFoundError, OSError):
    HAS_GPU = False

pytestmark = pytest.mark.skipif(
    not (HAS_GPU and HAS_FAISS),
    reason="Requires libvectordb.so + faiss-cpu"
)

RNG = np.random.default_rng(0)


@pytest.mark.parametrize("n,d,nq,k", [
    (1000, 64,  50, 10),
    (5000, 128, 100, 5),
])
def test_recall_vs_faiss(n, d, nq, k):
    """
    Recall@k must be >= 95% compared to FAISS exact L2 search.
    Because our kernel returns squared L2, distances differ by sqrt
    but indices must agree.
    """
    vecs    = RNG.random((n, d), dtype=np.float32)
    queries = RNG.random((nq, d), dtype=np.float32)

    # FAISS reference
    fi = faiss.IndexFlatL2(d)
    fi.add(vecs)
    _, faiss_ids = fi.search(queries, k)

    # ArithmosDB
    idx = FlatIndex(d=d)
    idx.add(vecs)
    _, our_ids = idx.search(queries, k)

    # Recall: fraction of FAISS top-k that appear in our top-k
    hits = 0
    total = nq * k
    for q in range(nq):
        hits += len(set(our_ids[q]) & set(faiss_ids[q]))

    recall = hits / total
    assert recall >= 0.95, f"Recall@{k} = {recall:.3f} < 0.95"
