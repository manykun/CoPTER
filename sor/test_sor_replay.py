import os
import tempfile

import numpy as np

from sor_replay import SORReplayConfig, StructuredSORReplayBuffer


def test_sor_replay_push_sample_and_persist():
    config = SORReplayConfig(
        rb_size=16,
        recent_size=8,
        boundary_size=8,
        max_clusters=4,
        prototype_distance=0.25,
        boundary_threshold=0.1,
        stats_window=8,
    )
    replay = StructuredSORReplayBuffer(config)
    for idx in range(12):
        state = np.full(18, idx * 0.1, dtype=np.float32)
        next_state = state + 0.01
        embedding = np.full(32, 0.0 if idx < 6 else 2.0, dtype=np.float32)
        replay.push(state, (idx % 6, idx % 4, idx % 10), float(idx) / 10.0, next_state, embedding)

    assert len(replay) == 12
    assert len(replay.prototype_manager.prototypes) >= 2
    assert len(replay.boundary_memory) > 0

    batch = replay.sample(4)
    assert len(batch) == 4
    arrays = replay.to_arrays(batch)
    assert arrays[0].shape == (4, 18)
    assert arrays[1].shape == (4, 3)
    replay.update_td_errors(batch, [0.1, 0.2, 0.3, 0.4])

    with tempfile.NamedTemporaryFile(delete=False) as handle:
        path = handle.name
    try:
        replay.save(path)
        restored = StructuredSORReplayBuffer(config)
        restored.load(path)
        assert len(restored) == len(replay)
        assert len(restored.prototype_manager.prototypes) == len(replay.prototype_manager.prototypes)
    finally:
        os.unlink(path)


def test_extreme_td_error_sampling_no_crash():
    """Sampling must never raise even when td errors are extreme (1e4+)."""
    config = SORReplayConfig(rb_size=64, recent_size=16, boundary_size=16, max_clusters=4, stats_window=16)
    replay = StructuredSORReplayBuffer(config)
    rng = np.random.default_rng(0)
    for idx in range(128):
        state = rng.normal(size=18).astype(np.float32)
        embedding = rng.normal(size=32).astype(np.float32)
        replay.push(state, (0, 0, 0), float(idx), state + 0.01, embedding, td_error=1e6 if idx % 2 == 0 else 1e-9)
    for _ in range(10):
        batch = replay.sample(32)
        assert len(batch) == 32
    # all stored td errors must be clipped
    for buf in replay.cluster_memory.values():
        for t in buf:
            assert t.td_error <= config.td_clip + 1e-9


def test_new_samples_not_evicted_first():
    """A fresh sample (td unknown) must not be the immediate eviction victim."""
    config = SORReplayConfig(rb_size=8, recent_size=4, boundary_size=4, max_clusters=1,
                             prototype_distance=1e9, stats_window=8)
    replay = StructuredSORReplayBuffer(config)
    state = np.zeros(18, dtype=np.float32)
    embedding = np.zeros(32, dtype=np.float32)
    # fill with high-td old samples
    for idx in range(8):
        replay.push(state, (0, 0, 0), 0.0, state, embedding, td_error=5.0)
    # fresh sample without td -> inherits current max td, must survive the eviction
    fresh = replay.push(state, (1, 1, 1), 0.0, state, embedding)
    cluster = list(replay.cluster_memory.values())[0]
    assert any(t is fresh for t in cluster), "fresh sample was evicted immediately"


def test_eviction_is_score_based():
    """The lowest-score transition must be evicted, not the oldest (FIFO)."""
    config = SORReplayConfig(rb_size=4, recent_size=2, boundary_size=2, max_clusters=1,
                             prototype_distance=1e9, beta_under_sample=0.0, gamma_drift=0.0,
                             rho_boundary=0.0, stats_window=4)
    replay = StructuredSORReplayBuffer(config)
    state = np.zeros(18, dtype=np.float32)
    embedding = np.zeros(32, dtype=np.float32)
    # oldest has the HIGHEST td -> FIFO would evict it, score-based must keep it
    tds = [9.0, 0.1, 0.2, 0.3]
    kept = []
    for idx, td in enumerate(tds):
        kept.append(replay.push(state, (idx % 6, 0, 0), 0.0, state, embedding, td_error=td))
    overflow = replay.push(state, (5, 0, 0), 0.0, state, embedding, td_error=0.05)
    cluster = list(replay.cluster_memory.values())[0]
    assert len(cluster) == 4
    assert any(t is kept[0] for t in cluster), "high-td oldest sample was evicted (FIFO behavior)"


def test_eviction_order_strictly_by_score():
    """Push n_keep+m samples with monotonically increasing td -> the lowest-td
    samples must be the ones evicted, regardless of insertion order. This is
    stronger than test_eviction_is_score_based which only checks the oldest."""
    config = SORReplayConfig(rb_size=4, recent_size=2, boundary_size=2, max_clusters=1,
                             prototype_distance=1e9, beta_under_sample=0.0, gamma_drift=0.0,
                             rho_boundary=0.0, alpha_td=1.0, stats_window=4)
    replay = StructuredSORReplayBuffer(config)
    state = np.zeros(18, dtype=np.float32)
    embedding = np.zeros(32, dtype=np.float32)
    # Push 8 samples with td_error = idx (so newest has highest td).
    pushed = []
    for idx in range(8):
        pushed.append(replay.push(state, (idx % 6, 0, 0), 0.0, state, embedding, td_error=float(idx + 1)))
    cluster = list(replay.cluster_memory.values())[0]
    assert len(cluster) == 4, "rb_size=4 cap should hold"
    remaining_tds = sorted([t.td_error for t in cluster])
    # The 4 lowest-td samples must have been evicted; the 4 highest remain.
    assert remaining_tds == [5.0, 6.0, 7.0, 8.0], f"got {remaining_tds}"


def test_push_existing_copies_and_reassigns_locally():
    """Imported transitions must have independent local cluster semantics."""
    config = SORReplayConfig(rb_size=8, recent_size=4, boundary_size=4, max_clusters=4,
                             prototype_distance=0.1, stats_window=4)
    src = StructuredSORReplayBuffer(config)
    state = np.zeros(18, dtype=np.float32)
    emb_a = np.zeros(32, dtype=np.float32)
    emb_b = np.ones(32, dtype=np.float32) * 5.0
    t1 = src.push(state, (0, 0, 0), 0.0, state, emb_a, td_error=1.0)
    t2 = src.push(state, (1, 0, 0), 0.0, state, emb_b, td_error=2.0)
    assert t1.cluster_id != t2.cluster_id

    dst = StructuredSORReplayBuffer(config)
    copied1 = dst.push_existing(t1)
    copied2 = dst.push_existing(t2)
    assert len(dst.prototype_manager.prototypes) == 2
    assert copied1 is not t1
    assert copied2 is not t2
    assert copied1.state is not t1.state
    copied1.td_error = 9.0
    copied1.sample_count += 1
    assert t1.td_error == 1.0
    assert t1.sample_count == 0
    assert dst.prototype_manager.get(copied1.cluster_id) is not None
    assert dst.prototype_manager.get(copied2.cluster_id) is not None


def test_boundary_detection_is_stream_local():
    """Interleaved ports must not create artificial cross-port boundaries."""
    config = SORReplayConfig(
        rb_size=16,
        recent_size=8,
        boundary_size=8,
        max_clusters=2,
        prototype_distance=1e9,
        boundary_threshold=1.0,
        stats_window=8,
    )
    replay = StructuredSORReplayBuffer(config)
    zero = np.zeros(18, dtype=np.float32)
    far = np.full(18, 100.0, dtype=np.float32)
    embedding = np.zeros(32, dtype=np.float32)
    first_port0 = replay.push(
        zero, (0, 0, 0), 0.0, zero, embedding, stream_id=0
    )
    first_port1 = replay.push(
        far, (0, 0, 0), 0.0, far, embedding, stream_id=1
    )
    second_port0 = replay.push(
        zero + 0.01, (0, 0, 0), 0.0, zero, embedding, stream_id=0
    )
    assert not first_port0.boundary
    assert not first_port1.boundary
    assert not second_port0.boundary

    changed_port0 = replay.push(
        far, (0, 0, 0), 1.0, far, embedding, stream_id=0
    )
    assert changed_port0.boundary


def test_candidate_cache_invalidates_on_push():
    config = SORReplayConfig(rb_size=8, recent_size=4, boundary_size=4, max_clusters=2,
                             prototype_distance=1e9, stats_window=4)
    replay = StructuredSORReplayBuffer(config)
    state = np.zeros(18, dtype=np.float32)
    emb = np.zeros(32, dtype=np.float32)
    replay.push(state, (0, 0, 0), 0.0, state, emb, td_error=1.0)
    cands_before = replay._unique_candidates()
    n_before = len(cands_before)
    replay.push(state, (1, 0, 0), 0.0, state, emb, td_error=1.0)
    cands_after = replay._unique_candidates()
    assert len(cands_after) == n_before + 1


def test_sampling_probability_valid():
    """Mixed-with-uniform probabilities are strictly positive and sum to 1."""
    config = SORReplayConfig(rb_size=32, recent_size=8, boundary_size=8, max_clusters=2, stats_window=8)
    replay = StructuredSORReplayBuffer(config)
    rng = np.random.default_rng(1)
    for idx in range(64):
        state = rng.normal(size=18).astype(np.float32)
        embedding = rng.normal(size=32).astype(np.float32)
        replay.push(state, (0, 0, 0), 0.0, state, embedding, td_error=10.0 if idx == 0 else 0.0)
    candidates = replay._unique_candidates()
    weights = np.asarray([replay._score(t) for t in candidates], dtype=np.float64)
    probs = replay._softmax(weights / config.temperature)
    mix = config.uniform_mix
    probs = (1.0 - mix) * probs + mix / len(candidates)
    probs = probs / probs.sum()
    assert np.all(probs > 0)
    assert abs(probs.sum() - 1.0) < 1e-9


if __name__ == "__main__":
    test_sor_replay_push_sample_and_persist()
    test_extreme_td_error_sampling_no_crash()
    test_new_samples_not_evicted_first()
    test_eviction_is_score_based()
    test_eviction_order_strictly_by_score()
    test_push_existing_copies_and_reassigns_locally()
    test_boundary_detection_is_stream_local()
    test_candidate_cache_invalidates_on_push()
    test_sampling_probability_valid()
    print("SOR replay tests passed (9/9)")
