import math
import pickle
import random
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass
class Transition:
    state: np.ndarray
    action: Tuple[int, int, int]
    reward: float
    next_state: np.ndarray
    embedding: np.ndarray
    cluster_id: int
    boundary: bool = False
    td_error: float = 1.0
    sample_count: int = 0


@dataclass
class SORReplayConfig:
    rb_size: int = 1000
    rb_size_global: int = 100000
    global_total_cap: Optional[int] = None
    recent_size: int = 2000
    boundary_size: int = 20000
    max_clusters: int = 32
    prototype_distance: float = 1.0
    prototype_eta: float = 0.05
    boundary_threshold: float = 0.5
    alpha_td: float = 1.0
    beta_under_sample: float = 0.2
    gamma_drift: float = 0.5
    rho_boundary: float = 0.5
    temperature: float = 1.0
    td_clip: float = 10.0
    uniform_mix: float = 0.01
    eps: float = 1e-6
    drift_reward_weight: float = 0.3
    drift_td_weight: float = 0.4
    drift_embedding_weight: float = 0.3
    stats_window: int = 256


class PrototypeManager:
    def __init__(self, max_clusters: int, distance_threshold: float, eta: float):
        self.max_clusters = max_clusters
        self.distance_threshold = distance_threshold
        self.eta = eta
        self.prototypes: Dict[int, np.ndarray] = {}
        self.counts: Dict[int, int] = defaultdict(int)
        self._next_id = 0

    def assign(self, embedding: np.ndarray) -> int:
        embedding = np.asarray(embedding, dtype=np.float32)
        if not self.prototypes:
            return self._create(embedding)

        cluster_id, distance = min(
            ((cid, float(np.linalg.norm(embedding - proto))) for cid, proto in self.prototypes.items()),
            key=lambda item: item[1],
        )
        if distance > self.distance_threshold and len(self.prototypes) < self.max_clusters:
            return self._create(embedding)
        return cluster_id

    def update(self, cluster_id: int, embedding: np.ndarray) -> None:
        embedding = np.asarray(embedding, dtype=np.float32)
        if cluster_id not in self.prototypes:
            self.prototypes[cluster_id] = embedding.copy()
            self.counts[cluster_id] = 1
            self._next_id = max(self._next_id, cluster_id + 1)
            return
        self.prototypes[cluster_id] = (1.0 - self.eta) * self.prototypes[cluster_id] + self.eta * embedding
        self.counts[cluster_id] += 1

    def get(self, cluster_id: int) -> Optional[np.ndarray]:
        return self.prototypes.get(cluster_id)

    def state_dict(self) -> dict:
        return {
            "max_clusters": self.max_clusters,
            "distance_threshold": self.distance_threshold,
            "eta": self.eta,
            "prototypes": self.prototypes,
            "counts": dict(self.counts),
            "next_id": self._next_id,
        }

    def load_state_dict(self, state: dict) -> None:
        self.max_clusters = int(state.get("max_clusters", self.max_clusters))
        self.distance_threshold = float(state.get("distance_threshold", self.distance_threshold))
        self.eta = float(state.get("eta", self.eta))
        self.prototypes = {int(k): np.asarray(v, dtype=np.float32) for k, v in state.get("prototypes", {}).items()}
        self.counts = defaultdict(int, {int(k): int(v) for k, v in state.get("counts", {}).items()})
        self._next_id = int(state.get("next_id", len(self.prototypes)))

    def _create(self, embedding: np.ndarray) -> int:
        cluster_id = self._next_id
        self._next_id += 1
        self.prototypes[cluster_id] = embedding.copy()
        self.counts[cluster_id] = 1
        return cluster_id


class DriftTracker:
    def __init__(self, config: SORReplayConfig):
        self.config = config
        self.reference: Dict[int, dict] = {}
        self.current: Dict[int, dict] = defaultdict(lambda: {"embeddings": deque(maxlen=config.stats_window), "td": deque(maxlen=config.stats_window), "reward": deque(maxlen=config.stats_window)})
        self.drift_scores: Dict[int, float] = defaultdict(float)

    def update(self, cluster_id: int, embedding: np.ndarray, reward: float, td_error: float) -> float:
        cur = self.current[cluster_id]
        cur["embeddings"].append(np.asarray(embedding, dtype=np.float32))
        cur["td"].append(float(abs(td_error)))
        cur["reward"].append(float(reward))

        if cluster_id not in self.reference and len(cur["embeddings"]) >= max(8, self.config.stats_window // 8):
            self.reference[cluster_id] = self._summarize(cur)
            self.drift_scores[cluster_id] = 0.0
            return 0.0

        if cluster_id in self.reference:
            ref = self.reference[cluster_id]
            now = self._summarize(cur)
            emb_distance = float(np.linalg.norm(now["embedding_mean"] - ref["embedding_mean"]))
            td_shift = abs(now["td_mean"] - ref["td_mean"])
            reward_shift = abs(now["reward_mean"] - ref["reward_mean"])
            drift = (
                self.config.drift_embedding_weight * emb_distance
                + self.config.drift_td_weight * td_shift
                + self.config.drift_reward_weight * reward_shift
            )
            self.drift_scores[cluster_id] = float(drift)
        return float(self.drift_scores[cluster_id])

    def get(self, cluster_id: int) -> float:
        return float(self.drift_scores.get(cluster_id, 0.0))

    def state_dict(self) -> dict:
        current = {}
        for cid, stats in self.current.items():
            current[int(cid)] = {
                "embeddings": list(stats["embeddings"]),
                "td": list(stats["td"]),
                "reward": list(stats["reward"]),
            }
        return {
            "reference": self.reference,
            "current": current,
            "drift_scores": dict(self.drift_scores),
        }

    def load_state_dict(self, state: dict) -> None:
        self.reference = {int(k): v for k, v in state.get("reference", {}).items()}
        self.current = defaultdict(lambda: {"embeddings": deque(maxlen=self.config.stats_window), "td": deque(maxlen=self.config.stats_window), "reward": deque(maxlen=self.config.stats_window)})
        for cid, stats in state.get("current", {}).items():
            self.current[int(cid)] = {
                "embeddings": deque([np.asarray(v, dtype=np.float32) for v in stats.get("embeddings", [])], maxlen=self.config.stats_window),
                "td": deque([float(v) for v in stats.get("td", [])], maxlen=self.config.stats_window),
                "reward": deque([float(v) for v in stats.get("reward", [])], maxlen=self.config.stats_window),
            }
        self.drift_scores = defaultdict(float, {int(k): float(v) for k, v in state.get("drift_scores", {}).items()})

    @staticmethod
    def _summarize(stats: dict) -> dict:
        embeddings = np.stack(list(stats["embeddings"]), axis=0)
        return {
            "embedding_mean": embeddings.mean(axis=0),
            "td_mean": float(np.mean(stats["td"])) if stats["td"] else 0.0,
            "reward_mean": float(np.mean(stats["reward"])) if stats["reward"] else 0.0,
        }


class StructuredSORReplayBuffer:
    def __init__(self, config: SORReplayConfig):
        self.config = config
        self.prototype_manager = PrototypeManager(config.max_clusters, config.prototype_distance, config.prototype_eta)
        self.drift_tracker = DriftTracker(config)
        self.cluster_memory: Dict[int, List[Transition]] = defaultdict(list)
        self.boundary_memory: List[Transition] = []
        self.recent_memory: Deque[Transition] = deque(maxlen=config.recent_size)
        self._last_state: Optional[np.ndarray] = None
        self._last_reward: Optional[float] = None
        self._recent_cluster_hits: Deque[int] = deque(maxlen=config.stats_window)
        self._cluster_hit_counts: Dict[int, int] = defaultdict(int)
        self._td_max: float = 1.0
        # Cached candidate list for sample(); invalidated on push/evict.
        self._candidates_cache: Optional[List[Transition]] = None
        self._evict_counter: int = 0

    def _clip_td(self, td_error: float) -> float:
        return float(min(abs(float(td_error)), self.config.td_clip))

    def push(self, state, action, reward, next_state, embedding, td_error: Optional[float] = None) -> Transition:
        state_arr = np.asarray(state, dtype=np.float32)
        next_state_arr = np.asarray(next_state, dtype=np.float32)
        embedding_arr = np.asarray(embedding, dtype=np.float32)
        cluster_id = self.prototype_manager.assign(embedding_arr)
        self.prototype_manager.update(cluster_id, embedding_arr)
        boundary = self._is_boundary(state_arr, float(reward))
        # New samples enter with current max td so they are never the first
        # eviction candidates before being trained on at least once.
        td_value = self._td_max if td_error is None else self._clip_td(td_error)
        self._td_max = max(self._td_max, td_value)

        transition = Transition(
            state=state_arr,
            action=tuple(int(v) for v in action),
            reward=float(reward),
            next_state=next_state_arr,
            embedding=embedding_arr,
            cluster_id=cluster_id,
            boundary=boundary,
            td_error=td_value,
        )
        self.cluster_memory[cluster_id].append(transition)
        self._enforce_capacity(cluster_id)
        self.recent_memory.append(transition)
        if boundary:
            self.boundary_memory.append(transition)
            while len(self.boundary_memory) > self.config.boundary_size:
                self._evict_lowest(self.boundary_memory)
        self.drift_tracker.update(cluster_id, embedding_arr, float(reward), td_value)
        self._last_state = state_arr
        self._last_reward = float(reward)
        self._candidates_cache = None
        return transition

    def push_existing(self, transition: Transition) -> Transition:
        """Append an already-clustered transition without redoing prototype
        assignment, drift updates, or boundary detection. Used by sync()
        broadcast paths where the source buffer has already done that work.
        Only enforces per-cluster / global capacity.
        """
        cluster_id = int(transition.cluster_id)
        self.cluster_memory[cluster_id].append(transition)
        self._enforce_capacity(cluster_id)
        self.recent_memory.append(transition)
        self._td_max = max(self._td_max, float(transition.td_error))
        self._candidates_cache = None
        return transition

    def sample(self, batch_size: int) -> List[Transition]:
        candidates = self._unique_candidates()
        if not candidates:
            return []
        sample_size = min(batch_size, len(candidates))
        weights = np.asarray([self._score(item) for item in candidates], dtype=np.float64)
        weights = self._softmax(weights / max(self.config.temperature, self.config.eps))
        # Mix with uniform so every candidate keeps non-zero probability and
        # np.random.choice(replace=False) can always draw sample_size items.
        mix = float(np.clip(self.config.uniform_mix, 0.0, 1.0))
        weights = (1.0 - mix) * weights + mix / len(candidates)
        weights = weights / weights.sum()
        indices = np.random.choice(len(candidates), size=sample_size, replace=False, p=weights)
        sampled = [candidates[int(idx)] for idx in indices]
        for item in sampled:
            item.sample_count += 1
            self._record_cluster_hit(item.cluster_id)
        return sampled

    def _record_cluster_hit(self, cluster_id: int) -> None:
        if len(self._recent_cluster_hits) == self._recent_cluster_hits.maxlen:
            oldest = self._recent_cluster_hits.popleft()
            self._cluster_hit_counts[oldest] = max(0, self._cluster_hit_counts[oldest] - 1)
        self._recent_cluster_hits.append(cluster_id)
        self._cluster_hit_counts[cluster_id] += 1

    def update_td_errors(self, transitions: Iterable[Transition], td_errors: Iterable[float]) -> None:
        for transition, td_error in zip(transitions, td_errors):
            transition.td_error = self._clip_td(td_error)
            self._td_max = max(self._td_max, transition.td_error)
            self.drift_tracker.update(transition.cluster_id, transition.embedding, transition.reward, transition.td_error)
        # Scores depend on td_error; invalidate cached candidate ordering used
        # by the next sample()'s probability weights (the list itself is fine
        # but the score weights need fresh computation).
        # The list of candidates is unchanged, so we keep _candidates_cache.

    def to_arrays(self, transitions: List[Transition]):
        return (
            np.asarray([t.state for t in transitions], dtype=np.float32),
            np.asarray([t.action for t in transitions], dtype=np.int64),
            np.asarray([t.reward for t in transitions], dtype=np.float32),
            np.asarray([t.next_state for t in transitions], dtype=np.float32),
            np.asarray([t.embedding for t in transitions], dtype=np.float32),
            np.asarray([t.cluster_id for t in transitions], dtype=np.int64),
            np.asarray([t.boundary for t in transitions], dtype=np.float32),
        )

    def state_dict(self) -> dict:
        return {
            "config": self.config,
            "prototype_manager": self.prototype_manager.state_dict(),
            "drift_tracker": self.drift_tracker.state_dict(),
            "cluster_memory": {int(k): list(v) for k, v in self.cluster_memory.items()},
            "boundary_memory": list(self.boundary_memory),
            "recent_memory": list(self.recent_memory),
            "last_state": self._last_state,
            "last_reward": self._last_reward,
            "recent_cluster_hits": list(self._recent_cluster_hits),
        }

    def load_state_dict(self, state: dict) -> None:
        self.prototype_manager.load_state_dict(state.get("prototype_manager", {}))
        self.drift_tracker.load_state_dict(state.get("drift_tracker", {}))
        self.cluster_memory = defaultdict(list)
        for cid, entries in state.get("cluster_memory", {}).items():
            self.cluster_memory[int(cid)] = list(entries)
        self.boundary_memory = list(state.get("boundary_memory", []))
        while len(self.boundary_memory) > self.config.boundary_size:
            self._evict_lowest(self.boundary_memory)
        self.recent_memory = deque(state.get("recent_memory", []), maxlen=self.config.recent_size)
        self._last_state = state.get("last_state")
        self._last_reward = state.get("last_reward")
        self._recent_cluster_hits = deque(state.get("recent_cluster_hits", []), maxlen=self.config.stats_window)
        self._cluster_hit_counts = defaultdict(int)
        for cid in self._recent_cluster_hits:
            self._cluster_hit_counts[int(cid)] += 1
        self._td_max = max([1.0] + [self._clip_td(t.td_error) for buf in self.cluster_memory.values() for t in buf])
        self._candidates_cache = None
        if self.config.global_total_cap is not None:
            self._enforce_capacity(cluster_id=-1)
        else:
            for cid in list(self.cluster_memory.keys()):
                while len(self.cluster_memory[cid]) > self.config.rb_size:
                    self._evict_lowest(self.cluster_memory[cid])

    def save(self, path: str) -> None:
        with open(path, "wb") as handle:
            pickle.dump(self.state_dict(), handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> None:
        with open(path, "rb") as handle:
            self.load_state_dict(pickle.load(handle))

    def __len__(self) -> int:
        return sum(len(buffer) for buffer in self.cluster_memory.values())

    def _is_boundary(self, state: np.ndarray, reward: float) -> bool:
        if self._last_state is None or self._last_reward is None:
            return False
        discontinuity = float(np.linalg.norm(state - self._last_state) + abs(reward - self._last_reward))
        return discontinuity > self.config.boundary_threshold

    def _unique_candidates(self) -> List[Transition]:
        if self._candidates_cache is not None:
            return self._candidates_cache
        seen = set()
        candidates: List[Transition] = []
        for memory in list(self.cluster_memory.values()) + [self.boundary_memory, self.recent_memory]:
            for item in memory:
                item_id = id(item)
                if item_id not in seen:
                    seen.add(item_id)
                    candidates.append(item)
        self._candidates_cache = candidates
        return candidates

    def _score(self, transition: Transition) -> float:
        cluster_id = transition.cluster_id
        recent_hits = self._cluster_hit_counts.get(cluster_id, 0)
        under_sample = 1.0 / (recent_hits + 1.0 + self.config.eps)
        return (
            self.config.alpha_td * self._clip_td(transition.td_error)
            + self.config.beta_under_sample * under_sample
            + self.config.gamma_drift * self.drift_tracker.get(cluster_id)
            + self.config.rho_boundary * float(transition.boundary)
        )

    def _evict_lowest(self, container: List[Transition]) -> None:
        if not container:
            return
        idx_min = min(range(len(container)), key=lambda i: self._score(container[i]))
        evicted = container.pop(idx_min)
        self._candidates_cache = None
        self._evict_counter += 1
        if self._evict_counter % 1000 == 0:
            try:
                tds = [self._clip_td(t.td_error) for t in container[:256]]
                td_min = min(tds) if tds else 0.0
                td_max = max(tds) if tds else 0.0
                td_mean = float(sum(tds) / len(tds)) if tds else 0.0
                # idx_min relative position before pop: 0 == oldest, len-1 == newest
                pos_ratio = float(idx_min) / max(1, len(container))
                # Use plain print to avoid loguru import overhead in hot path.
                print(
                    f"[SOR-Replay] evictions={self._evict_counter} "
                    f"evicted_score={self._score(evicted):.4f} td={evicted.td_error:.4f} "
                    f"position_ratio={pos_ratio:.2f} (0=oldest 1=newest) "
                    f"remain_td_min/mean/max={td_min:.4f}/{td_mean:.4f}/{td_max:.4f}"
                )
            except Exception:
                pass

    def _enforce_capacity(self, cluster_id: int) -> None:
        if self.config.global_total_cap is not None:
            total = sum(len(b) for b in self.cluster_memory.values())
            while total > self.config.global_total_cap:
                # Approximate score-based eviction: scan only the largest
                # cluster instead of all transitions. Evicting from the
                # largest cluster also keeps cluster sizes balanced, which is
                # what protects old regimes from being flushed out.
                target_cid = max(self.cluster_memory, key=lambda cid: len(self.cluster_memory[cid]))
                self._evict_lowest(self.cluster_memory[target_cid])
                total -= 1
        else:
            buf = self.cluster_memory[cluster_id]
            while len(buf) > self.config.rb_size:
                self._evict_lowest(buf)

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        values = values - np.max(values)
        exp_values = np.exp(values)
        total = float(exp_values.sum())
        if not math.isfinite(total) or total <= 0:
            return np.ones_like(exp_values) / len(exp_values)
        return exp_values / total
