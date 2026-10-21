"""Hilbert-Partitioned B-Tree of Subgraphs with Temporal Merging (Full Implementation).

Key Architectural Components:
  1. HilbertSpace: Multi-attribute quantization & recursive hypercube interval decomposition.
  2. GraphSparsityTuner: Sampled Average Pairwise Distance (APD) to dynamically tune local graph degree M_i.
  3. LeafSubgraph: Dedicated HNSW graph over a disjoint Hilbert key interval:
       - Mode A (Fully Enclosed): Pure vanilla unconstrained HNSW (0% filter overhead).
       - Mode B (Boundary, Small Count <= scan_threshold): Contiguous SIMD flat scan.
       - Mode C (Boundary, Large Count > scan_threshold): Hilbert-Seeded 2-hop HNSW traversal.
  4. HilbertBTreeRouter: B-Tree router maintaining disjoint leaf partitions [K_i, K_{i+1}) with Zone Maps.
  5. TemporalBuffer: L0 MemTable for stream appends; flushes & triggers localized median leaf splits
     (rebuilding ONLY the split leaves with ZERO cross-partition graph rewiring).
  6. HilbertHNSWEngine: Unified query optimizer and execution engine.
"""

import concurrent.futures
import heapq
import json
import os
import struct
import tempfile
import threading
import time
import numpy as np
import faiss
try:
    from hilbertcurve.hilbertcurve import HilbertCurve
except ImportError:
    class HilbertCurve:
        """Pure-Python fallback implementation of Skilling's Hilbert curve algorithm."""
        def __init__(self, p, n, n_procs=0):
            self.p = p
            self.n = n

        def distance_from_point(self, point):
            point = [int(el) for el in point]
            m = 1 << (self.p - 1)
            q = m
            while q > 1:
                p = q - 1
                for i in range(self.n):
                    if point[i] & q:
                        point[0] ^= p
                    else:
                        t = (point[0] ^ point[i]) & p
                        point[0] ^= t
                        point[i] ^= t
                q >>= 1
            for i in range(1, self.n):
                point[i] ^= point[i - 1]
            t = 0
            q = m
            while q > 1:
                if point[self.n - 1] & q:
                    t ^= q - 1
                q >>= 1
            for i in range(self.n):
                point[i] ^= t
            x_bit_str = [bin(point[i])[2:].zfill(self.p) for i in range(self.n)]
            return int("".join([y[i] for i in range(self.p) for y in x_bit_str]), 2)

        def distances_from_points(self, points):
            return [self.distance_from_point(pt) for pt in points]


class SharedSearchState:
    """Thread-safe shared candidate tracker with dual branch-and-bound thresholds:

      1. worst_dist_k: Distance of the k-th best candidate globally. Used for exact leaf-level
         centroid bounding hypersphere pruning and SIMD flat scan thresholding.
      2. worst_dist_ef: Distance of the ef-th best candidate globally. Used for HNSW beam search
         waypoint exploration, preserving full beam width (ef) while pruning unpromising paths.
    """

    def __init__(self, k, ef=64):
        self.k = k
        self.ef = max(k, ef)
        self.lock = threading.Lock()
        self.heap = []  # max-heap storing up to self.ef entries: (-dist, gid)
        self.seen_gids = set()
        self.worst_dist_k = float("inf")
        self.worst_dist_ef = float("inf")
        self.early_exits = 0

    def get_bound(self):
        """Returns the k-th best distance bound (for leaf centroid and scan pruning)."""
        return self.worst_dist_k

    def get_ef_bound(self):
        """Returns the ef-th best distance bound (for HNSW beam search traversal)."""
        return self.worst_dist_ef

    def is_full(self):
        """Returns True if at least k candidates have been found."""
        return len(self.heap) >= self.k

    def is_ef_full(self):
        """Returns True if the beam capacity ef has been filled."""
        return len(self.heap) >= self.ef

    def _recalc_bounds_locked(self):
        if len(self.heap) >= self.ef:
            self.worst_dist_ef = -self.heap[0][0]
        else:
            self.worst_dist_ef = float("inf")

        if len(self.heap) >= self.k:
            dists = sorted(-item[0] for item in self.heap)
            self.worst_dist_k = dists[self.k - 1]
        else:
            self.worst_dist_k = float("inf")

    def update(self, gid, dist):
        """Inserts candidate (gid, dist) if it beats worst_dist_ef."""
        dist = float(dist)
        gid = int(gid)
        if dist >= self.worst_dist_ef and len(self.heap) >= self.ef:
            return False
        with self.lock:
            if dist >= self.worst_dist_ef and len(self.heap) >= self.ef:
                return False
            if gid in self.seen_gids:
                return False
            self.seen_gids.add(gid)
            if len(self.heap) < self.ef:
                heapq.heappush(self.heap, (-dist, gid))
            else:
                heapq.heappushpop(self.heap, (-dist, gid))
            self._recalc_bounds_locked()
            return True

    def update_batch(self, gids, dists):
        """Inserts a batch of candidates under a single lock acquisition."""
        if len(gids) == 0:
            return
        with self.lock:
            modified = False
            for gid, d in zip(gids, dists):
                d = float(d)
                gid = int(gid)
                if gid in self.seen_gids:
                    continue
                if len(self.heap) < self.ef:
                    self.seen_gids.add(gid)
                    heapq.heappush(self.heap, (-d, gid))
                    modified = True
                elif d < self.worst_dist_ef:
                    self.seen_gids.add(gid)
                    heapq.heappushpop(self.heap, (-d, gid))
                    modified = True
            if modified:
                self._recalc_bounds_locked()

    def record_early_exit(self):
        with self.lock:
            self.early_exits += 1

    def get_results(self):
        """Returns sorted (global_ids, dists) of the top-k candidates."""
        with self.lock:
            res = sorted((-d, gid) for d, gid in self.heap)[:self.k]
        out_ids = np.array([gid for _, gid in res], dtype=np.int64)
        out_dists = np.array([d for d, _ in res], dtype=np.float32)
        return out_ids, out_dists


class MapReduceCollation:
    """Multi-threaded hierarchical MapReduce tree collation for top-k candidates across subgraphs."""

    @staticmethod
    def merge_pair(cand_a, cand_b, k):
        """Merges two (gids, dists) candidate sets and retains the top-k in O(len(a) + len(b))."""
        ids_a, dists_a = cand_a
        ids_b, dists_b = cand_b
        if len(ids_a) == 0:
            return ids_b[:k], dists_b[:k]
        if len(ids_b) == 0:
            return ids_a[:k], dists_a[:k]

        comb_ids = np.concatenate([ids_a, ids_b])
        comb_dists = np.concatenate([dists_a, dists_b])

        # If overlapping global IDs exist across boundaries, deduplicate
        if len(comb_ids) > len(np.unique(comb_ids)):
            _, uniq_idx = np.unique(comb_ids, return_index=True)
            comb_ids = comb_ids[uniq_idx]
            comb_dists = comb_dists[uniq_idx]

        if len(comb_ids) <= k:
            order = np.argsort(comb_dists)
            return comb_ids[order], comb_dists[order]

        top_k_idx = np.argpartition(comb_dists, k)[:k]
        sorted_sub = top_k_idx[np.argsort(comb_dists[top_k_idx])]
        return comb_ids[sorted_sub], comb_dists[sorted_sub]

    @classmethod
    def tree_reduce(cls, candidate_lists, k, executor=None, parallel=True):
        """Hierarchically merges P candidate lists into global top-k using a binary reduction tree."""
        valid_lists = [(c[0], c[1]) for c in candidate_lists if len(c[0]) > 0]
        if not valid_lists:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        if len(valid_lists) == 1:
            ids, dists = valid_lists[0]
            order = np.argsort(dists)[:k]
            return ids[order], dists[order]

        current = valid_lists
        while len(current) > 1:
            pairs = []
            pass_through = []
            for i in range(0, len(current) - 1, 2):
                pairs.append((current[i], current[i + 1]))
            if len(current) % 2 == 1:
                pass_through.append(current[-1])

            if parallel and executor is not None and len(pairs) > 1:
                futures = [executor.submit(cls.merge_pair, p[0], p[1], k) for p in pairs]
                merged = [f.result() for f in futures]
            else:
                merged = [cls.merge_pair(p[0], p[1], k) for p in pairs]

            current = merged + pass_through

        return current[0]

    @classmethod
    def flat_reduce(cls, candidate_lists, k):
        """Baseline single-threaded flat collation via concatenation + np.argpartition."""
        valid_lists = [c for c in candidate_lists if len(c[0]) > 0]
        if not valid_lists:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        all_ids = np.concatenate([c[0] for c in valid_lists])
        all_dists = np.concatenate([c[1] for c in valid_lists])
        if len(all_ids) > len(np.unique(all_ids)):
            _, uniq_idx = np.unique(all_ids, return_index=True)
            all_ids = all_ids[uniq_idx]
            all_dists = all_dists[uniq_idx]
        if len(all_ids) <= k:
            order = np.argsort(all_dists)
            return all_ids[order], all_dists[order]
        top_k_idx = np.argpartition(all_dists, k)[:k]
        sorted_sub = top_k_idx[np.argsort(all_dists[top_k_idx])]
        return all_ids[sorted_sub], all_dists[sorted_sub]


def interleave_subspace_keys(keys_list, bits=16):
    """Bit-interleaves S subspace Hilbert keys into a single uint64 master clustering key."""
    S = len(keys_list)
    if S == 1:
        return np.asarray(keys_list[0], dtype=np.uint64)
    n = len(keys_list[0])
    res = np.zeros(n, dtype=np.uint64)
    for bit_idx in range(bits):
        for s_idx in range(S):
            b = (np.uint64(keys_list[s_idx]) >> np.uint64(bit_idx)) & np.uint64(1)
            target_shift = np.uint64(bit_idx * S + (S - 1 - s_idx))
            if target_shift < 64:
                res |= (b << target_shift)
    return res


def deinterleave_subspace_keys(master_keys, S, bits=16):
    """Recovers the S subspace Hilbert keys from the interleaved master key."""
    master_keys = np.asarray(master_keys, dtype=np.uint64)
    out = [np.zeros_like(master_keys) for _ in range(S)]
    for bit_idx in range(bits):
        for s_idx in range(S):
            src_shift = np.uint64(bit_idx * S + (S - 1 - s_idx))
            if src_shift < 64:
                b = (master_keys >> src_shift) & np.uint64(1)
                out[s_idx] |= (b << np.uint64(bit_idx))
    return out


class LeafColumnarStorage:
    """Parquet-style columnar storage engine for secondary attributes within a leaf subgraph.

    Architectural Features:
      1. Columnar Memory-Mapped Layout: Secondary attributes [4..m-1] are stored contiguously by
         column chunk rather than row-by-row.
      2. Column Projection: Queries only touch and read the memory for active filtered columns,
         leaving unqueried secondary attributes unread in cache/memory.
      3. Row Group Chunking & Zone Maps: Divides rows into groups (default 2048 rows), storing
         min/max statistics per column per row group to skip non-matching blocks in O(1).
      4. Vectorized SIMD Filtering: Surviving row groups are evaluated via NumPy SIMD boolean masks.
      5. File Backing: Writes dedicated columnar files ('leaf_{leaf_id:05d}_secondary.pcol')
         with magic bytes b'PAR1_COL', header metadata, column chunks, and footer.
    """
    MAGIC = b"PAR1_COL"

    def __init__(self, leaf_id, secondary_attrs, col_names=None, row_group_size=2048, storage_dir=None):
        self.leaf_id = leaf_id
        self.row_group_size = row_group_size
        self.secondary_attrs = np.ascontiguousarray(secondary_attrs, dtype=np.float32)
        if self.secondary_attrs.ndim == 1:
            self.secondary_attrs = self.secondary_attrs[:, None]
        self.num_rows, self.num_cols = self.secondary_attrs.shape
        self.col_names = col_names or [f"sec_attr_{i}" for i in range(self.num_cols)]

        if storage_dir is None:
            self.storage_dir = os.path.join(tempfile.gettempdir(), "cse532_leaf_columnar")
        else:
            self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.file_path = os.path.join(self.storage_dir, f"leaf_{self.leaf_id:05d}_secondary.pcol")

        self.num_row_groups = int(np.ceil(self.num_rows / self.row_group_size)) if self.num_rows > 0 else 0
        self.row_group_stats = []
        self._fp = None
        self._mmap = None
        self._write_columnar_file()
        self._open_mmap()

    def _write_columnar_file(self):
        """Serializes secondary attributes into a binary Parquet-style columnar file."""
        with open(self.file_path, "wb") as f:
            f.write(self.MAGIC)
            header_dict = {
                "leaf_id": self.leaf_id,
                "num_rows": self.num_rows,
                "num_cols": self.num_cols,
                "num_row_groups": self.num_row_groups,
                "row_group_size": self.row_group_size,
                "col_names": self.col_names,
            }
            hdr_bytes = json.dumps(header_dict).encode("utf-8")
            f.write(struct.pack("<I", len(hdr_bytes)))
            f.write(hdr_bytes)

            stats = []
            for g in range(self.num_row_groups):
                r_start = g * self.row_group_size
                r_end = min(self.num_rows, (g + 1) * self.row_group_size)
                chunk_slice = self.secondary_attrs[r_start:r_end]

                group_stats = []
                for c in range(self.num_cols):
                    col_data = np.ascontiguousarray(chunk_slice[:, c], dtype=np.float32)
                    c_min = float(col_data.min()) if len(col_data) > 0 else 0.0
                    c_max = float(col_data.max()) if len(col_data) > 0 else 0.0
                    offset = f.tell()
                    raw_bytes = col_data.tobytes()
                    f.write(raw_bytes)
                    group_stats.append({
                        "min": c_min,
                        "max": c_max,
                        "offset": offset,
                        "nbytes": len(raw_bytes),
                        "nrows": len(col_data),
                    })
                stats.append(group_stats)

            # Footer: metadata and row-group Zone Maps
            meta_bytes = json.dumps(stats).encode("utf-8")
            f.write(meta_bytes)
            f.write(struct.pack("<I", len(meta_bytes)))
            f.write(self.MAGIC)

        self.row_group_stats = stats
        self.file_size_bytes = os.path.getsize(self.file_path)

    def _open_mmap(self):
        if self.file_size_bytes > 0:
            self._fp = open(self.file_path, "rb")
            self._mmap = np.memmap(self._fp, mode="r")

    def filter_secondary(self, lo_sec, hi_sec, active_cols=None):
        """Applies column projection and row-group Zone Map pruning on secondary attributes.

        Returns:
            np.ndarray: Indices of rows inside this leaf matching the secondary attribute predicate.
        """
        if self.num_rows == 0 or self.num_cols == 0:
            return np.arange(self.num_rows, dtype=np.int64)

        lo_sec = np.asarray(lo_sec, dtype=np.float32)
        hi_sec = np.asarray(hi_sec, dtype=np.float32)

        if active_cols is None:
            active_cols = [c for c in range(min(len(lo_sec), self.num_cols))
                           if lo_sec[c] > -1e30 or hi_sec[c] < 1e30]

        if not active_cols:
            return np.arange(self.num_rows, dtype=np.int64)

        matching_chunks = []
        for g, group_stat in enumerate(self.row_group_stats):
            r_start = g * self.row_group_size

            # 1. Row-group Zone Map Pruning: skip chunk if any active column is disjoint
            skip_chunk = False
            for c in active_cols:
                if group_stat[c]["min"] > hi_sec[c] or group_stat[c]["max"] < lo_sec[c]:
                    skip_chunk = True
                    break
            if skip_chunk:
                continue

            # 2. Column Projection & SIMD vectorized evaluation on surviving chunk
            nrows = group_stat[active_cols[0]]["nrows"]
            chunk_mask = np.ones(nrows, dtype=bool)

            for c in active_cols:
                offset = group_stat[c]["offset"]
                nbytes = group_stat[c]["nbytes"]
                if self._mmap is not None:
                    col_data = np.frombuffer(self._mmap[offset:offset + nbytes], dtype=np.float32)
                else:
                    self._fp.seek(offset)
                    col_data = np.frombuffer(self._fp.read(nbytes), dtype=np.float32)

                chunk_mask &= (col_data >= lo_sec[c]) & (col_data <= hi_sec[c])
                if not np.any(chunk_mask):
                    break

            if np.any(chunk_mask):
                local_match = np.where(chunk_mask)[0] + r_start
                matching_chunks.append(local_match)

        if matching_chunks:
            return np.concatenate(matching_chunks)
        return np.empty(0, dtype=np.int64)

    def close(self):
        if hasattr(self, "_mmap") and self._mmap is not None:
            del self._mmap
            self._mmap = None
        if hasattr(self, "_fp") and self._fp is not None:
            self._fp.close()
            self._fp = None

    def __del__(self):
        self.close()


class HilbertSpace:
    """Handles multi-attribute quantization via a 4-Dimensional Hilbert Curve
    over primary clustering attributes [0..3], delegating secondary attributes [4..m-1]
    to per-leaf Parquet-style columnar storage files.
    """

    def __init__(self, amin, amax, p=14, m=4, primary_dim=4, **kwargs):
        self.m = m
        self.primary_dim = min(primary_dim, m)
        # Precision p for 4D Hilbert curve (4 * 15 = 60 <= 63 bits in uint64)
        if self.primary_dim > 0:
            self.p = min(15, 60 // self.primary_dim)
        else:
            self.p = p
        self.amin = np.asarray(amin, dtype=np.float64)
        self.amax = np.asarray(amax, dtype=np.float64)
        self.span = np.where(self.amax > self.amin, self.amax - self.amin, 1.0)

        # True 4-Dimensional Hilbert Curve over primary dimensions
        self.curve = HilbertCurve(self.p, self.primary_dim)
        total_bits = self.primary_dim * self.p
        self.max_key = np.uint64((1 << total_bits) - 1)

    def quantize(self, A):
        """Maps continuous attributes to the p-bit integer grid [0, 2^p - 1]."""
        A = np.asarray(A, dtype=np.float64)
        q = np.floor((A - self.amin) / self.span * (2**self.p - 1) + 0.5)
        return np.clip(q, 0, 2**self.p - 1).astype(np.int64)

    def quantize_primary(self, A):
        """Quantizes the primary clustering attributes [0 .. primary_dim-1]."""
        A = np.asarray(A, dtype=np.float64)
        if A.ndim == 1:
            A = A[None, :]
        q = np.floor((A[:, :self.primary_dim] - self.amin[:self.primary_dim]) /
                     self.span[:self.primary_dim] * (2**self.p - 1) + 0.5)
        return np.clip(q, 0, 2**self.p - 1).astype(np.int64)

    def points_to_keys(self, Aq):
        """Converts quantized points to 1D uint64 keys via the 4-Dimensional Hilbert Curve."""
        Aq = np.asarray(Aq, dtype=np.int64)
        if len(Aq) == 0:
            return np.empty(0, dtype=np.uint64)
        primary_pts = Aq[:, :self.primary_dim]
        keys = self.curve.distances_from_points(primary_pts.tolist())
        return np.array(keys, dtype=np.uint64)

    def decompose(self, lo_raw, hi_raw, max_depth=6):
        """Decomposes a 4D bounding box [lo_raw[:4], hi_raw[:4]] into disjoint
        merged 1D Hilbert key intervals [start, end).
        """
        lo = self.quantize_primary(lo_raw)[0]
        hi = self.quantize_primary(hi_raw)[0]
        d = self.primary_dim
        max_depth = min(self.p, max_depth)
        out = []

        def rec(level, corner):
            side = 1 << (self.p - level)
            cmax = corner + side - 1
            if np.any(cmax < lo) or np.any(corner > hi):
                return
            if (np.all(corner >= lo) and np.all(cmax <= hi)) or level == max_depth:
                size = 1 << (d * (self.p - level))
                ck = self.points_to_keys(corner[None])[0]
                start = (ck // size) * size
                out.append((start, start + size))
                return
            half = side >> 1
            for bits in range(1 << d):
                off = np.array([(bits >> j) & 1 for j in range(d)]) * half
                rec(level + 1, corner + off)

        rec(0, np.zeros(d, dtype=np.int64))
        if not out:
            return []
        out.sort()
        merged = []
        for s, e in out:
            if merged and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e])
        return merged


class GraphSparsityTuner:
    """Adapts HNSW graph degree M_i based on local vector dispersion (Sampled APD)."""

    def __init__(self, global_apd, M_min=12, M_max=32, M_base=20):
        self.global_apd = max(global_apd, 1e-6)
        self.M_min = M_min
        self.M_max = M_max
        self.M_base = M_base

    @staticmethod
    def compute_apd(X, sample_size=64, rng=None):
        """Computes Sampled Average Pairwise Distance (APD) in O(sample_size^2)."""
        rng = rng or np.random.default_rng(0)
        n = len(X)
        if n <= 1:
            return 1.0
        k = min(n, sample_size)
        idx = rng.choice(n, size=k, replace=False)
        sample = X[idx]
        diff = sample[:, None, :] - sample[None, :, :]
        dist_sq = np.sum(diff * diff, axis=-1)
        distances = np.sqrt(np.maximum(dist_sq, 0.0))
        triu_idx = np.triu_indices(k, k=1)
        return float(np.mean(distances[triu_idx]))

    def compute_degree(self, X_leaf, rng=None):
        """Computes adaptive M_i for a leaf partition."""
        if len(X_leaf) < 100:
            return self.M_base
        local_apd = self.compute_apd(X_leaf, rng=rng)
        delta = (self.global_apd - local_apd) / self.global_apd
        scale = (self.M_max - self.M_min) / 2.0
        deg = int(np.round(self.M_base - delta * scale))
        return int(np.clip(deg, self.M_min, self.M_max))


class LeafSubgraph:
    """A dedicated, independent HNSW subgraph over a disjoint Hilbert key range [k_min, k_max)."""

    def __init__(self, leaf_id, k_min, k_max, X, A, keys, global_ids, M=20, ef_construction=100, primary_dim=4, storage_dir=None):
        self.leaf_id = leaf_id
        self.k_min = np.uint64(k_min)
        self.k_max = np.uint64(k_max)
        self.X = np.ascontiguousarray(X, dtype=np.float32)
        self.A = np.ascontiguousarray(A, dtype=np.float32)
        self.keys = np.asarray(keys, dtype=np.uint64)
        self.global_ids = np.asarray(global_ids, dtype=np.int64)
        self.n, self.d = self.X.shape
        self.m = self.A.shape[1] if self.A.ndim > 1 else 1
        self.primary_dim = min(primary_dim, self.m)
        self.M = M
        self.ef_construction = ef_construction

        # 1. Parquet-style Columnar Storage File for secondary attributes [primary_dim .. m-1]
        if self.m > self.primary_dim and self.n > 0:
            sec_attrs = self.A[:, self.primary_dim:]
            col_names = [f"sec_attr_{i}" for i in range(self.primary_dim, self.m)]
            self.columnar_store = LeafColumnarStorage(
                leaf_id=self.leaf_id,
                secondary_attrs=sec_attrs,
                col_names=col_names,
                storage_dir=storage_dir,
            )
        else:
            self.columnar_store = None

        # Sort internal elements by Hilbert key for fast local binary searches
        self.order = np.argsort(self.keys, kind="stable")
        self.skeys = self.keys[self.order]

        # Bounding box in raw attribute space for fast zone pruning & bounding hypersphere for branch-and-bound
        if self.n > 0:
            self.attr_min = self.A.min(axis=0)
            self.attr_max = self.A.max(axis=0)
            self.centroid = np.mean(self.X, axis=0).astype(np.float32)
            diff = self.X - self.centroid
            self.radius = float(np.sqrt(np.max(np.sum(diff * diff, axis=-1))))
        else:
            self.attr_min = np.zeros(self.A.shape[1] if self.A.ndim > 1 else 1)
            self.attr_max = np.zeros_like(self.attr_min)
            self.centroid = np.zeros(self.d, dtype=np.float32)
            self.radius = 0.0

        # Build dedicated local HNSW graph
        self._build_hnsw()

    def _build_hnsw(self):
        if self.n == 0:
            self.index = None
            return
        self.index = faiss.IndexHNSWFlat(self.d, self.M)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.add(self.X)
        h = self.index.hnsw
        self.nbrs = faiss.vector_to_array(h.neighbors)
        self.offs = faiss.vector_to_array(h.offsets)
        self.nb0 = h.nb_neighbors(0)

    def close(self):
        """Closes any underlying columnar storage file handles."""
        if hasattr(self, "columnar_store") and self.columnar_store is not None:
            self.columnar_store.close()

    def __del__(self):
        self.close()

    def get_memory_bytes(self):
        """Returns detailed memory breakdown in bytes for this leaf partition."""
        faiss_bytes = len(faiss.serialize_index(self.index)) if self.index is not None else 0
        raw_vec_bytes = self.X.nbytes
        attr_bytes = self.A.nbytes
        key_bytes = self.keys.nbytes + self.skeys.nbytes + self.order.nbytes
        id_bytes = self.global_ids.nbytes
        col_file_bytes = self.columnar_store.file_size_bytes if self.columnar_store is not None else 0
        meta_bytes = self.attr_min.nbytes + self.attr_max.nbytes + self.centroid.nbytes + 8
        pure_graph_bytes = max(0, faiss_bytes - raw_vec_bytes)
        return {
            "faiss_total": faiss_bytes,
            "pure_graph_bytes": pure_graph_bytes,
            "vector_bytes": raw_vec_bytes,
            "attr_bytes": attr_bytes,
            "key_bytes": key_bytes,
            "id_bytes": id_bytes,
            "col_file_bytes": col_file_bytes,
            "meta_bytes": meta_bytes,
            "total_bytes": faiss_bytes + attr_bytes + key_bytes + id_bytes + meta_bytes + col_file_bytes,
        }

    def dist(self, q, local_ids):
        """Computes squared L2 distances from query q to local indices."""
        diff = self.X[local_ids] - q
        return np.einsum("ij,ij->i", diff, diff)

    def find_matching_local(self, intervals, lo_raw, hi_raw, space):
        """Finds local matching indices inside this leaf using 4D Hilbert key intervals
        and Parquet-style columnar storage for secondary attributes.
        """
        p_dim = self.primary_dim
        lo_p = lo_raw[:p_dim]
        hi_p = hi_raw[:p_dim]

        # 1. Primary Attributes Filter [0 .. primary_dim-1] via 4D Hilbert Curve / Zone Map
        if intervals:
            lo_q = space.quantize_primary(lo_p)
            hi_q = space.quantize_primary(hi_p)
            sk = self.skeys
            slices = []
            for s, e in intervals:
                if s >= self.k_max or e <= self.k_min:
                    continue
                ls = max(s, self.k_min)
                le = min(e, self.k_max)
                a = np.searchsorted(sk, np.uint64(ls))
                b = np.searchsorted(sk, np.uint64(le))
                if b > a:
                    slices.append((a, b))

            if not slices:
                return np.empty(0, dtype=np.int64)
            cands = np.concatenate([self.order[a:b] for a, b in slices])
            primary_mask = np.all((self.A[cands, :p_dim] >= lo_p) & (self.A[cands, :p_dim] <= hi_p), axis=1)
            primary_matches = cands[primary_mask]
        else:
            primary_mask = np.all((self.A[:, :p_dim] >= lo_p) & (self.A[:, :p_dim] <= hi_p), axis=1)
            primary_matches = np.where(primary_mask)[0]

        if len(primary_matches) == 0:
            return np.empty(0, dtype=np.int64)

        # 2. Secondary Attributes Filter [primary_dim .. m-1] via Leaf Columnar Storage File
        if self.columnar_store is not None and self.m > p_dim:
            lo_sec = lo_raw[p_dim:]
            hi_sec = hi_raw[p_dim:]
            has_sec_constraints = np.any((lo_sec > -1e30) | (hi_sec < 1e30))
            if has_sec_constraints:
                sec_matches = self.columnar_store.filter_secondary(lo_sec, hi_sec)
                if len(sec_matches) == 0:
                    return np.empty(0, dtype=np.int64)
                if len(primary_matches) < self.n:
                    return np.intersect1d(primary_matches, sec_matches, assume_unique=True)
                return sec_matches

        return primary_matches

    def search_unconstrained(self, q, k=10, ef=64, shared_state=None):
        """Mode 1: Pure Vanilla Unconstrained HNSW (0% filtering overhead)."""
        if self.n == 0 or self.index is None:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32), 0

        # Branch-and-bound centroid check against shared state
        if shared_state is not None and shared_state.is_full():
            dist_to_c = float(np.linalg.norm(q - self.centroid))
            min_dist_to_leaf = max(0.0, dist_to_c - self.radius)
            if (min_dist_to_leaf * min_dist_to_leaf) >= shared_state.get_bound():
                shared_state.record_early_exit()
                return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32), 1

        actual_k = min(k, self.n)
        self.index.hnsw.efSearch = max(ef, actual_k)
        D, I = self.index.search(q[None], actual_k)
        valid_idx = I[0] >= 0
        local_res = I[0][valid_idx]
        dists = D[0][valid_idx].astype(np.float32)
        gids = self.global_ids[local_res]
        ndis = self.index.hnsw.efSearch * 2

        if shared_state is not None:
            shared_state.update_batch(gids, dists)

        return gids, dists, ndis

    def search_scan(self, q, matching_local_ids, k=10, shared_state=None):
        """Mode 2: Contiguous SIMD Flat Scan over matching slice."""
        if len(matching_local_ids) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32), 0

        # Branch-and-bound centroid check against shared state
        if shared_state is not None and shared_state.is_full():
            dist_to_c = float(np.linalg.norm(q - self.centroid))
            min_dist_to_leaf = max(0.0, dist_to_c - self.radius)
            if (min_dist_to_leaf * min_dist_to_leaf) >= shared_state.get_bound():
                shared_state.record_early_exit()
                return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32), 1

        d = self.dist(q, matching_local_ids)
        if shared_state is not None and shared_state.is_full():
            bound = shared_state.get_bound()
            mask = d < bound
            if not np.any(mask):
                shared_state.record_early_exit()
                return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32), len(matching_local_ids)
            valid_indices = np.where(mask)[0]
            sub_d = d[valid_indices]
            top = valid_indices[np.argsort(sub_d)[:k]]
        else:
            top = np.argsort(d)[:k]

        top_ids = matching_local_ids[top]
        top_dists = d[top].astype(np.float32)
        top_gids = self.global_ids[top_ids]
        if shared_state is not None:
            shared_state.update_batch(top_gids, top_dists)
        return top_gids, top_dists, len(matching_local_ids)

    def search_seeded_hnsw(self, q, matching_local_ids, k=10, ef=64, n_seeds=8, seed_budget=64, nav_seeds=4, shared_state=None):
        """Mode 3: Hilbert-Seeded HNSW Traversal with 2-hop navigation for boundary leaves."""
        if len(matching_local_ids) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32), 0

        # Branch-and-bound centroid check against shared state
        if shared_state is not None and shared_state.is_full():
            dist_to_c = float(np.linalg.norm(q - self.centroid))
            min_dist_to_leaf = max(0.0, dist_to_c - self.radius)
            if (min_dist_to_leaf * min_dist_to_leaf) >= shared_state.get_bound():
                shared_state.record_early_exit()
                return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32), 1

        valid_mask = np.zeros(self.n, dtype=bool)
        valid_mask[matching_local_ids] = True

        seeds = []
        ndis_total = 0

        # Optional navigation seeds from standard HNSW descent
        if nav_seeds and self.index is not None:
            self.index.hnsw.efSearch = max(nav_seeds, 16)
            _, I = self.index.search(q[None], nav_seeds)
            seeds += [i for i in I[0] if i >= 0]
            ndis_total += self.index.hnsw.efSearch * 2

        # Hilbert seeds: sample valid candidates closest to q
        pool = matching_local_ids if len(matching_local_ids) <= seed_budget else np.random.choice(
            matching_local_ids, seed_budget, replace=False)
        dp = self.dist(q, pool)
        ndis_total += len(pool)
        seeds += pool[np.argsort(dp)[:n_seeds]].tolist()
        seeds = np.unique(seeds)

        # Beam search at layer 0 (2-hop hopping over invalid neighbors)
        visited = np.zeros(self.n, dtype=bool)
        visited[seeds] = True
        cand = [(float(d), int(i)) for d, i in zip(self.dist(q, seeds), seeds)]
        heapq.heapify(cand)
        res = [(-float(d), int(i)) for d, i in cand if valid_mask[i]]
        heapq.heapify(res)

        while len(res) > ef:
            heapq.heappop(res)

        # Update shared state with initial valid seeds
        if shared_state is not None and res:
            for neg_d, i in res:
                shared_state.update(self.global_ids[i], -neg_d)

        early_exited = False
        while cand:
            d, u = heapq.heappop(cand)

            local_worst = -res[0][0] if len(res) >= ef else np.inf
            ef_bound = shared_state.get_ef_bound() if (shared_state is not None and shared_state.is_ef_full()) else np.inf

            if (len(res) >= ef and d > local_worst) or (d > ef_bound):
                if d > ef_bound and (shared_state is not None and shared_state.is_ef_full()):
                    early_exited = True
                break

            nb = self.nbrs[self.offs[u]: self.offs[u] + self.nb0]
            nb = nb[nb >= 0]

            # 2-hop navigation over invalid neighbors
            hop = nb[~valid_mask[nb]]
            if hop.size > 0:
                nb2 = np.concatenate([self.nbrs[self.offs[h]: self.offs[h] + self.nb0] for h in hop])
                nb = np.concatenate([nb, nb2[nb2 >= 0]])

            nb = np.unique(nb)
            nb = nb[~visited[nb]]
            if nb.size == 0:
                continue
            visited[nb] = True
            dn = self.dist(q, nb)
            ndis_total += nb.size

            local_worst = -res[0][0] if len(res) >= ef else np.inf
            ef_bound = shared_state.get_ef_bound() if (shared_state is not None and shared_state.is_ef_full()) else np.inf
            effective_bound = min(local_worst, ef_bound)

            for dv, v in zip(dn, nb):
                if dv < effective_bound:
                    heapq.heappush(cand, (float(dv), int(v)))
                    if valid_mask[v]:
                        heapq.heappush(res, (-float(dv), int(v)))
                        if len(res) > ef:
                            heapq.heappop(res)
                        local_worst = -res[0][0] if len(res) >= ef else np.inf
                        if shared_state is not None:
                            shared_state.update(self.global_ids[v], dv)
                            if shared_state.is_ef_full():
                                ef_bound = shared_state.get_ef_bound()
                        effective_bound = min(local_worst, ef_bound)

        if early_exited and shared_state is not None:
            shared_state.record_early_exit()

        # res stores (-dist, i) -> extract positive distances!
        out = sorted((-neg_d, i) for neg_d, i in res)[:k]
        top_local = np.array([i for _, i in out], dtype=np.int64)
        top_dists = np.array([dist for dist, _ in out], dtype=np.float32)
        top_gids = self.global_ids[top_local]
        if shared_state is not None and len(top_gids) > 0:
            shared_state.update_batch(top_gids, top_dists)
        return top_gids, top_dists, ndis_total


class HilbertBTreeRouter:
    """Maintains a set of non-overlapping LeafSubgraph instances partitioned along Hilbert keys."""

    def __init__(self):
        self.leaves = []

    def add_leaf(self, leaf):
        self.leaves.append(leaf)
        self.leaves.sort(key=lambda l: l.k_min)

    def route(self, lo_raw, hi_raw, intervals):
        """Classifies leaves using Zone Maps and Hilbert intervals:

          - enclosed: leaves completely covered by query (pure vanilla HNSW).
          - boundary: leaves that partially intersect query.
          - disjoint: leaves completely outside (pruned with 0 compute).
        """
        enclosed = []
        boundary = []

        for leaf in self.leaves:
            # 1. Zone Map Check: Attribute Bounding Box
            if np.all(leaf.attr_min >= lo_raw) and np.all(leaf.attr_max <= hi_raw):
                enclosed.append(leaf)
                continue
            if np.any(leaf.attr_min > hi_raw) or np.any(leaf.attr_max < lo_raw):
                # Disjoint: Pruned!
                continue

            # 2. Key interval check (if intervals provided and non-empty)
            if intervals:
                overlap = False
                for s, e in intervals:
                    if not (e <= leaf.k_min or s >= leaf.k_max):
                        overlap = True
                        break
                if overlap:
                    boundary.append(leaf)
            else:
                boundary.append(leaf)

        return enclosed, boundary


class TemporalBuffer:
    """L0 Ingestion MemTable: buffers incoming stream appends and executes localized compaction/splits."""

    def __init__(self, space, tuner, router, max_leaf_size=10000, buf_capacity=2000):
        self.space = space
        self.tuner = tuner
        self.router = router
        self.max_leaf_size = max_leaf_size
        self.buf_capacity = buf_capacity

        self.buf_X = []
        self.buf_A = []
        self.buf_keys = []
        self.buf_ids = []
        self.next_leaf_id = len(router.leaves)

    def append(self, x, a, global_id):
        x = np.asarray(x, dtype=np.float32)
        a = np.asarray(a, dtype=np.float32)
        aq = self.space.quantize(a[None])[0]
        key = self.space.points_to_keys(aq[None])[0]

        self.buf_X.append(x)
        self.buf_A.append(a)
        self.buf_keys.append(key)
        self.buf_ids.append(global_id)

        if len(self.buf_X) >= self.buf_capacity:
            self.flush()

    def flush(self):
        """Flushes L0 buffer into respective Hilbert leaf partitions and triggers localized splits."""
        if not self.buf_X:
            return
        bX = np.array(self.buf_X, dtype=np.float32)
        bA = np.array(self.buf_A, dtype=np.float32)
        bKeys = np.array(self.buf_keys, dtype=np.uint64)
        bIds = np.array(self.buf_ids, dtype=np.int64)

        self.buf_X.clear()
        self.buf_A.clear()
        self.buf_keys.clear()
        self.buf_ids.clear()

        self.primary_dim = getattr(space, "primary_dim", 4)
        if not self.router.leaves:
            deg = self.tuner.compute_degree(bX)
            leaf = LeafSubgraph(0, np.uint64(0), self.space.max_key, bX, bA, bKeys, bIds, M=deg,
                                primary_dim=self.primary_dim)
            self.router.add_leaf(leaf)
            return

        for leaf in list(self.router.leaves):
            mask = (bKeys >= leaf.k_min) & (bKeys < leaf.k_max)
            if np.any(mask):
                nX = np.vstack([leaf.X, bX[mask]])
                nA = np.vstack([leaf.A, bA[mask]])
                nKeys = np.concatenate([leaf.keys, bKeys[mask]])
                nIds = np.concatenate([leaf.global_ids, bIds[mask]])

                if len(nX) > self.max_leaf_size:
                    # LOCALIZED SPLIT: Split at median Hilbert key!
                    order = np.argsort(nKeys)
                    mid = len(nKeys) // 2
                    mid_key = nKeys[order[mid]]

                    left_idx = order[:mid]
                    right_idx = order[mid:]

                    leaf.close()
                    self.router.leaves.remove(leaf)

                    deg_l = self.tuner.compute_degree(nX[left_idx])
                    deg_r = self.tuner.compute_degree(nX[right_idx])

                    leaf_l = LeafSubgraph(self.next_leaf_id, leaf.k_min, mid_key,
                                          nX[left_idx], nA[left_idx], nKeys[left_idx], nIds[left_idx], M=deg_l,
                                          primary_dim=self.primary_dim)
                    self.next_leaf_id += 1
                    leaf_r = LeafSubgraph(self.next_leaf_id, mid_key, leaf.k_max,
                                          nX[right_idx], nA[right_idx], nKeys[right_idx], nIds[right_idx], M=deg_r,
                                          primary_dim=self.primary_dim)
                    self.next_leaf_id += 1

                    self.router.add_leaf(leaf_l)
                    self.router.add_leaf(leaf_r)
                else:
                    leaf.close()
                    self.router.leaves.remove(leaf)
                    deg = self.tuner.compute_degree(nX)
                    new_leaf = LeafSubgraph(leaf.leaf_id, leaf.k_min, leaf.k_max,
                                            nX, nA, nKeys, nIds, M=deg,
                                            primary_dim=self.primary_dim)
                    self.router.add_leaf(new_leaf)


class HilbertHNSWEngine:
    """The Complete Hilbert-Partitioned B-Tree Vector Engine with 4D Hilbert Curve
    and Parquet-Style Columnar Storage for Secondary Attributes.
    """

    def __init__(self, X, A, p=14, max_leaf_size=10000, buf_capacity=2000,
                 M_min=12, M_max=32, M_base=20, scan_threshold=None, num_threads=None,
                 primary_dim=4, storage_dir=None, **kwargs):
        self.n, self.d = X.shape
        self.m = A.shape[1]
        self.primary_dim = min(primary_dim, self.m)
        self.storage_dir = storage_dir
        self.p = p
        self.max_leaf_size = max_leaf_size
        self.buf_capacity = buf_capacity
        if scan_threshold is None:
            self.scan_threshold = max(500, int(0.15 * max_leaf_size))
        else:
            self.scan_threshold = scan_threshold
        if num_threads is None:
            self.num_threads = min(8, os.cpu_count() or 4)
        else:
            self.num_threads = max(1, num_threads)
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads)

        amin = A.min(axis=0)
        amax = A.max(axis=0)
        self.space = HilbertSpace(amin, amax, p=p, m=self.m, primary_dim=self.primary_dim)
        global_apd = GraphSparsityTuner.compute_apd(X)
        self.tuner = GraphSparsityTuner(global_apd, M_min=M_min, M_max=M_max, M_base=M_base)

        self.router = HilbertBTreeRouter()
        self.buffer = TemporalBuffer(self.space, self.tuner, self.router,
                                     max_leaf_size=max_leaf_size, buf_capacity=buf_capacity)

        self._bulk_load(X, A)

    def close(self):
        """Shuts down executor and closes leaf columnar storage files."""
        if hasattr(self, "router") and self.router is not None:
            for leaf in self.router.leaves:
                leaf.close()
        if hasattr(self, "executor") and self.executor is not None:
            self.executor.shutdown(wait=False)
            self.executor = None

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def _bulk_load(self, X, A):
        """Initial bulk partitioning along 4D Hilbert space."""
        Aq = self.space.quantize_primary(A)
        keys = self.space.points_to_keys(Aq)
        order = np.argsort(keys)

        num_leaves = max(1, int(np.ceil(self.n / (self.max_leaf_size * 0.75))))
        chunks = np.array_split(order, num_leaves)

        k_start = np.uint64(0)
        for i, chunk_idx in enumerate(chunks):
            if len(chunk_idx) == 0:
                continue
            chunk_keys = keys[chunk_idx]
            if i == len(chunks) - 1:
                k_end = self.space.max_key
            else:
                next_start_key = keys[chunks[i + 1][0]]
                k_end = next_start_key

            deg = self.tuner.compute_degree(X[chunk_idx])
            leaf = LeafSubgraph(i, k_start, k_end,
                                X[chunk_idx], A[chunk_idx], chunk_keys, chunk_idx, M=deg,
                                primary_dim=self.primary_dim, storage_dir=self.storage_dir)
            self.router.add_leaf(leaf)
            k_start = k_end

    def insert(self, x, a, global_id):
        self.buffer.append(x, a, global_id)

    def flush(self):
        self.buffer.flush()

    def get_memory_footprint(self):
        """Returns memory footprint breakdown across all leaf subgraphs and router."""
        self.flush()
        leaf_stats = [leaf.get_memory_bytes() for leaf in self.router.leaves]
        total_faiss = sum(s["faiss_total"] for s in leaf_stats)
        pure_graph = sum(s["pure_graph_bytes"] for s in leaf_stats)
        vec_bytes = sum(s["vector_bytes"] for s in leaf_stats)
        attr_bytes = sum(s["attr_bytes"] for s in leaf_stats)
        key_bytes = sum(s["key_bytes"] for s in leaf_stats)
        id_bytes = sum(s["id_bytes"] for s in leaf_stats)
        meta_bytes = sum(s["meta_bytes"] for s in leaf_stats)
        total_bytes = sum(s["total_bytes"] for s in leaf_stats)
        return {
            "num_leaves": len(self.router.leaves),
            "faiss_total_bytes": total_faiss,
            "pure_graph_bytes": pure_graph,
            "vector_bytes": vec_bytes,
            "attr_bytes": attr_bytes,
            "key_bytes": key_bytes,
            "id_bytes": id_bytes,
            "meta_bytes": meta_bytes,
            "total_bytes": total_bytes,
            "faiss_mb": total_faiss / (1024 * 1024),
            "pure_graph_mb": pure_graph / (1024 * 1024),
            "total_mb": total_bytes / (1024 * 1024),
        }

    def search(self, q, lo_raw=None, hi_raw=None, k=10, ef=64, parallel=True, collation="mapreduce"):
        """Cost-based adaptive search supporting:

          1. Pure Unconstrained ANN (Bypass with Shared Top-K Bound).
          2. Zero-Filter Vanilla HNSW on fully enclosed leaves.
          3. SIMD Scan on small boundary slices.
          4. Hilbert-Seeded 2-hop HNSW on large boundary slices.
          5. Parallel multi-subgraph execution with shared-state branch-and-bound early exit.
          6. Multi-threaded MapReduce hierarchical tree collation across leaf candidates.

        collation: 'mapreduce' (Hybrid branch-and-bound Map + parallel tree reduce),
                   'pure_mapreduce' (Decoupled Map tasks + parallel tree reduce),
                   'shared_heap' (Concurrent locked insertions into SharedSearchState),
                   'flat' (Map + flat numpy argpartition reduction).
        """
        q = np.asarray(q, dtype=np.float32).ravel()
        stats = {
            "ndis": 0,
            "enclosed_leaves": 0,
            "boundary_leaves": 0,
            "pruned_leaves": 0,
            "early_exits": 0,
            "parallel": parallel and (self.num_threads > 1),
            "mode": "hybrid",
            "collation": collation,
        }

        self.flush()
        # In pure_mapreduce, workers run completely decoupled with zero lock contention
        shared_state = None if collation == "pure_mapreduce" else SharedSearchState(k=k, ef=ef)
        use_parallel = parallel and (self.num_threads > 1) and (self.executor is not None)

        # Fast Path 1: Pure Unconstrained ANN Search
        if lo_raw is None or hi_raw is None:
            stats["mode"] = "unconstrained"
            leaf_tasks = []
            for leaf in self.router.leaves:
                dist_to_c = float(np.linalg.norm(q - leaf.centroid))
                min_dist = max(0.0, dist_to_c - leaf.radius)
                leaf_tasks.append((min_dist, leaf))
            leaf_tasks.sort(key=lambda t: t[0])

            def _search_unconstrained_task(leaf):
                ids, dists, nd = leaf.search_unconstrained(q, k=k, ef=ef, shared_state=shared_state)
                return ids, dists, nd

            if use_parallel and len(leaf_tasks) > 1:
                futures = [self.executor.submit(_search_unconstrained_task, leaf) for _, leaf in leaf_tasks]
                task_results = [f.result() for f in concurrent.futures.as_completed(futures)]
            else:
                task_results = [_search_unconstrained_task(leaf) for _, leaf in leaf_tasks]

            total_ndis = sum(r[2] for r in task_results)
            cand_lists = [(r[0], r[1]) for r in task_results]

            if collation == "shared_heap" and shared_state is not None:
                out_ids, _ = shared_state.get_results()
            elif collation == "flat":
                out_ids, _ = MapReduceCollation.flat_reduce(cand_lists, k=k)
            else:  # 'mapreduce' or 'pure_mapreduce' (default)
                out_ids, _ = MapReduceCollation.tree_reduce(cand_lists, k=k, executor=self.executor, parallel=use_parallel)

            stats["ndis"] = total_ndis
            stats["early_exits"] = shared_state.early_exits if shared_state is not None else 0
            return out_ids, stats

        # Filtered Path: Zone Map & Range decomposition routing
        intervals = self.space.decompose(lo_raw, hi_raw, max_depth=6)
        enclosed_leaves, boundary_leaves = self.router.route(lo_raw, hi_raw, intervals)
        stats["enclosed_leaves"] = len(enclosed_leaves)
        stats["boundary_leaves"] = len(boundary_leaves)
        stats["pruned_leaves"] = len(self.router.leaves) - (len(enclosed_leaves) + len(boundary_leaves))

        tasks = []
        for leaf in enclosed_leaves:
            dist_to_c = float(np.linalg.norm(q - leaf.centroid))
            min_dist = max(0.0, dist_to_c - leaf.radius)
            tasks.append((min_dist, leaf, "enclosed"))

        for leaf in boundary_leaves:
            dist_to_c = float(np.linalg.norm(q - leaf.centroid))
            min_dist = max(0.0, dist_to_c - leaf.radius)
            tasks.append((min_dist, leaf, "boundary"))

        tasks.sort(key=lambda t: t[0])

        def _execute_filtered_task(task_tuple):
            _, leaf, kind = task_tuple
            if kind == "enclosed":
                ids, dists, nd = leaf.search_unconstrained(q, k=k, ef=ef, shared_state=shared_state)
                return ids, dists, nd
            else:  # boundary
                if shared_state is not None and shared_state.is_full():
                    dist_to_c = float(np.linalg.norm(q - leaf.centroid))
                    min_dist_to_leaf = max(0.0, dist_to_c - leaf.radius)
                    if (min_dist_to_leaf * min_dist_to_leaf) >= shared_state.get_bound():
                        shared_state.record_early_exit()
                        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32), 1
                matching_ids = leaf.find_matching_local(intervals, lo_raw, hi_raw, self.space)
                if len(matching_ids) == 0:
                    return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32), 0
                if len(matching_ids) <= self.scan_threshold:
                    ids, dists, nd = leaf.search_scan(q, matching_ids, k=k, shared_state=shared_state)
                else:
                    ids, dists, nd = leaf.search_seeded_hnsw(q, matching_ids, k=k, ef=ef, shared_state=shared_state)
                return ids, dists, nd

        if use_parallel and len(tasks) > 1:
            futures = [self.executor.submit(_execute_filtered_task, t) for t in tasks]
            task_results = [f.result() for f in concurrent.futures.as_completed(futures)]
        else:
            task_results = [_execute_filtered_task(t) for t in tasks]

        total_ndis = sum(r[2] for r in task_results)
        cand_lists = [(r[0], r[1]) for r in task_results]

        if collation == "shared_heap" and shared_state is not None:
            out_ids, _ = shared_state.get_results()
        elif collation == "flat":
            out_ids, _ = MapReduceCollation.flat_reduce(cand_lists, k=k)
        else:  # 'mapreduce' or 'pure_mapreduce' (default)
            out_ids, _ = MapReduceCollation.tree_reduce(cand_lists, k=k, executor=self.executor, parallel=use_parallel)

        stats["ndis"] = total_ndis
        stats["early_exits"] = shared_state.early_exits if shared_state is not None else 0
        return out_ids, stats
