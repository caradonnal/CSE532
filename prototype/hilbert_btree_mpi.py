"""Distributed MPI Hilbert-Partitioned Vector Engine for HPC & Slurm Environments.

Architecture:
  - Range-Partitioned Distributed Hilbert B-Tree: Each MPI rank hosts a contiguous
    segment of the Hilbert curve (one or more LeafSubgraph partitions).
  - Zone Map Distributed Pruning: Ranks whose bounding boxes [attr_min, attr_max]
    do not intersect the query box prune themselves locally with zero network chatter.
  - Broadcast Query -> Parallel Local Subgraph Search -> MPI Gather & Reduction.
  - Designed for hybrid MPI + OpenMP on Slurm clusters (e.g., srun -n $SLURM_NTASKS).
"""

import heapq
import os
import sys
import time
import numpy as np
import faiss

# Support graceful import of mpi4py with local fallback for testing without MPI
try:
    from mpi4py import MPI
    HAVE_MPI = True
except ImportError:
    HAVE_MPI = False

from hilbert_btree import (
    HilbertSpace,
    LeafSubgraph,
    GraphSparsityTuner,
    MapReduceCollation,
    interleave_subspace_keys,
    deinterleave_subspace_keys,
)


class MockMPIComm:
    """Single-process fallback communicator when mpi4py is not installed."""
    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def bcast(self, obj, root=0):
        return obj

    def gather(self, obj, root=0):
        return [obj]

    def Barrier(self):
        pass


def get_mpi_comm():
    """Returns MPI.COMM_WORLD if mpi4py is available, else a single-node mock."""
    if HAVE_MPI:
        return MPI.COMM_WORLD
    return MockMPIComm()


class DistributedMPIHilbertEngine:
    """Distributed Vector Engine distributing Hilbert partition subgraphs across MPI ranks."""

    def __init__(self, X=None, A=None, p=14, max_leaf_size=20000,
                 primary_dim=4, comm=None, **kwargs):
        self.comm = comm or get_mpi_comm()
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.p = p
        self.max_leaf_size = max_leaf_size
        self.primary_dim = primary_dim
        self.local_leaves = []
        self.space = None

        if self.rank == 0:
            assert X is not None and A is not None, "Rank 0 must provide X and A for initialization"
            self.n, self.d = X.shape
            self.m = A.shape[1]
            self.primary_dim = min(4, self.m)
            amin = A.min(axis=0)
            amax = A.max(axis=0)
            meta = {
                "n": self.n,
                "d": self.d,
                "m": self.m,
                "primary_dim": self.primary_dim,
                "amin": amin,
                "amax": amax,
                "p": self.p,
            }
        else:
            meta = None

        # Step 1: Broadcast dataset metadata to all ranks
        meta = self.comm.bcast(meta, root=0)
        self.n = meta["n"]
        self.d = meta["d"]
        self.m = meta["m"]
        self.primary_dim = meta.get("primary_dim", min(4, self.m))
        self.space = HilbertSpace(meta["amin"], meta["amax"], p=meta["p"], m=self.m, primary_dim=self.primary_dim)

        # Step 2: Bulk partition and distribute to ranks
        self._distribute_data(X, A)

    def _distribute_data(self, X, A):
        """Rank 0 partitions data by 4D Hilbert key and scatters assigned chunks to ranks."""
        if self.rank == 0:
            t0 = time.time()
            Aq = self.space.quantize_primary(A)
            keys = self.space.points_to_keys(Aq)
            order = np.argsort(keys)

            # Determine number of global leaf partitions
            num_leaves = max(self.size, int(np.ceil(self.n / (self.max_leaf_size * 0.75))))
            leaf_chunks = np.array_split(order, num_leaves)

            # Assign leaf partitions to MPI ranks (round-robin or contiguous block)
            rank_assignments = [[] for _ in range(self.size)]
            k_start = np.uint64(0)

            for i, chunk_idx in enumerate(leaf_chunks):
                if len(chunk_idx) == 0:
                    continue
                chunk_keys = keys[chunk_idx]
                k_end = self.space.max_key if i == len(leaf_chunks) - 1 else keys[leaf_chunks[i + 1][0]]
                assigned_rank = i % self.size
                rank_assignments[assigned_rank].append({
                    "leaf_id": i,
                    "k_min": k_start,
                    "k_max": k_end,
                    "X": X[chunk_idx],
                    "A": A[chunk_idx],
                    "keys": chunk_keys,
                    "gids": chunk_idx,
                })
                k_start = k_end
            partition_time = time.time() - t0
        else:
            rank_assignments = None

        # Scatter partition payloads to respective ranks
        if HAVE_MPI and self.size > 1:
            my_leaves_data = self.comm.scatter(rank_assignments, root=0)
        else:
            my_leaves_data = rank_assignments[0] if rank_assignments else []

        # Each rank builds its assigned local leaf subgraphs and Parquet-style columnar files
        t_build_start = time.time()
        for leaf_dict in my_leaves_data:
            deg = GraphSparsityTuner(GraphSparsityTuner.compute_apd(leaf_dict["X"])).compute_degree(leaf_dict["X"])
            leaf = LeafSubgraph(
                leaf_dict["leaf_id"],
                leaf_dict["k_min"],
                leaf_dict["k_max"],
                leaf_dict["X"],
                leaf_dict["A"],
                leaf_dict["keys"],
                leaf_dict["gids"],
                M=deg,
                primary_dim=self.primary_dim,
            )
            self.local_leaves.append(leaf)

        self.local_build_time = time.time() - t_build_start
        self.comm.Barrier()

    def close(self):
        """Closes all leaf columnar file resources on this rank."""
        if hasattr(self, "local_leaves") and self.local_leaves:
            for leaf in self.local_leaves:
                leaf.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def get_rank_memory_footprint(self):
        """Calculates memory breakdown for the subgraphs hosted on this MPI rank."""
        leaf_stats = [leaf.get_memory_bytes() for leaf in self.local_leaves]
        total_bytes = sum(s["total_bytes"] for s in leaf_stats)
        vec_bytes = sum(s["vector_bytes"] for s in leaf_stats)
        graph_bytes = sum(s["pure_graph_bytes"] for s in leaf_stats)
        aux_bytes = total_bytes - (vec_bytes + graph_bytes)
        num_vecs = sum(leaf.n for leaf in self.local_leaves)
        return {
            "rank": self.rank,
            "num_leaves": len(self.local_leaves),
            "num_vectors": num_vecs,
            "total_bytes": total_bytes,
            "total_mb": total_bytes / (1024 * 1024),
            "vec_mb": vec_bytes / (1024 * 1024),
            "graph_mb": graph_bytes / (1024 * 1024),
            "aux_mb": aux_bytes / (1024 * 1024),
        }

    def search(self, q, lo_raw=None, hi_raw=None, k=10, ef=64):
        """Executes a distributed filtered ANN query across all MPI ranks."""
        # Step 1: Broadcast query payload from Rank 0
        if self.rank == 0:
            query_payload = (q, lo_raw, hi_raw, k, ef)
        else:
            query_payload = None

        t_bcast0 = time.perf_counter()
        q, lo_raw, hi_raw, k, ef = self.comm.bcast(query_payload, root=0)
        t_bcast = (time.perf_counter() - t_bcast0) * 1000.0

        t_local0 = time.perf_counter()
        local_cand_lists = []
        local_ndis = 0
        local_pruned = 0
        local_enclosed = 0
        local_boundary = 0

        # Fast Path: Pure Unconstrained ANN Search
        if lo_raw is None or hi_raw is None:
            for leaf in self.local_leaves:
                ids, dists, nd = leaf.search_unconstrained(q, k=k, ef=ef)
                local_cand_lists.append((ids, dists))
                local_ndis += nd
        else:
            # Filtered Path: Zone Map Pruning & Leaf Search
            for leaf in self.local_leaves:
                # 1. Zone Map Check: Is leaf disjoint from query bounding box?
                if np.any(leaf.attr_min > hi_raw) or np.any(leaf.attr_max < lo_raw):
                    local_pruned += 1
                    continue

                # 2. Zone Map Enclosed Check
                if np.all(leaf.attr_min >= lo_raw) and np.all(leaf.attr_max <= hi_raw):
                    local_enclosed += 1
                    ids, dists, nd = leaf.search_unconstrained(q, k=k, ef=ef)
                    local_cand_lists.append((ids, dists))
                    local_ndis += nd
                    continue

                # 3. Boundary Leaf Check
                local_boundary += 1
                matching_ids = leaf.find_matching_local([], lo_raw, hi_raw, self.space)
                if len(matching_ids) == 0:
                    continue

                scan_thresh = max(1000, int(0.25 * self.max_leaf_size))
                if len(matching_ids) <= scan_thresh:
                    ids, dists, nd = leaf.search_scan(q, matching_ids, k=k)
                else:
                    ids, dists, nd = leaf.search_seeded_hnsw(q, matching_ids, k=k, ef=ef)

                local_cand_lists.append((ids, dists))
                local_ndis += nd

        # Reduce locally on rank to top-k candidates before MPI gather
        if local_cand_lists:
            top_ids, top_dists = MapReduceCollation.flat_reduce(local_cand_lists, k=k)
        else:
            top_ids = np.empty(0, dtype=np.int64)
            top_dists = np.empty(0, dtype=np.float32)

        t_local_ms = (time.perf_counter() - t_local0) * 1000.0

        # Step 2: MPI Gather local top-k candidates to Rank 0
        t_gather0 = time.perf_counter()
        local_result = {
            "ids": top_ids,
            "dists": top_dists,
            "ndis": local_ndis,
            "pruned": local_pruned,
            "enclosed": local_enclosed,
            "boundary": local_boundary,
            "local_time_ms": t_local_ms,
        }
        all_results = self.comm.gather(local_result, root=0)
        t_gather = (time.perf_counter() - t_gather0) * 1000.0

        # Step 3: Collate global top-k on Rank 0
        if self.rank == 0:
            rank_cand_lists = [(r["ids"], r["dists"]) for r in all_results if len(r["ids"]) > 0]
            if rank_cand_lists:
                global_ids, global_dists = MapReduceCollation.flat_reduce(rank_cand_lists, k=k)
            else:
                global_ids = np.empty(0, dtype=np.int64)
                global_dists = np.empty(0, dtype=np.float32)

            total_ndis = sum(r["ndis"] for r in all_results)
            total_pruned = sum(r["pruned"] for r in all_results)
            total_enclosed = sum(r["enclosed"] for r in all_results)
            total_boundary = sum(r["boundary"] for r in all_results)
            max_local_time = max(r["local_time_ms"] for r in all_results)

            stats = {
                "ndis": total_ndis,
                "pruned_leaves": total_pruned,
                "enclosed_leaves": total_enclosed,
                "boundary_leaves": total_boundary,
                "bcast_time_ms": t_bcast,
                "gather_time_ms": t_gather,
                "max_local_time_ms": max_local_time,
                "active_ranks": sum(1 for r in all_results if r["ndis"] > 0),
            }
            return global_ids, global_dists, stats
        else:
            return None, None, None


class DistributedShardedHNSW:
    """Traditional Distributed Sharded HNSW Baseline without Hilbert partitioning.

    Standard architecture used in distributed vector databases (e.g. Milvus / Qdrant):
      - Ranks receive random/horizontal shards of the global dataset (N / P vectors each).
      - Each rank builds a standard monolithic Faiss HNSW index over its shard.
      - Shards have NO attribute awareness: attributes are uniformly distributed across all ranks.
      - Query execution: Every single rank MUST execute search (0% rank pruning).
      - Supports:
          1. Distributed Post-Filtering: Rank searches top-ef nearest neighbors, filters by attributes.
          2. Distributed In-Graph Filtered Beam Search: Rank executes in-graph predicate filtering during traversal.
    """

    def __init__(self, X=None, A=None, d=32, M=20, ef_construction=100, comm=None):
        self.comm = comm or get_mpi_comm()
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.d = d
        self.M = M
        self.ef_construction = ef_construction

        if self.rank == 0:
            assert X is not None and A is not None
            self.n = len(X)
            self.m = A.shape[1]
            chunks_X = np.array_split(X, self.size)
            chunks_A = np.array_split(A, self.size)
            indices = np.arange(self.n, dtype=np.int64)
            chunks_gids = np.array_split(indices, self.size)
            scatter_data = [
                {"X": chunks_X[i], "A": chunks_A[i], "gids": chunks_gids[i], "n": self.n, "m": self.m}
                for i in range(self.size)
            ]
        else:
            scatter_data = None

        if HAVE_MPI and self.size > 1:
            my_shard = self.comm.scatter(scatter_data, root=0)
        else:
            my_shard = scatter_data[0] if scatter_data else None

        self.X_local = my_shard["X"]
        self.A_local = my_shard["A"]
        self.gids_local = my_shard["gids"]
        self.n_local = len(self.X_local)

        # Build local HNSW index on shard
        t0 = time.time()
        self.index = faiss.IndexHNSWFlat(self.d, self.M)
        self.index.hnsw.efConstruction = self.ef_construction
        self.index.add(self.X_local)
        self.build_time = time.time() - t0

        # Extract graph adjacency for in-graph beam search
        h = self.index.hnsw
        self.nbrs = faiss.vector_to_array(h.neighbors)
        self.offs = faiss.vector_to_array(h.offsets)
        self.nb0 = h.nb_neighbors(0)

        self.comm.Barrier()

    def dist_sq(self, q, idxs):
        diff = self.X_local[idxs] - q
        return np.einsum("ij,ij->i", diff, diff)

    def search(self, q, lo_raw=None, hi_raw=None, k=10, ef=64, mode="beam"):
        """Executes distributed search over traditional sharded HNSW graph.

        mode: 'beam' (in-graph filtered beam search) or 'post' (post-filtering).
        """
        payload = (q, lo_raw, hi_raw, k, ef, mode) if self.rank == 0 else None
        q, lo_raw, hi_raw, k, ef, mode = self.comm.bcast(payload, root=0)

        t_local0 = time.perf_counter()

        if lo_raw is None or hi_raw is None:
            # Unconstrained search on shard
            actual_k = min(k, self.n_local)
            self.index.hnsw.efSearch = max(ef, actual_k)
            D, I = self.index.search(q[None], actual_k)
            valid_idx = I[0] >= 0
            local_res = I[0][valid_idx]
            local_dists = D[0][valid_idx].astype(np.float32)
            local_ids = self.gids_local[local_res]
            local_ndis = self.index.hnsw.efSearch * 2
        elif mode == "post":
            # Distributed Post-Filtering
            self.index.hnsw.efSearch = ef
            D, I = self.index.search(q[None], ef)
            local_matches = []
            local_match_dists = []
            valid_mask = np.all((self.A_local >= lo_raw) & (self.A_local <= hi_raw), axis=1)
            for dist, idx in zip(D[0], I[0]):
                if idx >= 0 and valid_mask[idx]:
                    local_matches.append(self.gids_local[idx])
                    local_match_dists.append(dist)
                    if len(local_matches) >= k:
                        break
            local_ids = np.array(local_matches, dtype=np.int64)
            local_dists = np.array(local_match_dists, dtype=np.float32)
            local_ndis = ef * 2
        else:
            # Distributed In-Graph Filtered Beam Search
            valid_mask = np.all((self.A_local >= lo_raw) & (self.A_local <= hi_raw), axis=1)
            self.index.hnsw.efSearch = 16
            _, I = self.index.search(q[None], 4)
            seeds = [i for i in I[0] if i >= 0]
            if not seeds:
                seeds = [0]

            visited = np.zeros(self.n_local, dtype=bool)
            visited[seeds] = True
            cand = [(float(d), int(i)) for d, i in zip(self.dist_sq(q, seeds), seeds)]
            heapq.heapify(cand)
            res = [(-float(d), int(i)) for d, i in cand if valid_mask[i]]
            heapq.heapify(res)

            local_ndis = len(seeds) + 16 * 2
            while len(res) > ef:
                heapq.heappop(res)

            while cand:
                d, u = heapq.heappop(cand)
                if len(res) >= ef and d > -res[0][0]:
                    break
                nb = self.nbrs[self.offs[u]: self.offs[u] + self.nb0]
                nb = nb[nb >= 0]
                nb = nb[~visited[nb]]
                if nb.size == 0:
                    continue
                visited[nb] = True
                dn = self.dist_sq(q, nb)
                local_ndis += nb.size
                worst = -res[0][0] if len(res) >= ef else np.inf
                for dv, v in zip(dn, nb):
                    if dv < worst:
                        heapq.heappush(cand, (float(dv), int(v)))
                        if valid_mask[v]:
                            heapq.heappush(res, (-float(dv), int(v)))
                            if len(res) > ef:
                                heapq.heappop(res)
                            worst = -res[0][0] if len(res) >= ef else np.inf

            out = sorted((-neg_d, i) for neg_d, i in res)[:k]
            top_local = np.array([i for _, i in out], dtype=np.int64)
            local_dists = np.array([dist for dist, _ in out], dtype=np.float32)
            local_ids = self.gids_local[top_local] if len(top_local) > 0 else np.empty(0, dtype=np.int64)

        local_time_ms = (time.perf_counter() - t_local0) * 1000.0

        # Step 2: MPI Gather to Rank 0
        local_result = {
            "ids": local_ids,
            "dists": local_dists,
            "ndis": local_ndis,
            "local_time_ms": local_time_ms,
        }
        all_results = self.comm.gather(local_result, root=0)

        if self.rank == 0:
            rank_cand_lists = [(r["ids"], r["dists"]) for r in all_results if len(r["ids"]) > 0]
            if rank_cand_lists:
                global_ids, global_dists = MapReduceCollation.flat_reduce(rank_cand_lists, k=k)
            else:
                global_ids = np.empty(0, dtype=np.int64)
                global_dists = np.empty(0, dtype=np.float32)
            total_ndis = sum(r["ndis"] for r in all_results)
            max_local_time = max(r["local_time_ms"] for r in all_results)
            stats = {
                "ndis": total_ndis,
                "max_local_time_ms": max_local_time,
                "active_ranks": self.size,  # 0% rank pruning: every rank searched!
                "pruned_ranks": 0,
            }
            return global_ids, global_dists, stats
        else:
            return None, None, None

    def get_rank_memory_footprint(self):
        faiss_bytes = len(faiss.serialize_index(self.index))
        vec_bytes = self.X_local.nbytes
        graph_bytes = max(0, faiss_bytes - vec_bytes)
        attr_bytes = self.A_local.nbytes
        id_bytes = self.gids_local.nbytes
        total_bytes = faiss_bytes + attr_bytes + id_bytes
        return {
            "rank": self.rank,
            "num_vectors": self.n_local,
            "total_bytes": total_bytes,
            "total_mb": total_bytes / (1024 * 1024),
            "vec_mb": vec_bytes / (1024 * 1024),
            "graph_mb": graph_bytes / (1024 * 1024),
            "aux_mb": (attr_bytes + id_bytes) / (1024 * 1024),
        }
