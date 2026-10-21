"""HPC Distributed MPI Benchmark: Hilbert-Partitioned B-Tree vs. Traditional Sharded HNSW.

Evaluates distributed performance in HPC / Slurm cluster environments:
  1. Distributed Sharded HNSW (Post-Filtering): Industry baseline (random horizontal vector sharding).
  2. Distributed Sharded HNSW (In-Graph Filtered Beam Search): Standard in-graph filtering across all ranks.
  3. Distributed Hilbert B-Tree (Our Approach): Range-partitioned distributed Hilbert curve + Zone Map rank pruning.

Usage:
  # On Slurm HPC cluster:
  srun --mpi=pmi2 python prototype/bench_mpi.py --n 1000000 --m 12 --max-leaf-size 20000 --nq 20

  # Local test:
  python prototype/bench_mpi.py --n 20000 --m 12 --max-leaf-size 5000 --nq 10
"""

import sys
import time
import numpy as np

from hilbert_btree_mpi import (
    DistributedMPIHilbertEngine,
    DistributedShardedHNSW,
    get_mpi_comm,
    HAVE_MPI,
)


def run_mpi_benchmark():
    comm = get_mpi_comm()
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Parse CLI Arguments
    n = 100_000
    m = 12
    d = 32
    nq = 20
    k = 10
    ef = 64
    max_leaf_size = 20_000

    for i, arg in enumerate(sys.argv):
        if arg == "--n" and i + 1 < len(sys.argv):
            n = int(sys.argv[i + 1])
        elif arg == "--max-leaf-size" and i + 1 < len(sys.argv):
            max_leaf_size = int(sys.argv[i + 1])
        elif arg == "--nq" and i + 1 < len(sys.argv):
            nq = int(sys.argv[i + 1])
        elif arg in ("--m", "--n-attr") and i + 1 < len(sys.argv):
            m = int(sys.argv[i + 1])

    if rank == 0:
        print("=" * 115)
        print("CSE532 Distributed MPI Benchmark: Hilbert B-Tree vs. Traditional Sharded HNSW (HPC / Slurm)")
        print(f"MPI Topology: {size} Ranks | Dataset: N={n:,}, d={d}, m={m} attributes | Max Leaf Size: {max_leaf_size:,}")
        if not HAVE_MPI:
            print(">> Notice: mpi4py not detected. Running in single-node mock MPI emulation mode.")
        print("=" * 115)

    # 1. Dataset Generation on Rank 0
    rng = np.random.default_rng(42)
    if rank == 0:
        print(f"\n[1] Generating Dataset (N={n:,}, d={d}, m={m})...")
        t0 = time.time()
        X = rng.standard_normal((n, d)).astype(np.float32)
        A = rng.random((n, m)).astype(np.float32)
        Q = rng.standard_normal((nq, d)).astype(np.float32)
        print(f"    Generated vectors and attributes in {time.time() - t0:.2f}s")
    else:
        X, A, Q = None, None, None

    Q = comm.bcast(Q, root=0)

    # 2. Build Distributed Sharded HNSW (Traditional Baseline)
    if rank == 0:
        print(f"\n[2] Building Traditional Distributed Sharded HNSW Baseline ({size} Ranks)...")
    comm.Barrier()
    t0 = time.time()
    engine_sharded = DistributedShardedHNSW(X=X, A=A, d=d, M=20, ef_construction=100, comm=comm)
    comm.Barrier()
    sharded_build_time = time.time() - t0

    sharded_mem = engine_sharded.get_rank_memory_footprint()
    all_sharded_mem = comm.gather(sharded_mem, root=0)

    # 3. Build Distributed Hilbert-Partitioned B-Tree (Our Approach)
    if rank == 0:
        print(f"\n[3] Building Distributed Hilbert-Partitioned B-Tree ({size} Ranks, max_leaf_size={max_leaf_size:,})...")
    comm.Barrier()
    t0 = time.time()
    engine_hilbert = DistributedMPIHilbertEngine(
        X=X, A=A, p=14, max_leaf_size=max_leaf_size,
        primary_dim=4, comm=comm
    )
    comm.Barrier()
    hilbert_build_time = time.time() - t0

    hilbert_mem = engine_hilbert.get_rank_memory_footprint()
    all_hilbert_mem = comm.gather(hilbert_mem, root=0)

    # 4. Memory Footprint Comparison Across Ranks
    if rank == 0:
        print("\n[4] Distributed Memory Footprint Comparison Across MPI Ranks:")
        print(f"{'Component':<35} | {'Trad. Sharded HNSW':>20} | {'Hilbert B-Tree':>20} | {'Difference':>15}")
        print("-" * 98)

        total_sharded_mb = sum(m["total_mb"] for m in all_sharded_mem)
        total_hilbert_mb = sum(m["total_mb"] for m in all_hilbert_mem)
        total_sharded_graph = sum(m["graph_mb"] for m in all_sharded_mem)
        total_hilbert_graph = sum(m["graph_mb"] for m in all_hilbert_mem)
        total_sharded_vec = sum(m["vec_mb"] for m in all_sharded_mem)
        total_hilbert_vec = sum(m["vec_mb"] for m in all_hilbert_mem)
        total_sharded_aux = sum(m["aux_mb"] for m in all_sharded_mem)
        total_hilbert_aux = sum(m["aux_mb"] for m in all_hilbert_mem)

        print(f"{'Raw Vector Storage (float32)':<35} | {total_sharded_vec:>17.2f} MB | {total_hilbert_vec:>17.2f} MB | {total_hilbert_vec - total_sharded_vec:>12.2f} MB")
        print(f"{'Graph Index Edges (HNSW Links)':<35} | {total_sharded_graph:>17.2f} MB | {total_hilbert_graph:>17.2f} MB | {total_hilbert_graph - total_sharded_graph:>12.2f} MB")
        print(f"{'Auxiliary Data (Attrs, Keys, IDs)':<35} | {total_sharded_aux:>17.2f} MB | {total_hilbert_aux:>17.2f} MB | {total_hilbert_aux - total_sharded_aux:>12.2f} MB")
        print("-" * 98)
        print(f"{'Total Distributed Memory':<35} | {total_sharded_mb:>17.2f} MB | {total_hilbert_mb:>17.2f} MB | {total_hilbert_mb - total_sharded_mb:>12.2f} MB")
        print(f"{'Index Build Wall-Clock Time':<35} | {sharded_build_time:>18.2f}s | {hilbert_build_time:>18.2f}s | {hilbert_build_time - sharded_build_time:>13.2f}s")

    def calc_recall(found, truth):
        if len(truth) == 0:
            return 1.0
        return len(set(found.tolist()) & set(truth.tolist())) / len(truth)

    def make_query_box(sel, qi):
        side = np.sqrt(sel)
        lo = np.zeros(m, dtype=np.float32)
        hi = np.ones(m, dtype=np.float32)
        subspace_idx = qi % max(1, m // 2)
        a1 = subspace_idx * 2
        a2 = min(m - 1, a1 + 1)
        if a1 == a2:
            c = rng.random() * (1.0 - sel)
            lo[a1] = c
            hi[a1] = c + sel
        else:
            c = rng.random(2) * (1.0 - side)
            lo[a1] = c[0]
            hi[a1] = c[0] + side
            lo[a2] = c[1]
            hi[a2] = c[1] + side
        return lo, hi

    # 5. Distributed Filtered Search Comparison Sweep
    if rank == 0:
        print(f"\n[5] Distributed Filtered Query Performance across Selectivity Spectrum ({m} Attributes):")
        header = (f"{'Sel':>6} | "
                  f"{'Sharded Post (Rec / Lat)':>25} | "
                  f"{'Sharded Beam (Rec / Lat / Dist)':>31} | "
                  f"{'Hilbert B-Tree (Rec / Lat / Dist)':>33} | "
                  f"{'Active Ranks (Hil / Shd)':>24} | "
                  f"{'Speedup':>9}")
        print(header)
        print("-" * len(header))

    for sel in [0.005, 0.02, 0.05, 0.10, 0.25]:
        r_post, lat_post = [], []
        r_beam, lat_beam, nd_beam = [], [], []
        r_hil, lat_hil, nd_hil = [], [], []
        active_hil_list = []

        for qi in range(nq):
            if rank == 0:
                lo, hi = make_query_box(sel, qi)
                in_range = np.all((A >= lo) & (A <= hi), axis=1)
                valid_gids = np.where(in_range)[0]
                if len(valid_gids) < k:
                    lo, hi = None, None
                    truth = np.empty(0, dtype=np.int64)
                else:
                    diff = X[valid_gids] - Q[qi]
                    true_dists = np.einsum("ij,ij->i", diff, diff)
                    truth = valid_gids[np.argsort(true_dists)[:k]]
            else:
                lo, hi = None, None
                truth = None

            # 1. Sharded HNSW Post-Filtering
            comm.Barrier()
            t0 = time.perf_counter()
            ids_post, _, _ = engine_sharded.search(Q[qi], lo_raw=lo, hi_raw=hi, k=k, ef=ef, mode="post")
            t_post = (time.perf_counter() - t0) * 1000.0

            # 2. Sharded HNSW In-Graph Filtered Beam Search
            comm.Barrier()
            t0 = time.perf_counter()
            ids_beam, _, st_beam = engine_sharded.search(Q[qi], lo_raw=lo, hi_raw=hi, k=k, ef=ef, mode="beam")
            t_beam = (time.perf_counter() - t0) * 1000.0

            # 3. Distributed Hilbert B-Tree
            comm.Barrier()
            t0 = time.perf_counter()
            ids_hil, _, st_hil = engine_hilbert.search(Q[qi], lo_raw=lo, hi_raw=hi, k=k, ef=ef)
            t_hil = (time.perf_counter() - t0) * 1000.0

            if rank == 0 and len(truth) >= k:
                r_post.append(calc_recall(ids_post, truth))
                lat_post.append(t_post)

                r_beam.append(calc_recall(ids_beam, truth))
                lat_beam.append(t_beam)
                nd_beam.append(st_beam["ndis"])

                r_hil.append(calc_recall(ids_hil, truth))
                lat_hil.append(t_hil)
                nd_hil.append(st_hil["ndis"])
                active_hil_list.append(st_hil["active_ranks"])

        if rank == 0:
            post_str = f"{np.mean(r_post):.3f} / {np.mean(lat_post):5.2f}ms"
            beam_str = f"{np.mean(r_beam):.3f} / {np.mean(lat_beam):5.2f}ms / {np.mean(nd_beam):5.0f}"
            hil_str = f"{np.mean(r_hil):.3f} / {np.mean(lat_hil):5.2f}ms / {np.mean(nd_hil):5.0f}"
            ranks_str = f"{np.mean(active_hil_list):.1f} / {size}"
            sp = np.mean(lat_beam) / np.mean(lat_hil) if np.mean(lat_hil) > 0 else 1.0

            print(f"{sel:>6.3f} | {post_str:>25} | {beam_str:>31} | {hil_str:>33} | {ranks_str:>24} | {sp:>8.2f}x")

    # 6. Pure Unconstrained ANN Search Comparison
    if rank == 0:
        print("\n[6] Distributed Pure Unconstrained ANN Search (No Filters):")
        r_shd_u, lat_shd_u = [], []
        r_hil_u, lat_hil_u = [], []

    for qi in range(nq):
        if rank == 0:
            diff = X - Q[qi]
            true_dists = np.einsum("ij,ij->i", diff, diff)
            truth = np.argsort(true_dists)[:k]
        else:
            truth = None

        # Sharded unconstrained
        comm.Barrier()
        t0 = time.perf_counter()
        ids_shd, _, _ = engine_sharded.search(Q[qi], lo_raw=None, hi_raw=None, k=k, ef=ef)
        t_shd = (time.perf_counter() - t0) * 1000.0

        # Hilbert unconstrained
        comm.Barrier()
        t0 = time.perf_counter()
        ids_hil, _, _ = engine_hilbert.search(Q[qi], lo_raw=None, hi_raw=None, k=k, ef=ef)
        t_hil = (time.perf_counter() - t0) * 1000.0

        if rank == 0:
            r_shd_u.append(calc_recall(ids_shd, truth))
            lat_shd_u.append(t_shd)
            r_hil_u.append(calc_recall(ids_hil, truth))
            lat_hil_u.append(t_hil)

    if rank == 0:
        print(f"    Traditional Sharded HNSW : Recall@10 = {np.mean(r_shd_u):.3f} | Latency = {np.mean(lat_shd_u):.2f}ms")
        print(f"    Distributed Hilbert B-Tree: Recall@10 = {np.mean(r_hil_u):.3f} | Latency = {np.mean(lat_hil_u):.2f}ms")
        print("\n" + "=" * 115)
        print("Distributed Comparative Benchmark Completed Successfully!")
        print("=" * 115)

    engine_sharded.close()
    engine_hilbert.close()


if __name__ == "__main__":
    run_mpi_benchmark()
