import time
import math
from multiprocessing import Pool
import matplotlib.pyplot as plt

def lorenz_rhs(x, y, z, sigma, rho, beta):
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return dx, dy, dz


def integrate_lorenz(sigma, rho, beta, x0, y0, z0, t_start, t_end, dt):
    steps = int((t_end - t_start) / dt)
    x = x0
    y = y0
    z = z0
    for _ in range(steps):
        dx1, dy1, dz1 = lorenz_rhs(x, y, z, sigma, rho, beta)
        kx1 = dt * dx1
        ky1 = dt * dy1
        kz1 = dt * dz1

        dx2, dy2, dz2 = lorenz_rhs(
            x + 0.5 * kx1, y + 0.5 * ky1, z + 0.5 * kz1, sigma, rho, beta
        )
        kx2 = dt * dx2
        ky2 = dt * dy2
        kz2 = dt * dz2

        dx3, dy3, dz3 = lorenz_rhs(
            x + 0.5 * kx2, y + 0.5 * ky2, z + 0.5 * kz2, sigma, rho, beta
        )
        kx3 = dt * dx3
        ky3 = dt * dy3
        kz3 = dt * dz3

        dx4, dy4, dz4 = lorenz_rhs(
            x + kx3, y + ky3, z + kz3, sigma, rho, beta
        )
        kx4 = dt * dx4
        ky4 = dt * dy4
        kz4 = dt * dz4

        x = x + (kx1 + 2.0 * kx2 + 2.0 * kx3 + kx4) / 6.0
        y = y + (ky1 + 2.0 * ky2 + 2.0 * ky3 + ky4) / 6.0
        z = z + (kz1 + 2.0 * kz2 + 2.0 * kz3 + kz4) / 6.0
    return x, y, z


def run_one_system(args):
    sigma, rho, beta, x0, y0, z0, t_start, t_end, dt = args
    return integrate_lorenz(sigma, rho, beta, x0, y0, z0, t_start, t_end, dt)


def run_sequential(task_list):
    start_time = time.perf_counter()
    results = []
    for args in task_list:
        results.append(run_one_system(args))
    end_time = time.perf_counter()
    return results, end_time - start_time


def run_parallel_with_pool(task_list, pool, process_count):
    start_time = time.perf_counter()
    task_len = len(task_list)
    if task_len == 0:
        return [], 0.0
    chunk_size = max(1, task_len // (process_count * 4))
    results = pool.map(run_one_system, task_list, chunksize=chunk_size)
    end_time = time.perf_counter()
    return results, end_time - start_time


def max_difference(list_a, list_b):
    max_diff = 0.0
    for (xa, ya, za), (xb, yb, zb) in zip(list_a, list_b):
        dx = xa - xb
        dy = ya - yb
        dz = za - zb
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        if d > max_diff:
            max_diff = d
    return max_diff


def main():
    sigma_value = 10.0
    beta_value = 8.0 / 3.0
    base_rho = 20.0
    t_start = 0.0
    t_end = 80.0
    dt = 0.01
    initial_state = (1.0, 1.0, 1.0)

    system_counts = [16, 32, 64, 128, 256]
    process_options = [2, 4]

    seq_times = []
    par_times_2 = []
    par_times_4 = []
    speedups_2 = []
    speedups_4 = []

    pools = {p: Pool(p) for p in process_options}
    try:
        for system_count in system_counts:
            rho_values = [base_rho + i for i in range(system_count)]
            tasks = []
            for rho in rho_values:
                tasks.append(
                    (
                        sigma_value,
                        rho,
                        beta_value,
                        initial_state[0],
                        initial_state[1],
                        initial_state[2],
                        t_start,
                        t_end,
                        dt,
                    )
                )

            seq_results, seq_time = run_sequential(tasks)
            seq_times.append(seq_time)
            print(f"\nsystems={system_count}, sequential time={seq_time:.3f}s")

            par_time_for_2 = None
            par_time_for_4 = None
            speedup_for_2 = None
            speedup_for_4 = None

            for procs in process_options:
                pool = pools[procs]
                par_results, par_time = run_parallel_with_pool(
                    tasks, pool, procs
                )
                speedup = seq_time / par_time
                diff = max_difference(seq_results, par_results)
                print(
                    f"  processes={procs}, parallel time={par_time:.3f}s, "
                    f"speedup={speedup:.2f}x, max_diff={diff:.3e}"
                )
                if procs == 2:
                    par_time_for_2 = par_time
                    speedup_for_2 = speedup
                elif procs == 4:
                    par_time_for_4 = par_time
                    speedup_for_4 = speedup

            par_times_2.append(par_time_for_2)
            par_times_4.append(par_time_for_4)
            speedups_2.append(speedup_for_2)
            speedups_4.append(speedup_for_4)
    finally:
        for pool in pools.values():
            pool.close()
        for pool in pools.values():
            pool.join()

    plt.figure()
    plt.plot(system_counts, seq_times, marker="o", label="Sequential")
    plt.plot(system_counts, par_times_2, marker="o", label="Parallel (2 processes)")
    plt.plot(system_counts, par_times_4, marker="o", label="Parallel (4 processes)")
    plt.xlabel("Number of systems")
    plt.ylabel("Runtime (s)")
    plt.title("Sequential vs Parallel runtimes")
    plt.legend()
    plt.grid(True)
    plt.savefig("lorenz_seq_vs_par.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("\nSummary table (times in seconds):")
    print("systems  seq_time   par2_time  speedup2   par4_time  speedup4")
    for i, systems in enumerate(system_counts):
        print(
            f"{systems:7d}  "
            f"{seq_times[i]:8.3f}  "
            f"{par_times_2[i]:9.3f}  "
            f"{speedups_2[i]:8.2f}x  "
            f"{par_times_4[i]:9.3f}  "
            f"{speedups_4[i]:8.2f}x"
        )


if __name__ == "__main__":
    main()
