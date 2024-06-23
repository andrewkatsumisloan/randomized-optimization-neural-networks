import mlrose_hiive as mlrose
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from mlrose_hiive import QueensGenerator, MaxKColorGenerator, TSPGenerator


# Create output directory if it doesn't exist
output_dir = "./output/kcolor/"
os.makedirs(output_dir, exist_ok=True)


# Run, track different algorithms
def run_algorithm(algorithm, problem, max_attempts, max_iters):
    start_time = time.time()
    if algorithm == "rhc":
        runner = mlrose.RHCRunner(
            problem=problem,
            experiment_name="RHC_Experiment",
            output_directory="./output/kcolor",
            seed=1,
            iteration_list=[max_iters],
            max_attempts=max_attempts,
            restart_list=[0],
        )
        df_run_stats, df_run_curves = runner.run()
        best_fitness = df_run_stats["Fitness"].iloc[-1]
        fitness_curve = df_run_curves["Fitness"]
    elif algorithm == "sa":
        # Define a custom exponential decay schedule
        schedule = mlrose.ExpDecay(init_temp=750, exp_const=0.045, min_temp=1e-10)

        runner = mlrose.SARunner(
            problem=problem,
            experiment_name="SA_Experiment",
            output_directory="./output/kcolor",
            seed=4534,
            iteration_list=[max_iters],
            max_attempts=max_attempts,
            temperature_list=[schedule],
        )
        df_run_stats, df_run_curves = runner.run()

        best_fitness = df_run_stats["Fitness"].max()
        fitness_curve = df_run_curves["Fitness"]

    elif algorithm == "ga":
        runner = mlrose.GARunner(
            problem=problem,
            experiment_name="GA_Experiment",
            output_directory="./output/kcolor",
            seed=52342,
            iteration_list=[max_iters],
            max_attempts=max_attempts,
            population_sizes=[300],
            mutation_rates=[0.25],
        )
        df_run_stats, df_run_curves = runner.run()
        best_fitness = df_run_stats["Fitness"].iloc[-1]
        fitness_curve = df_run_curves["Fitness"]
    else:
        raise ValueError("Unknown algorithm: {}".format(algorithm))

    execution_time = df_run_curves["Time"].iloc[-1]  # Use the last row's Time value
    return best_fitness, fitness_curve, execution_time


# Parameters
max_attempts = 125
max_iters = 12000

# 500 && 2000
# 125 && 12000, schedule = mlrose.ExpDecay(init_temp=750, exp_const=0.045, min_temp=1e-10),

# Test on different problem sizes
problem_sizes = [20, 50, 75, 90, 100, 150, 200, 500]
algorithms = ["rhc", "sa", "ga"]
results = {alg: [] for alg in algorithms}
curves = {alg: [] for alg in algorithms}
times = {alg: [] for alg in algorithms}
eval_counts = {alg: [] for alg in algorithms}

for size in problem_sizes:
    edges = [(i, (i + 1) % size) for i in range(size)]
    fitness = mlrose.MaxKColor(edges)
    problem = mlrose.DiscreteOpt(
        length=size, fitness_fn=fitness, maximize=True, max_val=5
    )

    for algorithm in algorithms:
        best_fitness, fitness_curve, elapsed_time = run_algorithm(
            algorithm, problem, max_attempts=max_attempts, max_iters=max_iters
        )
        results[algorithm].append(best_fitness)
        curves[algorithm].append(fitness_curve)
        times[algorithm].append(elapsed_time)
        eval_counts[algorithm].append(len(fitness_curve))
        print(
            f"Size: {size}, Algorithm: {algorithm}, Best Fitness: {best_fitness}, Time: {elapsed_time:.4f} seconds"
        )

# Plot fitness vs iterations for different problem sizes
# Visualize fitness score at each iteration
for i, size in enumerate(problem_sizes):
    plt.figure(figsize=(12, 8))
    for algorithm in algorithms:
        plt.plot(curves[algorithm][i], label=f"{algorithm}")
    plt.xlabel("Iterations")
    plt.ylabel("Fitness")
    plt.title(f"Fitness vs. Iterations for Problem Size {size}")
    plt.legend()
    plt.savefig(os.path.join(output_dir, f"fitness_vs_iterations_size_{size}_2.png"))
    plt.close()

# Plot fitness / problem size
# Focusing on a single size can misrepresent how well an algorithm is actually doing. To counteract this, try multiple problem sizes...
plt.figure(figsize=(12, 8))
markers = ["o-", "s--", "d-."]  # Different markers for each algorithm
for idx, algorithm in enumerate(algorithms):
    plt.plot(
        problem_sizes,
        results[algorithm],
        markers[idx],
        label=f"{algorithm} Best Fitness",
        markersize=8,
    )

plt.xlabel("Problem Size (Number of Nodes in Graph)")
plt.ylabel("Best Fitness Achieved")
plt.title("Comparison of Optimization Algorithms Across Different Problem Sizes")
plt.grid(True)
plt.legend()
plt.xticks(problem_sizes)  # Ensure all problem sizes are marked

# Annotate each point with its corresponding fitness value
for algorithm in algorithms:
    for i, size in enumerate(problem_sizes):
        plt.text(
            size,
            results[algorithm][i],
            f"{results[algorithm][i]:.2f}",
            ha="center",
            va="bottom",
        )

plt.savefig(os.path.join(output_dir, "enhanced_fitness_vs_problem_size_2.png"))
plt.show()
plt.close()

# Plot Function Evaluations vs. Problem Size
plt.figure(figsize=(12, 8))
for algorithm in algorithms:
    plt.plot(problem_sizes, eval_counts[algorithm], label=algorithm)
plt.xlabel("Problem Size")
plt.ylabel("Function Evaluations")
plt.title("Function Evaluations vs. Problem Size")
plt.legend()
plt.savefig(os.path.join(output_dir, "function_evaluations_vs_problem_size_2.png"))
plt.close()

# Plot Wall Clock Time vs. Problem Size
plt.figure(figsize=(12, 8))
for algorithm in algorithms:
    plt.plot(problem_sizes, times[algorithm], label=algorithm)
plt.xlabel("Problem Size")
plt.ylabel("Wall Clock Time (seconds)")
plt.title("Wall Clock Time vs. Problem Size")
plt.legend()
plt.savefig(os.path.join(output_dir, "wall_clock_time_vs_problem_size_2.png"))
plt.close()
