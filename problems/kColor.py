import mlrose_hiive as mlrose
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from mlrose_hiive import QueensGenerator, MaxKColorGenerator


# Create output directory if it doesn't exist
output_dir = "./output/kcolor/"
os.makedirs(output_dir, exist_ok=True)


# Run, track different algorithms
def run_algorithm(algorithm, problem, max_attempts, max_iters):
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
        best_fitness = df_run_stats["Fitness"].max()

    elif algorithm == "sa":
        exp_decay_schedule = mlrose.ExpDecay(
            init_temp=525, exp_const=0.5, min_temp=1e-12
        )

        runner = mlrose.SARunner(
            problem=problem,
            experiment_name="SA_Experiment",
            output_directory="./output/k_color",
            seed=1,
            iteration_list=[max_iters],
            max_attempts=max_attempts,
            # temperature_list=[0.1, 0.5, 0.75, 1.0, 2.0, 10.0],
            # decay_list=[
            #     mlrose.ExpDecay,
            # ],
            # temperature_list=[mlrose.ExpDecay()],
            temperature_list=[exp_decay_schedule],
            # decay_list=[mlrose.GeomDecay],
        )

        df_run_stats, df_run_curves = runner.run()
        best_fitness = df_run_stats["Fitness"].iloc[-1]
        fitness_curve = df_run_curves["Fitness"]
    elif algorithm == "ga":
        runner = mlrose.GARunner(
            problem=problem,
            experiment_name="GA_Experiment",
            output_directory="./output/kcolor",
            seed=1,
            iteration_list=[max_iters],
            max_attempts=max_attempts,
            population_sizes=[150, 200, 250],
            mutation_rates=[0.1, 0.15],
        )
        df_run_stats, df_run_curves = runner.run()
        best_fitness = df_run_stats["Fitness"].iloc[-1]
        fitness_curve = df_run_curves["Fitness"]
        best_fitness = df_run_stats["Fitness"].max()

    else:
        raise ValueError("Unknown algorithm: {}".format(algorithm))

    execution_time = df_run_curves["Time"].iloc[-1]  # Use the last row's Time value
    return best_fitness, fitness_curve, execution_time


# Parameters
max_attempts = 750
max_iters = 4000

# Test on different problem sizes
problem_sizes = [20, 50, 100, 150]
algorithms = ["rhc", "sa", "ga"]
results = {alg: [] for alg in algorithms}
curves = {alg: [] for alg in algorithms}
times = {alg: [] for alg in algorithms}
eval_counts = {alg: [] for alg in algorithms}

for size in problem_sizes:
    # Define edges for a simple ring graph (or customize as needed)
    edges = [(i, (i + 1) % size) for i in range(size)]  # Simple ring graph
    max_colors = 5  # Set the maximum number of colors
    fitness = mlrose.MaxKColor(edges)
    problem = mlrose.DiscreteOpt(
        length=size, fitness_fn=fitness, maximize=True, max_val=max_colors
    )

    # problem = MaxKColorGenerator().generate(
    #     seed=123456,
    #     number_of_nodes=10,
    #     max_connections_per_node=3,
    #     max_colors=3,
    # )

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
    plt.savefig(os.path.join(output_dir, f"fitness_vs_iterations_size_{size}.png"))
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

plt.savefig(os.path.join(output_dir, "enhanced_fitness_vs_problem_size.png"))
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
plt.savefig(os.path.join(output_dir, "function_evaluations_vs_problem_size.png"))
plt.close()

# Plot Wall Clock Time vs. Problem Size
plt.figure(figsize=(12, 8))
for algorithm in algorithms:
    plt.plot(problem_sizes, times[algorithm], label=algorithm)
plt.xlabel("Problem Size")
plt.ylabel("Wall Clock Time (seconds)")
plt.title("Wall Clock Time vs. Problem Size")
plt.legend()
plt.savefig(os.path.join(output_dir, "wall_clock_time_vs_problem_size.png"))
plt.close()
