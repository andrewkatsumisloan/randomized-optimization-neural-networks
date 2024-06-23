import mlrose_hiive as mlrose
import numpy as np
import time
import os
import matplotlib.pyplot as plt
from mlrose_hiive import QueensGenerator


# Create output directory if it doesn't exist
output_dir = "./output/nqueens/"
os.makedirs(output_dir, exist_ok=True)


# Run, track different algorithms
def run_algorithm(algorithm, problem, max_attempts=100, max_iters=1000):
    start_time = time.time()
    if algorithm == "rhc":
        runner = mlrose.RHCRunner(
            problem=problem,
            experiment_name="RHC_Experiment",
            output_directory="./output/nqueens",
            seed=25,
            iteration_list=[max_iters],
            max_attempts=max_attempts,
            restart_list=[1, 5, 10, 15, 20, 30],
        )
        df_run_stats, df_run_curves = runner.run()
        best_fitness = df_run_stats["Fitness"].iloc[-1]
        fitness_curve = df_run_curves["Fitness"]
    elif algorithm == "sa":
        exp_decay_schedule = mlrose.ExpDecay(
            init_temp=150, exp_const=0.15, min_temp=1e-10
        )

        runner = mlrose.SARunner(
            problem=problem,
            experiment_name="SA_Experiment",
            output_directory="./output/four_peaks",
            seed=25,
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
            output_directory="./output/nqueens",
            seed=25,
            iteration_list=[max_iters],
            max_attempts=max_attempts,
            population_sizes=[200],
            mutation_rates=[0.01],
        )
        df_run_stats, df_run_curves = runner.run()
        best_fitness = df_run_stats["Fitness"].iloc[-1]
        fitness_curve = df_run_curves["Fitness"]
    else:
        raise ValueError("Unknown algorithm: {}".format(algorithm))
    end_time = time.time()

    return best_fitness, fitness_curve, end_time - start_time


# Parameters
max_attempts = 500  # If the fitness remains static for max_attempts iterations, it will terminate that run
max_iters = 1500

# Test on different problem sizes
problem_sizes = [8, 16, 32, 64]
algorithms = ["rhc", "sa", "ga"]
results = {alg: [] for alg in algorithms}
curves = {alg: [] for alg in algorithms}
times = {alg: [] for alg in algorithms}
eval_counts = {alg: [] for alg in algorithms}

for size in problem_sizes:
    fitness = mlrose.Queens()
    problem = mlrose.DiscreteOpt(
        length=size,
        fitness_fn=fitness,
        maximize=True,
        max_val=size,
    )
    # problem = QueensGenerator().generate(seed=123456, size=size, maximize=True)

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


plt.xlabel("Problem Size")
plt.ylabel("Best Fitness Achieved")
plt.title("Optimization Algorithm Performance on N Queens Problem")
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
            fontsize=9,
            color="black",
        )

plt.savefig(os.path.join(output_dir, "enhanced_fitness_vs_problem_size_nQueens.png"))
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
