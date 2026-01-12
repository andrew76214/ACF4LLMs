"""Advanced Search Agent for strategy optimization using various search algorithms."""

import json
import random
import numpy as np
from typing import Dict, Optional, Any, List, Tuple
from datetime import datetime
from pathlib import Path
from langchain_core.tools import tool
import logging

logger = logging.getLogger(__name__)


@tool
def bayesian_optimization_search(
    objective_function: str,
    search_space: Dict[str, Any],
    num_iterations: int = 20,
    exploration_weight: float = 1.0,
    history: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """Perform Bayesian optimization for hyperparameter search.

    Args:
        objective_function: Name of objective to optimize
        search_space: Parameter search space
        num_iterations: Number of search iterations
        exploration_weight: UCB exploration weight
        history: Previous evaluation history

    Returns:
        Dictionary with next parameters to try
    """
    print(f"[Bayesian] Optimizing {objective_function} over {num_iterations} iterations")

    try:
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import Matern
        from scipy.stats import norm

        # Initialize Gaussian Process
        kernel = Matern(length_scale=1.0, nu=2.5)
        gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)

        # Parse history if available
        if history and len(history) > 0:
            X_observed = []
            y_observed = []

            for h in history:
                params = _extract_params_vector(h["parameters"], search_space)
                score = h.get("score", 0.5)
                X_observed.append(params)
                y_observed.append(score)

            X_observed = np.array(X_observed)
            y_observed = np.array(y_observed)

            # Fit GP to observed data
            gp.fit(X_observed, y_observed)

            # Acquisition function (Upper Confidence Bound)
            def acquisition(x):
                x = x.reshape(1, -1)
                mu, sigma = gp.predict(x, return_std=True)
                return mu + exploration_weight * sigma

            # Find next point to evaluate
            best_x = None
            best_acq = -np.inf

            for _ in range(100):  # Random search for acquisition maximum
                x_candidate = _sample_from_space(search_space)
                acq_value = acquisition(np.array(x_candidate))
                if acq_value > best_acq:
                    best_acq = acq_value
                    best_x = x_candidate

            next_params = _vector_to_params(best_x, search_space)
        else:
            # No history, sample randomly
            next_params = _sample_params_dict(search_space)

    except ImportError:
        logger.warning("scikit-learn not available, using random search")
        next_params = _sample_params_dict(search_space)

    print(f"[Bayesian] Next parameters to try: {next_params}")

    return {
        "search_method": "bayesian_optimization",
        "next_parameters": next_params,
        "iteration": len(history) + 1 if history else 1,
        "exploration_weight": exploration_weight,
        "search_space": search_space,
    }


@tool
def evolutionary_search(
    population_size: int = 20,
    num_generations: int = 10,
    mutation_rate: float = 0.1,
    crossover_rate: float = 0.7,
    search_space: Dict[str, Any] = None,
    fitness_history: Optional[List[Dict]] = None,
    fitness_weights: Optional[Dict[str, float]] = None,
    current_generation: int = 0,
) -> Dict[str, Any]:
    """Perform evolutionary search for strategy optimization.

    This function supports iterative evolution with real fitness values from
    external evaluation. Each call can advance one or more generations.

    The fitness_history should contain entries with either:
    - "fitness": Pre-computed fitness value, OR
    - "result": Evaluation result dict with accuracy, latency_ms, memory_gb
      (fitness will be computed automatically using _compute_fitness)

    Args:
        population_size: Size of population
        num_generations: Number of generations to evolve
        mutation_rate: Mutation probability per individual
        crossover_rate: Crossover probability between parents
        search_space: Parameter search space definition
        fitness_history: Previous evaluations with fitness or results.
            Each entry should have "parameters" and either "fitness" or "result".
        fitness_weights: Optional weights for multi-objective fitness.
            Keys: "accuracy" (default 1.0), "latency" (default 0.3), "memory" (default 0.2)
        current_generation: Current generation number for multi-call evolution

    Returns:
        Dictionary with:
        - search_method: "evolutionary"
        - best_individual: Best parameter configuration found
        - best_fitness: Best fitness score
        - candidates_to_evaluate: New individuals needing evaluation
        - final_population: Current population after evolution
        - current_generation: Generation number after this call
        - is_complete: Whether evolution is complete (all generations done)
    """
    print(f"[Evolution] Running evolutionary search (gen {current_generation + 1}-{current_generation + num_generations})")

    if not search_space:
        search_space = _get_default_compression_search_space()

    # Compute fitness from results if not already provided
    if fitness_history:
        for h in fitness_history:
            if "fitness" not in h and "result" in h:
                h["fitness"] = _compute_fitness(h["result"], fitness_weights)
            elif "fitness" not in h:
                # No fitness and no result - assign neutral fitness
                h["fitness"] = 0.5

    # Initialize population from history or randomly
    if fitness_history and len(fitness_history) >= population_size:
        # Use most recent history entries as initial population
        recent_history = fitness_history[-population_size:]
        population = [h["parameters"] for h in recent_history]
        fitness = [h["fitness"] for h in recent_history]
        has_real_fitness = True
    elif fitness_history and len(fitness_history) > 0:
        # Partial history - use what we have and fill with random
        population = [h["parameters"] for h in fitness_history]
        fitness = [h["fitness"] for h in fitness_history]
        # Fill remaining with random individuals
        while len(population) < population_size:
            population.append(_sample_params_dict(search_space))
            fitness.append(0.5)  # Neutral fitness for unevaluated
        has_real_fitness = True
    else:
        # No history - generate initial population for evaluation
        population = [_sample_params_dict(search_space) for _ in range(population_size)]
        print(f"[Evolution] Generated initial population of {population_size} candidates")
        return {
            "search_method": "evolutionary",
            "best_individual": population[0],
            "best_fitness": 0.0,
            "candidates_to_evaluate": population,
            "final_population": population,
            "current_generation": 0,
            "is_complete": False,
            "message": "Initial population generated. Please evaluate and provide fitness_history.",
        }

    best_individual = population[np.argmax(fitness)]
    best_fitness = max(fitness)

    print(f"[Evolution] Starting with best fitness: {best_fitness:.3f}")

    # Evolution loop
    for gen in range(num_generations):
        # Evolve one generation using helper function
        population = _evolve_one_generation(
            population=population,
            fitness=fitness,
            search_space=search_space,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
        )

        # For multi-generation evolution within one call, we need fitness
        # for intermediate generations. Use the last known fitness as proxy
        # since actual fitness requires external evaluation.
        # The final candidates will be returned for real evaluation.
        if gen < num_generations - 1:
            # Intermediate generation - estimate fitness from similarity to best
            # This is a heuristic; real fitness comes from external evaluation
            fitness = []
            for ind in population:
                # Similarity-based proxy fitness
                similarity = sum(
                    1 for k in ind if k in best_individual and ind[k] == best_individual[k]
                ) / max(len(ind), 1)
                fitness.append(best_fitness * (0.8 + 0.2 * similarity))

    final_generation = current_generation + num_generations

    print(f"[Evolution] Completed generation {final_generation}")
    print(f"[Evolution] Best fitness so far: {best_fitness:.3f}")

    return {
        "search_method": "evolutionary",
        "best_individual": best_individual,
        "best_fitness": best_fitness,
        "candidates_to_evaluate": population,
        "final_population": population,
        "population_size": population_size,
        "current_generation": final_generation,
        "is_complete": final_generation >= num_generations,
    }


@tool
def multi_armed_bandit_search(
    arms: List[str],
    rewards_history: Optional[Dict[str, List[float]]] = None,
    exploration_strategy: str = "ucb",
    exploration_param: float = 2.0,
) -> Dict[str, Any]:
    """Use multi-armed bandit for strategy selection.

    Args:
        arms: List of available strategies/methods
        rewards_history: Historical rewards for each arm
        exploration_strategy: Strategy ('ucb', 'thompson', 'epsilon_greedy')
        exploration_param: Exploration parameter

    Returns:
        Dictionary with selected arm and statistics
    """
    print(f"[Bandit] Selecting strategy using {exploration_strategy}")

    if not rewards_history:
        rewards_history = {arm: [] for arm in arms}

    # Calculate statistics for each arm
    arm_stats = {}
    for arm in arms:
        rewards = rewards_history.get(arm, [])
        if rewards:
            arm_stats[arm] = {
                "mean": np.mean(rewards),
                "std": np.std(rewards),
                "count": len(rewards),
                "total_reward": sum(rewards),
            }
        else:
            arm_stats[arm] = {
                "mean": 0.5,  # Optimistic initialization
                "std": 1.0,
                "count": 0,
                "total_reward": 0,
            }

    # Select arm based on strategy
    if exploration_strategy == "ucb":
        # Upper Confidence Bound
        total_pulls = sum(s["count"] for s in arm_stats.values())
        ucb_scores = {}

        for arm, stats in arm_stats.items():
            if stats["count"] == 0:
                ucb_scores[arm] = float("inf")  # Explore unplayed arms
            else:
                exploration_bonus = exploration_param * np.sqrt(
                    np.log(total_pulls + 1) / stats["count"]
                )
                ucb_scores[arm] = stats["mean"] + exploration_bonus

        selected_arm = max(ucb_scores, key=ucb_scores.get)

    elif exploration_strategy == "thompson":
        # Thompson Sampling
        samples = {}
        for arm, stats in arm_stats.items():
            # Beta distribution for Bernoulli rewards
            alpha = stats["total_reward"] + 1
            beta = stats["count"] - stats["total_reward"] + 1
            samples[arm] = np.random.beta(alpha, beta)

        selected_arm = max(samples, key=samples.get)

    elif exploration_strategy == "epsilon_greedy":
        # Epsilon-greedy
        epsilon = exploration_param
        if random.random() < epsilon:
            selected_arm = random.choice(arms)
        else:
            selected_arm = max(arm_stats, key=lambda x: arm_stats[x]["mean"])
    else:
        # Random selection
        selected_arm = random.choice(arms)

    print(f"[Bandit] Selected arm: {selected_arm}")
    if arm_stats[selected_arm]["count"] > 0:
        print(f"[Bandit] Historical performance: {arm_stats[selected_arm]['mean']:.3f}")

    return {
        "search_method": "multi_armed_bandit",
        "selected_arm": selected_arm,
        "arm_statistics": arm_stats,
        "exploration_strategy": exploration_strategy,
        "exploration_param": exploration_param,
        "total_trials": sum(s["count"] for s in arm_stats.values()),
    }


@tool
def random_search(
    search_space: Dict[str, Any],
    num_samples: int = 10,
    constraints: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Perform random search over parameter space.

    Args:
        search_space: Parameter search space
        num_samples: Number of random samples
        constraints: Optional constraints on parameters

    Returns:
        Dictionary with sampled parameters
    """
    print(f"[Random] Sampling {num_samples} random configurations")

    samples = []
    for _ in range(num_samples):
        sample = _sample_params_dict(search_space)

        # Apply constraints if any
        if constraints:
            sample = _apply_constraints(sample, constraints)

        samples.append(sample)

    return {
        "search_method": "random",
        "samples": samples,
        "num_samples": num_samples,
        "search_space": search_space,
        "constraints": constraints,
    }


@tool
def grid_search(
    search_space: Dict[str, List],
    max_combinations: int = 100,
) -> Dict[str, Any]:
    """Perform grid search over discrete parameter space.

    Args:
        search_space: Parameter grid (each param has list of values)
        max_combinations: Maximum combinations to try

    Returns:
        Dictionary with grid search configurations
    """
    print(f"[Grid] Performing grid search")

    # Calculate total combinations
    import itertools

    param_names = list(search_space.keys())
    param_values = [search_space[name] for name in param_names]

    # Generate all combinations
    all_combinations = list(itertools.product(*param_values))
    total_combinations = len(all_combinations)

    print(f"[Grid] Total combinations: {total_combinations}")

    # Limit if too many
    if total_combinations > max_combinations:
        print(f"[Grid] Limiting to {max_combinations} combinations")
        indices = random.sample(range(total_combinations), max_combinations)
        selected_combinations = [all_combinations[i] for i in indices]
    else:
        selected_combinations = all_combinations

    # Convert to dict format
    configurations = []
    for combo in selected_combinations:
        config = {name: value for name, value in zip(param_names, combo)}
        configurations.append(config)

    return {
        "search_method": "grid",
        "configurations": configurations,
        "total_combinations": total_combinations,
        "evaluated_combinations": len(configurations),
        "search_space": search_space,
    }


@tool
def analyze_search_history(
    history: List[Dict[str, Any]],
    objective: str = "accuracy",
) -> Dict[str, Any]:
    """Analyze search history to find patterns and best strategies.

    Args:
        history: List of previous search results
        objective: Objective to analyze

    Returns:
        Dictionary with analysis results
    """
    print(f"[Analysis] Analyzing {len(history)} historical results")

    if not history:
        return {"error": "No history to analyze"}

    # Extract scores
    scores = [h.get(objective, 0) for h in history]

    # Find best configuration
    best_idx = np.argmax(scores)
    best_config = history[best_idx]
    best_score = scores[best_idx]

    # Analyze parameter importance (simplified)
    param_importance = {}
    if len(history) > 10:
        # Get all parameter names
        all_params = set()
        for h in history:
            if "parameters" in h:
                all_params.update(h["parameters"].keys())

        for param in all_params:
            # Calculate correlation with objective
            param_values = []
            param_scores = []

            for h in history:
                if "parameters" in h and param in h["parameters"]:
                    try:
                        # Try to convert to float for correlation
                        val = float(h["parameters"][param]) if isinstance(h["parameters"][param], (int, float)) else 0
                        param_values.append(val)
                        param_scores.append(h.get(objective, 0))
                    except:
                        continue

            if len(param_values) > 2:
                correlation = np.corrcoef(param_values, param_scores)[0, 1]
                param_importance[param] = abs(correlation)

    # Convergence analysis
    convergence = {
        "best_score_progression": [],
        "is_converging": False,
    }

    current_best = -np.inf
    for score in scores:
        current_best = max(current_best, score)
        convergence["best_score_progression"].append(current_best)

    # Check if converging (best score hasn't improved in last 20% of iterations)
    if len(scores) > 10:
        recent_start = int(len(scores) * 0.8)
        recent_best = max(scores[recent_start:])
        overall_best = max(scores)
        convergence["is_converging"] = (overall_best - recent_best) < 0.001

    print(f"[Analysis] Best score: {best_score:.3f}")
    print(f"[Analysis] Converging: {convergence['is_converging']}")

    return {
        "num_evaluations": len(history),
        "best_configuration": best_config,
        "best_score": best_score,
        "score_statistics": {
            "mean": np.mean(scores),
            "std": np.std(scores),
            "min": min(scores),
            "max": max(scores),
            "median": np.median(scores),
        },
        "parameter_importance": param_importance,
        "convergence": convergence,
        "top_5_configurations": sorted(history, key=lambda x: x.get(objective, 0), reverse=True)[:5],
    }


def _get_default_compression_search_space() -> Dict[str, Any]:
    """Get default search space for compression strategies."""
    return {
        "quantization_bits": {"type": "choice", "values": [4, 8]},
        "quantization_method": {"type": "choice", "values": ["autoround", "gptq", "awq"]},
        "pruning_ratio": {"type": "continuous", "min": 0.1, "max": 0.8},
        "lora_rank": {"type": "choice", "values": [8, 16, 32, 64]},
        "learning_rate": {"type": "log_uniform", "min": 1e-5, "max": 1e-3},
        "distillation_temperature": {"type": "continuous", "min": 1.0, "max": 10.0},
        "distillation_alpha": {"type": "continuous", "min": 0.3, "max": 0.7},
    }


def _sample_params_dict(search_space: Dict[str, Any]) -> Dict[str, Any]:
    """Sample parameters from search space."""
    params = {}
    for name, spec in search_space.items():
        params[name] = _sample_param_value(spec)
    return params


def _sample_param_value(spec: Any):
    """Sample a single parameter value."""
    if isinstance(spec, dict):
        if spec.get("type") == "choice":
            return random.choice(spec["values"])
        elif spec.get("type") == "continuous":
            return random.uniform(spec["min"], spec["max"])
        elif spec.get("type") == "integer":
            return random.randint(spec["min"], spec["max"])
        elif spec.get("type") == "log_uniform":
            log_min = np.log(spec["min"])
            log_max = np.log(spec["max"])
            return np.exp(random.uniform(log_min, log_max))
    elif isinstance(spec, list):
        return random.choice(spec)
    else:
        return spec


def _extract_params_vector(params: Dict, search_space: Dict) -> List[float]:
    """Convert parameter dict to vector for Gaussian Process."""
    vector = []
    for name, spec in search_space.items():
        if name in params:
            value = params[name]
            # Convert to numeric
            if isinstance(value, (int, float)):
                vector.append(float(value))
            elif isinstance(value, str) and "values" in spec:
                # Categorical -> one-hot or index
                vector.append(spec["values"].index(value) if value in spec["values"] else 0)
            else:
                vector.append(0)
    return vector


def _vector_to_params(vector: List[float], search_space: Dict) -> Dict:
    """Convert vector back to parameter dict."""
    params = {}
    idx = 0
    for name, spec in search_space.items():
        if idx < len(vector):
            if spec.get("type") == "choice":
                # Convert index back to choice
                index = int(vector[idx]) % len(spec["values"])
                params[name] = spec["values"][index]
            else:
                params[name] = vector[idx]
            idx += 1
    return params


def _sample_from_space(search_space: Dict) -> List[float]:
    """Sample a vector from search space."""
    params = _sample_params_dict(search_space)
    return _extract_params_vector(params, search_space)


def _apply_constraints(params: Dict, constraints: Dict) -> Dict:
    """Apply constraints to parameters."""
    # Example constraints: {"quantization_bits": lambda x: x in [4, 8]}
    for param, constraint in constraints.items():
        if param in params and callable(constraint):
            if not constraint(params[param]):
                # Resample if constraint violated
                # This is simplified; in practice, would need proper handling
                pass
    return params


def _compute_fitness(
    result: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None
) -> float:
    """Compute weighted fitness score from evaluation result.

    Combines multiple objectives (accuracy, latency, memory) into a single
    scalar fitness value for evolutionary optimization.

    Args:
        result: Evaluation result dictionary with keys like:
            - accuracy or average_accuracy: Model accuracy (0-1, higher is better)
            - latency_ms: Inference latency in milliseconds (lower is better)
            - memory_gb: Memory usage in GB (lower is better)
        weights: Optional weight dict with keys "accuracy", "latency", "memory".
            Defaults to {"accuracy": 1.0, "latency": 0.3, "memory": 0.2}

    Returns:
        Fitness score (higher is better), typically in range 0-1.5
    """
    if weights is None:
        weights = {
            "accuracy": 1.0,
            "latency": 0.3,
            "memory": 0.2,
        }

    # Extract metrics with sensible defaults
    accuracy = result.get("accuracy", result.get("average_accuracy", 0.5))
    latency_ms = result.get("latency_ms", 100.0)
    memory_gb = result.get("memory_gb", 8.0)

    # Normalize: accuracy already 0-1
    # For latency and memory, lower is better, so we invert
    # Using sigmoid-like transformation for bounded output
    latency_score = 1.0 / (1.0 + latency_ms / 100.0)  # 100ms -> 0.5, 50ms -> 0.67
    memory_score = 1.0 / (1.0 + memory_gb / 8.0)      # 8GB -> 0.5, 4GB -> 0.67

    fitness = (
        weights.get("accuracy", 1.0) * accuracy +
        weights.get("latency", 0.3) * latency_score +
        weights.get("memory", 0.2) * memory_score
    )

    return fitness


def _evolve_one_generation(
    population: List[Dict],
    fitness: List[float],
    search_space: Dict[str, Any],
    crossover_rate: float = 0.7,
    mutation_rate: float = 0.1,
) -> List[Dict]:
    """Perform one generation of evolution (selection, crossover, mutation).

    Args:
        population: Current population of parameter dictionaries
        fitness: Fitness scores for each individual
        search_space: Parameter search space definition
        crossover_rate: Probability of crossover between parents
        mutation_rate: Probability of mutation per individual

    Returns:
        New population after evolution
    """
    population_size = len(population)
    new_population = []

    # Tournament selection
    for _ in range(population_size):
        tournament_size = min(3, population_size)
        tournament_indices = random.sample(range(population_size), tournament_size)
        tournament_fitness = [fitness[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        new_population.append(population[winner_idx].copy())

    # Crossover
    for i in range(0, population_size - 1, 2):
        if random.random() < crossover_rate:
            for param in search_space:
                if param in new_population[i] and param in new_population[i + 1]:
                    if random.random() < 0.5:
                        new_population[i][param], new_population[i + 1][param] = \
                            new_population[i + 1][param], new_population[i][param]

    # Mutation
    for individual in new_population:
        if random.random() < mutation_rate:
            param_to_mutate = random.choice(list(search_space.keys()))
            individual[param_to_mutate] = _sample_param_value(search_space[param_to_mutate])

    return new_population


def get_search_subagent(spec: Dict[str, Any]) -> Dict[str, Any]:
    """Create the search subagent configuration.

    Args:
        spec: Model specification

    Returns:
        Subagent configuration dictionary
    """
    model_name = spec.get("model_name", "unknown")

    prompt = f"""You are a Search Optimization Agent for finding optimal compression strategies.

Model: {model_name}

Your responsibilities:
1. Use Bayesian optimization for continuous hyperparameters
2. Apply evolutionary algorithms for complex strategy spaces
3. Use multi-armed bandits for method selection
4. Perform grid/random search when appropriate
5. Analyze search history to guide optimization

Available tools:
- bayesian_optimization_search: Efficient hyperparameter optimization
- evolutionary_search: Population-based strategy evolution
- multi_armed_bandit_search: Adaptive method selection
- random_search: Simple random sampling
- grid_search: Exhaustive grid evaluation
- analyze_search_history: Extract insights from history

Search Strategy:
1. Start with multi-armed bandit to select promising methods
2. Use Bayesian optimization for method hyperparameters
3. Apply evolutionary search for complex multi-method strategies
4. Analyze history to identify patterns
5. Focus on promising regions of search space

Guidelines:
- Early exploration: Use random/grid search
- After 5-10 evaluations: Switch to Bayesian optimization
- For discrete choices: Use multi-armed bandits
- For multi-objective: Use evolutionary algorithms
- Monitor convergence and adjust strategy

When you receive a search request:
1. Analyze the search space complexity
2. Choose appropriate search method
3. Consider historical results
4. Generate next configuration(s)
5. Report search progress and insights
"""

    return {
        "name": "search",
        "description": "Optimizes compression strategies using advanced search algorithms",
        "prompt": prompt,
        "tools": [
            bayesian_optimization_search,
            evolutionary_search,
            multi_armed_bandit_search,
            random_search,
            grid_search,
            analyze_search_history,
        ],
        "model": "anthropic:claude-3-haiku-20240307",
    }


__all__ = ["get_search_subagent", "bayesian_optimization_search", "evolutionary_search"]