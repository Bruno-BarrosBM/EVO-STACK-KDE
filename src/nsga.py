from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
from deap import base, creator, tools
from tqdm import tqdm

from .eval import complexity, nll_kfold, stability_bootstrap
from .genome import ExpertConfig, ModelConfig, decode_to_model, random_model_config


def _ensure_creators() -> None:
    if "FitnessMin3" not in creator.__dict__:
        creator.create("FitnessMin3", base.Fitness, weights=(-1.0, -1.0, -1.0))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", ModelConfig, fitness=creator.FitnessMin3)


def _clone_expert(expert: ExpertConfig) -> ExpertConfig:
    return ExpertConfig(alpha=expert.alpha, feature_mask=expert.feature_mask.copy())


def _clone_config(config: ModelConfig) -> ModelConfig:
    return ModelConfig(
        weight_logits=config.weight_logits.copy(),
        experts=[_clone_expert(exp) for exp in config.experts],
    )


def _make_individual(config: ModelConfig) -> ModelConfig:
    return creator.Individual(weight_logits=config.weight_logits.copy(), experts=[_clone_expert(e) for e in config.experts])


def _crossover(config_a: ModelConfig, config_b: ModelConfig, rng: np.random.Generator) -> Tuple[ModelConfig, ModelConfig]:
    m = len(config_a.experts)
    point = rng.integers(1, m) if m > 1 else 0
    experts1 = [_clone_expert(exp) for exp in (config_a.experts[:point] + config_b.experts[point:])]
    experts2 = [_clone_expert(exp) for exp in (config_b.experts[:point] + config_a.experts[point:])]
    blend = rng.uniform(0.25, 0.75)
    logits1 = blend * config_a.weight_logits + (1.0 - blend) * config_b.weight_logits
    logits2 = blend * config_b.weight_logits + (1.0 - blend) * config_a.weight_logits
    return (
        ModelConfig(weight_logits=logits1.copy(), experts=experts1),
        ModelConfig(weight_logits=logits2.copy(), experts=experts2),
    )


def _mutate(config: ModelConfig, rng: np.random.Generator) -> ModelConfig:
    mutated = _clone_config(config)
    for expert in mutated.experts:
        log_alpha = np.log(expert.alpha)
        log_alpha += rng.normal(0.0, 0.15)
        expert.alpha = float(np.clip(np.exp(log_alpha), 0.1, 5.0))
        mask = expert.feature_mask.copy()
        flip_prob = 1.0 / mask.size
        flips = rng.random(mask.size) < flip_prob
        mask = np.logical_xor(mask, flips)
        if not mask.any():
            mask[rng.integers(0, mask.size)] = True
        expert.feature_mask = mask
    mutated.weight_logits = mutated.weight_logits + rng.normal(0.0, 0.3, size=mutated.weight_logits.shape)
    return mutated


def run_nsga(
    X: np.ndarray,
    pop_size: int = 20,
    n_gen: int = 10,
    seed: int = 42,
    kfold: int = 2,
    bootstraps: int = 3,
    outdir: str = "outputs",
) -> Tuple[List[dict], Path]:
    _ensure_creators()
    rng = np.random.default_rng(seed)

    toolbox = base.Toolbox()
    d = X.shape[1]

    def init_individual() -> ModelConfig:
        config = random_model_config(d, rng, m=5)
        return _make_individual(config)

    toolbox.register("individual", init_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def evaluate(individual: ModelConfig) -> Tuple[float, float, float]:
        config = individual
        factory = decode_to_model(config)
        f1 = nll_kfold(factory, X, k=kfold, seed=seed)
        f2 = stability_bootstrap(factory, X, B=bootstraps, k=kfold, seed=seed)
        f3 = complexity(config, n=X.shape[0], d=d)
        return f1, f2, f3

    toolbox.register("evaluate", evaluate)

    def mate(ind1: ModelConfig, ind2: ModelConfig) -> Tuple[ModelConfig, ModelConfig]:
        child1, child2 = _crossover(ind1, ind2, rng)
        return _make_individual(child1), _make_individual(child2)

    def mutate(ind: ModelConfig) -> Tuple[ModelConfig]:
        mutated = _mutate(ind, rng)
        return (_make_individual(mutated),)

    toolbox.register("mate", mate)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selNSGA2)

    pop = toolbox.population(n=pop_size)

    for ind in tqdm(pop, desc="Avaliando população inicial", unit="ind"):
        ind.fitness.values = toolbox.evaluate(ind)

    # Assign crowding distance before first tournament selection
    pop = toolbox.select(pop, len(pop))

    for gen in tqdm(range(1, n_gen + 1), desc="Evoluindo gerações", unit="ger"):
        offspring = tools.selTournamentDCD(pop, len(pop))
        offspring = [_clone_config(ind) for ind in offspring]

        for i in range(1, len(offspring), 2):
            if rng.random() < 0.9:
                child1, child2 = _crossover(offspring[i - 1], offspring[i], rng)
                offspring[i - 1], offspring[i] = child1, child2

        for i in range(len(offspring)):
            if rng.random() < 0.4:
                offspring[i] = _mutate(offspring[i], rng)

        offspring = [_make_individual(cfg) for cfg in offspring]

        for ind in tqdm(
            offspring,
            desc=f"Geração {gen}: avaliando descendentes",
            unit="ind",
            leave=False,
        ):
            ind.fitness.values = toolbox.evaluate(ind)

        pop = toolbox.select(pop + offspring, pop_size)

    pareto_front = tools.sortNondominated(pop, k=len(pop), first_front_only=True)[0]

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(outdir) / "runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    pareto_data: List[dict] = []
    for ind in pareto_front:
        pareto_data.append({"config": ind.to_jsonable(), "fitness": list(ind.fitness.values)})

    with open(run_dir / "pareto.json", "w", encoding="utf-8") as f:
        json.dump(pareto_data, f, indent=2)

    return pareto_data, run_dir
