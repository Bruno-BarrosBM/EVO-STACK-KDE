from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import List

import joblib
import numpy as np

from .data import load_wine_red, persist_splits, split_and_scale
from .genome import decode_to_model, model_config_from_jsonable
from .nsga import run_nsga
from .plots import plot_hist_neglogp_test, plot_pareto_2d


def _log_step(message: str) -> None:
    print(f"[EVO-STACK-KDE] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EVO-STACK-KDE MVP pipeline")
    parser.add_argument("--csv", required=True, help="Path to winequality-red.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", default="outputs")
    parser.add_argument("--pop", type=int, default=20)
    parser.add_argument("--gens", type=int, default=10)
    parser.add_argument("--kfold", type=int, default=2)
    parser.add_argument("--bootstraps", type=int, default=3)
    parser.add_argument("--load_model", help="Path to previously trained model (.pkl)")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass


def _select_knee(pareto: List[dict]) -> dict:
    fitness = np.array([entry["fitness"] for entry in pareto], dtype=float)
    mins = fitness.min(axis=0)
    maxs = fitness.max(axis=0)
    ranges = np.where(maxs - mins == 0, 1.0, maxs - mins)
    normalized = (fitness - mins) / ranges
    distances = np.linalg.norm(normalized, axis=1)
    idx = int(np.argmin(distances))
    return pareto[idx]


def main() -> None:
    _log_step("Iniciando pipeline EVO-STACK-KDE")
    args = parse_args()

    if args.load_model:
        _log_step(f"Carregando modelo salvo de '{args.load_model}'")
        payload = joblib.load(args.load_model)
        model = payload["model"]
        scaler = payload["scaler"]
        metadata = payload.get("metadata", {})

        if metadata:
            meta_str = json.dumps(metadata, indent=2)
            _log_step("Metadados do modelo carregado:\n" + meta_str)

        _log_step(f"Carregando novos dados a partir de '{args.csv}'")
        new_df = load_wine_red(args.csv)

        if "quality" in new_df.columns:
            new_df = new_df.drop(columns=["quality"])

        X_new = new_df.values.astype(float)
        X_new_scaled = scaler.transform(X_new)

        _log_step(f"Gerando predições para {X_new_scaled.shape[0]} amostras")
        densities = model.predict(X_new_scaled)
        logp = model.logpdf(X_new_scaled)

        inference_dir = Path(args.outdir)
        inference_dir.mkdir(parents=True, exist_ok=True)

        preds_path = inference_dir / "predictions.npy"
        np.save(preds_path, densities)

        summary = {
            "mean_density": float(np.mean(densities)),
            "median_density": float(np.median(densities)),
            "mean_neg_log_likelihood": float(-np.mean(logp)),
            "n_samples": int(X_new_scaled.shape[0]),
            "source_csv": str(args.csv),
        }

        summary_path = inference_dir / "inference_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        _log_step(
            "Resumo das métricas de inferência: "
            f"NLL médio={summary['mean_neg_log_likelihood']:.4f}, "
            f"densidade média={summary['mean_density']:.4f}"
        )
        _log_step(f"Predições salvas em '{preds_path}'")
        _log_step(f"Resumo da inferência salvo em '{summary_path}'")
        _log_step("Pipeline concluído utilizando modelo pré-treinado")
        return

    _log_step(f"Seed global definida como {args.seed}")
    seed_everything(args.seed)

    _log_step(f"Carregando dados a partir de '{args.csv}'")
    data_df = load_wine_red(args.csv)
    _log_step("Dividindo dataset em conjuntos de treino, validação e teste e aplicando padronização")
    splits = split_and_scale(data_df, seed=args.seed)
    _log_step("Persistindo divisões processadas em 'data/processed'")
    persist_splits(splits, Path("data/processed"))

    _log_step(
        "Iniciando busca evolutiva NSGA-II "
        f"(população={args.pop}, gerações={args.gens}, kfold={args.kfold}, bootstraps={args.bootstraps})"
    )
    pareto, run_dir = run_nsga(
        splits.X_train,
        pop_size=args.pop,
        n_gen=args.gens,
        seed=args.seed,
        kfold=args.kfold,
        bootstraps=args.bootstraps,
        outdir=args.outdir,
    )
    _log_step(f"Busca evolutiva concluída. Resultados registrados em '{run_dir}'")

    _log_step("Selecionando solução joelho da frente de Pareto")
    knee_entry = _select_knee(pareto)
    best_config = model_config_from_jsonable(knee_entry["config"])

    _log_step("Treinando modelo final com dados de treino e validação combinados")
    X_train_full = np.vstack([splits.X_train, splits.X_val])
    factory = decode_to_model(best_config)
    model = factory()
    model.fit(X_train_full)

    _log_step("Avaliando modelo no conjunto de teste")
    test_logp = model.logpdf(splits.X_test)
    test_nll = float(-np.mean(test_logp))

    _log_step(f"Criando diretórios de saída em '{args.outdir}'")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    figures_dir = outdir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir = outdir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    _log_step("Gerando visualizações e salvando artefatos do modelo")
    plot_pareto_2d([entry["fitness"] for entry in pareto], figures_dir / "pareto_2d.png")
    plot_hist_neglogp_test(-test_logp, figures_dir / "hist_test_logp.png")

    joblib.dump({"model": model, "scaler": splits.scaler, "config": best_config.to_jsonable()}, models_dir / "best_model.pkl")

    final_model_payload = {
        "model": model,
        "scaler": splits.scaler,
        "config": best_config.to_jsonable(),
        "metadata": {
            "seed": args.seed,
            "csv_path": str(args.csv),
            "pop": args.pop,
            "gens": args.gens,
            "kfold": args.kfold,
            "bootstraps": args.bootstraps,
            "run_dir": str(run_dir),
        },
    }

    final_model_path = outdir / "final_model.pkl"
    joblib.dump(final_model_payload, final_model_path)
    _log_step(f"Modelo final salvo em '{final_model_path}'")

    metrics_path = outdir / "metrics.json"

    metrics = {
        "test_nll": test_nll,
        "knee_fitness": knee_entry["fitness"],
        "knee_config": knee_entry["config"],
        "pareto_run_dir": str(run_dir),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    _log_step(f"Pipeline concluído com sucesso. Métricas salvas em '{metrics_path}'")


if __name__ == "__main__":
    main()
