from __future__ import annotations

import argparse

from .causal.discovery import bootstrap_edges
from .config import load_config
from .data.synthetic import make_synthetic
from .train import train_eval
from .utils import make_run_dir, save_json, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="ml-showcase-causal-genomics-demo")
    p.add_argument("--config", default="configs/demo.yaml", help="Path to YAML config")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg.seed)
    run = make_run_dir()

    dcfg = cfg.data
    train = make_synthetic(
        n_samples=int(dcfg.get("n_train", 2000)),
        vocab_size=int(dcfg.get("vocab_size", 512)),
        max_set_len=int(dcfg.get("n_mutations_max", 60)),
        n_modules=int(dcfg.get("n_modules", 12)),
        shift_strength=0.0,
        seed=cfg.seed,
    )
    test = make_synthetic(
        n_samples=int(dcfg.get("n_test", 1000)),
        vocab_size=int(dcfg.get("vocab_size", 512)),
        max_set_len=int(dcfg.get("n_mutations_max", 60)),
        n_modules=int(dcfg.get("n_modules", 12)),
        shift_strength=float(dcfg.get("shift_strength_test", 0.8)),
        seed=cfg.seed + 1,
    )

    mcfg = cfg.model
    tcfg = cfg.training

    res = train_eval(
        train_mut_idx=train.mut_idx,
        train_offsets=train.offsets,
        train_y=train.y,
        test_mut_idx=test.mut_idx,
        test_offsets=test.offsets,
        test_y=test.y,
        vocab_size=int(dcfg.get("vocab_size", 512)),
        embed_dim=int(mcfg.get("embed_dim", 64)),
        set_hidden=int(mcfg.get("set_hidden", 128)),
        attn_heads=int(mcfg.get("attn_heads", 4)),
        dropout=float(mcfg.get("dropout", 0.1)),
        lr=float(tcfg.get("lr", 1e-3)),
        weight_decay=float(tcfg.get("weight_decay", 5e-4)),
        epochs=int(tcfg.get("epochs", 12)),
        batch_size=int(tcfg.get("batch_size", 128)),
        device=str(cfg.device),
    )

    ccfg = cfg.causal
    edges = bootstrap_edges(
        modules=test.modules,
        n_boot=int(ccfg.get("bootstrap", 10)),
        edge_threshold=float(ccfg.get("edge_threshold", 0.6)),
        seed=cfg.seed,
    )

    edges.edges.to_csv(run.causal_path, index=False)
    metrics = {"auc_test": res.auc, "run_dir": str(run.run_dir)}
    save_json(metrics, run.metrics_path)

    print(f"AUC(test)={res.auc:.4f}")
    print(f"Wrote: {run.metrics_path}")
    print(f"Wrote: {run.causal_path}")


if __name__ == "__main__":
    main()
