# ML Showcase: Set-based Genomics + Causal Modules (Demo)

A clean, self-contained Python ML demo showcasing:

- **Permutation-invariant mutation set encoders** (DeepSets + lightweight attention pooling)
- **Generalization-first evaluation** under dataset shift
- **Module-level causal discovery** (partial-correlation baseline + bootstrap uncertainty)

The synthetic generator mimics:

**mutations → cell-state programs → therapy response**

Swap the generator with real cohorts when available.

## Quickstart

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac:
# source .venv/bin/activate

pip install -r requirements.txt

python -m ml_showcase.run_experiment --config configs/demo.yaml
```

Outputs are written to `runs/<timestamp>/` (gitignored).

## Repo layout

- `ml_showcase/` core library (models, training, causal discovery, runner)
- `configs/` YAML experiment configs
- `scripts/` convenience wrappers
- `runs/` outputs (gitignored)
