# PARA-PV

**Physics-Aware Retrieval-Augmented PV Prediction Based on Frozen Foundation
Model and Distribution Shift Correction**

PARA-PV is a PV forecasting framework, which addresses weather variability, day-night
transitions, regime-dependent dynamics, and physical constraints by embedding
PV-specific knowledge throughout the forecasting pipeline:

- **Physics-aware retrieval (PA-RAL)** matches patch-level observations with
  historical patterns and analog trajectories according to temporal shape,
  power level, operating state, and intra-day period.
- **Frozen foundation-model calibration** uses a lightweight residual adapter
  to incorporate a Chronos time-series prior without overriding the
  physically grounded base forecast.
- **Distribution shift correction (PA-DSC)** selectively adjusts forecast mean
  and scale using power, weather, timestamp, and day/night conditions.
- **Physics-constrained learning** balances errors across peak, ramping,
  night-time, and regular operating regimes.

Experiments on two geographically distinct PV farms show consistent gains over
seven representative baselines across multiple forecasting horizons, with
improved point errors and sharper prediction intervals using fewer than one
million trainable parameters.

This repository contains the runnable model and its training, evaluation,
data-loading, loss, and metric dependencies. Experiment outputs, checkpoints,
and pretrained foundation-model weights are excluded.

## Included code

- `src/PARA_PV/model.py`: PARA-PV, PA-RAL, PA-DSC, and Chronos residual adapter.
- `src/PARA_PV/temporal_prior.py`: optional frozen Chronos2 prior branch.
- `layers/Embed.py`: patch embedding used by the model.
- `exp/`: training, validation, and test workflow.
- `data_provider/`: Excel PV dataset loader and data splits.
- `utils/`: losses, metrics, augmentation, time features, and training helpers.
- `run.py`: command-line entry point.

## Installation

Python 3.9+ is sufficient for the base model. A CUDA-enabled PyTorch build is
recommended for training.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Chronos2 is optional and requires Python 3.10+:

```bash
pip install -r requirements-chronos.txt
```

## Data

Put a supported Excel file in `dataset/`; see `dataset/README.md`. Data files
are ignored by Git so that private or large research data are not published by
accident.

## Run

Point forecasting with the base PARA-PV configuration:

```bash
python run.py --model PARA-PV --forecast_mode point \
  --point_train_loss adaptive_segment_pv \
  --root_path ./dataset/ \
  --data_path Solarstation1-50MW-2019.xlsx
```

Enable all optional model branches:

```bash
python run.py --model PARA-PV --forecast_mode point \
  --use_pa_ral true --use_pa_ral_analog true \
  --use_chronos_prior true \
  --chronos_model_path ./pretrained_models/chronos-2 \
  --use_chronos_adapter true --use_pa_dsc true
```

Probabilistic forecasting:

```bash
python run.py --model PARA-PV --forecast_mode prob \
  --quantiles 0.05,0.1,0.5,0.9,0.95
```

Run `python run.py --help` for all settings.
