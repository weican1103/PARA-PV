# Pretrained models

Pretrained weights are intentionally excluded from Git.

To enable the optional Chronos2 prior, place a compatible checkpoint under
`pretrained_models/chronos-2/` or pass its local path with
`--chronos_model_path`. Install the optional dependency set first:

```bash
pip install -r requirements-chronos.txt
```

PARA-PV runs without Chronos2 when `--use_chronos_prior false` (the default).
