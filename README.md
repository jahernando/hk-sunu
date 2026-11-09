# hk-sunu

Phenomenological studies of supernova neutrinos in Hyper-Kamiokande, starting
with the Diffuse Supernova Neutrino Background (DSNB).

This repository uses only public information. It is not an official product of
the Hyper-Kamiokande collaboration.

## Layout

| Path | Content |
|---|---|
| `src/hk_sunu/` | Python package: flux, IBD cross section, detector, event rates |
| `NB/` | Notebooks mixing narrative, code and figures |
| `tests/` | Physics and normalisation checks |
| `data/` | Local data, not tracked except for small essential files |

## Setup

```bash
pip install -e ".[dev]"
pytest
```

## Notebooks

- `NB/01_sk_dsnb_analysis.ipynb`: the Super-Kamiokande DSNB analysis, signal and backgrounds.
