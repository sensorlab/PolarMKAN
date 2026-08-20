# PolarMKAN

**Interpretable Feature Learning for RF Fingerprinting via Polar MKANs**

Mikhail Krasnov\*, Ljupcho Milosheski\*, Carolina Fortuna — Jožef Stefan Institute, Ljubljana, Slovenia
<sub>\*equal contribution</sub>

Code and configurations for the IEEE Wireless Communications Letters submission.
This repository extends [**ZeroUED**](https://github.com/sensorlab/ZeroUED)
(paper: [arXiv:2511.07026](https://arxiv.org/abs/2511.07026)).

For a longer, plain-language introduction to RF fingerprinting, the
self-supervised training strategies and the ZeroUED background, see
[`README copy.md`](README%20copy.md).

---

## What the letter proposes

Every radio transmitter carries small manufacturing imperfections — carrier
frequency offset, I/Q imbalance, amplifier non-linearity — that distort its
signal in a device-specific way. That distortion is the **RF fingerprint**, and
it can identify a device without any cryptographic exchange.

A **feature extractor** compresses each burst into a handful of numbers. The
usual deep-learning extractors are accurate but opaque: nothing constrains what
any given latent coordinate means.

**Polar MKAN** is a feature extractor with that constraint built into the
architecture. It reads each burst in polar form (magnitude and principal phase
rather than raw I/Q) and partitions its output so that:

- `F_R` connects **only** to the magnitude inputs,
- `F_phi1` and `F_phi2` connect **only** to the phase inputs,

with every connected mapping componentwise non-decreasing. Raising the phase
inputs can therefore only raise the phase features, and cannot move `F_R` at
all. The separation is a property of the wiring, not of how training happened to
converge — so it holds at inference time and can be checked.

On the synthetic gain/CFO benchmark this yields **57.2% DCI Disentanglement**,
against ≤12.9% for the unpartitioned baselines, at a measurable cost in
detection accuracy on real WiSig data.

## What is in this repository

| Path | Contents |
| --- | --- |
| `DCI_measure.ipynb` | Driver notebook: runs every experiment in the letter and draws its two generated figures. |
| `src/experiments.py` | The experiment implementations — one `run_*` function per experiment. |
| `src/figures.py` | The two figures in the letter. |
| `src/datasets.py` | WiSig / ORACLE / LoRa loaders **and** the synthetic benchmark. |
| `src/metrics.py` | Clustering and detection scores, plus the DCI metric. |
| `results/` | The CSVs behind the letter's tables. |

Contributions specific to this work:

- **`WiSig_Dataset_SingleDay`** (`src/datasets.py`) — the compact 28-transmitter,
  10-receiver, single-day WiSig subset.
- **Blind CFO compensation** — a `cfo_compensate` flag on all three WiSig
  datasets and on the synthetic dataset. Estimates the bulk offset from the mean
  phase slope, `nu_hat = mean(arg(x[n] x*[n-1])) / 2pi`, and de-rotates the burst
  before the Cartesian-to-polar conversion. This produces the `AUC_comp` columns.
- **Polar CNN / Polar KAN configs** — the polar-input controls, which isolate the
  effect of the polar representation from the effect of the block partition.
- **`measure_wisig_impairments.py`** — impairment measurement on real WiSig captures.

---

## Setup

```bash
pip install -r reqs.txt
```

`reqs.txt` pins the versions the reported results were produced with; `env.yaml`
is the equivalent conda environment. The PyTorch pin is CUDA-agnostic — install
the build matching your machine from
[pytorch.org](https://pytorch.org/get-started/locally/) if you need a specific
CUDA or ROCm version. `tqdm` is optional; the capacity sweep uses it for progress
bars and falls back to plain iteration without it.

The notebook's first code cell is a lazy dependency check: it installs only
what is missing, into the kernel's own environment via `sys.executable -m pip`
rather than a bare `!pip`, which often reaches a different environment than the
kernel. PyTorch is reported but never installed automatically, since the right
build depends on your CUDA / ROCm version.

Launch notebooks **from the repository root** so Python can find `src/`.

### Datasets

The synthetic experiments generate their own data and need no download. The
WiSig experiments expect `SingleDay.pkl`, `ManyTx.pkl` and `ManySig.pkl` in the
repository root:

```bash
python download_data.py
```

These files are several GB and are excluded by `.gitignore`, along with
`ckpt/` and any `*.zip`.

---

## Reproducing the letter

Open `DCI_measure.ipynb` and run the sections you need — each is a few lines
calling into `src/experiments.py`.

| Notebook section | Produces | Function |
| --- | --- | --- |
| 6 | Table II — DCI Disentanglement / Completeness | `run_dci_benchmark` |
| 7 | Table III, Synthetic columns | `run_synthetic_ued` |
| 8 | Table III, SingleDay and ManyTx columns | `run_wisig_ued` |
| 9 | Parameter counts and forward-pass timing *(supplementary)* | `model_size_table` |
| 10 | Grid-size capacity sweep *(supplementary)* | `run_capacity_sweep` |
| 11 | Multi-seed perturbation response, guarantee check | `run_perturbation_analysis`, `run_guarantee_check` |
| 12 | Sign consistency vs. phase-wrap fraction | `run_branch_cut_validation` |
| 13 | Fig. 3 and Fig. 4 | `plot_phase_wrap`, `plot_tradeoff_scatter` |

Sections 9 and 10 are marked supplementary because the letter does not report
them. They are kept as supporting evidence — the capacity sweep in particular
addresses whether the disentanglement gap could be a capacity artefact.

The letter's settings are `N_RUNS = 10`, `N_EPOCHS = 120`. Both are slow on CPU
— lower them in the setup cell for a first pass.

### Which result file backs which table

Each of these was matched value-by-value against the letter:

| Table | File |
| --- | --- |
| Table II | `results/synthetic_dci_20260812_105624.csv` (+ `_perrun` companion) |
| Table III, Synthetic | `results/synthetic_ued_20260806_144410.csv` (+ `_perseed` companion) |
| Table III, SingleDay | `results/wisig_ued_20260804_151644.csv` |
| Table III, ManyTx | `results/ManyTx/wisig_ued_20260730_002754.csv` |

Section 13 redraws Fig. 4 from the first two files, so the figure can be
regenerated without retraining anything. Point the two paths at your own run to
redraw it from new results.

Training checkpoints are cached under `ckpt/`, so re-running the perturbation
and branch-cut sections reuses the trained models instead of retraining them.

### Config-driven experiments

The original config-file workflow still works for the standard sweeps:

```bash
python run_experiments.py configs/wisig/manytx/ae_config_cnn.yaml
```

Configs are grouped by dataset and subset under `configs/`; `singleday/` holds
the six configurations added for this work. For experiment logging, create a
free account at [wandb.ai](https://wandb.ai) and run `wandb login` first.

---

## Architectures compared

All six share a 3-dimensional bottleneck and are trained identically with the
autoencoder strategy, so the comparison isolates the feature extractor. The
registries are `AE_CONFIGS` (DCI benchmark) and `ARCHS` (UED experiments) in
`src/experiments.py`; add an entry there to include your own.

| Name | Input | Partition | Monotonic |
| --- | --- | --- | --- |
| CNN | raw I/Q | — | — |
| KAN | raw I/Q | no | no |
| MKAN | raw I/Q | no | yes |
| Polar CNN | polar | — | — |
| Polar KAN | polar | no | no |
| **Polar MKAN** | polar | **yes** | **yes** |

Polar CNN and Polar KAN are the controls that show the polar transform alone
does not recover the separation.

---

## Repository structure

```
DCI_measure.ipynb           Driver notebook for every experiment in the letter.
mde_calculation.ipynb       Minimum detectable effect analysis.
calculate_mde.py            Supporting MDE computation.
measure_wisig_impairments.py  Impairment measurement on real WiSig captures.
run_experiments.py          Run a pre-defined experiment from a config file.
download_data.py            Download the WiSig and ORACLE datasets.
create_oracle_dataset.py    Build the ORACLE dataset files.
reqs.txt / env.yaml         Python dependencies.
configs/                    Experiment settings (dataset + model + training):
    oracle/{raw_iq,2d_const}
    wisig/{manysig,manytx,singleday}
results/                    CSVs backing the letter's tables.
src/
    experiments.py          One run_* function per experiment in the letter.
    figures.py              plot_tradeoff_scatter, plot_phase_wrap.
    datasets.py             WiSig / ORACLE / LoRa loaders + synthetic benchmark.
    metrics.py              Clustering and detection scores, DCI metric.
    trainers.py             Training loops (AE, Deep Clustering, SimCLR, PCA).
    config_manager.py       Reads the YAML config files.
    architectures/
        features_extractors/mkan.py   Polar MKAN and MKAN.
        features_extractors/kan.py    Plain KAN autoencoder.
        features_extractors/cnn.py    Convolutional autoencoder baseline.
        features_extractors/...       ResNet, transformer and ViT variants.
        side_networks.py, viewmakers.py
```

---

## References

- DCI framework — [Eastwood & Williams, ICLR 2018](https://openreview.net/forum?id=By-7dz-AZ)
- WiSig dataset — [Hanna, Karunaratne & Cabric, IEEE Access 2022](https://cores.ee.ucla.edu/downloads/datasets/wisig/)
- ORACLE dataset — [Northeastern University](https://repository.library.northeastern.edu/files/neu:m044q520q)
- Kolmogorov–Arnold Networks — [Liu et al., 2024](https://arxiv.org/abs/2404.19756)
- ZeroUED — [arXiv:2511.07026](https://arxiv.org/abs/2511.07026)

## Licence

See [`LICENSE`](LICENSE).
