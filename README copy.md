# PolarMKAN

**Telling radio devices apart by their tiny hardware quirks — with a neural
network that learns to organise those quirks neatly.**

Mikhail Krasnov, Ljupcho Milosheski, Mihael Mohorčič and Carolina Fortuna

This repository extends [**ZeroUED**](https://github.com/sensorlab/ZeroUED)
(paper: [arXiv:2511.07026](https://arxiv.org/abs/2511.07026)).

---

## What is this project about? (start here)

Every radio transmitter (a WiFi card, a phone, an IoT sensor) has small
manufacturing imperfections. Two devices sending the *exact same message* still
produce slightly different signals because of these imperfections. That hidden,
unintentional signature is called an **RF fingerprint**, and it can be used to
recognise a device even when it never tells us who it is.

The heart of the project is a **feature extractor**: a small neural network that
turns each raw signal into just a handful of numbers — the fingerprint. The new
contribution in this fork is **PolarMKAN**, a new feature extractor.

The tricky part is that we want to learn this fingerprint **without anyone
labelling the data** (we never tell the model which device is which). To do that
we need a *training strategy*. The one used here is the **autoencoder** approach:
we temporarily attach a second network (a *decoder*) that tries to rebuild the
original signal from those few numbers. To make the rebuild possible, the
feature extractor is forced to pack the most important information into its
output — and the device's fingerprint turns out to be exactly that. Once
training is done we **keep the feature extractor and throw the decoder away**.

> **The key distinction:** *PolarMKAN is the feature extractor* (the thing we
> keep and use). *The autoencoder is just one way to train it.* Other training
> strategies exist (Deep Clustering, SimCLR — see below) and could train the
> same feature extractor instead.

PolarMKAN is designed so that the few numbers it produces are **organised
cleanly**: ideally, each number corresponds to *one* physical property of the
device (for example, one number for the frequency error, another for the
amplitude error). When a representation is organised this way we call it
**disentangled**, and disentangled fingerprints are easier to interpret and more
reliable.

## Key ideas in plain words

| Term | Plain-language meaning |
| --- | --- |
| **RF fingerprint** | The tiny, device-specific distortion every transmitter adds to its signal because no hardware is perfect. |
| **Feature extractor** | The network that turns a raw signal into a few meaningful numbers (the fingerprint). This is the part we keep and use. **PolarMKAN is a feature extractor.** |
| **Embedding / latent space** | The few numbers a feature extractor produces for each signal. Here it is just **3 numbers** per signal, so we can plot it in 3-D. |
| **Autoencoder** | A *training strategy*, not a goal in itself: bolt a decoder onto the feature extractor, train the pair to reconstruct the input, then discard the decoder. One way to learn features without labels. |
| **Self-supervised** | Training *without labels*. The only goal is "rebuild the input", so we never tell the model which device is which. |
| **Disentanglement** | How cleanly each embedding number maps to a single real-world property of the device. Higher = tidier, more interpretable. |
| **KAN** | *Kolmogorov–Arnold Network*: a recent alternative to the usual neural-network layer that learns flexible curves instead of plain weights. PolarMKAN is built from KAN layers. |
| **PolarMKAN** | Our feature extractor. "Polar" = it reads each signal as *magnitude + phase* instead of raw I/Q; "M" = monotonic, a constraint that helps disentanglement. |

## What's new in this fork

* **PolarMKAN** — the new feature extractor, in
  `src/architectures/features_extractors/mkan.py`. It is built so that each of
  its embedding numbers tends to capture a single physical device property
  (frequency offset, amplitude, I/Q imbalance). It is trained here with the
  autoencoder approach, but the trainers in `src/trainers.py` could learn it
  other ways too.
* **`DCI_measure.ipynb`** — a notebook that *measures* how disentangled a feature
  extractor's embedding is, on synthetic data where we know the right answer.
* **`WiSig_evaluation.ipynb`** — a notebook that compares PolarMKAN against other
  feature extractors on a **real** WiFi dataset.

---

## Learning strategies (how the feature extractor is trained)

Remember the key distinction: **PolarMKAN is the feature extractor we keep; a
*learning strategy* is just one recipe for training it without labels.** The
same PolarMKAN (or any baseline) can be plugged into any of the strategies
below. Each one is implemented as a `Trainer` in `src/trainers.py`, so you can
swap strategies without touching the model.

All of them are **self-supervised** — they never see which device produced which
signal. They differ only in the *pretext task* they invent to force the feature
extractor to learn a useful fingerprint.

| Strategy | `Trainer` class | The trick it uses |
| --- | --- | --- |
| **Autoencoder** | `AE_Trainer` | Reconstruct the input from the embedding. |
| **Deep Clustering** | `Deep_Clustering_Trainer` | Cluster the embeddings, then predict the cluster. |
| **SimCLR (contrastive)** | `SIM_CLR_Trainer` | Pull augmented copies of one signal together, push others apart. |
| **PCA** | `PCA_Trainer` | Linear baseline: project onto the top principal components. |

### Autoencoder (the default here)

Bolt a **decoder** onto the feature extractor and train the pair to rebuild the
original signal from the few-number embedding. To make the rebuild possible, the
feature extractor is forced to pack the most important information into its
output — and the device fingerprint turns out to be exactly that. After training
we keep the feature extractor and throw the decoder away. This is the strategy
used in both notebooks, because it is the simplest to reason about and pairs
naturally with PolarMKAN's reconstruction-friendly design. (Optionally it can
add a small *distance loss* that nudges nearby embeddings into tighter clusters.)

### Deep Clustering

Alternate between two steps: (1) run every signal through the feature extractor
and **cluster** the resulting embeddings (k-means) to invent temporary
pseudo-labels; (2) train the feature extractor to **predict** those
pseudo-labels, as if they were real classes. Repeating this loop makes the
embedding progressively more cluster-friendly. There is no decoder — the
"supervision" is the model's own clustering from the previous round, refreshed
every few epochs. Based on Caron et al., *Deep Clustering for Unsupervised
Learning of Visual Features* (ECCV 2018).

### SimCLR (contrastive learning)

Make two randomly **augmented** copies of the same signal and teach the feature
extractor that they should land **close together** in embedding space, while
copies coming from *different* signals should be pushed **apart**. No decoder and
no clustering — the learning signal comes entirely from this "same vs. different"
contrast. This implementation also supports learnable augmentations and optional
hard-positive / hard-negative mining. Inspired by Hao et al., *Contrastive
Self-Supervised Clustering for Specific Emitter Identification* (IEEE IoT
Journal, 2023).

### PCA (linear baseline)

Not a neural network at all: just fit **Principal Component Analysis** to the raw
signals and keep the top components as the embedding. It is fast and parameter-free
and serves as a sanity-check floor — a good neural strategy should beat it.

> **Which should I use?** Start with the **autoencoder** — it is the default in
> the notebooks and the easiest to interpret. Reach for **Deep Clustering** or
> **SimCLR** when you care more about how cleanly devices *separate into groups*
> than about reconstructing the signal, and use **PCA** as a baseline to make
> sure the fancier strategies are actually earning their keep.

---

## Quickstart (no data download needed)

The fastest way to see the project in action is the synthetic notebook — it
**creates its own data**, so you can run it immediately:

```bash
pip install -r reqs.txt        # install dependencies
pip install --upgrade numpy pandas matplotlib pyarrow numexpr bottleneck
jupyter notebook DCI_measure.ipynb
```

Run the cells top to bottom. It trains a few small autoencoders and prints a
table showing that PolarMKAN produces a more disentangled embedding than the
baselines. Tip: lower `N_RUNS` and `N_EPOCHS` near the top of the training
section for a quick first run.

> Always launch the notebooks **from the repository root** so Python can find the
> `src/` package.

---

## The two evaluations

### 1. `DCI_measure.ipynb` — measuring disentanglement on synthetic data

We *invent* radio signals ourselves, so we know exactly which physical
properties ("factors") went into each one. That lets us check whether a model
recovered them cleanly. The notebook:

1. **Builds a synthetic dataset.** Each signal is a 16-QAM radio burst (a common
   digital modulation) with two device factors we care about — a
   carrier-frequency offset (`cfo`) and an amplitude offset (`ampl`) — plus some
   realistic noise. Because we generate them, the ground truth is exact.
2. **Trains each feature extractor** with the autoencoder approach — reconstruct
   the signal, no labels used.
3. **Scores the embeddings** with the **DCI** framework:
   * **Disentanglement** — does each embedding number describe just one factor?
   * **Completeness** — is each factor described by just one embedding number?
   * **Spearman correlation** — a simple sanity check of the overall trend.

Feature extractors compared out of the box:

| Feature extractor | What it is |
| --- | --- |
| **PolarMKAN** | Our model: KAN feature extractor with the monotonic + polar design. |
| **KAN-AE (plain)** | The same KAN feature extractor *without* our extra constraints (a comparison to show the constraints help). |
| **CNN-AE (1D)** | A standard convolutional feature extractor — a familiar baseline. |

(All three are trained the same way — with the autoencoder approach — so the
comparison is purely about the feature extractor.) You can add your own with a
single line in the `ARCHITECTURES` dictionary.

### 2. WiSig — testing on a real WiFi dataset

Synthetic data is convenient, but the real test is real signals. This notebook
uses [**WiSig**](https://cores.ee.ucla.edu/downloads/datasets/wisig/), a public
dataset of WiFi transmissions from many real devices. It:

1. Loads WiSig for a handful of transmitters. PolarMKAN reads the signals in
   *polar* form; the other feature extractors read the raw signal.
2. Trains every feature extractor (again with the autoencoder approach) down to
   the same **3-number embedding**.
3. **Groups the embeddings** (with k-means clustering) and checks how well those
   groups match the true devices, using standard clustering scores
   (NMI, ARI, homogeneity, completeness, silhouette). It then draws a 3-D plot of
   each embedding, coloured by device — well-separated colours mean a good
   fingerprint.

The following intrusctins are essentional:

---

## Full installation (for the real datasets)

You only need this if you want to run `WiSig_evaluation.ipynb` or the original
config-driven experiments. `DCI_measure.ipynb` needs none of it.

1. Install dependencies:
   ```bash
   pip install -r reqs.txt
   pip install --upgrade numpy pandas matplotlib pyarrow numexpr bottleneck
   ```
2. (For experiment logging) Create a free account at
   https://wandb.ai, then log in with your token from
   https://wandb.ai/quickstart?product=models:
   ```bash
   wandb login
   ```
3. Download the datasets:
   ```bash
   python download_data.py
   ```
5. Run a pre-defined experiments from a config file (our exepriments are in wisig folder):
   ```bash
   python run_experiments.py configs/wisig/manytx/ae_config_cnn.yaml
   ```

## Repository structure

```
DCI_measure.ipynb        Start here: disentanglement on synthetic data (this fork).
mde_calculation.ipynb    Minimum detectable effect analysis.
run_experiments.py       Run a pre-defined experiment from a config file.
download_data.py         Download the WiSig and ORACLE datasets.
reqs.txt / env.yaml      Python dependencies.
configs/                 Ready-made experiment settings (dataset + model + training).
src/                     Source code:
    src/trainers.py          Training loops for each learning approach.
    src/datasets.py          Loading the WiSig / ORACLE / LoRa datasets.
    src/metrics.py           Scoring functions (clustering quality, etc.).
    src/config_manager.py    Reads the YAML config files.
    src/architectures/       The neural-network building blocks:
        features_extractors/mkan.py   PolarMKAN, the new model (this fork).
        features_extractors/kan.py    Plain KAN autoencoder.
        features_extractors/cnn.py    Convolutional autoencoder baseline.
```

## Background and references

The original ZeroUED study explores how to build detectors for *unknown*
radio emitters across two situations (devices sending the same message vs.
different messages). It evaluates three families of self-supervised methods:

  - [Autoencoders](https://ieeexplore.ieee.org/document/10623390)
  - [Deep Clustering](https://openaccess.thecvf.com/content_ECCV_2018/html/Mathilde_Caron_Deep_Clustering_for_ECCV_2018_paper.html)
  - [SimCLR (contrastive learning)](https://www.researchgate.net/publication/371460960_Contrastive_Self-supervised_Clustering_for_Specific_Emitter_Identification)

on two public datasets:

  - [WiSig](https://cores.ee.ucla.edu/downloads/datasets/wisig/)
  - [ORACLE](https://repository.library.northeastern.edu/files/neu:m044q520q)

The disentanglement score used here is the
[DCI framework](https://openreview.net/forum?id=By-7dz-AZ).
