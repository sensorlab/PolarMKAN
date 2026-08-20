"""Experiment drivers for the Polar MKAN letter.

Each public ``run_*`` function corresponds to one experiment in the paper and
returns its results as pandas objects, so the accompanying notebook only has to
call them and display the output.

    run_dci_benchmark        -> Table I  (DCI Disentanglement / Completeness)
    run_synthetic_ued        -> Table II, Synthetic columns
    run_wisig_ued            -> Table II, SingleDay and ManyTx columns
    model_size_table         -> parameter counts and forward-pass timing
    run_capacity_sweep       -> grid-size sweep for Polar MKAN vs KAN
    run_perturbation_analysis-> multi-seed phase/amplitude response
    run_guarantee_check      -> monotonicity vs circularity probes
    run_branch_cut_validation-> sign consistency vs phase-wrap fraction
"""

import copy
import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from scipy import stats
from torch import nn
from torch.utils.data import DataLoader, Subset

from src import config_manager as cm
from src.architectures.features_extractors.cnn import AE_CNN_1D
from src.architectures.features_extractors.mkan import Autoencoder as KAN_AE
from src.datasets import (
    RFFingerprintDataset,
    SyntheticUEDDataset,
    get_synthetic_loaders,
)
from src.metrics import compute_dci, mean_ci95, reconstruction_mse
from src.trainers import AE_Trainer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

METRIC_NAMES = ["roc_auc", "f1", "precision", "recall", "accuracy"]

CH, LEN = 2, 256                      # synthetic/WiSig input is (batch, 2, 256)
MAG_SCALE, TWO_PI = 1.5, 2 * np.pi


# =============================================================================
# Architecture registries
# =============================================================================

# Autoencoder factories for the DCI benchmark. `data_polars` selects the input
# representation; `polars` inside the model selects the block partition.
AE_CONFIGS = [
    {"name": "MKAN",
     "model": lambda: KAN_AE(input_size=512, hidden_size=3, grid_size=8, monotonic=True, polars=False),
     "data_polars": False},
    {"name": "PolarMKAN",
     "model": lambda: KAN_AE(input_size=512, hidden_size=3, grid_size=8, monotonic=True, polars=True),
     "data_polars": True},
    {"name": "KAN",
     "model": lambda: KAN_AE(input_size=512, hidden_size=3, grid_size=8, monotonic=False, polars=False),
     "data_polars": False},
    {"name": "CNN AE",
     "model": lambda: AE_CNN_1D(input_signal_length=256, features_size=3),
     "data_polars": False},
    # Polar-input controls: polar input, no block partition.
    {"name": "Polar KAN",
     "model": lambda: KAN_AE(input_size=512, hidden_size=3, grid_size=8, monotonic=False, polars=False),
     "data_polars": True},
    {"name": "Polar CNN",
     "model": lambda: AE_CNN_1D(input_signal_length=256, features_size=3),
     "data_polars": True},
]

# config_manager-style specs for the UED experiments.
ARCHS = {
    "CNN":        dict(fe="AE_CNN_1D", fe_cfg=dict(input_signal_length=256, in_channels=2, features_size=3), polars=False),
    "KAN":        dict(fe="AE_MKAN",   fe_cfg=dict(input_size=512, hidden_size=3, polars=False, monotonic=False), polars=False),
    "MKAN":       dict(fe="AE_MKAN",   fe_cfg=dict(input_size=512, hidden_size=3, polars=False, monotonic=True),  polars=False),
    "Polar MKAN": dict(fe="AE_MKAN",   fe_cfg=dict(input_size=512, hidden_size=3, polars=True,  monotonic=True),  polars=True),
    "Polar KAN":  dict(fe="AE_MKAN",   fe_cfg=dict(input_size=512, hidden_size=3, polars=False, monotonic=False), polars=True),
    "Polar CNN":  dict(fe="AE_CNN_1D", fe_cfg=dict(input_signal_length=256, in_channels=2, features_size=3), polars=True),
}

SUBSETS = {
    "SingleDay": dict(name="WiSig_SingleDay", file="SingleDay.pkl", total_devices=28,  ratio=7, clusters=(280,)),
    "ManyTx":    dict(name="WiSig_ManyTx",    file="ManyTx.pkl",    total_devices=150, ratio=5, clusters=(1000,)),
    "ManySig":   dict(name="WiSig_ManySig",   file="ManySig.pkl",   total_devices=6,   ratio=6, clusters=(80,)),
}


def seed_everything(s):
    """Pin every RNG a run touches."""
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


# =============================================================================
# Experiment 1 - DCI disentanglement on the synthetic benchmark (Table I)
# =============================================================================

def evaluate_architecture(config, iq_imb=True, lr=0.01, n_runs=10, n_epochs=120,
                          full_range=True, cfo_compensate=False, device=DEVICE,
                          verbose=True):
    """Train one architecture `n_runs` times; score DCI on the held-out set.

    The autoencoder is trained on the 1000-device training split and every
    metric is computed on the independent 100-device evaluation split.
    """
    trainloader, valloader = get_synthetic_loaders(
        iq_imb, config["data_polars"],
        full_range=full_range, cfo_compensate=cfo_compensate,
    )

    factors = torch.cat([f for _, f in valloader]).numpy()
    results = {"disentanglement": [], "completeness": [], "spearman": [], "recon_mse": []}

    for run in range(n_runs):
        torch.manual_seed(run)

        models = nn.ModuleDict({"feature_extractor": config["model"]()})
        optimizers = {"main_optimizer": torch.optim.Adam(models.parameters(), lr=lr)}
        trainer = AE_Trainer(models=models, optimizers=optimizers,
                             num_epochs=n_epochs, device=device)

        for _ in range(n_epochs):
            trainer.train_epoch(trainloader)

        features = trainer.get_features(valloader).numpy()
        dci = compute_dci(features, factors)
        spearman = abs(stats.spearmanr(features.sum(-1), factors.sum(-1)).statistic)
        mse = reconstruction_mse(models["feature_extractor"].to(device), valloader, device)

        results["disentanglement"].append(dci["disentanglement"])
        results["completeness"].append(dci["completeness"])
        results["spearman"].append(spearman)
        results["recon_mse"].append(mse)

        if verbose:
            print(f"  run {run:>2}: D={dci['disentanglement']:.3f}  "
                  f"C={dci['completeness']:.3f}  rho={spearman:.3f}  mse={mse:.4f}")

    return results


def run_dci_benchmark(configs=None, **kwargs):
    """Run `evaluate_architecture` over every architecture.

    Returns:
        (all_results, summary) - the raw per-run dict and a mean +/- 95% CI table.
    """
    configs = AE_CONFIGS if configs is None else configs
    all_results = {}
    for config in configs:
        print(f"=== {config['name']} ===")
        all_results[config["name"]] = evaluate_architecture(config, **kwargs)
    return all_results, summarise_dci(all_results)


def summarise_dci(all_results):
    """Mean +/- 95% CI table over the per-run DCI results."""
    rows = []
    for name, res in all_results.items():
        row = {"architecture": name}
        for metric, values in res.items():
            mean, ci = mean_ci95(values)
            row[metric] = f"{mean:.3f} +/- {ci:.3f}"
        rows.append(row)
    return pd.DataFrame(rows).set_index("architecture")


# =============================================================================
# Experiment 2 - synthetic UED classification (Table II, Synthetic)
# =============================================================================

def build_synth_ued(polar, cfo_compensate, n_devices=200, bursts=20,
                    unknown_frac=0.25, burst_len=256, sps=4, seed=7):
    """Build the synthetic open-set split: a fixed fraction of devices is unknown.

    Returns:
        (train_set, test_set, targets) where targets[i] is 1 for an unknown device.
    """
    full = SyntheticUEDDataset(
        n_devices=n_devices, n_bursts_per_device=bursts, burst_len=burst_len,
        sps=sps, snr_range=(10, 30), seed=seed, polar=polar, iq_imb=True,
        full_range=True, cfo_compensate=cfo_compensate,
    )
    full.return_indices = False

    dev = np.asarray(full.device_id)
    rng = np.random.default_rng(seed)
    unknown = set(rng.choice(n_devices, int(n_devices * unknown_frac), replace=False).tolist())

    train_idx, test_idx, test_targets = [], [], []
    for d in range(n_devices):
        idxs = np.where(dev == d)[0]
        if d in unknown:
            test_idx += idxs.tolist()
            test_targets += [1] * len(idxs)
        else:
            k = int(len(idxs) * 0.8)
            train_idx += idxs[:k].tolist()
            test_idx += idxs[k:].tolist()
            test_targets += [0] * len(idxs[k:])

    train_set, test_set = Subset(full, np.array(train_idx)), Subset(full, np.array(test_idx))
    train_set.return_indices = False
    test_set.return_indices = False
    return train_set, test_set, np.array(test_targets)


def train_eval_synth_ued(arch, train_set, test_set, targets, seed,
                         n_epochs=120, clusters=(500,), lr=0.01, device=DEVICE):
    """Train one architecture on the synthetic open-set split and score UED."""
    a = ARCHS[arch]
    seed_everything(seed)                      # seed before model init and loader
    g = torch.Generator().manual_seed(seed)    # reproducible shuffling

    tl = DataLoader(train_set, batch_size=50, shuffle=True, generator=g)
    vl = DataLoader(test_set, batch_size=50, shuffle=False)

    exp = {
        "feature_extractor": {"name": a["fe"], "config": a["fe_cfg"]},
        "approach": {
            "name": "AE",
            "config": {"noise_std": 0.01, "device": device, "num_epochs": n_epochs},
            "trainer": {"main_optimizer": {"name": "Adam", "config": {"lr": lr}}},
        },
    }
    trainer = cm.get_trainer(exp)
    for _ in range(trainer.num_epochs):
        trainer.train_epoch(tl)

    m = trainer.evaluate(train_loader=tl, test_loader=vl, targets=targets,
                         clusters_numbers=clusters)
    return {name: float(np.mean([v for k, v in m.items() if k.endswith("_" + name)]) or np.nan)
            for name in METRIC_NAMES}


def run_synthetic_ued(seeds=(0, 1, 2), n_epochs=120, archs=None, save=True, outdir="results"):
    """Synthetic UED for every architecture, uncompensated and CFO-compensated.

    Returns:
        (summary, perseed) DataFrames.
    """
    archs = ARCHS if archs is None else archs
    rows, perseed_rows = [], []

    for arch in archs:
        a = ARCHS[arch]
        for comp in (False, True):
            # The split is deterministic, so it is built once per arch/compensation.
            train_set, test_set, targets = build_synth_ued(a["polars"], comp)
            vals = {m: [] for m in METRIC_NAMES}

            for seed in seeds:
                print(f"=== synthetic UED | {arch} | comp={comp} | seed={seed} ===", flush=True)
                r = train_eval_synth_ued(arch, train_set, test_set, targets, seed,
                                         n_epochs=n_epochs)
                for m in METRIC_NAMES:
                    vals[m].append(r[m])
                perseed_rows.append({"architecture": arch, "cfo_compensate": comp,
                                     "seed": seed, **r})

            agg = {}
            for m in METRIC_NAMES:
                agg[m + "_mean"] = float(np.mean(vals[m]))
                agg[m + "_std"] = float(np.std(vals[m], ddof=1)) if len(vals[m]) > 1 else 0.0
            rows.append({"architecture": arch, "cfo_compensate": comp, **agg})
            print(f"  -> {arch} comp={comp}: AUC {agg['roc_auc_mean'] * 100:.1f} "
                  f"+/- {agg['roc_auc_std'] * 100:.1f}", flush=True)

    summary = pd.DataFrame(rows).set_index(["architecture", "cfo_compensate"])
    perseed = pd.DataFrame(perseed_rows)

    if save:
        ts = _timestamp(outdir)
        summary.to_csv(f"{outdir}/synthetic_ued_{ts}.csv")
        perseed.to_csv(f"{outdir}/synthetic_ued_perseed_{ts}.csv", index=False)
        print(f"saved {outdir}/synthetic_ued_{ts}.csv and {outdir}/synthetic_ued_perseed_{ts}.csv")

    return summary, perseed


# =============================================================================
# Experiment 3 - WiSig UED (Table II, SingleDay and ManyTx)
# =============================================================================

def build_wisig_config(subset, arch, cfo_compensate, num_epochs=120, lr=0.001,
                       device=DEVICE):
    """config_manager experiment dict for one (subset, architecture) pair."""
    s, a = SUBSETS[subset], ARCHS[arch]
    return {
        "num_iterations": 1, "starting_iteration": 0, "exp_id": f"{arch}_{subset}",
        "logs_dir": "logs", "report_interval": 10,
        "dataset": {"name": s["name"], "config": {
            "selected_receivers": (0,), "file": s["file"], "polars": a["polars"],
            "cfo_compensate": cfo_compensate,
            "k_fold": {"total_devices": s["total_devices"], "ratio": s["ratio"]},
            "test_config": {"type": "validation"}, "train_config": {"type": "train"}}},
        "train_loader": {"num_workers": 0, "batch_size": 50, "shuffle": True},
        "test_loader": {"num_workers": 0, "batch_size": 50, "shuffle": False},
        "evaluation": {"clusters_numbers": s["clusters"]},
        "approach": {"name": "AE",
                     "config": {"noise_std": 0.01, "device": device, "num_epochs": num_epochs},
                     "trainer": {"main_optimizer": {"name": "Adam", "config": {"lr": lr}}}},
        "feature_extractor": {"name": a["fe"], "config": a["fe_cfg"]},
    }


def run_wisig(exp_config, verbose=True, progress=None):
    """Run one WiSig experiment over its device folds.

    A no-wandb replica of ``config_manager.evauate_config``.

    Returns:
        (agg, per_fold) - aggregate mean/std dict and per-metric fold lists.
    """
    exp_config = copy.deepcopy(exp_config)
    train_cfgs, test_cfgs, unknown_folds = cm.get_data_configs(exp_config["dataset"]["config"])
    dataset_cls = cm.DATASETS[exp_config["dataset"]["name"]]
    clusters = exp_config["evaluation"]["clusters_numbers"]

    per_fold = {n: [] for n in METRIC_NAMES}
    for fold, (tr, te, unk) in enumerate(zip(train_cfgs, test_cfgs, unknown_folds)):
        train_set, test_set = dataset_cls(**tr), dataset_cls(**te)
        targets = np.array([test_set[i][1] in unk for i in range(len(test_set))])
        tl = DataLoader(train_set, **exp_config["train_loader"])
        vl = DataLoader(test_set, **exp_config["test_loader"])

        trainer = cm.get_trainer(exp_config)
        epochs = range(trainer.num_epochs)
        if progress is not None:
            epochs = progress(epochs, desc=f"{exp_config['exp_id']} fold{fold}", leave=False)
        for ep in epochs:
            loss = trainer.train_epoch(tl)
            if progress is None and verbose and (ep % 10 == 0 or ep == trainer.num_epochs - 1):
                print(f"      fold {fold} | epoch {ep + 1}/{trainer.num_epochs} | loss={loss:.4f}",
                      flush=True)

        m = trainer.evaluate(train_loader=tl, test_loader=vl, targets=targets,
                             clusters_numbers=clusters)
        for name in METRIC_NAMES:
            vals = [v for k, v in m.items() if k.endswith("_" + name)]
            per_fold[name].append(float(np.mean(vals)) if vals else np.nan)
        if verbose:
            print(f"    fold {fold}: " + "  ".join(f"{n}={per_fold[n][-1]:.3f}" for n in METRIC_NAMES))

    agg = {**{n + "_mean": float(np.nanmean(per_fold[n])) for n in METRIC_NAMES},
           **{n + "_std": float(np.nanstd(per_fold[n])) for n in METRIC_NAMES}}
    return agg, per_fold


def run_wisig_ued(subset="SingleDay", n_epochs=120, archs=None, save=True, outdir="results"):
    """WiSig UED for every architecture on one subset, uncompensated and compensated.

    Returns:
        (summary, per_fold) DataFrames.
    """
    archs = ARCHS if archs is None else archs
    rows, perfold_rows = [], []

    for arch in archs:
        for comp in (False, True):
            print(f"=== {subset} | {arch} | cfo_compensate={comp} ===")
            agg, pf = run_wisig(build_wisig_config(subset, arch, comp, num_epochs=n_epochs))
            rows.append({"architecture": arch, "cfo_compensate": comp, **agg})
            for f in range(len(pf["roc_auc"])):
                perfold_rows.append({"architecture": arch, "cfo_compensate": comp, "fold": f,
                                     **{n: pf[n][f] for n in METRIC_NAMES}})

    summary = pd.DataFrame(rows).set_index(["architecture", "cfo_compensate"])
    perfold = pd.DataFrame(perfold_rows)

    if save:
        ts = _timestamp(outdir)
        summary.to_csv(f"{outdir}/wisig_ued_{ts}.csv")
        perfold.to_csv(f"{outdir}/wisig_perfold_{ts}.csv", index=False)
        print(f"saved {outdir}/wisig_ued_{ts}.csv and {outdir}/wisig_perfold_{ts}.csv")

    return summary, perfold


# =============================================================================
# Model size and compute
# =============================================================================

def model_size_table(device="cpu", batch_size=256, reps=50, latex=False):
    """Parameter counts and forward-pass timing for the six feature extractors.

    "Active" counts scalars with non-zero gradient support over a large random
    batch, so only structurally pruned edges (Polar MKAN's block masks) are
    excluded rather than merely inactive non-linearities.
    """
    models = {
        "CNN":        lambda: AE_CNN_1D(input_signal_length=LEN, features_size=3),
        "KAN":        lambda: KAN_AE(input_size=512, hidden_size=3, grid_size=8, monotonic=False, polars=False),
        "MKAN":       lambda: KAN_AE(input_size=512, hidden_size=3, grid_size=8, monotonic=True,  polars=False),
        "Polar CNN":  lambda: AE_CNN_1D(input_signal_length=LEN, features_size=3),
        "Polar KAN":  lambda: KAN_AE(input_size=512, hidden_size=3, grid_size=8, monotonic=False, polars=False),
        "Polar MKAN": lambda: KAN_AE(input_size=512, hidden_size=3, grid_size=8, monotonic=True,  polars=True),
    }

    def total_params(m):
        return sum(p.numel() for p in m.parameters())

    def active_params(m):
        m.zero_grad(set_to_none=True)
        x = torch.randn(batch_size, CH, LEN, device=device)
        out = m(x)
        out = out if isinstance(out, (tuple, list)) else (out,)
        loss = sum(o.float().abs().sum() for o in out)
        loss.backward()
        return sum(int(torch.count_nonzero(p.grad)) for p in m.parameters() if p.grad is not None)

    def fwd_ms(m):
        x = torch.randn(1, CH, LEN, device=device)
        with torch.no_grad():
            for _ in range(5):
                m(x)                                  # warm-up
            t = time.perf_counter()
            for _ in range(reps):
                m(x)
        return (time.perf_counter() - t) / reps * 1e3

    rows = []
    for name, build in models.items():
        m = build().to(device).eval()
        tot, act = total_params(m), active_params(m)
        rows.append({"Architecture": name, "Params": tot, "Active": act,
                     "Active %": round(100 * act / tot, 1), "Fwd (ms)": round(fwd_ms(m), 3)})

    df = pd.DataFrame(rows).set_index("Architecture")
    if latex:
        print("\n% --- LaTeX table body ---\n\\begin{tabular}{lccc}\n\\toprule")
        print("FE & Params & Active & Active\\% \\\\ \\midrule")
        for n, r in df.iterrows():
            print(f"{n} & {int(r.Params):,} & {int(r.Active):,} & {r['Active %']:.0f}\\% \\\\")
        print("\\bottomrule\n\\end{tabular}")
    return df


# =============================================================================
# Capacity sweep
# =============================================================================

_SWEEP_SPEC = {
    "KAN":        dict(monotonic=False, polars=False, data_polars=False),
    "MKAN":       dict(monotonic=True,  polars=False, data_polars=False),
    "Polar KAN":  dict(monotonic=False, polars=False, data_polars=True),
    "Polar MKAN": dict(monotonic=True,  polars=True,  data_polars=True),
}


def run_capacity_sweep(grid_sizes=(16, 32), archs=("Polar MKAN", "KAN"),
                       subset="ManySig", wisig_epochs=120, dci_runs=1,
                       dci_epochs=120, dci_lr=0.01, full_range=True,
                       cfo_compensate=False, device=DEVICE):
    """Vary the KAN grid size and report DCI together with WiSig UED."""
    try:
        from tqdm.auto import tqdm
    except ImportError:                       # graceful fallback if tqdm missing
        def tqdm(it, **k):
            return it

    def dci_for(model_fn, data_polars):
        tl, _ = get_synthetic_loaders(True, data_polars, full_range=full_range,
                                      cfo_compensate=cfo_compensate)
        factors = torch.cat([f for _, f in tl]).numpy()
        ds, cs = [], []
        for run in range(dci_runs):
            torch.manual_seed(run)
            models = nn.ModuleDict({"feature_extractor": model_fn()})
            opt = {"main_optimizer": torch.optim.Adam(models.parameters(), lr=dci_lr)}
            trainer = AE_Trainer(models=models, optimizers=opt,
                                 num_epochs=dci_epochs, device=device)
            for _ in tqdm(range(dci_epochs), desc=f"  run {run}", leave=False):
                trainer.train_epoch(tl)
            dci = compute_dci(trainer.get_features(tl).numpy(), factors)
            ds.append(dci["disentanglement"])
            cs.append(dci["completeness"])
        return np.mean(ds) * 100, np.mean(cs) * 100

    rows = []
    for grid, arch in tqdm([(g, a) for g in grid_sizes for a in archs], desc="sweep"):
        spec = _SWEEP_SPEC[arch]
        model_fn = (lambda g=grid, s=spec: KAN_AE(input_size=512, hidden_size=3, grid_size=g,
                                                  monotonic=s["monotonic"], polars=s["polars"]))
        d, c = dci_for(model_fn, spec["data_polars"])

        cfg = build_wisig_config(subset, arch, False, num_epochs=wisig_epochs, device=device)
        cfg["feature_extractor"]["config"]["grid_size"] = grid
        agg, _ = run_wisig(cfg, verbose=False, progress=tqdm)

        rows.append({"arch": arch, "grid_size": grid, "D": round(d, 1), "C": round(c, 1),
                     f"{subset}_AUC": round(agg["roc_auc_mean"] * 100, 1),
                     f"{subset}_F1": round(agg["f1_mean"] * 100, 1)})
        print(rows[-1])

    return pd.DataFrame(rows).set_index(["arch", "grid_size"])


# =============================================================================
# Perturbation analysis
# =============================================================================

# (name, model factory, input mode). Distinct from ARCHS above: these are direct
# model factories, not config_manager specs.
PERTURB_ARCHS = [
    ("PolarMKAN", lambda: KAN_AE(input_size=512, hidden_size=3, grid_size=8,
                                 monotonic=True, polars=True), "polar"),
    ("MKAN",      lambda: KAN_AE(input_size=512, hidden_size=3, grid_size=8,
                                 monotonic=True, polars=False), "iq"),
]

# Polar MKAN feature order in the current implementation: [F_R, F_phi1, F_phi2].
F_R_IDX, F_PHI_IDX = [0], [1, 2]

DEFAULT_ALPHAS = np.linspace(-0.6, 0.6, 25)     # phase rotations, rad
DEFAULT_DELTAS = np.linspace(-0.30, 0.30, 25)   # relative amplitude changes


def get_or_train(name, factory, seed, ckpt_dir="ckpt", n_epochs=120, iq_imb=True,
                 lr=0.01, device=DEVICE):
    """Load ``ckpt/{name}_seed{seed}.pt`` if present, otherwise train and save it."""
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"{name}_seed{seed}.pt")

    model = factory()
    if os.path.exists(path):
        model.load_state_dict(torch.load(path, map_location=device))
        return model.to(device).eval()

    torch.manual_seed(seed)
    model = factory()                                    # seeded re-init
    polar_data = name.lower().startswith("polar")
    trainloader, _ = get_synthetic_loaders(iq_imb, polar_data)

    models = nn.ModuleDict({"feature_extractor": model})
    opt = {"main_optimizer": torch.optim.Adam(models.parameters(), lr=lr)}
    trainer = AE_Trainer(models=models, optimizers=opt, num_epochs=n_epochs, device=device)
    for _ in range(n_epochs):
        trainer.train_epoch(trainloader)

    torch.save(model.state_dict(), path)
    return model.to(device).eval()


@torch.no_grad()
def encode(model, mag, phase, mode, device=DEVICE):
    """Encode a (magnitude, phase) batch through the model in the given input mode."""
    if mode == "polar":
        x = torch.stack([mag / MAG_SCALE, phase / TWO_PI], dim=1)
    else:
        x = torch.stack([mag * torch.cos(phase), mag * torch.sin(phase)], dim=1)
    out = model(x.to(device))
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out.detach().cpu().float()


def _eval_loader(batch_size=256, n_devices=100, seed=41, iq_imb=True):
    """Polar-formatted evaluation set; the polar flag only changes formatting."""
    ds = RFFingerprintDataset(n_devices=n_devices, n_bursts_per_device=1, burst_len=LEN,
                              sps=4, snr_range=(10, 30), seed=seed, polar=True,
                              iq_imb=iq_imb, full_range=True, cfo_compensate=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def sweep_perturbations(model, mode, loader, alphas=DEFAULT_ALPHAS, deltas=DEFAULT_DELTAS):
    """Sweep phase and amplitude perturbations over the evaluation set.

    Returns:
        (dP, dA, wr) - feature deltas under phase rotation, under amplitude
        change, and the per-burst fraction of samples that wrap.
    """
    d_phase, d_amp, wrap = [], [], []
    for batch in loader:
        x = batch[0]
        mag, phase = x[:, 0] * MAG_SCALE, x[:, 1] * TWO_PI
        base = encode(model, mag, phase, mode)

        rp, rw = [], []
        for a in alphas:
            s = phase + a
            rw.append(((s >= TWO_PI) | (s < 0)).float().mean(1).numpy())
            rp.append((encode(model, mag, s % TWO_PI, mode) - base).numpy())
        d_phase.append(np.stack(rp, 1))
        wrap.append(np.stack(rw, 1))

        d_amp.append(np.stack([(encode(model, mag * (1 + d), phase, mode) - base).numpy()
                               for d in deltas], 1))

    return np.concatenate(d_phase), np.concatenate(d_amp), np.concatenate(wrap)


def perturbation_metrics(d_phase, d_amp, wrap, alphas=DEFAULT_ALPHAS,
                         deltas=DEFAULT_DELTAS, eps=0.01):
    """Sign consistency, cross-channel leakage and wrap-binned consistency.

    `eps` is a dead zone: a feature change smaller than this never counts as a
    sign violation.
    """
    nz_a, nz_d = np.abs(alphas) > 1e-12, np.abs(deltas) > 1e-12
    sa, sd = np.sign(alphas[nz_a])[None], np.sign(deltas[nz_d])[None]

    out = {}
    for k in F_PHI_IDX:
        v = d_phase[:, nz_a, k]
        out[f"sign_phi{k}"] = ((np.sign(v) == sa) | (np.abs(v) < eps)).mean()

    v = d_amp[:, nz_d, F_R_IDX[0]]
    out["sign_R"] = ((np.sign(v) == sd) | (np.abs(v) < eps)).mean()

    out["leak_phase->R"] = (np.abs(d_phase[:, nz_a][..., F_R_IDX]).mean()
                            / max(np.abs(d_phase[:, nz_a][..., F_PHI_IDX]).mean(), 1e-12))
    out["leak_amp->phi"] = (np.abs(d_amp[:, nz_d][..., F_PHI_IDX]).mean()
                            / max(np.abs(d_amp[:, nz_d][..., F_R_IDX]).mean(), 1e-12))

    # Sign consistency of the dominant phase feature, binned by wrap fraction.
    dom = max(F_PHI_IDX, key=lambda k: np.abs(d_phase[:, nz_a, k]).mean())
    ok = (np.sign(d_phase[:, nz_a, dom]) == sa) | (np.abs(d_phase[:, nz_a, dom]) < eps)
    w = wrap[:, nz_a]
    for lo, hi in [(0, .02), (.02, .05), (.05, .10), (.10, 1.0)]:
        m = (w >= lo) & (w < hi)
        out[f"sign_dom_wrap[{lo:.2f},{hi:.2f})"] = ok[m].mean() if m.any() else np.nan
    return out


def run_perturbation_analysis(n_seeds=10, ckpt_dir="ckpt", n_epochs=120, verbose=True):
    """Multi-seed phase/amplitude perturbation sweep for each perturbation arch.

    Mean response curves are saved to ``{ckpt_dir}/curves_{name}.npz``.

    Returns:
        dict mapping architecture name to a per-seed list of metric dicts.
    """
    loader = _eval_loader()
    results = {}

    for name, factory, mode in PERTURB_ARCHS:
        runs, curves_p, curves_a = [], [], []
        for seed in range(n_seeds):
            model = get_or_train(name, factory, seed, ckpt_dir=ckpt_dir, n_epochs=n_epochs)
            d_phase, d_amp, wrap = sweep_perturbations(model, mode, loader)
            runs.append(perturbation_metrics(d_phase, d_amp, wrap))
            curves_p.append(d_phase.mean(0))
            curves_a.append(d_amp.mean(0))
            if verbose:
                print(f"{name} seed {seed}: " + "  ".join(
                    f"{k}={v:.3f}" for k, v in runs[-1].items() if not k.startswith("sign_dom")))

        results[name] = runs
        np.savez(os.path.join(ckpt_dir, f"curves_{name}.npz"),
                 ALPHAS=DEFAULT_ALPHAS, DELTAS=DEFAULT_DELTAS,
                 phase=np.stack(curves_p), amp=np.stack(curves_a))

        if verbose:
            print(f"\n=== {name}: mean +/- std over {n_seeds} seeds ===")
            for k in runs[0]:
                vals = np.array([r[k] for r in runs], dtype=float)
                print(f"  {k:28s} {np.nanmean(vals):.3f} +/- {np.nanstd(vals):.3f}")
            print()

    return results


def run_guarantee_check(n_seeds=10, ckpt_dir="ckpt",
                        alphas_fine=(0.002, 0.005, 0.01, 0.02), tol=1e-6):
    """Separate the monotonicity guarantee from the branch-cut (circularity) effect.

    Probe A (clamped): phi' = min(phi + a, 2pi - eps). Every input change is
        >= 0 and no sample wraps, so the MKAN guarantee applies exactly and the
        sign-consistency rate should be 1.000. Anything less is an implementation bug.
    Probe B (wrap-free bursts): tiny physical (mod 2pi) rotations restricted to
        bursts in which no sample wraps; the guarantee applies there too.
    """
    loader = _eval_loader()
    name, factory, mode = PERTURB_ARCHS[0]                 # PolarMKAN

    ok_clamp, ok_free, n_free = [], [], 0
    for seed in range(n_seeds):
        model = get_or_train(name, factory, seed, ckpt_dir=ckpt_dir)
        for batch in loader:
            x = batch[0]
            mag, ph = x[:, 0] * MAG_SCALE, x[:, 1] * TWO_PI
            base = encode(model, mag, ph, mode)

            for a in alphas_fine:
                # Probe A: clamped shift, monotone by construction, no wrap.
                d_f = encode(model, mag, torch.clamp(ph + a, max=TWO_PI - 1e-6), mode) - base
                ok_clamp.append((d_f[:, F_PHI_IDX].numpy() >= -tol).mean())

                # Probe B: physical rotation, wrap-free bursts only.
                wrap_free = ((ph + a) < TWO_PI).all(dim=1)
                if wrap_free.any():
                    d_fm = encode(model, mag, (ph + a) % TWO_PI, mode) - base
                    ok_free.append((d_fm[wrap_free][:, F_PHI_IDX].numpy() >= -tol).mean())
                    n_free += int(wrap_free.sum())

    result = {"probe_a_clamped": float(np.mean(ok_clamp)),
              "probe_b_wrap_free": float(np.mean(ok_free)),
              "n_wrap_free_pairs": n_free}
    print(f"Probe A  clamped shift, sign>=0 rate : {result['probe_a_clamped']:.4f}  (expected 1.0000)")
    print(f"Probe B  wrap-free bursts, mod shift : {result['probe_b_wrap_free']:.4f}  "
          f"over {n_free} wrap-free burst-alpha pairs (expected 1.0000)")
    return result


WRAP_BIN_LABELS = ["0%", "(0,2%]", "(2,5%]", "(5,10%]", ">10%"]


def _wrap_bin_masks(w):
    return [
        w == 0,
        (w > 0) & (w <= 0.02),
        (w > 0.02) & (w <= 0.05),
        (w > 0.05) & (w <= 0.10),
        w > 0.10,
    ]


def _get_polar_mkan_wrap(seed, ckpt_dir="ckpt", n_epochs=120, lr=0.01, device=DEVICE):
    """Polar MKAN trained on the iq_imb polar split, cached per seed."""
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = os.path.join(ckpt_dir, f"PolarMKAN_wrap_seed{seed}.pt")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = KAN_AE(input_size=512, hidden_size=3, grid_size=8, monotonic=True, polars=True)
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        return model.to(device).eval()

    trainloader, _ = get_synthetic_loaders(True, True, full_range=True, cfo_compensate=False)
    models = nn.ModuleDict({"feature_extractor": model})
    optimizers = {"main_optimizer": torch.optim.Adam(models.parameters(), lr=lr)}
    trainer = AE_Trainer(models=models, optimizers=optimizers, num_epochs=n_epochs, device=device)
    for _ in range(n_epochs):
        trainer.train_epoch(trainloader)

    torch.save(model.state_dict(), ckpt)
    return model.to(device).eval()


def run_branch_cut_validation(n_seeds=10, n_epochs=120, ckpt_dir="ckpt", tol=1e-6,
                              alphas=None, verbose=True):
    """Sign consistency of the phase features against the phase-wrap fraction.

    Applies the physical rotation phi' = (phi + alpha) mod 2pi and, for each
    burst/shift pair, records the fraction of samples crossing the branch cut
    and whether Delta F_phi1 and Delta F_phi2 have the expected sign. At 0%
    wrapping the consistency should be ~1.0.

    Returns:
        dict with `means` and `ci95` of shape (2, n_bins), `counts`, and `labels`.
    """
    if alphas is None:
        alphas = DEFAULT_ALPHAS[np.abs(DEFAULT_ALPHAS) > 1e-12]   # drop alpha = 0

    _, valloader = get_synthetic_loaders(iq_imb=True, polar=True, full_range=True,
                                         cfo_compensate=False)
    n_bins = len(WRAP_BIN_LABELS)
    seed_results = np.full((n_seeds, 2, n_bins), np.nan)
    seed_counts = np.zeros((n_seeds, n_bins), dtype=int)

    for seed in range(n_seeds):
        if verbose:
            print(f"=== Polar MKAN seed {seed} ===")
        model = _get_polar_mkan_wrap(seed, ckpt_dir=ckpt_dir, n_epochs=n_epochs)

        bin_ok = [[[], []] for _ in WRAP_BIN_LABELS]
        bin_counts = np.zeros(n_bins, dtype=int)

        for batch, _ in valloader:
            # Dataset polar representation: channel 0 = R / 1.5, channel 1 = phi / 2pi.
            mag = batch[:, 0] * MAG_SCALE
            phase = batch[:, 1] * TWO_PI
            base = encode(model, mag, phase, "polar")

            for alpha in alphas:
                phase_unwrapped = phase + alpha
                wrap_fraction = ((phase_unwrapped < 0) |
                                 (phase_unwrapped >= TWO_PI)).float().mean(dim=1)
                shifted = encode(model, mag, torch.remainder(phase_unwrapped, TWO_PI), "polar")
                d_f = shifted - base

                # Expected direction from monotonicity.
                if alpha > 0:
                    ok1, ok2 = d_f[:, F_PHI_IDX[0]] >= -tol, d_f[:, F_PHI_IDX[1]] >= -tol
                else:
                    ok1, ok2 = d_f[:, F_PHI_IDX[0]] <= tol, d_f[:, F_PHI_IDX[1]] <= tol

                w = wrap_fraction.numpy()
                for b, mask in enumerate(_wrap_bin_masks(w)):
                    if not np.any(mask):
                        continue
                    bin_ok[b][0].extend(ok1.numpy()[mask].astype(float))
                    bin_ok[b][1].extend(ok2.numpy()[mask].astype(float))
                    bin_counts[b] += int(mask.sum())

        for b in range(n_bins):
            for k in range(2):
                if bin_ok[b][k]:
                    seed_results[seed, k, b] = np.mean(bin_ok[b][k])
            seed_counts[seed, b] = bin_counts[b]

        if verbose:
            print("  F_phi1:", np.round(seed_results[seed, 0], 3))
            print("  F_phi2:", np.round(seed_results[seed, 1], 3))

    means = np.nanmean(seed_results, axis=0)
    stds = np.nanstd(seed_results, axis=0, ddof=1)
    n_valid = np.sum(~np.isnan(seed_results), axis=0)
    ci95 = 1.96 * stds / np.sqrt(np.maximum(n_valid, 1))
    counts = seed_counts.sum(axis=0)

    if verbose:
        print("\n=== Mean +/- 95% CI over seeds ===")
        for b, label in enumerate(WRAP_BIN_LABELS):
            print(f"{label:>8s} | F_phi1 = {means[0, b]:.3f} +/- {ci95[0, b]:.3f} | "
                  f"F_phi2 = {means[1, b]:.3f} +/- {ci95[1, b]:.3f} | N = {counts[b]}")

        if np.nanmin(means[:, 0]) >= 0.999:
            print("\nPASS: zero-wrap phase shifts satisfy the monotonic sign guarantee.")
        else:
            print("\nWARNING: zero-wrap consistency is below 0.999. "
                  "Inspect implementation before using this figure.")

    return {"means": means, "ci95": ci95, "counts": counts, "labels": WRAP_BIN_LABELS,
            "per_seed": seed_results}


# =============================================================================
# Result persistence
# =============================================================================

RESULT_LABELS = {
    "summary": "synthetic_dci",              # DCI disentanglement / completeness
    "all_results": "synthetic_dci_perrun",   # raw per-run DCI values
    "wisig_summary": "wisig_ued",            # real-data UED
    "synth_ued_summary": "synthetic_ued",    # synthetic UED
}


def _timestamp(outdir):
    os.makedirs(outdir, exist_ok=True)
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_results(outdir="results", **objects):
    """Write named result objects to timestamped CSVs under `outdir`.

    Keys are matched against RESULT_LABELS for the filename prefix; unknown keys
    use the key itself. Pass only the objects that exist, e.g.::

        save_results(summary=summary, all_results=all_results)
    """
    stamp = _timestamp(outdir)
    saved = []
    for var, obj in objects.items():
        if obj is None:
            continue
        label = RESULT_LABELS.get(var, var)
        path = os.path.join(outdir, f"{label}_{stamp}.csv")
        try:
            if isinstance(obj, pd.DataFrame):
                obj.to_csv(path)
            elif isinstance(obj, dict):        # {arch: {metric: [runs]}}
                pd.concat({a: pd.DataFrame(r) for a, r in obj.items()},
                          names=["architecture", "run"]).to_csv(path)
            else:
                pd.DataFrame(obj).to_csv(path)
            saved.append(path)
            print("saved:", path)
        except Exception as e:                 # noqa: BLE001 - report and continue
            print(f"skip {var}: {e}")

    print(f"\nSaved {len(saved)} file(s) to {os.path.abspath(outdir)}")
    return saved
