"""
Estimate the empirical hardware-impairment spread in a WiSig subset, to calibrate
the synthetic dataset's sigmas to something comparable to WiSig.

Usage:
    python measure_wisig_impairments.py ManySig.pkl
    python measure_wisig_impairments.py ManyTx.pkl --rx 0 --max-bursts 80

Pure numpy. Loads the whole pickle (needs enough RAM: ManySig ~2.4GB, ManyTx ~4.2GB),
then samples bursts per device on receiver `--rx` and reports:
  * gain      : across-device spread of per-device RMS amplitude (fractional -> compare to synthetic 'ampl' sigma=0.02)
  * cfo       : per-device mean normalised CFO, across-device std (compare to synthetic 'cfo' sigma=2e-4)
  * iq (rough): device-aggregated non-circularity |E[x^2]|/E[|x|^2] as a COMBINED amplitude+phase
                imbalance proxy (see note); compare to synthetic ~sqrt((0.115*0.3)^2 + 0.03^2) ~ 0.046
Prints diagnostics first so we can see what WiSig actually contains (raw vs normalised/equalised).
"""
import sys, pickle, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("pkl")
ap.add_argument("--rx", type=int, default=0)
ap.add_argument("--max-bursts", type=int, default=80, help="bursts sampled per device")
ap.add_argument("--max-devices", type=int, default=None)
args = ap.parse_args()

print(f"loading {args.pkl} ... (this can take a while for multi-GB files)")
with open(args.pkl, "rb") as f:
    data = pickle.load(f)

print("\n=== top-level structure ===")
if isinstance(data, dict):
    for k in data:
        v = data[k]
        info = f"len={len(v)}" if hasattr(v, "__len__") else str(type(v))
        print(f"  key '{k}': {type(v).__name__}  {info}")
tx_list = data.get("tx_list", None)
rx_list = data.get("rx_list", None)
days = data.get("capture_date_list", None)
n_tx = len(tx_list) if tx_list is not None else len(data["data"])
n_days = len(days) if days is not None else None
print(f"  devices={n_tx}  receivers={len(rx_list) if rx_list is not None else '?'}  days={n_days}")

def to_complex(burst):
    b = np.asarray(burst, dtype=np.float64)
    if b.ndim != 2:
        b = b.reshape(-1, 2)
    if b.shape[0] == 2 and b.shape[1] != 2:   # (2,N)
        I, Q = b[0], b[1]
    else:                                     # (N,2)
        I, Q = b[:, 0], b[:, 1]
    return I + 1j * Q

def device_bursts(tx, rx):
    """Yield bursts for a device on receiver rx across all days."""
    out = []
    entry = data["data"][tx][rx]
    day_range = range(len(entry))
    for m in day_range:
        try:
            sigs = entry[m][1]     # matches loader: data['data'][tx][rx][day][1]
        except Exception:
            sigs = entry[m]
        for s in sigs:
            out.append(s)
    return out

# ---- inspect one burst ----
b0 = to_complex(device_bursts(0, args.rx)[0])
print(f"\nexample burst: len={b0.size}, mean|x|^2={np.mean(np.abs(b0)**2):.4g}, "
      f"complex-dtype-ok={np.iscomplexobj(b0)}")

dev_rms, dev_cfo, dev_circ = [], [], []
ndev = n_tx if args.max_devices is None else min(n_tx, args.max_devices)
for tx in range(ndev):
    bursts = device_bursts(tx, args.rx)
    if len(bursts) == 0:
        continue
    idx = np.linspace(0, len(bursts) - 1, min(args.max_bursts, len(bursts))).astype(int)
    rms, cfo = [], []
    sx2 = 0.0 + 0j; sxx = 0.0   # for aggregated circularity
    for j in idx:
        x = to_complex(bursts[j])
        x = x - x.mean()                       # remove DC
        p = np.mean(np.abs(x) ** 2)
        if p <= 0: 
            continue
        rms.append(np.sqrt(p))
        d = np.angle(x[1:] * np.conj(x[:-1]))  # per-sample phase increment
        cfo.append(np.mean(d) / (2 * np.pi))   # normalised CFO (cycles/sample)
        sx2 += np.sum(x ** 2); sxx += np.sum(np.abs(x) ** 2)
    if not rms:
        continue
    dev_rms.append(np.mean(rms))
    dev_cfo.append(np.mean(cfo))
    dev_circ.append(abs(sx2) / sxx if sxx > 0 else np.nan)
    if tx < 6:
        print(f"  dev {tx:3d}: bursts={len(idx):3d}  RMS={dev_rms[-1]:.4g}  "
              f"cfo_norm={dev_cfo[-1]:+.3e}  circ={dev_circ[-1]:.4f}")

dev_rms = np.array(dev_rms); dev_cfo = np.array(dev_cfo); dev_circ = np.array(dev_circ)

def stats(a): 
    a = a[np.isfinite(a)]
    return a.mean(), a.std()

print("\n=== IMPAIRMENT SPREAD ACROSS DEVICES (rx=%d, %d devices) ===" % (args.rx, len(dev_rms)))
# gain: fractional spread of per-device RMS amplitude
gm, gs = stats(dev_rms)
print(f"gain  : per-device RMS amplitude mean={gm:.4g}  std={gs:.4g}  "
      f"-> fractional sigma ~= {gs/gm:.4f}   (synthetic 'ampl' sigma=0.02)")
cm, cs = stats(dev_cfo)
print(f"cfo   : per-device mean normalised CFO: mean={cm:+.3e}  ACROSS-DEVICE std={cs:.3e}  "
      f"(synthetic 'cfo' sigma=2e-4)")
print(f"        mean |CFO| = {np.mean(np.abs(dev_cfo)):.3e}  (if ~0, WiSig is CFO-compensated)")
im, iss = stats(dev_circ)
print(f"iq    : device non-circularity |E[x^2]|/E[|x|^2]: mean={im:.4f}  std={iss:.4f}")
print(f"        (COMBINED amp+phase imbalance proxy; paper 0.3dB/0.03rad -> ~0.046. "
      f"Finite-sample floor ~1/sqrt(Nsamp).)")
print("\nNote: if gain fractional sigma ~0 -> signals are power-normalised; "
      "if mean|CFO|~0 -> CFO already removed. Either changes what's realistic to inject.")
