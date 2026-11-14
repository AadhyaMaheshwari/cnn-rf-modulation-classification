
# CNN-based Automatic RF Modulation Classification

It analyses MacLellan, Crockett & Stewart, *RFSoC Modulation Classification With Streaming CNN: Data Set Generation & Quantized-Aware Training*, IEEE OJCAS 6, 2025 (doi:10.1109/OJCAS.2024.3509627), and reproduces its quantisation study **in software on simulated data**.


## Contents
| Path | Purpose |
|---|---|
| `src/gen.py` | Simulator: 8 classes, raised-cosine pulses, 3-tap Rician channel, CFO (±5 kHz), AWGN at 14 SNRs; writes `data.npz` (53,760 windows, split by long frame 40/10/10) |
| `src/net.py` | Network (conv 64×1×3 → conv 16×2×3 → dense 1984×128 → dense 128×8), hand-written backprop, Adam, fake-quantised weights/activations with a straight-through estimator |
| `src/bench.py` | Speed test and finite-difference gradient check |
| `src/run.py` | Float32 baseline, PTQ 16/8/4, per-layer 4-bit PTQ, QAT 16/8/4 and three mixed allocations → `results/results.json` |
| `src/run2.py` | PTQ at 3 and 2 bits (all layers, per layer) and the float fine-tune control → `results/results2.json` |
| `src/run3.py` | QAT 2-bit and the two 2-bit mixed allocations, one name per call → `results/results2.json` |
| `src/analytic.py` | Layer-wise weights, MACs, cycles and consistency checks on the paper's numbers |
| `src/figs1.py`, `src/figs2.py` | Report figures → `figs/` |
| `results/` | Raw numbers behind every table and chart in the report |
| `checkpoints/fp32_ckpt.npz` | Trained float model that `run2.py` and `run3.py` start from |

## Run (from the repository root)
```bash
pip install -r requirements.txt
python src/gen.py                       
python src/bench.py                     
python src/run.py                      
python src/run2.py
python src/run3.py QAT_2w16a
python src/run3.py MP_c4_d2
python src/run3.py MP_c8_d2
python src/analytic.py
python src/figs1.py && python src/figs2.py
```
`run.py` retrains the float model and overwrites `checkpoints/fp32_ckpt.npz`, so a rerun is a fresh single-seed run and will differ slightly from the stored results.

## Key results (SNR ≥ 20 dB, simulated data, one run per model)
Float32 63.0% · float + 10 extra epochs 63.9% · QAT 16/8/4-bit 64.1 / 64.2 / 62.5% · uniform 2-bit 13.4% · 8/8/2/2 mixed 40.2%.
Differences under about 3 points are within run-to-run noise (about 2.7-point standard error). The paper's 76/70/63% staircase was **not** reproduced.

## Limitations
- Single seed and simulated data; no converter, filter or spur effects.
- Network reconstructed from the paper's text (its Table 1 was not legible); dropout 0.3 is our choice.
- Packed-storage and block-RAM figures are derived bounds, not synthesis results.
