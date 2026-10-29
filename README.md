# T24 - Over-the-air federated learning: power control under fading

Team 24, Communication Networks (VIT SCOPE). Saurabh Suresh (24BCE2750) and Advaith Sivakumar (24BCE2715).

This folder holds the code, results and web demo of the revised paper (second-level corrections). Every number in the paper is computed from the result files here.

## What is compared

All policies run on the same data and channels, with unit-power symbols, so the SNR is the per-device transmit SNR.

| Code name | In the paper | What it is |
|---|---|---|
| `M1` | M1 | full channel inversion (sanity check) |
| `M2` | M2 | truncated channel inversion |
| `M3` | M3 | multi-antenna truncated inversion (4 antennas, max-min combiner) |
| `M2-EIG` | M2+EIG | control: M3 with a fixed eigenvector combiner |
| `M1-BF` | M1+BF | control: M3 combiner without truncation |
| `OPT` | OPT | threshold structure of Cao et al. 2020 [7] |
| `RCI` | RCI | regularised channel inversion with a tuned regulariser |
| `C22` | C12 | Case I policy of Cao et al. 2022 [12] |
| `Z10` | Z10 | Theorem 1 of Zhang and Tao 2021 [10] |
| `Y14` | Y14 | policy with the structure described for Yu et al. 2023 [14] |
| `Y4` | Y4 | device selection and beamforming of Yang et al. 2020 [4] (DC algorithm) |
| `debias=True` | suffix U | unit-gain normalisation: the server divides by the sum of the received amplitudes |

These are our re-implementations, with tuned or replaced constants, not the authors' code.

## Setup

Python 3.10 or newer.

    pip install numpy scipy scikit-learn matplotlib cvxpy torch pillow

`cvxpy` is needed only for Y4 and `torch` only for CIFAR-10.

## Files

Simulator and experiments (digits task, run from this folder):

- `ota_fl_sim.py` - the simulator: channels, all policies, one training run. With its default settings it reproduces the earlier version (fixed scaling).
- `runner.py`, `stats_v3.py` - resumable job runner and the helpers for means and paired confidence intervals
- `explore_ref.py` - choice of the reference SNR on the tuning rounds
- `exp_tune.py` - tuning of the thresholds and constants on rounds 0-99 (writes `v3/tune.jsonl`)
- `exp_main.py <stage>` - main experiments; stages `main`, `csi`, `thr`, `rule`, `jakes`, `alpha`, `n8tune`, `n8`, `debias`, `uvar` (writes `v3/<stage>.jsonl`)
- `y4_pre.py`, `exp_y4.py` - the Y4 selections (cached in `y4cache/`) and the Y4 runs
- `check_z10.py` - checks the closed form of Z10 against a direct numerical optimum
- `demo.py` - quick check, about 20 seconds: the main policies at -35 dB with three seeds

Larger tasks:

- `mnist_cnn_sim.py`, `mnist_v3.py` - MNIST with a 13,610-parameter CNN in NumPy (writes `v3/mnist_v3.jsonl`)
- `cifar_sim.py`, `cifar_v3.py`, `cifar_prepare.py` - CIFAR-10 with a 136,874-parameter CNN in PyTorch (writes `v3/cifar_v3.jsonl`)

Figures, numbers and demo:

- `figs_v3.py` - Figures 2-10 of the paper (into `figs/`)
- `num_v3.py` - every number and table of the paper, computed from `v3/`
- `fig_pipeline.py` - Figure 1
- `p1_channel_generation.py` - the channel-generation script that the simulator follows
- `demo_data_v4.py` - data for the web demo
- `demo.html` - the web demo; open it in a browser, it works offline

Results:

- `v3/*.jsonl` - one line per run (configuration and seed) of the revised study
- `results.json`, `results_ext*.json`, `results_norm.json` - the earlier version with fixed scaling (Section 4.8 of the paper)
- `earlier_version/` - scripts of the earlier version

## Reproducing the paper

    python exp_tune.py
    for s in main csi thr rule jakes alpha n8tune debias uvar; do python exp_main.py $s; done
    N8_THR=0.7 python exp_main.py n8
    python y4_pre.py tune && python exp_y4.py tune
    Y4_THR=0.9 python y4_pre.py test && Y4_THR=0.9 python exp_y4.py run
    python mnist_v3.py tune 2 && C22_TAU=0.8 python mnist_v3.py main 2
    python cifar_v3.py lr && CIFAR_LR=0.2 python cifar_v3.py tune
    CIFAR_LR=0.2 C22_TAU=1.2 python cifar_v3.py main
    python figs_v3.py

Every script skips runs that are already in its result file, so it can be stopped and started again. Set `OMP_NUM_THREADS=1` when running several in parallel. On two CPU cores the digits experiments take about one hour, Y4 about two hours, MNIST about two hours and CIFAR-10 about four hours.

## Data

The digits data set ships with scikit-learn. MNIST: put these four files next to the scripts, without unzipping them:

- [train-images-idx3-ubyte.gz](https://raw.githubusercontent.com/golbin/TensorFlow-MNIST/master/mnist/data/train-images-idx3-ubyte.gz)
- [train-labels-idx1-ubyte.gz](https://raw.githubusercontent.com/golbin/TensorFlow-MNIST/master/mnist/data/train-labels-idx1-ubyte.gz)
- [t10k-images-idx3-ubyte.gz](https://raw.githubusercontent.com/golbin/TensorFlow-MNIST/master/mnist/data/t10k-images-idx3-ubyte.gz)
- [t10k-labels-idx1-ubyte.gz](https://raw.githubusercontent.com/golbin/TensorFlow-MNIST/master/mnist/data/t10k-labels-idx1-ubyte.gz)

CIFAR-10 was taken from a public PNG copy of the official release:

    git clone --depth 1 https://github.com/YoongiKim/CIFAR-10-images cifar_png
    python cifar_prepare.py cifar_png ../data/cifar10.npz

`cifar_sim.py` reads `../data/cifar10.npz` (or the path in the environment variable `CIFAR_NPZ`).
