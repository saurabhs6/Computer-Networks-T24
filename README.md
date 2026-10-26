# T24 - Over-the-Air Federated Learning: Power Control Under Fading
Team 24, Communication Networks (VIT SCOPE). Authors: Saurabh Suresh (24BCE2750), Advaith Sivakumar (24BCE2715).

Simulation of three power-control methods for over-the-air federated learning under Rayleigh fading:
M1 full channel inversion, M2 truncated inversion, M3 receive beamforming + truncation under imperfect CSI.
Extra controls: M1+BF, M2+EIG (fixed combiner), OPT (baseline with the threshold structure of Cao et al. 2020), RCI (generic regularised inversion), and C12 (re-implementation of the Case I power control of Cao et al. 2022, with tuned constants).

## Setup
Python 3.10+ with: numpy, scikit-learn, matplotlib  (tested with numpy 2.5, scikit-learn 1.9, matplotlib 3.11)
    pip install numpy scikit-learn matplotlib

## Files
- `demo.html`             interactive web demo (open in any browser, works offline; uses the saved results)
- `demo.py`               quick demo, about 1 minute (3 seeds, -25 dB)
- `ota_fl_sim.py`         full simulation; `python ota_fl_sim.py` regenerates `results.json` and the figures
- `ext_experiments.py`    extra experiments (receive SNR, M2+EIG, OPT, K = 20/50/100 threshold rule); needs `results.json`; writes `results_ext.json`
- `ext2_rci.py`           regularised channel inversion baseline (generic form, tuned lambda); writes `results_ext2.json`
- `ext3_c22.py`          C12 (policy of Cao et al. 2022, Case I): tunes tau and kappa, SNR and CSI sweeps; writes `results_ext3.json`
- `ext4_norm.py`         unit-power-symbol variant (`S.NORM = True`) of the SNR sweep; writes `results_norm.json`
- `ext5_c22_var.py`      C12 with kappa = 0 (closed form of Proposition 3) and kappa -> infinity (unbiased case); writes `results_ext5.json`
- `mnist_cnn_sim.py`      MNIST + small CNN (NumPy), K = 100; `mnist_scan.py` (SNR scan, 1 seed) and `mnist_full.py` (3 seeds at -30 dB) produce `mnist_scan.json` and `mnist_full.json`; `mnist_c22_tune.py` tunes the C12 target; `mnist_batch.py` adds seeds 3-4, C12, and the -35/-25/-20 dB sweep (results in `mnist_batch.jsonl`, resumable)
- `p1_channel_generation.py`  channel-trace generation
- `results*.json`, `mnist_*.json*`  all numbers reported in the paper
- `fig_*.png`             figures used in the paper

## Setup used in the paper
sklearn digits (1,347 train / 450 test), softmax regression (d = 650), Dirichlet 0.5 non-IID split, K = 50 devices,
100 rounds per run, 10 seeds, reference SNR -25 dB. Channel-trace indices 0-99 for tuning, 400-499 for reporting.

## MNIST files (not included, public data set)
`mnist_cnn_sim.py` expects the four gzip files in the same folder:
    for f in train-images-idx3-ubyte train-labels-idx1-ubyte t10k-images-idx3-ubyte t10k-labels-idx1-ubyte; do
      curl -O https://raw.githubusercontent.com/golbin/TensorFlow-MNIST/master/mnist/data/$f.gz; done
Run single-threaded for speed: `OMP_NUM_THREADS=1 python mnist_full.py` (about 30 minutes on 2 cores for `mnist_full.py`; `mnist_batch.py` takes about 1.5 hours on 2 cores).
