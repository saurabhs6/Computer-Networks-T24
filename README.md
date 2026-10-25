# T24 - Over-the-Air Federated Learning: Power Control Under Fading
Team 24, Communication Networks (VIT SCOPE). Authors: Saurabh Suresh (24BCE2750), Advaith Sivakumar (24BCE2715).

Simulation of three power-control methods for over-the-air federated learning under Rayleigh fading:
M1 full channel inversion, M2 truncated inversion, M3 receive beamforming + truncation under imperfect CSI.
Extra controls: M1+BF, M2+EIG (fixed combiner), OPT (baseline with the threshold structure of Cao et al. 2020).

## Setup
Python 3.10+ with: numpy, scikit-learn, matplotlib  (tested with numpy 2.5, scikit-learn 1.9, matplotlib 3.11)
    pip install numpy scikit-learn matplotlib

## Files
- `demo.html`             interactive web demo (open in any browser, works offline; uses the saved results)
- `demo.py`               quick demo, about 1 minute (3 seeds, -25 dB)
- `ota_fl_sim.py`         full simulation; `python ota_fl_sim.py` regenerates `results.json` and the figures
- `ext_experiments.py`    extra experiments (receive SNR, M2+EIG, OPT, K = 20/50/100 threshold rule); needs `results.json`; writes `results_ext.json`
- `p1_channel_generation.py`  channel-trace generation
- `results.json`, `results_ext.json`  all numbers reported in the paper
- `fig_*.png`             figures used in the paper

## Setup used in the paper
sklearn digits (1,347 train / 450 test), softmax regression (d = 650), Dirichlet 0.5 non-IID split, K = 50 devices,
100 rounds per run, 10 seeds, reference SNR -25 dB. Channel-trace indices 0-99 for tuning, 400-499 for reporting.
