"""
P1 - Data / Signal / Channel Generation
Project T24: Over-the-Air Federated Learning - Power Control Under Fading

WHAT THIS SCRIPT DOES
----------------------
Generates the baseline fading-channel traces that M1, M2, and M3 will all
share and reuse. Using the SAME channel realizations across all three
methodologies is what makes the later comparison fair (per Phase P5 in the
roadmap: "ensure fair comparison - same seed, split").

CHANNEL MODEL
-------------
We use i.i.d. Rayleigh fading, the standard model used across the OTA-FL
literature this project's lit review is based on (e.g. Cao et al. [7],
Zhu et al. [9], Cao et al. [12], Yu et al. [14]).

Under Rayleigh fading, each device's channel coefficient h_k is a complex
Gaussian random variable: h_k ~ CN(0, 1). Its magnitude |h_k| follows a
Rayleigh distribution, and its squared magnitude |h_k|^2 follows an
exponential distribution (this is what "fading" means physically - the
signal randomly gets stronger or weaker as if bouncing off obstacles).

OUTPUT
------
- channel_traces.npy   : the actual channel data, shape (num_rounds, K), complex
- channel_traces.csv   : same data in a human-readable table (magnitude only)
- channel_distribution.png : histogram verifying the Rayleigh shape
- A printed summary you can paste into your report's "Data / Materials" section
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# SIMULATION PARAMETERS - tune these here, every later script imports from
# this file so you only need to change them in one place.
# ---------------------------------------------------------------------------
SEED = 42                  # fixed seed -> same channel data every run (fair comparison)
NUM_DEVICES = 50           # K: number of edge devices (matches typical OTA-FL sim scale)
NUM_ROUNDS = 500           # number of communication rounds to simulate
NOISE_VARIANCE = 1.0       # sigma^2 for additive white Gaussian noise (AWGN)

TRAIN_ROUNDS = 400         # rounds 0-399 used while developing/tuning M1/M2/M3
TEST_ROUNDS = 100          # rounds 400-499 held out for final reported results


def generate_rayleigh_channel(num_rounds: int, num_devices: int, seed: int) -> np.ndarray:
    """
    Generate i.i.d. Rayleigh fading channel coefficients.

    Returns
    -------
    h : np.ndarray, shape (num_rounds, num_devices), dtype complex128
        h[t, k] is device k's channel coefficient in round t.
        Real and imaginary parts are each ~ N(0, 0.5), so that
        E[|h|^2] = 1 (unit average channel gain, standard normalization).
    """
    rng = np.random.default_rng(seed)
    real_part = rng.normal(loc=0.0, scale=np.sqrt(0.5), size=(num_rounds, num_devices))
    imag_part = rng.normal(loc=0.0, scale=np.sqrt(0.5), size=(num_rounds, num_devices))
    h = real_part + 1j * imag_part
    return h


def main():
    h = generate_rayleigh_channel(NUM_ROUNDS, NUM_DEVICES, SEED)

    # ---- Save the full complex channel data for reuse by M1/M2/M3 ----
    np.save("channel_traces.npy", h)

    # ---- Also save a human-readable CSV of magnitudes (|h_k|) ----
    magnitudes = np.abs(h)
    np.savetxt("channel_traces_magnitude.csv", magnitudes, delimiter=",",
               header=",".join(f"device_{k}" for k in range(NUM_DEVICES)),
               comments="")

    # ---- Verify the distribution looks Rayleigh (sanity check) ----
    plt.figure(figsize=(7, 4.5))
    plt.hist(magnitudes.flatten(), bins=50, density=True, alpha=0.7,
             color="#1C7293", edgecolor="white", label="Simulated |h_k|")

    # overlay the theoretical Rayleigh PDF for comparison
    x = np.linspace(0, magnitudes.max(), 200)
    sigma = np.sqrt(0.5)  # matches our real/imag part scale
    theoretical_pdf = (x / sigma**2) * np.exp(-x**2 / (2 * sigma**2))
    plt.plot(x, theoretical_pdf, color="#0B1F3A", linewidth=2,
              label="Theoretical Rayleigh PDF")

    plt.xlabel("Channel magnitude |h_k|")
    plt.ylabel("Probability density")
    plt.title("P1: Simulated Fading Channel Distribution (K=%d, %d rounds)"
               % (NUM_DEVICES, NUM_ROUNDS))
    plt.legend()
    plt.tight_layout()
    plt.savefig("channel_distribution.png", dpi=150)
    plt.close()

    # ---- Print a summary for your report's "Data / Materials" section ----
    print("=" * 60)
    print("P1 CHANNEL GENERATION - SUMMARY")
    print("=" * 60)
    print(f"Model            : i.i.d. Rayleigh fading, h_k ~ CN(0,1)")
    print(f"Devices (K)      : {NUM_DEVICES}")
    print(f"Total rounds     : {NUM_ROUNDS}")
    print(f"  - Train rounds : {TRAIN_ROUNDS} (rounds 0-{TRAIN_ROUNDS-1})")
    print(f"  - Test rounds  : {TEST_ROUNDS} (rounds {TRAIN_ROUNDS}-{NUM_ROUNDS-1})")
    print(f"Noise variance   : {NOISE_VARIANCE}")
    print(f"Random seed      : {SEED}  (fixed -> reproducible, fair M1/M2/M3 comparison)")
    print(f"Mean |h_k|       : {magnitudes.mean():.4f}  (theoretical: {sigma*np.sqrt(np.pi/2):.4f})")
    print(f"Mean |h_k|^2     : {(magnitudes**2).mean():.4f}  (theoretical: 1.0)")
    print()
    print("Files written:")
    print("  - channel_traces.npy            (full complex data, for M1/M2/M3 scripts)")
    print("  - channel_traces_magnitude.csv  (human-readable magnitudes)")
    print("  - channel_distribution.png      (sanity-check plot)")
    print("=" * 60)


if __name__ == "__main__":
    main()
