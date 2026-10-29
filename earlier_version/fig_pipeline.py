"""Figure 1: OTA-FL signal chain shared by M1, M2, M3 (block diagram)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(6.4, 3.0))
ax.set_xlim(0, 12.8); ax.set_ylim(-0.15, 5.9); ax.axis("off")


def box(x, y, w, h, text, fc="#E8EEF7", ec="#1F3864", fs=7.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                                fc=fc, ec=ec, lw=1.1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def arrow(x1, y1, x2, y2, text=None, dy=0.22, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
                                 lw=1.1, color="#333333", linestyle=ls))
    if text:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + dy, text, ha="center", va="bottom", fontsize=6.8)


# top row: devices -> precoding -> channel -> receiver -> model update
box(0.2, 3.3, 2.3, 2.4, "K devices\nlocal gradient $g_k$\n$s_k=g_k/\\rho$")
box(3.2, 3.3, 2.5, 2.4, "Power control\n$b_k=\\sqrt{\\eta}\\,\\hat c_k^{*}/|\\hat c_k|^2$\n(M1: all devices\nM2, M3: $|\\hat c_k|\\geq h_{th}$)")
box(6.4, 3.3, 2.5, 2.4, "Fading MAC\n$y=\\sum_k c_k b_k s_k+n$\nRayleigh $h_k\\sim CN(0,1)$\nnoise $n\\sim CN(0,\\sigma^2)$")
box(9.6, 3.3, 3.0, 2.4, "Server\n(M3: combiner $a$, $c_k=a^{H}h_k$)\n$\\hat g=\\rho\\,\\mathrm{Re}\\{y\\}/(|S|\\sqrt{\\eta})$")
arrow(2.5, 4.5, 3.2, 4.5); arrow(5.7, 4.5, 6.4, 4.5); arrow(8.9, 4.5, 9.6, 4.5)

# bottom: model update and broadcast
box(9.6, 0.5, 3.0, 1.5, "Global update\n$w \\leftarrow w-\\mu\\,\\hat g$", fc="#FDF1E3", ec="#8A5A14")
box(0.2, 0.5, 2.3, 1.5, "Local models\n(broadcast $w$)", fc="#FDF1E3", ec="#8A5A14")
arrow(11.1, 3.3, 11.1, 2.0)
arrow(9.6, 1.6, 2.5, 1.6, "downlink broadcast (assumed error-free)", dy=0.08)
arrow(1.35, 2.0, 1.35, 3.3)
box(3.2, 0.0, 5.2, 1.0, "Channel estimate $\\hat h_k=(h_k+w_k)/(1+\\nu)$,\nerror variance $\\epsilon$ (imperfect CSI)", fc="#F2F2F2", ec="#666666", fs=7)
fig.tight_layout(pad=0.3)
fig.savefig("fig_pipeline.png", dpi=200)
