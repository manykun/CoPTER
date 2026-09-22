#!/usr/bin/env python3
"""Reproduce the figures for the seed-1 WebServer→CacheFollower ACC run."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "docs" / "figures"
OUTPUT.mkdir(parents=True, exist_ok=True)

PORTS = np.array([323, 321, 320, 345, 346, 347])

OVERALL = {
    "A acquisition": {
        "reward": (0.3692704438, 0.3750239422),
        "p95": (606.60675, 616.19950),
        "completion": (0.9923443456, 0.9925349428),
    },
    "B acquisition": {
        "reward": (0.4580758633, 0.4630654147),
        "p95": (7681.90320, 7576.04010),
        "completion": (0.9009106678, 0.9017779705),
    },
    "A forgetting": {
        "reward": (0.3750239422, 0.3732108950),
        "p95": (616.43600, 638.87550),
        "completion": (0.9925349428, 0.9925031766),
    },
}

FORGETTING_CHANGE = {
    "Reward": np.array([13.88, 1.99, -0.51, 0.42, -0.57, -6.05]),
    "Queue p95": np.array([-54.55, -12.64, -14.08, -9.27, 24.82, -18.40]),
    "ECN mean": np.array([213.22, 778.47, 43.45, -8.15, -3.47, 327.04]),
}

PARAMETERS = {
    "Kmin (KB)": (
        np.array([63, 80, 32, 8, 8, 57]),
        np.array([8, 16, 20, 8, 16, 16]),
    ),
    "Kmax (KB)": (
        np.array([120, 160, 64, 24, 24, 110]),
        np.array([24, 32, 40, 24, 32, 32]),
    ),
    "Pmax": (
        np.array([0.40, 0.40, 0.20, 0.60, 0.40, 1.00]),
        np.array([1.00, 0.80, 0.60, 0.40, 0.20, 1.00]),
    ),
}

TASK_SIGNALS = {
    "Queue p95 (KB)": (
        np.array([123.84, 82.09, 103.97, 82.34, 70.51, 113.59]),
        np.array([70.87, 66.52, 93.13, 51.17, 99.27, 72.60]),
    ),
    "ECN marking rate (%)": (
        100 * np.array([0.0090, 0.0002, 0.0161, 0.0354, 0.0248, 0.0067]),
        100 * np.array([0.0656, 0.0333, 0.0483, 0.0269, 0.0706, 0.0102]),
    ),
}

TRAINING = {
    "A: WebServer": {
        "reward_first": np.array([0.5676, 0.5101, 0.3751, 0.1946, 0.6380, 0.3296]),
        "reward_last": np.array([0.5804, 0.5037, 0.3302, 0.5002, 0.5738, 0.4895]),
        "loss_median": np.array([37.7872, 183.1698, 92.4920, 35.2847, 92.9263, 85.0307]),
        "loss_max": np.array([529.5695, 2228.0843, 1027.4346, 579.6810, 1168.7947, 1205.3977]),
    },
    "B: CacheFollower": {
        "reward_first": np.array([0.5990, 0.5052, 0.3664, 0.5167, 0.5741, 0.5013]),
        "reward_last": np.array([0.5873, 0.7295, 0.6900, 0.5916, 0.5065, 0.5834]),
        "loss_median": np.array([0.0902, 0.2108, 0.2460, 0.1012, 0.3925, 0.0428]),
        "loss_max": np.array([3.1926, 11.8884, 2.6620, 1.8828, 8.9014, 1.6483]),
    },
}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "figure.titlesize": 15,
        "legend.fontsize": 9,
    })


def save(figure, name):
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(OUTPUT / name, dpi=220, bbox_inches="tight")
    plt.close(figure)


def plot_overall():
    labels = list(OVERALL)
    metrics = (
        ("reward", "All-congested reward"),
        ("p95", "Common-flow p95 FCT (us)"),
        ("completion", "Completion ratio"),
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    x = np.arange(len(labels))
    width = 0.34
    for axis, (key, title) in zip(axes, metrics):
        before = [OVERALL[label][key][0] for label in labels]
        after = [OVERALL[label][key][1] for label in labels]
        axis.bar(x - width / 2, before, width, label="Before", color="#4C78A8")
        axis.bar(x + width / 2, after, width, label="After", color="#F58518")
        axis.set_title(title)
        axis.set_xticks(x, labels, rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.25)
        if key == "completion":
            axis.set_ylim(0.88, 1.0)
        for idx, (left, right) in enumerate(zip(before, after)):
            if key == "completion":
                change = (right - left) * 100
                text = f"{change:+.3f} pp"
            else:
                change = (right - left) / abs(left) * 100
                text = f"{change:+.2f}%"
            axis.text(idx, max(left, right), text, ha="center", va="bottom", fontsize=8)
    axes[0].legend(loc="best")
    figure.suptitle("ACC realistic workload curriculum: network-level results")
    save(figure, "realistic_acc_overall.png")


def plot_forgetting_heatmap():
    rows = list(FORGETTING_CHANGE)
    values = np.vstack([FORGETTING_CHANGE[row] for row in rows])
    display = np.sign(values) * np.log1p(np.abs(values))
    bound = np.max(np.abs(display))
    figure, axis = plt.subplots(figsize=(9.4, 3.6))
    image = axis.imshow(display, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
    axis.set_xticks(np.arange(len(PORTS)), PORTS)
    axis.set_yticks(np.arange(len(rows)), rows)
    axis.set_xlabel("Port agent")
    axis.set_title("Old-task change after learning CacheFollower")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            axis.text(column, row, f"{value:+.2f}%", ha="center", va="center", fontsize=9)
    colorbar = figure.colorbar(image, ax=axis, fraction=0.032, pad=0.03)
    colorbar.set_label("Signed log scale")
    figure.suptitle("Port-local forgetting diagnostics")
    save(figure, "realistic_acc_port_forgetting.png")


def plot_parameters():
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    x = np.arange(len(PORTS))
    width = 0.36
    for axis, (label, (webserver, cachefollower)) in zip(axes, PARAMETERS.items()):
        axis.bar(x - width / 2, webserver, width, label="WebServer after-A", color="#4C78A8")
        axis.bar(x + width / 2, cachefollower, width, label="CacheFollower after-B", color="#E45756")
        axis.set_xticks(x, PORTS)
        axis.set_xlabel("Port agent")
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper right")
    figure.suptitle("Dominant greedy ACC parameters on congested samples")
    save(figure, "realistic_acc_parameters.png")


def plot_task_signals():
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    x = np.arange(len(PORTS))
    width = 0.36
    for axis, (label, (webserver, cachefollower)) in zip(axes[:2], TASK_SIGNALS.items()):
        axis.bar(x - width / 2, webserver, width, label="WebServer after-A", color="#4C78A8")
        axis.bar(x + width / 2, cachefollower, width, label="CacheFollower after-B", color="#F58518")
        axis.set_xticks(x, PORTS)
        axis.set_xlabel("Port agent")
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="best")
    axes[2].axis("off")
    axes[2].text(
        0.5,
        0.58,
        "PFC pause events\n\nWebServer: unavailable (n/a)\nCacheFollower: 0 on all watched ports",
        ha="center",
        va="center",
        fontsize=13,
        bbox={"boxstyle": "round,pad=0.8", "facecolor": "#F2F2F2", "edgecolor": "#888888"},
    )
    figure.suptitle("Queue, ECN and PFC after learning each workload")
    save(figure, "realistic_acc_network_signals.png")


def plot_training():
    figure, axes = plt.subplots(2, 2, figsize=(12.2, 7.3), sharex=True)
    x = np.arange(len(PORTS))
    width = 0.36
    for column, (phase, data) in enumerate(TRAINING.items()):
        reward_axis = axes[0, column]
        loss_axis = axes[1, column]
        reward_axis.bar(x - width / 2, data["reward_first"], width, label="First", color="#72B7B2")
        reward_axis.bar(x + width / 2, data["reward_last"], width, label="Last", color="#F58518")
        reward_axis.set_title(phase)
        reward_axis.set_ylabel("Batch reward")
        reward_axis.grid(axis="y", alpha=0.25)
        loss_axis.bar(x - width / 2, data["loss_median"], width, label="Median", color="#B279A2")
        loss_axis.bar(x + width / 2, data["loss_max"], width, label="Maximum", color="#E45756")
        loss_axis.set_yscale("log")
        loss_axis.set_ylabel("TD loss (log scale)")
        loss_axis.set_xlabel("Port agent")
        loss_axis.set_xticks(x, PORTS)
        loss_axis.grid(axis="y", alpha=0.25)
    axes[0, 0].legend(loc="best")
    axes[1, 0].legend(loc="best")
    figure.suptitle("Per-port training summary (600 optimizer updates per task)")
    save(figure, "realistic_acc_training.png")


def main():
    style()
    plot_overall()
    plot_forgetting_heatmap()
    plot_parameters()
    plot_task_signals()
    plot_training()
    for path in sorted(OUTPUT.glob("realistic_acc_*.png")):
        print(path)


if __name__ == "__main__":
    main()
