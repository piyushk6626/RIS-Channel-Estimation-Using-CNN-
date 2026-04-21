from __future__ import annotations

from pathlib import Path

import numpy as np


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_loss_curve(history: list[dict[str, float]], destination: str | Path) -> None:
    plt = _plt()
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [row["train_loss"] for row in history], label="Train")
    plt.plot(epochs, [row["val_loss"] for row in history], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Training and Validation Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination, dpi=200)
    plt.close()


def plot_nmse_curve(history: list[dict[str, float]], destination: str | Path) -> None:
    plt = _plt()
    epochs = [row["epoch"] for row in history]
    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [row["train_nmse_db"] for row in history], label="Train")
    plt.plot(epochs, [row["val_nmse_db"] for row in history], label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("NMSE (dB)")
    plt.title("Training and Validation NMSE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination, dpi=200)
    plt.close()


def plot_snr_vs_nmse(
    cnn_summary: dict[str, object],
    ls_summary: dict[str, object],
    destination: str | Path,
) -> None:
    plt = _plt()
    snr_keys = sorted(cnn_summary["per_snr"], key=lambda value: float(value))
    snr_values = [float(key) for key in snr_keys]
    cnn_nmse = [float(cnn_summary["per_snr"][key]["nmse_db_mean"]) for key in snr_keys]
    ls_nmse = [float(ls_summary["per_snr"][key]["nmse_db_mean"]) for key in snr_keys]

    plt.figure(figsize=(7, 4))
    plt.plot(snr_values, cnn_nmse, marker="o", label="CNN")
    plt.plot(snr_values, ls_nmse, marker="s", label="LS")
    plt.xlabel("SNR (dB)")
    plt.ylabel("NMSE (dB)")
    plt.title("Test NMSE vs SNR")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination, dpi=200)
    plt.close()


def plot_error_histogram(cnn_nmse_db: np.ndarray, ls_nmse_db: np.ndarray, destination: str | Path) -> None:
    plt = _plt()
    plt.figure(figsize=(7, 4))
    plt.hist(cnn_nmse_db, bins=20, alpha=0.65, label="CNN")
    plt.hist(ls_nmse_db, bins=20, alpha=0.65, label="LS")
    plt.xlabel("Per-sample NMSE (dB)")
    plt.ylabel("Count")
    plt.title("Test Error Distribution")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination, dpi=200)
    plt.close()


def plot_channel_examples(
    true_channels: np.ndarray,
    predicted_channels: np.ndarray,
    nmse_db: np.ndarray,
    destination: str | Path,
    *,
    examples: int = 3,
) -> None:
    plt = _plt()
    total_examples = min(examples, true_channels.shape[0])
    sorted_indices = np.argsort(nmse_db)
    if total_examples == 0:
        return
    ranked_positions = np.linspace(0, len(sorted_indices) - 1, total_examples, dtype=int)
    selection = [int(sorted_indices[position]) for position in ranked_positions]

    fig, axes = plt.subplots(total_examples, 3, figsize=(10, 3.5 * total_examples))
    if total_examples == 1:
        axes = np.asarray([axes])

    for row, index in enumerate(selection):
        true_magnitude = np.abs(true_channels[index])
        predicted_magnitude = np.abs(predicted_channels[index])
        error_magnitude = np.abs(true_channels[index] - predicted_channels[index])
        panels = (
            (true_magnitude, "True |H|"),
            (predicted_magnitude, "Predicted |H|"),
            (error_magnitude, f"Absolute Error |H| ({nmse_db[index]:.2f} dB)"),
        )
        for column, (panel, title) in enumerate(panels):
            axis = axes[row, column]
            image = axis.imshow(panel, aspect="auto", cmap="viridis")
            axis.set_title(title)
            axis.set_xlabel("RIS elements")
            axis.set_ylabel("BS antennas")
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(destination, dpi=200)
    plt.close(fig)


def plot_pilot_length_vs_nmse(summary_rows: list[dict[str, float]], destination: str | Path) -> None:
    plt = _plt()
    pilot_lengths = [row["pilot_length"] for row in summary_rows]
    cnn_nmse = [row["cnn_nmse_db_mean"] for row in summary_rows]
    ls_nmse = [row["ls_nmse_db_mean"] for row in summary_rows]

    plt.figure(figsize=(7, 4))
    plt.plot(pilot_lengths, cnn_nmse, marker="o", label="CNN")
    plt.plot(pilot_lengths, ls_nmse, marker="s", label="LS")
    plt.xlabel("Pilot length")
    plt.ylabel("Test NMSE (dB)")
    plt.title("Pilot Length vs Test NMSE")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(destination, dpi=200)
    plt.close()
