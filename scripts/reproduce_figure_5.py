"""Reproduce the multipole-convergence plot in Fig. 5 of the article."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from tractor_magnet.constants import CENTIMETRE, MILLINEWTON  # noqa: E402
from tractor_magnet.reproduction import figure_5_force_curves  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figure_5_multipole_convergence.png"),
        help="output image path",
    )
    parser.add_argument("--points", type=int, default=401, help="number of x samples")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.points < 2:
        raise ValueError("--points must be at least two")

    x_coordinates = np.linspace(1.0, 5.0, arguments.points) * CENTIMETRE
    curves = figure_5_force_curves(x_coordinates)

    figure, axis = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    styles = {
        0: {"label": "MPEX0", "linewidth": 1.8},
        2: {"label": "MPEX2", "linewidth": 1.8},
        4: {"label": "MPEX4", "linewidth": 2.0},
        6: {"label": "MPEX6", "linewidth": 1.5, "linestyle": "--"},
    }
    for order, force in curves.items():
        axis.plot(x_coordinates / CENTIMETRE, force / MILLINEWTON, **styles[order])

    axis.axhline(0.0, color="0.55", linewidth=0.8, linestyle="--")
    axis.set(
        xlabel=r"$x$ [cm]",
        ylabel=r"$F_x$ [mN]",
        xlim=(1.0, 5.0),
        ylim=(-50.0, 150.0),
    )
    axis.grid(alpha=0.22)
    axis.legend(frameon=True)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(arguments.output, dpi=200)
    plt.close(figure)
    print(arguments.output.resolve())


if __name__ == "__main__":
    main()
