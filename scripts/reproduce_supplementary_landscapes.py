"""Reproduce the five energy landscapes in Supplementary Figs. 3-7."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from tractor_magnet.constants import CENTIMETRE, MILLIJOULE  # noqa: E402
from tractor_magnet.reproduction import (  # noqa: E402
    EnergyLandscape,
    compute_supplementary_energy_landscape,
    supplementary_energy_landscape_cases,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/supplementary_energy_landscapes"),
        help="directory for individual plots, overview, and equilibrium table",
    )
    parser.add_argument("--points", type=int, default=201, help="grid points per axis")
    parser.add_argument("--order", type=int, default=4, help="even multipole order")
    return parser.parse_args()


def plot_landscape(
    figure: Figure,
    axis: Axes,
    landscape: EnergyLandscape,
    *,
    include_colorbar: bool = True,
) -> None:
    """Draw one landscape with the supplement's limits and annotations."""

    x_cm = landscape.x_grid / CENTIMETRE
    y_cm = landscape.y_grid / CENTIMETRE
    energy_mj = landscape.energy / MILLIJOULE
    lower, upper = (value / MILLIJOULE for value in landscape.case.energy_limits)

    image = axis.pcolormesh(
        x_cm,
        y_cm,
        energy_mj,
        cmap="turbo",
        vmin=lower,
        vmax=upper,
        shading="auto",
        rasterized=True,
    )
    contour_levels = np.linspace(lower, upper, 6)[1:-1]
    axis.contour(x_cm, y_cm, energy_mj, levels=contour_levels, colors="white", linewidths=0.8)

    saddle_energies = sorted(
        {
            round(equilibrium.energy / MILLIJOULE, 12)
            for equilibrium in landscape.equilibria
            if not equilibrium.stable
        }
    )
    for saddle_energy in saddle_energies:
        if lower < saddle_energy < upper:
            axis.contour(
                x_cm,
                y_cm,
                energy_mj,
                levels=[saddle_energy],
                colors="black",
                linewidths=1.0,
                linestyles="--",
            )

    stable = landscape.stable_equilibrium
    axis.plot(
        stable.point[0] / CENTIMETRE,
        stable.point[1] / CENTIMETRE,
        marker="o",
        markersize=4.5,
        color="red",
        markeredgecolor="white",
        markeredgewidth=0.4,
    )
    axis.axhline(0.0, color="black", linewidth=0.6, linestyle=":", alpha=0.65)
    axis.set(
        title=f"{landscape.case.title} ({landscape.case.table})",
        xlabel=r"$x$ [cm]",
        ylabel=r"$y$ [cm]",
        xlim=(1.0, 3.0),
        ylim=(-1.0, 1.0),
        aspect="equal",
    )
    if include_colorbar:
        colorbar = figure.colorbar(image, ax=axis, fraction=0.047, pad=0.04)
        colorbar.set_label(r"$U$ [mJ]")


def write_individual_plot(output_path: Path, landscape: EnergyLandscape) -> None:
    figure, axis = plt.subplots(figsize=(5.4, 4.5), constrained_layout=True)
    plot_landscape(figure, axis, landscape)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def write_overview(output_path: Path, landscapes: list[EnergyLandscape]) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(13.0, 7.8), constrained_layout=True)
    for axis, landscape in zip(axes.flat, landscapes, strict=False):
        plot_landscape(figure, axis, landscape)

    summary_axis = axes.flat[-1]
    summary_axis.axis("off")
    summary_axis.text(
        0.05,
        0.95,
        "Supplementary Figs. 3-7\n\n"
        "MPEX4\n"
        "Domain: 1 <= x <= 3 cm\n"
        "             -1 <= y <= 1 cm\n\n"
        "Red: computed stable equilibrium\n"
        "Dashed: saddle-point energy",
        ha="left",
        va="top",
        fontsize=11,
        linespacing=1.35,
    )
    figure.suptitle("Supplementary energy landscapes", fontsize=15)
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def write_equilibrium_table(output_path: Path, landscapes: list[EnergyLandscape]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "case",
                "table",
                "expected_x_cm",
                "computed_x_cm",
                "difference_mm",
                "energy_mJ",
                "curvature_x_J_per_m2",
                "curvature_y_J_per_m2",
            )
        )
        for landscape in landscapes:
            stable = landscape.stable_equilibrium
            writer.writerow(
                (
                    landscape.case.slug,
                    landscape.case.table,
                    landscape.case.expected_equilibrium_x / CENTIMETRE,
                    stable.point[0] / CENTIMETRE,
                    (stable.point[0] - landscape.case.expected_equilibrium_x) * 1.0e3,
                    stable.energy / MILLIJOULE,
                    stable.hessian[0][0],
                    stable.hessian[1][1],
                )
            )


def main() -> None:
    arguments = parse_arguments()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)

    landscapes = [
        compute_supplementary_energy_landscape(
            case,
            order=arguments.order,
            points=arguments.points,
        )
        for case in supplementary_energy_landscape_cases()
    ]
    for landscape in landscapes:
        write_individual_plot(
            arguments.output_dir / f"{landscape.case.slug}_mpex{arguments.order}.png",
            landscape,
        )

    write_overview(
        arguments.output_dir / f"supplementary_landscapes_mpex{arguments.order}.png",
        landscapes,
    )
    write_equilibrium_table(arguments.output_dir / "equilibria.csv", landscapes)
    print(arguments.output_dir.resolve())


if __name__ == "__main__":
    main()
