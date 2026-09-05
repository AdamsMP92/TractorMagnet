"""Animate the MPEX4 energy landscape along the practical production line."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import numpy as np
from scipy.optimize import brentq

matplotlib.use("Agg")
from matplotlib import animation, colormaps  # noqa: E402
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.colors import PowerNorm  # noqa: E402

from tractor_magnet.configurations import article_minimal_configuration  # noqa: E402
from tractor_magnet.constants import MILLIJOULE, MILLIMETRE  # noqa: E402
from tractor_magnet.multipole import MultipoleExpansion  # noqa: E402
from tractor_magnet.stability_map import evaluate_stability_point  # noqa: E402


@dataclass(frozen=True, slots=True)
class ProductionLine:
    """Linear design rule and its practically stable interval, in metres."""

    slope: float
    intercept: float
    a_min: float
    a_max: float

    def b(self, a: float) -> float:
        return self.slope * a + self.intercept


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--grid-points", type=int, default=181)
    parser.add_argument("--energy-max-mj", type=float, default=5.0)
    parser.add_argument(
        "--map",
        type=Path,
        default=Path("results/figure_4_stability_maps/figure_4_theory_mpex4.npz"),
    )
    parser.add_argument(
        "--fits",
        type=Path,
        default=Path("results/figure_4_stability_maps/symmetry_line_fits.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/production_line_energy_landscape_mpex4.mp4"),
    )
    return parser.parse_args()


def _longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        if start is not None and (not value or index == mask.size - 1):
            end = index if value and index == mask.size - 1 else index - 1
            runs.append((start, end))
            start = None
    if not runs:
        raise ValueError("the production line does not intersect the practical stability mask")
    return max(runs, key=lambda run: run[1] - run[0])


def load_production_line(map_path: Path, fits_path: Path, order: int) -> ProductionLine:
    """Infer and refine the longest practically stable symmetry-line segment."""

    with fits_path.open(newline="", encoding="utf-8") as stream:
        fits = {int(row["order"]): row for row in csv.DictReader(stream)}
    if order not in fits:
        raise ValueError(f"no MPEX{order} symmetry-line fit in {fits_path}")
    slope = float(fits[order]["slope"])
    intercept = float(fits[order]["intercept_mm"]) * MILLIMETRE

    with np.load(map_path) as archive:
        if int(archive["order"]) != order:
            raise ValueError(f"{map_path} does not contain MPEX{order}")
        a_values = np.asarray(archive["a_values"], dtype=float)
        b_values = np.asarray(archive["b_values"], dtype=float)
        stable = (
            archive["passes_friction_threshold"]
            & archive["passes_centering_tolerance"]
            & ~archive["geometry_forbidden"]
        )
        tolerance = float(archive["centering_tolerance"][0, 0])

    line_b = slope * a_values + intercept
    rows = np.asarray([np.argmin(abs(b_values - b)) for b in line_b], dtype=int)
    line_mask = stable[rows, np.arange(a_values.size)]
    start, end = _longest_true_run(line_mask)

    def geometry_margin(a: float) -> float:
        b = slope * a + intercept
        return math.hypot(a, b) - 12.5 * MILLIMETRE

    a_min = float(a_values[start])
    lower_search_bound = max(0.0, float(a_values[0]) - 5.0 * MILLIMETRE)
    if geometry_margin(lower_search_bound) * geometry_margin(a_min) <= 0.0:
        a_min = float(brentq(geometry_margin, lower_search_bound, a_min, xtol=1.0e-12))

    def centering_margin(a: float) -> float:
        point = evaluate_stability_point(
            a,
            slope * a + intercept,
            order=order,
            centering_tolerance=tolerance,
            barrier_samples=401,
        )
        crossing = point.transverse_friction_crossing
        return tolerance - crossing if np.isfinite(crossing) else -math.inf

    # The stored Figure-4 grid may end while the line is still stable. Continue
    # by one grid spacing at a time so that a2 is the physical y10 boundary,
    # rather than an artefact of the rectangular map extent.
    step = float(np.median(np.diff(a_values)))
    left = float(a_values[end])
    left_margin = centering_margin(left)
    right = float(a_values[end + 1]) if end + 1 < a_values.size else left + step
    a_max = left
    for _ in range(50):
        right_margin = centering_margin(right)
        if left_margin * right_margin <= 0.0:
            a_max = float(brentq(centering_margin, left, right, xtol=1.0e-10))
            break
        left, left_margin = right, right_margin
        right += step
    else:
        raise RuntimeError("could not bracket the upper centering boundary")

    return ProductionLine(slope, intercept, a_min, a_max)


def _collision_mask(
    model: MultipoleExpansion,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
) -> np.ndarray:
    configuration = model.configuration
    forbidden = np.zeros(x_grid.shape, dtype=bool)
    for source in configuration.sources:
        contact_distance = configuration.follower.radius + source.magnet.radius
        forbidden |= (x_grid - source.position[0]) ** 2 + (
            y_grid - source.position[1]
        ) ** 2 < contact_distance**2
    return forbidden


def write_animation(
    output_path: Path,
    metadata_path: Path,
    production_line: ProductionLine,
    *,
    order: int,
    frames: int,
    fps: int,
    grid_points: int,
    energy_max_mj: float,
) -> None:
    """Compute and encode all energy-landscape frames sequentially."""

    a_values = np.linspace(production_line.a_min, production_line.a_max, frames)
    x_values = np.linspace(10.0, 32.0, grid_points) * MILLIMETRE
    y_values = np.linspace(-12.0, 12.0, grid_points) * MILLIMETRE
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    cmap = colormaps["viridis"].copy()
    cmap.set_bad("#707070")
    norm = PowerNorm(gamma=0.55, vmin=0.0, vmax=energy_max_mj)

    figure, axis = plt.subplots(figsize=(8.0, 6.1), constrained_layout=True)
    colorbar = figure.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=axis, pad=0.025)
    colorbar.set_label(r"$U(x,y)-U_\mathrm{eq}$ [mJ]")
    writer = animation.FFMpegWriter(
        fps=fps,
        codec="libx264",
        bitrate=3200,
        extra_args=("-pix_fmt", "yuv420p"),
        metadata={"title": f"MPEX{order} energy landscape along the production line"},
    )
    records: list[tuple[float | int, ...]] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(figure, output_path, dpi=160):
        for frame, a in enumerate(a_values):
            b = production_line.b(float(a))
            point = evaluate_stability_point(float(a), b, order=order, barrier_samples=401)
            model = MultipoleExpansion(article_minimal_configuration(float(a), b), order=order)
            relative_energy = (
                model.derivative_array(x_grid, y_grid) - model.energy((point.equilibrium_x, 0.0))
            ) / MILLIJOULE
            energy = np.ma.masked_where(
                _collision_mask(model, x_grid, y_grid),
                relative_energy,
            )

            axis.clear()
            axis.pcolormesh(
                x_grid / MILLIMETRE,
                y_grid / MILLIMETRE,
                energy,
                cmap=cmap,
                norm=norm,
                shading="auto",
                rasterized=True,
            )
            finite_energy = energy.compressed()
            contour_levels = [
                level
                for level in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0)
                if finite_energy.min() < level < finite_energy.max()
            ]
            if contour_levels:
                axis.contour(
                    x_grid / MILLIMETRE,
                    y_grid / MILLIMETRE,
                    energy,
                    levels=contour_levels,
                    colors="white",
                    linewidths=0.65,
                    alpha=0.85,
                )
            axis.plot(
                point.equilibrium_x / MILLIMETRE,
                0.0,
                marker="*",
                markersize=11,
                color="#ff3030",
                markeredgecolor="white",
                markeredgewidth=0.7,
            )
            progress = frame / (frames - 1) if frames > 1 else 1.0
            axis.text(
                0.02,
                0.025,
                f"production-line position: {100 * progress:5.1f}%",
                transform=axis.transAxes,
                color="white",
                fontsize=9,
                bbox={"facecolor": "black", "alpha": 0.55, "edgecolor": "none"},
            )
            axis.set(
                title=(
                    f"MPEX{order}:  b = {production_line.slope:.4f} a "
                    f"{production_line.intercept / MILLIMETRE:+.4f} mm\n"
                    f"a = {a / MILLIMETRE:.3f} mm,  b = {b / MILLIMETRE:.3f} mm,  "
                    f"y₁₀ = {point.transverse_friction_crossing / MILLIMETRE:.3f} mm"
                ),
                xlabel=r"$x$ [mm]",
                ylabel=r"$y$ [mm]",
                xlim=(10.0, 32.0),
                ylim=(-12.0, 12.0),
                aspect="equal",
            )
            writer.grab_frame()
            records.append(
                (
                    frame,
                    a / MILLIMETRE,
                    b / MILLIMETRE,
                    point.equilibrium_x / MILLIMETRE,
                    point.binding_energy / MILLIJOULE,
                    point.curvature_x,
                    point.curvature_y,
                    point.transverse_friction_crossing / MILLIMETRE,
                )
            )
            if frame == 0 or (frame + 1) % 10 == 0 or frame + 1 == frames:
                print(f"MPEX{order}: rendered frame {frame + 1}/{frames}")
    plt.close(figure)

    with metadata_path.open("w", newline="", encoding="utf-8") as stream:
        table = csv.writer(stream)
        table.writerow(
            (
                "frame",
                "a_mm",
                "b_mm",
                "equilibrium_x_mm",
                "binding_energy_mJ",
                "curvature_x_N_per_m",
                "curvature_y_N_per_m",
                "y10_mm",
            )
        )
        table.writerows(records)


def main() -> None:
    arguments = parse_arguments()
    if arguments.frames < 2 or arguments.grid_points < 3 or arguments.fps <= 0:
        raise ValueError("frames >= 2, grid-points >= 3, and fps > 0 are required")
    if arguments.energy_max_mj <= 0.0:
        raise ValueError("--energy-max-mj must be positive")

    production_line = load_production_line(arguments.map, arguments.fits, arguments.order)
    print(
        f"MPEX{arguments.order} interval: "
        f"a={production_line.a_min / MILLIMETRE:.6f}.."
        f"{production_line.a_max / MILLIMETRE:.6f} mm, "
        f"b={production_line.b(production_line.a_min) / MILLIMETRE:.6f}.."
        f"{production_line.b(production_line.a_max) / MILLIMETRE:.6f} mm"
    )
    metadata_path = arguments.output.with_suffix(".csv")
    write_animation(
        arguments.output,
        metadata_path,
        production_line,
        order=arguments.order,
        frames=arguments.frames,
        fps=arguments.fps,
        grid_points=arguments.grid_points,
        energy_max_mj=arguments.energy_max_mj,
    )
    print(arguments.output.resolve())


if __name__ == "__main__":
    main()
