"""Reproduce the six theoretical stability maps underlying article Fig. 4."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from tractor_magnet.constants import MILLIMETRE  # noqa: E402
from tractor_magnet.experimental import (  # noqa: E402
    ExperimentalPoint,
    load_figure_4_experimental_points,
)
from tractor_magnet.stability_map import (  # noqa: E402
    StabilityMap,
    SymmetryLineFit,
    compute_stability_map,
    fit_symmetry_line,
    symmetry_line_points,
)

_PANELS = (
    ("negative_x_barrier", r"$|F^x_{0,\min}|$ [mN]", 1.0e3, 0.0, 200.0, "a"),
    ("binding_energy", r"$U_B$ [mJ]", 1.0e3, 0.0, 5.0, "c"),
    ("equilibrium_x", r"$x_\mathrm{eq}$ [mm]", 1.0e3, 15.0, 25.0, "e"),
    ("positive_x_barrier", r"$F^x_{0,\max}$ [mN]", 1.0e3, 0.0, 500.0, "b"),
    ("transverse_barrier", r"$|F^y_{0,\max}|$ [mN]", 1.0e3, 0.0, 200.0, "d"),
    ("chi", r"$\chi$", 1.0, -1.0, 1.0, "f"),
)

_EXPERIMENTAL_STYLES = {
    "possible": ("black", "possible"),
    "partial": ("#00d7e9", "partial"),
    "not_possible": ("#e600a9", "not possible"),
    "transverse_unstable": ("#ffe000", "transverse unstable"),
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orders", type=int, nargs="+", default=(0, 2, 4))
    parser.add_argument("--points", type=int, default=101, help="points on each map axis")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/figure_4_stability_maps"),
    )
    parser.add_argument(
        "--experimental-data",
        type=Path,
        default=Path("data/figure_4_experimental_points.csv"),
    )
    parser.add_argument("--stl-root", type=Path, default=Path("STL_Samples"))
    parser.add_argument(
        "--centering-tolerance-mm",
        type=float,
        default=1.25,
        help="maximum displacement before the transverse force exceeds friction",
    )
    return parser.parse_args()


def _plot_experimental_points(
    axis: Axes,
    points: tuple[ExperimentalPoint, ...],
    *,
    labels: bool = False,
) -> None:
    for outcome, (color, label) in _EXPERIMENTAL_STYLES.items():
        selected = [point for point in points if point.outcome == outcome]
        axis.scatter(
            [point.a / MILLIMETRE for point in selected],
            [point.b / MILLIMETRE for point in selected],
            s=15,
            facecolor=color,
            edgecolor="white" if outcome == "possible" else "black",
            linewidth=0.35,
            label=label if labels else None,
            zorder=8,
        )


def _background(axis: Axes, stability_map: StabilityMap) -> None:
    a_grid, b_grid = np.meshgrid(
        stability_map.a_values / MILLIMETRE,
        stability_map.b_values / MILLIMETRE,
    )
    forbidden = stability_map.geometry_forbidden.astype(float)
    axis.pcolormesh(
        a_grid,
        b_grid,
        forbidden,
        cmap=ListedColormap(("#b8b8b8", "#666666")),
        vmin=0.0,
        vmax=1.0,
        shading="nearest",
        rasterized=True,
    )

    a_boundary = np.linspace(9.0, np.sqrt(12.5**2 - 5.0**2), 200)
    b_boundary = np.sqrt(12.5**2 - a_boundary**2)
    axis.plot(a_boundary, b_boundary, color="white", linewidth=1.5)
    axis.plot((a_boundary[-1], 19.0), (5.0, 5.0), color="white", linewidth=1.5)


def _plot_panel(
    figure: Figure,
    axis: Axes,
    stability_map: StabilityMap,
    panel: tuple[str, str, float, float, float, str],
    experimental_points: tuple[ExperimentalPoint, ...],
) -> None:
    name, label, scale, lower, upper, letter = panel
    _background(axis, stability_map)
    a_grid, b_grid = np.meshgrid(
        stability_map.a_values / MILLIMETRE,
        stability_map.b_values / MILLIMETRE,
    )
    values = np.asarray(getattr(stability_map, name) * scale, dtype=float)
    visible = np.ma.masked_where(~stability_map.stable | ~np.isfinite(values), values)
    image = axis.pcolormesh(
        a_grid,
        b_grid,
        visible,
        cmap="autumn_r",
        vmin=lower,
        vmax=upper,
        shading="nearest",
        rasterized=True,
    )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.025)
    colorbar.set_label(label)
    axis.text(0.025, 0.96, f"({letter})", transform=axis.transAxes, va="top", fontsize=11)
    axis.set(xlim=(9.0, 19.0), ylim=(4.0, 14.0), xlabel=r"$a$ [mm]")
    if letter in {"a", "b"}:
        axis.set_ylabel(r"$b$ [mm]")
    else:
        axis.tick_params(labelleft=False)

    if name == "chi":
        points = symmetry_line_points(stability_map)
        if points.size:
            stable_points = []
            for a, b in points:
                column = int(np.argmin(abs(stability_map.a_values - a)))
                row = int(np.argmin(abs(stability_map.b_values - b)))
                if stability_map.stable[row, column]:
                    stable_points.append((a, b))
            if stable_points:
                line = np.asarray(stable_points)
                axis.plot(
                    line[:, 0] / MILLIMETRE,
                    line[:, 1] / MILLIMETRE,
                    color="white",
                    linewidth=1.0,
                    label=r"$H_{yy}-H_{xx}=0$",
                )

    _plot_experimental_points(axis, experimental_points, labels=letter == "a")
    if letter == "a":
        axis.legend(loc="upper left", fontsize=6.5, framealpha=0.85)


def write_map(
    output_path: Path,
    stability_map: StabilityMap,
    experimental_points: tuple[ExperimentalPoint, ...],
) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(11.8, 7.4), constrained_layout=True)
    for axis, panel in zip(axes.flat, _PANELS, strict=True):
        _plot_panel(figure, axis, stability_map, panel, experimental_points)
    figure.suptitle(
        f"Figure 4 theoretical stability maps - MPEX{stability_map.order}\n"
        r"colored: all force barriers $\geq 10$ mN and "
        rf"$y_{{10}}\leq {stability_map.centering_tolerance[0, 0] / MILLIMETRE:.2f}$ mm; "
        r"transverse width $2(b+R_\mathrm{rep})$",
        fontsize=13,
    )
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def save_map(output_path: Path, stability_map: StabilityMap) -> None:
    np.savez_compressed(
        output_path,
        **{field: getattr(stability_map, field) for field in stability_map.__dataclass_fields__},
    )


def load_map(input_path: Path) -> StabilityMap:
    """Load a map written by :func:`save_map`."""

    with np.load(input_path) as archive:
        missing = set(StabilityMap.__dataclass_fields__) - set(archive.files)
        if missing:
            raise ValueError(
                f"{input_path.name} uses an older map schema; missing {sorted(missing)}"
            )
        values = {field: archive[field] for field in StabilityMap.__dataclass_fields__}
    values["order"] = int(values["order"])
    return StabilityMap(**values)


def write_chi_comparison(
    output_path: Path,
    maps: list[StabilityMap],
    fits: list[SymmetryLineFit],
    experimental_points: tuple[ExperimentalPoint, ...],
) -> None:
    figure, axes = plt.subplots(
        1, len(maps), figsize=(4.2 * len(maps), 4.0), constrained_layout=True, squeeze=False
    )
    for axis, stability_map, fit in zip(axes.flat, maps, fits, strict=True):
        _background(axis, stability_map)
        a_grid, b_grid = np.meshgrid(
            stability_map.a_values / MILLIMETRE,
            stability_map.b_values / MILLIMETRE,
        )
        chi = np.ma.masked_where(~stability_map.stable, stability_map.chi)
        image = axis.pcolormesh(
            a_grid,
            b_grid,
            chi,
            cmap="autumn_r",
            vmin=-1.0,
            vmax=1.0,
            shading="nearest",
            rasterized=True,
        )
        points = symmetry_line_points(stability_map)
        if points.size:
            axis.plot(
                points[:, 0] / MILLIMETRE,
                points[:, 1] / MILLIMETRE,
                color="white",
                linewidth=1.2,
            )
        a_fit = np.linspace(10.0, 14.5, 100)
        b_fit = fit.slope * a_fit + fit.intercept / MILLIMETRE
        axis.plot(a_fit, b_fit, color="#0066cc", linewidth=1.0, linestyle="--")
        _plot_experimental_points(axis, experimental_points)
        axis.set(
            title=(
                f"MPEX{stability_map.order}\n"
                f"b = {fit.slope:.3f} a {fit.intercept / MILLIMETRE:+.3f} mm"
            ),
            xlabel=r"$a$ [mm]",
            xlim=(9.0, 19.0),
            ylim=(4.0, 14.0),
        )
        if axis is axes.flat[0]:
            axis.set_ylabel(r"$b$ [mm]")
        else:
            axis.tick_params(labelleft=False)
    colorbar = figure.colorbar(image, ax=axes.ravel().tolist(), fraction=0.022, pad=0.02)
    colorbar.set_label(r"$\chi=\log_{10}|H_{yy}-H_{xx}|$")
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def write_localization_comparison(
    output_path: Path,
    maps: list[StabilityMap],
    experimental_points: tuple[ExperimentalPoint, ...],
) -> None:
    """Plot the static-friction dead-zone width underlying the practical mask."""

    figure, axes = plt.subplots(
        1, len(maps), figsize=(4.2 * len(maps), 4.0), constrained_layout=True, squeeze=False
    )
    for axis, stability_map in zip(axes.flat, maps, strict=True):
        _background(axis, stability_map)
        a_grid, b_grid = np.meshgrid(
            stability_map.a_values / MILLIMETRE,
            stability_map.b_values / MILLIMETRE,
        )
        crossing = np.ma.masked_where(
            ~stability_map.force_stable,
            stability_map.transverse_friction_crossing / MILLIMETRE,
        )
        image = axis.pcolormesh(
            a_grid,
            b_grid,
            crossing,
            cmap="viridis",
            vmin=0.0,
            vmax=3.0,
            shading="nearest",
            rasterized=True,
        )
        axis.contour(
            a_grid,
            b_grid,
            stability_map.stable.astype(float),
            levels=(0.5,),
            colors="white",
            linewidths=1.2,
        )
        _plot_experimental_points(axis, experimental_points)
        axis.set(
            title=f"MPEX{stability_map.order}",
            xlabel=r"$a$ [mm]",
            xlim=(9.0, 19.0),
            ylim=(4.0, 14.0),
        )
        if axis is axes.flat[0]:
            axis.set_ylabel(r"$b$ [mm]")
        else:
            axis.tick_params(labelleft=False)
    colorbar = figure.colorbar(image, ax=axes.ravel().tolist(), fraction=0.022, pad=0.02)
    colorbar.set_label(r"$y_{10}$ [mm]")
    figure.suptitle(r"Transverse static-friction dead zone; white: practical stability boundary")
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def write_fits(output_path: Path, fits: list[SymmetryLineFit]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("order", "slope", "intercept_mm", "rms_residual_mm", "points"))
        for fit in fits:
            writer.writerow(
                (
                    fit.order,
                    fit.slope,
                    fit.intercept / MILLIMETRE,
                    fit.rms_residual / MILLIMETRE,
                    fit.points,
                )
            )


def main() -> None:
    arguments = parse_arguments()
    if arguments.points < 3:
        raise ValueError("--points must be at least three")
    if arguments.centering_tolerance_mm <= 0.0:
        raise ValueError("--centering-tolerance-mm must be positive")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    experimental_points = load_figure_4_experimental_points(
        arguments.experimental_data,
        arguments.stl_root,
    )

    a_values = np.linspace(9.0, 19.0, arguments.points) * MILLIMETRE
    b_values = np.linspace(4.0, 14.0, arguments.points) * MILLIMETRE
    maps: list[StabilityMap] = []
    fits: list[SymmetryLineFit] = []
    for order in arguments.orders:
        stability_map = compute_stability_map(
            a_values,
            b_values,
            order=order,
            centering_tolerance=arguments.centering_tolerance_mm * MILLIMETRE,
            progress=True,
        )
        fit = fit_symmetry_line(stability_map)
        maps.append(stability_map)
        fits.append(fit)
        write_map(
            arguments.output_dir / f"figure_4_theory_mpex{order}.png",
            stability_map,
            experimental_points,
        )
        save_map(arguments.output_dir / f"figure_4_theory_mpex{order}.npz", stability_map)
        print(f"MPEX{order}: b = {fit.slope:.6f} a {fit.intercept / MILLIMETRE:+.6f} mm")

    # Include previously calculated orders in the comparison. This makes it
    # possible to add an expensive higher order without recomputing MPEX0-4.
    stored_paths = sorted(arguments.output_dir.glob("figure_4_theory_mpex*.npz"))
    comparison_maps = []
    for path in stored_paths:
        try:
            comparison_maps.append(load_map(path))
        except ValueError as error:
            print(f"Skipping incompatible stored map: {error}")
    comparison_maps.sort(key=lambda stability_map: stability_map.order)
    comparison_fits = [fit_symmetry_line(stability_map) for stability_map in comparison_maps]
    write_fits(arguments.output_dir / "symmetry_line_fits.csv", comparison_fits)
    write_chi_comparison(
        arguments.output_dir / "chi_symmetry_comparison.png",
        comparison_maps,
        comparison_fits,
        experimental_points,
    )
    write_localization_comparison(
        arguments.output_dir / "transverse_localization_comparison.png",
        comparison_maps,
        experimental_points,
    )
    print(arguments.output_dir.resolve())


if __name__ == "__main__":
    main()
