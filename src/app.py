"""Browser-native Shiny interface for the tractor-magnet stability model."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from shiny import App, Inputs, Outputs, Session, reactive, render, req, ui

from tractor_magnet.constants import MILLIMETRE
from tractor_magnet.stability_map import StabilityMap
from web_support import WebCalculation, WebInputs, calculate_web

APP_DIR = Path(__file__).parent


def _field(
    input_id: str,
    label: str,
    value: float,
    *,
    minimum: float,
    maximum: float | None = None,
    step: float,
    note: str,
) -> ui.Tag:
    return ui.div(
        ui.input_numeric(
            input_id,
            label,
            value,
            min=minimum,
            max=maximum,
            step=step,
            update_on="change",
        ),
        ui.tags.small(note, class_="field-note"),
        class_="field-block",
    )


def _metric_card(label: str, value: str, unit: str, detail: str, accent: str) -> ui.Tag:
    return ui.div(
        ui.div(
            ui.tags.span(label, class_="metric-label"),
            ui.tags.span(class_=f"metric-dot metric-dot-{accent}"),
            class_="metric-heading",
        ),
        ui.div(
            ui.tags.span(value, class_="metric-value"),
            ui.tags.span(unit, class_="metric-unit"),
            class_="metric-number",
        ),
        ui.tags.p(detail, class_="metric-detail"),
        class_=f"metric-card metric-card-{accent}",
    )


def _format(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}g}" if math.isfinite(value) else "—"


def _plot_map(
    axis: plt.Axes,
    stability_map: StabilityMap,
    *,
    title: str,
    norm: Normalize,
) -> object:
    a_mm = stability_map.a_values / MILLIMETRE
    b_mm = stability_map.b_values / MILLIMETRE
    eligible = stability_map.has_local_minimum & ~stability_map.geometry_forbidden
    values = np.ma.masked_where(~eligible | ~np.isfinite(stability_map.chi), stability_map.chi)
    image = axis.pcolormesh(a_mm, b_mm, values, cmap="coolwarm", norm=norm, shading="auto")
    axis.contourf(
        a_mm,
        b_mm,
        stability_map.geometry_forbidden.astype(float),
        levels=(0.5, 1.5),
        colors=("#aeb7c4",),
        alpha=0.72,
    )
    if np.any(stability_map.stable) and np.any(~stability_map.stable):
        axis.contour(
            a_mm,
            b_mm,
            stability_map.stable.astype(float),
            levels=(0.5,),
            colors=("#ffffff",),
            linewidths=1.8,
        )
        axis.contour(
            a_mm,
            b_mm,
            stability_map.stable.astype(float),
            levels=(0.5,),
            colors=("#172033",),
            linewidths=0.6,
        )
    difference = np.ma.masked_where(
        ~eligible | ~np.isfinite(stability_map.curvature_difference),
        stability_map.curvature_difference,
    )
    finite_difference = difference.compressed()
    if finite_difference.size and np.min(finite_difference) <= 0.0 <= np.max(finite_difference):
        axis.contour(
            a_mm,
            b_mm,
            difference,
            levels=(0.0,),
            colors=("#172033",),
            linewidths=3.2,
        )
        axis.contour(
            a_mm,
            b_mm,
            difference,
            levels=(0.0,),
            colors=("#ffd166",),
            linewidths=1.8,
        )
    else:
        axis.text(
            0.03,
            0.04,
            "no Hyy − Hxx zero contour in range",
            transform=axis.transAxes,
            fontsize=8,
            color="#172033",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78},
        )
    axis.set(title=title, xlabel="a [mm]", ylabel="b [mm]")
    axis.set_aspect("equal", adjustable="box")
    return image


def _stability_figure(calculation: WebCalculation):
    figure, axis = plt.subplots(1, 1, figsize=(7.2, 5.8))
    finite_values = calculation.refined_map.chi[np.isfinite(calculation.refined_map.chi)]
    if finite_values.size:
        centre = float(np.nanmedian(finite_values))
        spread = max(1.0, float(np.nanpercentile(abs(finite_values - centre), 95)))
        norm = Normalize(centre - spread, centre + spread)
    else:
        norm = Normalize(-1.0, 1.0)
    _plot_map(
        axis,
        calculation.refined_map,
        title=f"MPEX{calculation.inputs.refinement_order} curvature-symmetry map",
        norm=norm,
    )
    figure.suptitle(
        f"χ scale {norm.vmin:.2f}…{norm.vmax:.2f} · yellow: Hyy − Hxx = 0 · white: practical mask",
        fontsize=10,
    )
    figure.subplots_adjust(left=0.13, right=0.96, bottom=0.11, top=0.84)
    return figure


def _input_section(index: str, title: str) -> ui.Tag:
    return ui.div(
        ui.tags.span(index, class_="section-index"),
        ui.tags.span(title, class_="section-label"),
        class_="input-section-title",
    )


input_panel = ui.div(
    ui.tags.span("DESIGN INPUTS", class_="eyebrow"),
    ui.h2("Define the magnet set"),
    ui.tags.p(
        "All cylinders are coaxially magnetized. Dimensions are diametric radii and heights.",
        class_="panel-intro",
    ),
    _input_section("01", "Follower"),
    ui.div(
        _field(
            "follower_radius",
            "Radius [mm]",
            7.5,
            minimum=0.1,
            maximum=30,
            step=0.1,
            note="Movable cylinder",
        ),
        _field(
            "follower_height",
            "Height [mm]",
            10.0,
            minimum=0.1,
            maximum=50,
            step=0.1,
            note="Axial stack height",
        ),
        class_="input-pair",
    ),
    _field(
        "follower_remanence",
        "Remanence Br [T]",
        1.3,
        minimum=0.001,
        maximum=3,
        step=0.01,
        note="Positive model polarity",
    ),
    _input_section("02", "Attractor"),
    ui.div(
        _field(
            "attractor_radius",
            "Radius [mm]",
            7.5,
            minimum=0.1,
            maximum=30,
            step=0.1,
            note="Fixed at (-a, 0)",
        ),
        _field(
            "attractor_height",
            "Height [mm]",
            10.0,
            minimum=0.1,
            maximum=50,
            step=0.1,
            note="Axial stack height",
        ),
        class_="input-pair",
    ),
    _field(
        "attractor_remanence",
        "Remanence Br [T]",
        1.3,
        minimum=0.001,
        maximum=3,
        step=0.01,
        note="Opposite polarity to follower",
    ),
    _input_section("03", "Repulsor pair"),
    ui.div(
        _field(
            "repulsor_radius",
            "Radius [mm]",
            5.0,
            minimum=1.0,
            maximum=30,
            step=0.1,
            note="Fixed at (0, ±b)",
        ),
        _field(
            "repulsor_height",
            "Height [mm]",
            10.0,
            minimum=0.1,
            maximum=50,
            step=0.1,
            note="Both magnets identical",
        ),
        class_="input-pair",
    ),
    _field(
        "repulsor_remanence",
        "Remanence Br [T]",
        0.39,
        minimum=0.001,
        maximum=3,
        step=0.01,
        note="Same polarity as follower",
    ),
    _input_section("04", "Map and model"),
    ui.div(
        _field(
            "a_max",
            "a max [mm]",
            19.0,
            minimum=0.2,
            maximum=80,
            step=0.5,
            note="Horizontal map limit",
        ),
        _field(
            "b_max",
            "b max [mm]",
            14.0,
            minimum=0.2,
            maximum=80,
            step=0.5,
            note="Vertical map limit",
        ),
        class_="input-pair",
    ),
    ui.div(
        ui.div(
            ui.input_select(
                "order",
                "Refinement order",
                {"0": "MPEX0", "2": "MPEX2", "4": "MPEX4"},
                selected="4",
            ),
            ui.tags.small("0: dipole · 2/4: finite-size refinement", class_="field-note"),
            class_="field-block",
        ),
        ui.div(
            ui.input_select(
                "grid_points",
                "Grid per axis",
                {
                    "15": "15 × 15",
                    "21": "21 × 21",
                    "31": "31 × 31",
                    "41": "41 × 41",
                    "51": "51 × 51",
                    "61": "61 × 61",
                    "71": "71 × 71",
                    "81": "81 × 81",
                    "101": "101 × 101",
                },
                selected="21",
            ),
            ui.tags.small(
                "MPEX4 runtime grows approximately with the number of grid points",
                class_="field-note",
            ),
            class_="field-block",
        ),
        class_="input-pair",
    ),
    ui.div(
        _field(
            "friction",
            "Barrier threshold [mN]",
            10.0,
            minimum=0.001,
            maximum=1000,
            step=1.0,
            note="All three barriers must exceed it",
        ),
        _field(
            "centering",
            "Centering tolerance [mm]",
            1.25,
            minimum=0.001,
            maximum=30,
            step=0.05,
            note="Maximum half-width y₁₀ of the static-friction dead zone",
        ),
        class_="input-pair",
    ),
    ui.input_action_button("calculate", "Analyze configuration", class_="btn-calculate"),
    ui.div(
        ui.tags.span(class_="privacy-pulse"),
        ui.tags.span("The Python model runs locally in your browser."),
        class_="privacy-note",
    ),
    class_="input-panel",
)


calculator_panel = ui.div(
    ui.div(input_panel, class_="calculator-sidebar"),
    ui.tags.main(
        ui.output_ui("calculation_message"),
        ui.output_ui("summary_cards"),
        ui.div(
            ui.div(
                ui.div(
                    ui.tags.span("CURVATURE SYMMETRY", class_="eyebrow"),
                    ui.h3("χ map and zero contour"),
                    class_="card-title-group",
                ),
                ui.tags.span("yellow: Hyy − Hxx = 0 · white: practical", class_="method-chip"),
                class_="visual-card-header",
            ),
            ui.output_plot("stability_plot", height="470px"),
            class_="visual-card map-card",
        ),
        class_="calculator-results",
    ),
    class_="calculator-grid",
)


method_panel = ui.div(
    ui.div(
        ui.tags.span("MODEL", class_="eyebrow"),
        ui.h2("Three levels, one calculation core"),
        ui.tags.p(
            "The web layer changes neither the Cartesian moments nor the analytical "
            "derivatives. It only validates units and selects the requested map.",
            class_="docs-lead",
        ),
        class_="docs-intro",
    ),
    ui.div(
        ui.div(
            ui.tags.span("01", class_="docs-number"),
            ui.h3("Moment admissibility"),
            ui.tags.p("For the symmetric point-dipole configuration, a necessary condition is"),
            ui.HTML(
                r"<div class='equation-card'>\[\eta="
                r"\left|m_{\mathrm{rep}}/m_{\mathrm{att}}\right|<1/2.\]</div>"
            ),
            ui.tags.p("This is a fast exclusion test, not a proof of practical stability."),
            class_="docs-section",
        ),
        ui.div(
            ui.tags.span("02", class_="docs-number"),
            ui.h3("Dipole estimate"),
            ui.tags.p(
                "MPEX0 retains the magnetic dipole moments and locates the exterior "
                "axial minimum. It supplies the inexpensive first map and a reference "
                "for convergence."
            ),
            class_="docs-section",
        ),
        ui.div(
            ui.tags.span("03", class_="docs-number"),
            ui.h3("Finite-size refinement"),
            ui.tags.p(
                "MPEX2 and MPEX4 use the exact Cartesian moments of uniformly "
                "magnetized cylinders. Energy, force, and Hessian are differentiated "
                "analytically."
            ),
            class_="docs-section",
        ),
        ui.div(
            ui.tags.span("04", class_="docs-number"),
            ui.h3("Practical mask"),
            ui.tags.p(
                "A point is shown as practically stable only after overlap exclusion, "
                "a positive planar Hessian, three force barriers, and the transverse "
                "centering tolerance all pass."
            ),
            class_="docs-section",
        ),
        class_="docs-grid",
    ),
    ui.div(
        ui.h3("Scientific source"),
        ui.tags.p(
            "Michael P. Adams, “Understanding two-dimensional tractor magnets: Theory "
            "and realizations,” American Journal of Physics 92, 576–582 (2024)."
        ),
        ui.tags.a(
            "TractorMagnet on GitHub",
            href="https://github.com/AdamsMP92/TractorMagnet",
            target="_blank",
            class_="docs-link",
        ),
        class_="source-card docs-source-card",
    ),
    class_="documentation-panel",
)


app_ui = ui.page_fluid(
    ui.head_content(
        ui.tags.meta(name="viewport", content="width=device-width, initial-scale=1"),
        ui.tags.link(rel="preconnect", href="https://fonts.googleapis.com"),
        ui.tags.link(rel="preconnect", href="https://fonts.gstatic.com", crossorigin="anonymous"),
        ui.tags.link(
            href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Newsreader:wght@500;600&display=swap",
            rel="stylesheet",
        ),
        ui.tags.link(rel="stylesheet", href="styles.css"),
        ui.tags.script(
            src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js", async_="async"
        ),
        ui.tags.title("Tractor Magnet Designer"),
    ),
    ui.div(
        ui.tags.header(
            ui.div(
                ui.div(
                    ui.tags.img(
                        src="tractor-mark.svg", alt="Four-magnet plan-view mark", class_="hero-mark"
                    ),
                    ui.div(
                        ui.tags.span("BROWSER-NATIVE MAGNETOSTATICS", class_="hero-kicker"),
                        ui.h1("Tractor Magnet Designer"),
                        ui.tags.p(
                            "Explore the minimal symmetric configuration from the moment "
                            "inequality through dipole estimation to exact-cylinder "
                            "Cartesian multipole refinement."
                        ),
                    ),
                    class_="hero-copy",
                ),
                ui.div(
                    ui.tags.span(class_="runtime-dot"),
                    ui.div(
                        ui.tags.strong("Python in your browser"),
                        ui.tags.small("No server-side simulation"),
                    ),
                    class_="runtime-badge",
                ),
                class_="hero-inner",
            ),
            class_="app-hero",
        ),
        ui.div(
            ui.navset_card_tab(
                ui.nav_panel("Designer", calculator_panel),
                ui.nav_panel("Method", method_panel),
            ),
            class_="main-nav-wrap",
        ),
        ui.tags.footer(
            ui.tags.span("TractorMagnet"),
            ui.tags.span("·"),
            ui.tags.a("Michael P. Adams", href="https://adamsmp92.github.io/homepage/"),
            class_="app-footer",
        ),
        class_="app-shell",
    ),
    class_="app-page",
)


def server(input: Inputs, output: Outputs, session: Session) -> None:
    def collect_inputs() -> WebInputs:
        return WebInputs.from_millimetres(
            follower_radius_mm=input.follower_radius(),
            follower_height_mm=input.follower_height(),
            follower_remanence_t=input.follower_remanence(),
            attractor_radius_mm=input.attractor_radius(),
            attractor_height_mm=input.attractor_height(),
            attractor_remanence_t=input.attractor_remanence(),
            repulsor_radius_mm=input.repulsor_radius(),
            repulsor_height_mm=input.repulsor_height(),
            repulsor_remanence_t=input.repulsor_remanence(),
            a_max_mm=input.a_max(),
            b_max_mm=input.b_max(),
            grid_points=int(input.grid_points()),
            refinement_order=int(input.order()),
            friction_threshold_mn=input.friction(),
            centering_tolerance_mm=input.centering(),
        )

    @reactive.calc
    @reactive.event(input.calculate, ignore_init=True)
    def calculation() -> WebCalculation | ValueError:
        try:
            inputs = collect_inputs()
            with ui.Progress(min=0, max=1) as progress:
                progress.set(0.05, message="Evaluating dipole and multipole maps…")
                result = calculate_web(inputs)
                progress.set(1.0, message="Analysis complete")
            return result
        except ValueError as error:
            return error

    def result() -> WebCalculation:
        value = calculation()
        req(isinstance(value, WebCalculation))
        return value

    @output
    @render.ui
    def calculation_message() -> ui.Tag:
        value = calculation()
        if isinstance(value, ValueError):
            return ui.div(
                ui.tags.strong("Input error"),
                ui.tags.span(str(value)),
                class_="calculation-message message-error",
            )
        condition = "passes" if value.inputs.passes_moment_condition else "fails"
        stages = (
            "MPEX0 was evaluated"
            if value.inputs.refinement_order == 0
            else f"MPEX0 and MPEX{value.inputs.refinement_order} were evaluated"
        )
        return ui.div(
            ui.tags.strong("Three-stage check complete"),
            ui.tags.span(
                f"The moment condition {condition}; {stages} on "
                f"{value.inputs.grid_points} × {value.inputs.grid_points} grids in "
                f"{value.elapsed_s:.2f} s."
            ),
            class_="calculation-message message-success"
            if value.inputs.passes_moment_condition
            else "calculation-message message-warning",
        )

    @output
    @render.ui
    def summary_cards() -> ui.Tag:
        value = result()
        return ui.div(
            _metric_card(
                "Moment ratio η",
                _format(value.inputs.moment_ratio, 4),
                "",
                "required: η < 0.5",
                "teal" if value.inputs.passes_moment_condition else "red",
            ),
            _metric_card(
                "Approximation",
                f"MPEX{value.inputs.refinement_order}",
                "",
                "finite-size refinement order",
                "indigo",
            ),
            _metric_card(
                "Grid",
                f"{value.inputs.grid_points} × {value.inputs.grid_points}",
                "",
                f"calculated in {value.elapsed_s:.2f} s",
                "copper",
            ),
            class_="metrics-grid",
        )

    @output
    @render.plot
    def stability_plot():
        return _stability_figure(result())


app = App(app_ui, server, static_assets=APP_DIR / "www")
