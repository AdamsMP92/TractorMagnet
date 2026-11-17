"""Browser-native Shiny interface for the tractor-magnet stability model."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from shiny import App, Inputs, Outputs, Session, reactive, render, req, ui

from tractor_magnet.constants import MILLIMETRE
from tractor_magnet.materials import MAGNET_GRADES, magnet_grade
from tractor_magnet.stability_map import StabilityMap
from web_support import WebCalculation, WebInputs, calculate_web

APP_DIR = Path(__file__).parent
GRADE_CHOICES = {grade.key: grade.label for grade in MAGNET_GRADES}
DEFAULT_NDFEB_GRADE = magnet_grade("ndfeb_n42")
DEFAULT_FERRITE_GRADE = magnet_grade("ferrite_y30bh")


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


def _grade_field(input_id: str, *, selected: str) -> ui.Tag:
    return ui.div(
        ui.input_select(input_id, "Magnet grade", GRADE_CHOICES, selected=selected),
        ui.tags.small(
            "Room-temperature remanence range from the material grade",
            class_="field-note",
        ),
        class_="field-block",
    )


def _format(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}g}" if math.isfinite(value) else "—"


def _mask_to_practical_region(
    stability_map: StabilityMap,
    values: np.ndarray,
) -> np.ma.MaskedArray:
    """Apply the same practical-stability mask as the Figure-4 reproduction."""

    return np.ma.masked_where(~stability_map.stable | ~np.isfinite(values), values)


def _plot_forbidden_geometry(
    axis: plt.Axes,
    inputs: WebInputs,
    *,
    x_max_mm: float,
    b_min_mm: float,
) -> None:
    """Draw the exact overlap boundary independently of the numerical grid."""

    repulsor_radius_mm = inputs.repulsor_radius_m / MILLIMETRE
    radius_sum_mm = (inputs.attractor_radius_m + inputs.repulsor_radius_m) / MILLIMETRE
    intersection_mm = math.sqrt(max(0.0, radius_sum_mm**2 - repulsor_radius_mm**2))
    arc_end_mm = min(x_max_mm, intersection_mm)
    arc_x = np.linspace(0.0, arc_end_mm, 512)
    arc_y = np.sqrt(np.maximum(0.0, radius_sum_mm**2 - arc_x**2))
    boundary: list[tuple[float, float]] = []
    if x_max_mm > intersection_mm:
        boundary.extend(((x_max_mm, repulsor_radius_mm), (intersection_mm, repulsor_radius_mm)))
    boundary.extend(zip(arc_x[::-1], arc_y[::-1], strict=True))
    polygon = [(0.0, b_min_mm), (x_max_mm, b_min_mm), *boundary]
    axis.fill(
        *zip(*polygon, strict=True),
        facecolor="#aeb7c4",
        edgecolor="#7d8794",
        linewidth=0.9,
        alpha=0.72,
        zorder=2,
    )


def _plot_map(
    axis: plt.Axes,
    stability_map: StabilityMap,
    *,
    inputs: WebInputs,
    title: str,
    norm: Normalize,
) -> object:
    a_mm = stability_map.a_values / MILLIMETRE
    b_mm = stability_map.b_values / MILLIMETRE
    values = _mask_to_practical_region(stability_map, stability_map.chi)
    image = axis.pcolormesh(a_mm, b_mm, values, cmap="coolwarm", norm=norm, shading="auto")
    _plot_forbidden_geometry(
        axis,
        inputs,
        x_max_mm=float(a_mm[-1]),
        b_min_mm=float(b_mm[0]),
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
    difference = _mask_to_practical_region(stability_map, stability_map.curvature_difference)
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
    finite_values = calculation.refined_map.chi[
        calculation.refined_map.stable & np.isfinite(calculation.refined_map.chi)
    ]
    if finite_values.size:
        centre = float(np.nanmedian(finite_values))
        spread = max(1.0, float(np.nanpercentile(abs(finite_values - centre), 95)))
        norm = Normalize(centre - spread, centre + spread)
    else:
        norm = Normalize(-1.0, 1.0)
    _plot_map(
        axis,
        calculation.refined_map,
        inputs=calculation.inputs,
        title="Adaptive MPEX4 curvature-symmetry map",
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
        "All cylinders are coaxially magnetized and share one common height.",
        class_="panel-intro",
    ),
    _input_section("01", "Common Magnet Geometry"),
    _field(
        "magnet_height",
        "Height [mm]",
        10.0,
        minimum=0.1,
        maximum=50,
        step=0.1,
        note="Shared by follower, attractor, and both repulsors",
    ),
    _input_section("02", "Follower Magnet"),
    _field(
        "follower_radius",
        "Radius [mm]",
        7.5,
        minimum=0.1,
        maximum=30,
        step=0.1,
        note="Movable cylinder",
    ),
    _grade_field("follower_grade", selected=DEFAULT_NDFEB_GRADE.key),
    _field(
        "follower_remanence",
        "Remanence Br [T]",
        DEFAULT_NDFEB_GRADE.nominal_remanence,
        minimum=DEFAULT_NDFEB_GRADE.remanence_min,
        maximum=DEFAULT_NDFEB_GRADE.remanence_max,
        step=0.001,
        note="Selected-grade range · 1 mT steps · positive polarity",
    ),
    _input_section("03", "Attractor Magnet"),
    _field(
        "attractor_radius",
        "Radius [mm]",
        7.5,
        minimum=0.1,
        maximum=30,
        step=0.1,
        note="Fixed at (-a, 0)",
    ),
    _grade_field("attractor_grade", selected=DEFAULT_NDFEB_GRADE.key),
    _field(
        "attractor_remanence",
        "Remanence Br [T]",
        DEFAULT_NDFEB_GRADE.nominal_remanence,
        minimum=DEFAULT_NDFEB_GRADE.remanence_min,
        maximum=DEFAULT_NDFEB_GRADE.remanence_max,
        step=0.001,
        note="Selected-grade range · 1 mT steps · opposite polarity",
    ),
    _input_section("04", "Repulsor Magnet Pair"),
    _field(
        "repulsor_radius",
        "Radius [mm]",
        5.0,
        minimum=1.0,
        maximum=30,
        step=0.1,
        note="Fixed at (0, ±b)",
    ),
    _grade_field("repulsor_grade", selected=DEFAULT_FERRITE_GRADE.key),
    _field(
        "repulsor_remanence",
        "Remanence Br [T]",
        DEFAULT_FERRITE_GRADE.nominal_remanence,
        minimum=DEFAULT_FERRITE_GRADE.remanence_min,
        maximum=DEFAULT_FERRITE_GRADE.remanence_max,
        step=0.001,
        note="Selected-grade range · 1 mT steps · same polarity as follower",
    ),
    _input_section("05", "Physical friction criterion"),
    ui.div(
        _field(
            "friction",
            "Friction-force threshold [mN]",
            10.0,
            minimum=0.001,
            maximum=1000,
            step=1.0,
            note="Physical static-friction scale",
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
    _input_section("06", "Compute stability map"),
    _field(
        "resolution",
        "Resolution [mm]",
        0.2,
        minimum=0.1,
        maximum=2.0,
        step=0.1,
        note="Final MPEX4 spacing; search extent is determined automatically",
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
        ui.output_ui("line_fit_summary"),
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
        ui.h2("One automatic calculation pipeline"),
        ui.tags.p(
            "The web layer changes neither the Cartesian moments nor the analytical "
            "derivatives. It validates units and controls the adaptive MPEX4 search.",
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
            ui.h3("Adaptive MPEX0 scout"),
            ui.tags.p(
                "A cheap, automatically expanding MPEX0 search finds a relaxed candidate "
                "band around the practical stability region."
            ),
            class_="docs-section",
        ),
        ui.div(
            ui.tags.span("03", class_="docs-number"),
            ui.h3("Sparse MPEX4 refinement"),
            ui.tags.p(
                "MPEX4 is evaluated only inside the candidate band and expands it if a "
                "stable point reaches its boundary. Energy, force, and Hessian remain "
                "analytically differentiated."
            ),
            class_="docs-section",
        ),
        ui.div(
            ui.tags.span("04", class_="docs-number"),
            ui.h3("Practical mask"),
            ui.tags.p(
                "A point is shown as practically stable only after overlap exclusion, "
                "a positive planar Hessian, three force barriers, and the transverse "
                "centering tolerance all pass. The friction-force threshold and "
                "centering tolerance are physical parameters, not numerical map settings."
            ),
            class_="docs-section",
        ),
        ui.div(
            ui.tags.span("05", class_="docs-number"),
            ui.h3("Production-line fit"),
            ui.tags.p(
                "Inside the practical mask, zero crossings of Hyy − Hxx are interpolated "
                "column by column. Their least-squares line b(a) = m a + c is reported "
                "together with its actually resolved a interval and RMS residual."
            ),
            class_="docs-section",
        ),
        ui.div(
            ui.tags.span("06", class_="docs-number"),
            ui.h3("Material-grade bounds"),
            ui.tags.p(
                "Ferrite and NdFeB remanence intervals are room-temperature reference "
                "values. A grade selection sets the rounded midpoint and constrains Br "
                "to the published interval in 1 mT steps."
            ),
            ui.tags.a(
                "Physical magnet data",
                href="https://www.supermagnete.de/physical-magnet-data",
                target="_blank",
                class_="docs-link",
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
                            "inequality through an adaptive dipole scout to sparse "
                            "exact-cylinder MPEX4 refinement."
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
    def update_remanence(input_id: str, grade_key: str) -> None:
        grade = magnet_grade(grade_key)
        ui.update_numeric(
            input_id,
            value=grade.nominal_remanence,
            min=grade.remanence_min,
            max=grade.remanence_max,
            step=0.001,
            session=session,
        )

    @reactive.effect
    @reactive.event(input.follower_grade)
    def update_follower_grade() -> None:
        update_remanence("follower_remanence", str(input.follower_grade()))

    @reactive.effect
    @reactive.event(input.attractor_grade)
    def update_attractor_grade() -> None:
        update_remanence("attractor_remanence", str(input.attractor_grade()))

    @reactive.effect
    @reactive.event(input.repulsor_grade)
    def update_repulsor_grade() -> None:
        update_remanence("repulsor_remanence", str(input.repulsor_grade()))

    def collect_inputs() -> WebInputs:
        return WebInputs.from_millimetres(
            height_mm=input.magnet_height(),
            follower_radius_mm=input.follower_radius(),
            follower_grade=input.follower_grade(),
            follower_remanence_t=input.follower_remanence(),
            attractor_radius_mm=input.attractor_radius(),
            attractor_grade=input.attractor_grade(),
            attractor_remanence_t=input.attractor_remanence(),
            repulsor_radius_mm=input.repulsor_radius(),
            repulsor_grade=input.repulsor_grade(),
            repulsor_remanence_t=input.repulsor_remanence(),
            resolution_mm=input.resolution(),
            friction_threshold_mn=input.friction(),
            centering_tolerance_mm=input.centering(),
        )

    @reactive.calc
    @reactive.event(input.calculate, ignore_init=True)
    def calculation() -> WebCalculation | ValueError:
        try:
            inputs = collect_inputs()
            with ui.Progress(min=0, max=1) as progress:
                progress.set(0.05, message="Scouting and refining the adaptive MPEX4 map…")
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
        return ui.div(
            ui.tags.strong("Adaptive stability search complete"),
            ui.tags.span(
                f"The moment condition {condition}; the MPEX0 scout selected "
                f"{value.evaluated_points} of {value.full_grid_points} possible locations "
                f"for MPEX4 in {value.elapsed_s:.2f} s."
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
                "MPEX4",
                "",
                "automatic MPEX0 scout",
                "indigo",
            ),
            _metric_card(
                "Resolution",
                _format(value.inputs.resolution_m / MILLIMETRE),
                "mm",
                f"{value.evaluated_points} MPEX4 points · {value.elapsed_s:.2f} s",
                "copper",
            ),
            class_="metrics-grid",
        )

    @output
    @render.ui
    def line_fit_summary() -> ui.Tag:
        value = result()
        fit = value.line_fit
        if fit is None:
            return ui.div(
                ui.tags.span("PRODUCTION LINE", class_="eyebrow"),
                ui.h3("No resolved linear-fit interval"),
                ui.tags.p(
                    "The practical MPEX4 region contains fewer than two resolved "
                    "Hyy − Hxx zero crossings. Reduce the map resolution or change the design."
                ),
                class_="line-fit-card line-fit-unavailable",
            )
        intercept_mm = fit.intercept / MILLIMETRE
        return ui.div(
            ui.div(
                ui.div(
                    ui.tags.span("PRODUCTION LINE", class_="eyebrow"),
                    ui.h3("Linear fit of Hyy − Hxx = 0"),
                    class_="card-title-group",
                ),
                ui.tags.span(f"{fit.points} contour points", class_="method-chip"),
                class_="visual-card-header",
            ),
            ui.div(
                f"b(a) = {fit.slope:.5f} a {intercept_mm:+.4f} mm",
                class_="fit-equation",
            ),
            ui.div(
                ui.tags.span(f"slope m = {fit.slope:.6f}"),
                ui.tags.span(f"intercept c = {intercept_mm:+.4f} mm"),
                ui.tags.span(
                    f"fit range: {fit.a_min / MILLIMETRE:.2f} mm ≤ a ≤ "
                    f"{fit.a_max / MILLIMETRE:.2f} mm"
                ),
                ui.tags.span(f"RMS residual = {fit.rms_residual / MILLIMETRE:.4f} mm"),
                class_="fit-details",
            ),
            class_="line-fit-card",
        )

    @output
    @render.plot
    def stability_plot():
        return _stability_figure(result())


app = App(app_ui, server, static_assets=APP_DIR / "www")
