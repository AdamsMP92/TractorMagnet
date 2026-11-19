"""Browser-native Shiny interface for the tractor-magnet stability model."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from shiny import App, Inputs, Outputs, Session, reactive, render, req, ui

from tractor_magnet.constants import MILLIMETRE
from tractor_magnet.materials import MAGNET_GRADES, magnet_grade
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


def _design_curve_figure(calculation: WebCalculation):
    """Show the directly traced and fully validated design curve."""

    points = calculation.design_curve.points
    figure, axis = plt.subplots(1, 1, figsize=(7.2, 4.6))
    a_mm = points[:, 0] / MILLIMETRE
    b_mm = points[:, 1] / MILLIMETRE
    axis.plot(a_mm, b_mm, color="#136f7a", linewidth=2.2, alpha=0.9)
    axis.scatter(
        a_mm,
        b_mm,
        s=34,
        color="#ffd166",
        edgecolor="#172033",
        linewidth=0.75,
        zorder=3,
    )
    axis.set(xlabel="a [mm]", ylabel="b [mm]")
    axis.grid(color="#dfe5ec", linewidth=0.7, alpha=0.8)
    axis.set_aspect("equal", adjustable="datalim")
    axis.margins(0.08)
    figure.subplots_adjust(left=0.12, right=0.97, bottom=0.15, top=0.96)
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
    _input_section("06", "Compute design curve"),
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
        ui.div(
            ui.div(
                ui.div(
                    ui.tags.span("GEOMETRY CONVENTION", class_="eyebrow"),
                    ui.h3("Meaning of a and b"),
                    class_="card-title-group",
                ),
                ui.tags.span("schematic · not to scale", class_="method-chip"),
                class_="visual-card-header",
            ),
            ui.tags.img(
                src="geometry-convention.svg",
                alt=(
                    "Plan-view geometry with the attractor at minus a, two repulsors at "
                    "plus and minus b, and the movable follower at its positive x equilibrium"
                ),
                class_="geometry-convention-image",
            ),
            ui.tags.p(
                "The origin is the intersection of the fixed-magnet axes. The attractor "
                "is fixed at (-a, 0), the repulsor pair at (0, ±b), and the movable "
                "follower equilibrium x_eq > 0 is determined by the field model.",
                class_="geometry-caption",
            ),
            class_="visual-card geometry-convention-card",
        ),
        ui.output_ui("calculation_message"),
        ui.output_ui("summary_cards"),
        ui.div(
            ui.div(
                ui.div(
                    ui.tags.span("VALIDATED RESULT", class_="eyebrow"),
                    ui.h3("Design curve"),
                    class_="card-title-group",
                ),
                ui.tags.span("MPEX4 · all practical checks passed", class_="method-chip"),
                class_="visual-card-header",
            ),
            ui.output_plot("design_curve_plot", height="410px"),
            class_="visual-card design-curve-card",
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
            "derivatives. It validates units and controls the direct MPEX4 curve search.",
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
            ui.h3("Curvature-only MPEX0 scout"),
            ui.tags.p(
                "A cheap, automatically expanding MPEX0 search locates the candidate "
                "interval without constructing the expensive MPEX4 map."
            ),
            class_="docs-section",
        ),
        ui.div(
            ui.tags.span("03", class_="docs-number"),
            ui.h3("Direct MPEX4 root search"),
            ui.tags.p(
                "For every requested a, the solver directly finds b where the two planar "
                "curvatures agree. Energy, force, and Hessian remain analytically differentiated."
            ),
            class_="docs-section",
        ),
        ui.div(
            ui.tags.span("04", class_="docs-number"),
            ui.h3("Pointwise validation"),
            ui.tags.p(
                "A curve point is retained only after overlap exclusion, "
                "a positive planar Hessian, three force barriers, and the transverse "
                "centering tolerance all pass. Expensive barrier scans therefore run only "
                "at roots, rather than over a two-dimensional grid."
            ),
            class_="docs-section",
        ),
        ui.div(
            ui.tags.span("05", class_="docs-number"),
            ui.h3("Design curve"),
            ui.tags.p(
                "The output contains only the validated b(a) points. Their connecting line "
                "is a visual guide; no regression model is fitted."
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
                        ui.tags.p("Design your own tractor magnet."),
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
                progress.set(0.05, message="Tracing and validating the MPEX4 design curve…")
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
            ui.tags.strong("Design curve complete"),
            ui.tags.span(
                f"The moment condition {condition}; "
                f"{value.design_curve.a_values.size} fully validated MPEX4 curve points "
                f"were found in {value.elapsed_s:.2f} s."
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
                f"{value.design_curve.a_values.size} curve points · {value.elapsed_s:.2f} s",
                "copper",
            ),
            class_="metrics-grid",
        )

    @output
    @render.plot
    def design_curve_plot():
        return _design_curve_figure(result())


app = App(app_ui, server, static_assets=APP_DIR / "www")
