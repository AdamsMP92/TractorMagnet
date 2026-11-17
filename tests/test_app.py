from shiny import App

from app import _geometry_diagram, app, input_panel, method_panel
from web_support import WebInputs


def _inputs() -> WebInputs:
    return WebInputs.from_millimetres(
        follower_radius_mm=7.5,
        follower_height_mm=10.0,
        follower_remanence_t=1.3,
        attractor_radius_mm=7.5,
        attractor_height_mm=10.0,
        attractor_remanence_t=1.3,
        repulsor_radius_mm=5.0,
        repulsor_height_mm=10.0,
        repulsor_remanence_t=0.39,
        a_min_mm=9.0,
        a_max_mm=19.0,
        b_min_mm=4.0,
        b_max_mm=14.0,
        design_a_mm=14.0,
        design_b_mm=12.0,
        grid_points=11,
        refinement_order=4,
        friction_threshold_mn=10.0,
        centering_tolerance_mm=1.25,
    )


def test_shiny_app_constructs_with_staged_controls() -> None:
    assert isinstance(app, App)
    html = str(input_panel)
    assert "MPEX0" in html
    assert "MPEX2" in html
    assert "MPEX4" in html
    assert "Barrier threshold" in html


def test_method_panel_states_necessary_condition_and_full_mask() -> None:
    html = str(method_panel)
    assert "necessary condition" in html
    assert "m_{\\mathrm{rep}}" in html
    assert "three force barriers" in html


def test_geometry_diagram_contains_all_magnets_and_dimensions() -> None:
    diagram = _geometry_diagram(_inputs())
    assert diagram.count('class="magnet repulsor"') == 2
    assert 'class="magnet attractor"' in diagram
    assert 'class="magnet follower"' in diagram
    assert "a = 14.00 mm" in diagram
    assert "b = 12.00 mm" in diagram
