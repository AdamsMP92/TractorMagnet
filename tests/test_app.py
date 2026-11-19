from shiny import App

from app import app, calculator_panel, input_panel, method_panel


def test_shiny_app_exposes_only_the_adaptive_resolution_control() -> None:
    assert isinstance(app, App)
    html = str(input_panel)
    assert "Compute design curve" in html
    assert "Resolution [mm]" in html
    assert "Refinement order" not in html
    assert "Grid per axis" not in html
    assert "a max [mm]" not in html
    assert "b max [mm]" not in html
    assert "Physical friction criterion" in html
    assert "Friction-force threshold" in html
    friction_section = html[
        html.index("Physical friction criterion") : html.index("Compute design curve")
    ]
    assert "input-pair" not in friction_section
    assert friction_section.index("Friction-force threshold") < friction_section.index(
        "Centering tolerance"
    )
    assert html.index("Physical friction criterion") < html.index("Compute design curve")
    assert "Follower Magnet" in html
    assert "Attractor Magnet" in html
    assert "Repulsor Magnet Pair" in html
    assert html.count("Height [mm]") == 1
    assert "Ferrite Y35 (0.400–0.410 T)" in html
    assert "NdFeB N52 (1.420–1.470 T)" in html
    follower = html.index("Follower Magnet")
    attractor = html.index("Attractor Magnet")
    assert follower < html.index("Radius [mm]", follower, attractor)
    assert html.index("Radius [mm]", follower, attractor) < html.index(
        "Magnet grade", follower, attractor
    )
    assert html.index("Magnet grade", follower, attractor) < html.index(
        "Remanence Br [T]", follower, attractor
    )
    assert "Selected a" not in html
    assert "a min [mm]" not in html
    assert "b min [mm]" not in html
    assert "derived_map_bounds" not in html


def test_method_panel_states_direct_curve_search_and_full_validation() -> None:
    html = str(method_panel)
    assert "necessary condition" in html
    assert "m_{\\mathrm{rep}}" in html
    assert "three force barriers" in html
    assert "Direct MPEX4 root search" in html
    assert "without constructing the expensive MPEX4 map" in html
    assert "no regression model is fitted" in html
    assert "Material-grade bounds" in html
    assert "supermagnete.de/physical-magnet-data" in html


def test_standard_output_has_only_the_design_curve() -> None:
    html = str(calculator_panel)
    assert "Meaning of a and b" in html
    assert "geometry-convention.svg" in html
    assert "The origin is the intersection of the fixed-magnet axes" in html
    assert "follower equilibrium x_eq &gt; 0" in html
    assert "Design curve" in html
    assert "all practical checks passed" in html
    assert "CURVATURE SYMMETRY" not in html
    assert "χ map and zero contour" not in html
    assert "Hyy − Hxx = 0 data points" not in html
    assert "stability_plot" not in html
    assert "Selected design point" not in html
    assert "geometry_diagram" not in html
    assert "Linear fit" not in html


def test_hero_uses_the_simple_design_message() -> None:
    html = str(app.ui)
    assert "Design your own tractor magnet." in html
    assert "Explore the minimal symmetric configuration" not in html
