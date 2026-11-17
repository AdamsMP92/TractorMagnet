from types import SimpleNamespace

import numpy as np
from shiny import App

from app import _mask_to_practical_region, app, calculator_panel, input_panel, method_panel


def test_shiny_app_exposes_only_the_adaptive_resolution_control() -> None:
    assert isinstance(app, App)
    html = str(input_panel)
    assert "Compute stability map" in html
    assert "Resolution [mm]" in html
    assert "Refinement order" not in html
    assert "Grid per axis" not in html
    assert "a max [mm]" not in html
    assert "b max [mm]" not in html
    assert "Physical friction criterion" in html
    assert "Friction-force threshold" in html
    assert html.index("Physical friction criterion") < html.index("Compute stability map")
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


def test_method_panel_states_necessary_condition_and_full_mask() -> None:
    html = str(method_panel)
    assert "necessary condition" in html
    assert "m_{\\mathrm{rep}}" in html
    assert "three force barriers" in html
    assert "Production-line fit" in html
    assert "b(a) = m a + c" in html
    assert "Material-grade bounds" in html
    assert "supermagnete.de/physical-magnet-data" in html


def test_standard_output_has_only_the_chi_map() -> None:
    html = str(calculator_panel)
    assert "χ map and zero contour" in html
    assert "Selected design point" not in html
    assert "geometry_diagram" not in html


def test_chi_map_uses_the_original_practical_stability_mask() -> None:
    stability_map = SimpleNamespace(
        stable=np.asarray(((True, False), (False, True)), dtype=bool)
    )
    values = np.asarray(((1.0, 2.0), (np.nan, 4.0)))

    visible = _mask_to_practical_region(stability_map, values)

    assert visible.mask.tolist() == [[False, True], [True, False]]
    assert visible.compressed().tolist() == [1.0, 4.0]
