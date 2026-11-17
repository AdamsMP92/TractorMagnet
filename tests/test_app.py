from types import SimpleNamespace

import numpy as np
from shiny import App

from app import _mask_to_practical_region, app, calculator_panel, input_panel, method_panel


def test_shiny_app_constructs_with_staged_controls() -> None:
    assert isinstance(app, App)
    html = str(input_panel)
    assert "MPEX0" in html
    assert "MPEX2" in html
    assert "MPEX4" in html
    assert "101 × 101" in html
    assert "Barrier threshold" in html
    assert "Selected a" not in html
    assert "a min [mm]" not in html
    assert "b min [mm]" not in html
    assert "derived_map_bounds" not in html


def test_method_panel_states_necessary_condition_and_full_mask() -> None:
    html = str(method_panel)
    assert "necessary condition" in html
    assert "m_{\\mathrm{rep}}" in html
    assert "three force barriers" in html


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
