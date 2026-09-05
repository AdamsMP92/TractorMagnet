import numpy as np
import pytest

from tractor_magnet.constants import MILLIMETRE
from tractor_magnet.equilibrium import find_axial_equilibria, find_planar_equilibria
from tractor_magnet.multipole import MultipoleExpansion
from tractor_magnet.reproduction import supplementary_energy_landscape_cases


def test_supplement_contains_five_tabulated_configurations() -> None:
    cases = supplementary_energy_landscape_cases()

    assert [case.table for case in cases] == [
        "Table I",
        "Table II",
        "Table III",
        "Table IV",
        "Table V",
    ]
    assert [len(case.configuration.sources) for case in cases] == [3, 3, 3, 6, 6]


def test_table_positions_are_represented_in_si_units() -> None:
    cases = supplementary_energy_landscape_cases()
    minimal_1 = cases[0].configuration
    advanced_2 = cases[-1].configuration

    assert minimal_1.sources[0].position == pytest.approx((-14.0e-3, 0.0))
    assert minimal_1.sources[1].position == pytest.approx((0.0, 12.0e-3))
    assert advanced_2.sources[0].position == pytest.approx((-1.9799e-3, -18.2693e-3))
    assert advanced_2.sources[-1].position == pytest.approx((0.0, 5.4e-3))


@pytest.mark.parametrize("case", supplementary_energy_landscape_cases(), ids=lambda case: case.slug)
def test_mpex4_equilibria_reproduce_supplement_labels(case) -> None:
    model = MultipoleExpansion(case.configuration, order=4)
    equilibria = find_axial_equilibria(model, (10.0 * MILLIMETRE, 30.0 * MILLIMETRE))
    stable = [equilibrium for equilibrium in equilibria if equilibrium.stable]

    assert stable
    closest = min(
        stable,
        key=lambda equilibrium: abs(equilibrium.point[0] - case.expected_equilibrium_x),
    )
    assert closest.point[0] == pytest.approx(case.expected_equilibrium_x, abs=0.11 * MILLIMETRE)


def test_vectorized_energy_grid_matches_scalar_evaluation() -> None:
    case = supplementary_energy_landscape_cases()[0]
    model = MultipoleExpansion(case.configuration, order=2)
    x = np.asarray((0.018, 0.021))
    y = np.asarray((-0.002, 0.003))

    x_grid, y_grid, energy = model.energy_grid(x, y)

    for row, column in np.ndindex(energy.shape):
        assert energy[row, column] == pytest.approx(
            model.energy((x_grid[row, column], y_grid[row, column])),
            rel=2.0e-13,
        )


def test_advanced_example_1_has_two_off_axis_saddles() -> None:
    case = supplementary_energy_landscape_cases()[3]
    model = MultipoleExpansion(case.configuration, order=4)

    equilibria = find_planar_equilibria(
        model,
        (10.0 * MILLIMETRE, 30.0 * MILLIMETRE),
        (-10.0 * MILLIMETRE, 10.0 * MILLIMETRE),
    )
    saddles = [equilibrium for equilibrium in equilibria if not equilibrium.stable]

    assert len(saddles) == 2
    assert [equilibrium.point[0] for equilibrium in saddles] == pytest.approx(
        [17.66493 * MILLIMETRE, 17.66493 * MILLIMETRE], abs=0.001 * MILLIMETRE
    )
    assert [equilibrium.point[1] for equilibrium in saddles] == pytest.approx(
        [-7.23368 * MILLIMETRE, 7.23368 * MILLIMETRE], abs=0.001 * MILLIMETRE
    )
