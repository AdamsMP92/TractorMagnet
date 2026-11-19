# Tractor Magnet

Reproducible Python implementation of the calculations in Michael P. Adams,
“Understanding two-dimensional tractor magnets: Theory and realizations”,
*American Journal of Physics* **92**, 576–582 (2024).

## Environment

The environment is managed and locked with [Pixi](https://pixi.sh):

```bash
pixi shell
```

Useful checks can also be run without entering a shell:

```bash
pixi run test
pixi run lint
pixi run format
pixi run figure-4
pixi run figure-5
pixi run supplementary-landscapes
pixi run app
```

## Code layout

The Python package lives in `src/tractor_magnet`. Its public inputs use SI units.

- `models.py`: cylindrical magnets, placements, and configurations
- `moments.py`: Cartesian cylinder moments from Eqs. (B3)–(B7)
- `kernels.py`: analytical Cartesian derivatives of `1/r`
- `point_dipole.py`: MPEX0 and the analytical Appendix-A relations
- `multipole.py`: general even-order MPEX-N interaction model
- `equilibrium.py`: axial root search and planar stability classification
- `stability_map.py`: Fig. 4 force barriers, friction mask, and symmetry measure
- `design_curve.py`: direct, pointwise validated MPEX4 curvature-balance curve
- `configurations.py`: parameter sets used in the article
- `reproduction.py`: datasets for article figure reproductions

`MultipoleExpansion(configuration, order=4)` retains all nonzero Cartesian
moments of both the fixed magnets and the follower through fourth order. Energy,
force, and the planar Hessian are evaluated analytically; no finite-difference
derivatives are used.

## Minimal example

```python
from tractor_magnet import (
    MultipoleExpansion,
    article_minimal_configuration,
    find_axial_equilibria,
)

configuration = article_minimal_configuration(a=14e-3, b=12e-3)
model = MultipoleExpansion(configuration, order=4)
equilibria = find_axial_equilibria(model, (15e-3, 50e-3))
```

The `figure-5` task evaluates MPEX0, MPEX2, MPEX4, and MPEX6 sequentially and
writes `results/figure_5_multipole_convergence.png`.

The `figure-4` task evaluates the six theoretical maps successively with
MPEX0, MPEX2, and MPEX4. It writes one six-panel map per order, the numerical
map data, a comparison of
`chi = log10(abs(d2U/dy2 - d2U/dx2))`, and linear symmetry-line fits to
`results/figure_4_stability_maps/`. The plots include all 82 experimental
markers: 79 coordinates are decoded and verified from the filenames in
`STL_Samples/`, while the three additional minimal configurations come from the
supplementary tables. Their outcome labels are recorded in
`data/figure_4_experimental_points.csv`.

The transverse force barrier is evaluated over the geometry-dependent capture
interval `-(b + R_rep) <= y <= b + R_rep`. Mirror symmetry reduces the actual
calculation to its positive half. The map data include both the half-span and
the position at which the first restoring-force maximum occurs. Every axial
and transverse barrier path is clipped at the first possible contact between
the follower and a fixed magnet; the corresponding scan limits and all three
barrier positions are stored in the numerical map data.

The practical stability mask additionally limits the transverse static-friction
dead zone. Its width `y_10` is the first displacement at which the restoring
force reaches 10 mN. By default, a configuration must satisfy
`y_10 <= R_rep / 4 = 1.25 mm`; this explicit centering tolerance can be changed
with `--centering-tolerance-mm`. The original force-amplitude-only mask remains
available as `StabilityMap.force_stable`, and
`transverse_localization_comparison.png` visualizes the distinction.

Additional orders can be appended without recomputing existing maps, for
example `python scripts/reproduce_figure_4.py --orders 6`. The comparison plot
automatically includes every stored order.

The `supplementary-landscapes` task evaluates the five configurations from
Tables I-V of `Supplementary.pdf` with MPEX4. It writes individual energy
landscapes, a combined overview, and an `equilibria.csv` comparison to
`results/supplementary_energy_landscapes/`.

The tables describe the 5 mm constituent magnets. Following Sec. III and the
supplied MATLAB calculation, the interaction model uses two-magnet stacks with
an effective height of 10 mm.

The `production-line-animation` task samples 200 MPEX4 configurations along
the practically stable part of the fitted production line. It writes an MP4
with a fixed spatial domain and energy scale, plus a CSV containing the design
coordinates and stability diagnostics for every frame. Use, for example,
`pixi run production-line-animation --frames 200`.

## Browser app

The deployed app is available at
<https://adamsmp92.github.io/TractorMagnet/>.

The browser-native Shiny interface follows the same deployment pattern as the
other scientific apps at <https://adamsmp92.github.io/homepage/software/>. It
provides a direct design-curve analysis for arbitrary cylindrical follower,
attractor, and repulsor magnets:

1. the necessary point-dipole moment condition,
2. an automatically expanding, curvature-only MPEX0 scout,
3. a direct exact-cylinder MPEX4 root search for `Hyy - Hxx = 0`,
4. complete practical validation at every root.

The only numerical search input is the final spacing in `a` (0.2 mm by
default); all search limits are inferred automatically. Instead of evaluating
a two-dimensional MPEX4 map, the solver traces the balance directly as `b(a)`.
The expensive force-barrier and centering calculations therefore run only at
the roots. All four magnets share one height input. The validated points are
shown as a single design curve; no regression model is fitted.
Each magnet type is assigned a ferrite or NdFeB material grade from the
[physical magnet data published by supermagnete.de](https://www.supermagnete.de/physical-magnet-data).
Selecting a grade resets `Br` to its range midpoint rounded to 1 mT; `Br` can
then be adjusted in 1 mT increments within that grade's room-temperature
remanence interval.

Every displayed point retains the complete scientific core: fixed-magnet
overlap exclusion, an exterior follower equilibrium, a positive planar
Hessian, all three force-barrier thresholds, collision-limited scan paths, and
the transverse centering tolerance. The original two-dimensional Figure-4 map
pipeline remains available in the package for scientific reproduction work.

Run the app locally with `pixi run app` and export the static GitHub Pages
bundle with `pixi run export-site`. All numerical work then runs in the
visitor's browser; no simulation server is required.
