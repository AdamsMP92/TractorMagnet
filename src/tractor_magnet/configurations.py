"""Named configurations and parameter sets used in the article."""

from tractor_magnet.constants import MILLIMETRE
from tractor_magnet.models import CylinderMagnet, PlacedMagnet, PlanarConfiguration

_STACK_HEIGHT = 10.0 * MILLIMETRE


def _supplementary_magnets(
    magnet_height: float,
) -> tuple[CylinderMagnet, CylinderMagnet, CylinderMagnet]:
    """Return follower, attracting NdFeB, and repelling ferrite magnets."""

    follower = CylinderMagnet(
        radius=7.5 * MILLIMETRE,
        height=magnet_height,
        remanence=1.3,
        polarity=-1,
    )
    attractor = CylinderMagnet(
        radius=7.5 * MILLIMETRE,
        height=magnet_height,
        remanence=1.3,
        polarity=1,
    )
    repulsor = CylinderMagnet(
        radius=5.0 * MILLIMETRE,
        height=magnet_height,
        remanence=0.39,
        polarity=-1,
    )
    return follower, attractor, repulsor


def article_minimal_configuration(
    a: float = 14.0 * MILLIMETRE,
    b: float = 12.0 * MILLIMETRE,
    *,
    magnet_height: float = 10.0 * MILLIMETRE,
) -> PlanarConfiguration:
    """Return the minimal configuration from Sec. III and Fig. 5.

    ``a`` is the follower-to-attractor origin offset and ``b`` is the
    repulsor offset. The default height represents the stated stack of two
    5 mm magnets. The overall polarity convention follows Appendix A.
    """

    if a <= 0.0 or b <= 0.0:
        raise ValueError("a and b must be positive")

    follower = CylinderMagnet(
        radius=7.5 * MILLIMETRE,
        height=magnet_height,
        remanence=1.3,
        polarity=1,
    )
    attractor = CylinderMagnet(
        radius=7.5 * MILLIMETRE,
        height=magnet_height,
        remanence=1.3,
        polarity=-1,
    )
    repulsor = CylinderMagnet(
        radius=5.0 * MILLIMETRE,
        height=magnet_height,
        remanence=0.39,
        polarity=1,
    )

    return minimal_configuration(
        a,
        b,
        follower=follower,
        attractor=attractor,
        repulsor=repulsor,
    )


def minimal_configuration(
    a: float,
    b: float,
    *,
    follower: CylinderMagnet,
    attractor: CylinderMagnet,
    repulsor: CylinderMagnet,
) -> PlanarConfiguration:
    """Return the symmetric minimal configuration for arbitrary cylinders.

    The movable ``follower`` is evaluated in the common mid-plane. The fixed
    ``attractor`` is placed at ``(-a, 0)`` and two identical ``repulsor``
    magnets at ``(0, +/-b)``. Signs are carried by the supplied magnets, so
    this constructor is also useful outside the article's material choices.
    """

    if a <= 0.0 or b <= 0.0:
        raise ValueError("a and b must be positive")

    return PlanarConfiguration(
        follower=follower,
        sources=(
            PlacedMagnet(attractor, (-a, 0.0)),
            PlacedMagnet(repulsor, (0.0, b)),
            PlacedMagnet(repulsor, (0.0, -b)),
        ),
    )


def supplementary_configurations(
    *, magnet_height: float = _STACK_HEIGHT
) -> dict[str, PlanarConfiguration]:
    """Return the five configurations tabulated in the supplementary material.

    Tables I-V list the properties of each 5 mm constituent magnet. The model
    uses the 10 mm stacks stated in Sec. III and in the supplied MATLAB script.
    Positions and polarities follow the tables without an overall sign change.
    """

    follower, attractor, repulsor = _supplementary_magnets(magnet_height)

    def configuration(
        sources: tuple[tuple[CylinderMagnet, float, float], ...],
    ) -> PlanarConfiguration:
        return PlanarConfiguration(
            follower=follower,
            sources=tuple(
                PlacedMagnet(magnet, (x_mm * MILLIMETRE, y_mm * MILLIMETRE))
                for magnet, x_mm, y_mm in sources
            ),
        )

    return {
        "minimal_example_1": configuration(
            (
                (attractor, -14.0, 0.0),
                (repulsor, 0.0, 12.0),
                (repulsor, 0.0, -12.0),
            )
        ),
        "minimal_example_2": configuration(
            (
                (attractor, -11.2, 0.0),
                (repulsor, 0.0, 8.0),
                (repulsor, 0.0, -8.0),
            )
        ),
        "minimal_example_3": configuration(
            (
                (attractor, -10.5, 0.0),
                (repulsor, 0.0, 7.7),
                (repulsor, 0.0, -7.7),
            )
        ),
        "advanced_example_1": configuration(
            (
                (attractor, -1.4142, -25.8238),
                (repulsor, 0.0, -7.439),
                (repulsor, 16.9706, -24.4096),
                (attractor, -1.4142, 25.8238),
                (repulsor, 16.9706, 24.4096),
                (repulsor, 0.0, 7.439),
            )
        ),
        "advanced_example_2": configuration(
            (
                (attractor, -1.9799, -18.2693),
                (repulsor, 0.0, -5.4),
                (repulsor, 10.8894, -16.2894),
                (attractor, -1.9799, 18.2693),
                (repulsor, 10.8894, 16.2894),
                (repulsor, 0.0, 5.4),
            )
        ),
    }
