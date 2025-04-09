"""A script for plotting the emission model."""

import warnings

from synthesizer.emission_models import (
    EmissionModel,
    NebularEmission,
    ReprocessedEmission,
    StellarEmissionModel,
    TransmittedEmission,
)
from synthesizer.emission_models.transformers import PowerLaw
from synthesizer.grid import Grid
from unyt import Msun, Myr, angstrom

# Silence warnings (only because we now what we're doing)
warnings.filterwarnings("ignore")

# Msun needs to be respected
Msun = Msun.in_base("galactic")


class LOSStellarEmission(EmissionModel):
    """
    The stellar emission model used for in FLARES.

    This model is a subclass of the StellarEmissionModel class and is used
    to generate the stellar emission for galaxies in FLARES.
    """

    def __init__(self, grid):
        """
        Initialize the FLARESLOSEmission model.

        Args:
            grid (Grid): The grid to use for the model.
        """
        # Define the nebular emission models
        nebular = NebularEmission(
            grid=grid,
            label="nebular",
            mask_attr="ages",
            mask_op="<=",
            mask_thresh=10 * Myr,
        )

        # Define the transmitted models
        young_transmitted = TransmittedEmission(
            grid=grid,
            label="young_transmitted",
            mask_attr="ages",
            mask_op="<=",
            mask_thresh=10 * Myr,
        )
        old_transmitted = TransmittedEmission(
            grid=grid,
            label="old_transmitted",
            mask_attr="ages",
            mask_op=">",
            mask_thresh=10 * Myr,
        )
        transmitted = StellarEmissionModel(
            grid=grid,
            label="transmitted",
            combine=[young_transmitted, old_transmitted],
        )

        # Define the reprocessed models
        young_reprocessed = ReprocessedEmission(
            grid=grid,
            label="young_reprocessed",
            transmitted=young_transmitted,
            nebular=nebular,
            mask_attr="ages",
            mask_op="<=",
            mask_thresh=10 * Myr,
        )
        reprocessed = StellarEmissionModel(
            grid=grid,
            label="reprocessed",
            combine=[young_reprocessed, old_transmitted],
        )

        # Define the attenuated models
        young_attenuated_nebular = StellarEmissionModel(
            grid=grid,
            label="young_attenuated_nebular",
            apply_dust_to=young_reprocessed,
            tau_v="young_tau_v",
            dust_curve=PowerLaw(slope=-1),
            mask_attr="ages",
            mask_op="<=",
            mask_thresh=10 * Myr,
        )
        young_attenuated = StellarEmissionModel(
            grid=grid,
            label="young_attenuated",
            apply_dust_to=young_attenuated_nebular,
            tau_v="tau_v",
            dust_curve=PowerLaw(slope=-1),
            mask_attr="ages",
            mask_op="<=",
            mask_thresh=10 * Myr,
        )
        old_attenuated = StellarEmissionModel(
            grid=grid,
            label="old_attenuated",
            apply_dust_to=old_transmitted,
            tau_v="tau_v",
            dust_curve=PowerLaw(slope=-1),
            mask_attr="ages",
            mask_op=">",
            mask_thresh=10 * Myr,
        )

        # Finaly, combine to get the emergent emission
        EmissionModel.__init__(
            self,
            grid=grid,
            label="stellar_total",
            combine=[young_attenuated, old_attenuated],
            related_models=[
                nebular,
                transmitted,
                reprocessed,
                young_attenuated_nebular,
            ],
            emitter="stellar",
        )

        self.set_per_particle(True)


def get_emission_model(
    grid_name,
    grid_dir,
    fesc=0.0,
    fesc_ly_alpha=1.0,
):
    """Get the emission model to use for the observations."""
    grid = Grid(
        grid_name,
        grid_dir,
        lam_lims=(900 * angstrom, 6 * 10**5 * angstrom),
    )
    model = LOSStellarEmission(grid)

    # # Limit the spectra to be saved
    # model.save_spectra("attenuated", "reprocessed")

    return model


if __name__ == "__main__":
    # Get the emission model
    model = get_emission_model(
        grid_name="test_grid.hdf5",
        grid_dir="../../../flares_test/",
        fesc=0.0,
        fesc_ly_alpha=1.0,
    )

    fig, ax = model.plot_emission_tree()
    fig.savefig("../plots/emission_tree.png", dpi=300, bbox_inches="tight")
