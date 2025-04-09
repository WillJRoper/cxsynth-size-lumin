"""A script for plotting the emission model."""

import warnings

from my_emission_models import LOSStellarEmission
from synthesizer.grid import Grid
from unyt import Msun, angstrom

# Silence warnings (only because we now what we're doing)
warnings.filterwarnings("ignore")

# Msun needs to be respected
Msun = Msun.in_base("galactic")


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
