"""A script for plotting the size evolution of galaxies."""

import glob

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def plot_size_evolution(
    filepath,
    fig=None,
    ax=None,
    label=None,
    lstyle="-",
    color="r",
    outpath="/",
):
    """
    Plot the size evolution of galaxies.

    This will read all snapshots in the input directory and plot their size
    evolution.
    """
    # Are we writing?
    save = False
    if outpath is not None:
        save = True

    # Get all the files in the directory
    files = glob.glob(filepath)

    # We need to make sure we only have the combined files, luckily these all
    # have 1 extra element delineated by an underscore so we can use that to
    # filter out the other files
    lengths = [len(f.split("_")) for f in files]
    combined_lengths = min(lengths)
    files = [f for f in files if len(f.split("_")) == combined_lengths]

    print(f"Found {len(files)} files.")
    print(f"Files: {files}")

    # Loop over the files and extract the redshifts and sizes
    redshifts = []
    sizes = []
    for f in files:
        # Open the file and extract the sizes and redshifts
        with h5py.File(f, "r") as hdf:
            # Get the redshift (it's the same for all galaxies)
            redshift = hdf["Galaxies/Redshift"]
            sizes.extend(hdf["Galaxies/Stars/MassRadii/0p5"][...])
            redshifts.extend(redshift)

    # Convert the data to numpy arrays
    redshifts = np.array(redshifts)
    sizes = np.array(sizes)

    # Convert the sizes to kpc
    sizes = sizes * 1e3

    # Fit a line to the data
    popt, pcovr = curve_fit(lambda z, a, b: a * (1 + z) ** b, redshifts, sizes)
    fit_xs = np.linspace(min(redshifts), max(redshifts), 100)
    fit_ys = popt[0] * (1 + fit_xs) ** popt[1]

    print(f"Fitting parameters for {label}")
    print(f"Fitted parameters: {popt}")
    print(f"Fitted covariance: {pcovr}")

    # Plot the data
    if fig is None and ax is None:
        fig, ax = plt.subplots(figsize=(3.5, 3.5))

        # Add a grid and make sure its always at the back
        ax.grid(True)
        ax.set_axisbelow(True)

        # Set the axis labels
        ax.set_xlabel(r"$z$")
        ax.set_ylabel(r"$R_{1/2} / [\mathrm{kpc}]$")

    # Plot the fit
    ax.plot(
        fit_xs,
        fit_ys,
        label=label if label is not None else "Fit",
        linestyle=lstyle,
        color=color,
    )

    # If we are saving it go ahead and save it
    if save:
        ax.legend(
            loc="upper left",
            fontsize=8,
        )

        fig.savefig(
            f"{outpath}size_evolution.png",
            dpi=300,
            bbox_inches="tight",
        )

    return fig, ax


if __name__ == "__main__":
    outpath = "../plots/thermal_vs_hybrid_size_evo.png"

    # Plot the size evolution
    fig, ax = plot_size_evolution(
        filepath="../data/L050_m6/THERMAL_AGN_m6/Synthesized_imgs_*_test_grid.hdf5",
        fig=None,
        ax=None,
        label="L050_m6/Thermal",
        lstyle="--",
        color="r",
    )
    fig, ax = plot_size_evolution(
        filepath="../data/L050_m6/HYBRID_AGN_m6/Synthesized_imgs_*_test_grid.hdf5",
        fig=fig,
        ax=ax,
        label="L050_m6/Hybrid",
        lstyle="--",
        color="b",
    )
    fig, ax = plot_size_evolution(
        filepath="../data/L200_m7/THERMAL_AGN_m7/Synthesized_imgs_*_test_grid.hdf5",
        fig=None,
        ax=None,
        label="L200_m7/Thermal",
        lstyle="--",
        color="r",
    )
    fig, ax = plot_size_evolution(
        filepath="../data/L200_m7/HYBRID_AGN_m7/Synthesized_imgs_*_test_grid.hdf5",
        fig=fig,
        ax=ax,
        label="L200_m7/Hybrid",
        lstyle="--",
        color="b",
        outpath=outpath,
    )
