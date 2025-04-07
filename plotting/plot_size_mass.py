"""A script for plotting the size-mass relation."""

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binned_statistic
from unyt import Mpc


def plot_size_mass_hex(filepath, outpath):
    """
    Plot the size-luminosity relation.

    Args:
        filepath (str): The path to the file to plot.
    """
    # Open the file and extract the sizes and luminosities
    with h5py.File(filepath, "r") as hdf:
        # Get the redshift (it's the same for all galaxies)
        redshift = hdf["Galaxies/Redshift"][0]
        sizes = hdf["Galaxies/Stars/MassRadii/0p5"][...]
        mass = hdf["Galaxies/Stars/StellarMass"][...]

    # Create the plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)

    # Remove galaxies with no flux
    mask = np.logical_and(mass > 0, sizes > 0)
    sizes = sizes[mask] * Mpc
    mass = mass[mask]

    # Convert the sizes to kpc
    sizes = sizes.to("kpc").value

    # Plot the hexbin
    im = ax.hexbin(
        mass,
        sizes,
        gridsize=30,
        mincnt=1,
        cmap="viridis",
        xscale="log",
        linewidths=0.1,
    )

    ax.text(
        0.95,
        0.05,
        f"$z={redshift:.1f}$",
        bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="k", lw=1, alpha=0.8),
        transform=ax.transAxes,
        horizontalalignment="right",
        fontsize=8,
    )

    # Plot a median line
    median_xs = np.logspace(
        np.log10(np.min(mass)),
        np.log10(np.max(mass)),
        15,
    )
    median_ys = binned_statistic(
        mass,
        sizes,
        statistic="median",
        bins=median_xs,
    )[0]
    ax.plot(
        (median_xs[:-1] + median_xs[1:]) / 2,
        median_ys,
        color="r",
        linestyle="--",
        label="Median",
    )

    # Set the axis labels
    ax.set_xlabel(r"$M_\star / [\mathrm{M}_\odot]$")
    ax.set_ylabel(r"$R_{1/2} / [\mathrm{kpc}]$")

    # Make and label the colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$N_{\mathrm{gal}}$")

    ax.legend(
        loc="best",
        fontsize=8,
        fancybox=True,
    )

    fig.savefig(
        outpath,
        bbox_inches="tight",
        dpi=300,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-name",
        type=str,
        help="The name of the simulation (the directory in run-dir).",
        default="L025_m7",
    )
    parser.add_argument(
        "--variant",
        type=str,
        help="The variant of the simulation (e.g. THERMAL_AGN_m6/HYBRID_AGN_m7).",
        default="THERMAL_AGN_m7",
    )
    parser.add_argument(
        "--snap",
        type=int,
        help="The snapshot number to plot.",
        default=0,
    )
    args = parser.parse_args()

    # Define input and output paths
    run_name = args.run_name
    variant = args.variant
    path = (
        f"../data/{run_name}/{variant}/Synthesized_imgs_{args.snap:04d}_test_grid.hdf5"
    )
    outpath = f"../plots/{run_name}/{variant}/size_mass_{args.snap:04d}.png"

    # Check if the file exists
    if not os.path.exists(args.filepath):
        raise FileNotFoundError(f"{args.filepath} does not exist.")

    plot_size_mass_hex(path, outpath)
