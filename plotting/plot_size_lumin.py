"""A script for plotting the size-luminosity relation."""

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np


def plot_size_flux_hex(filepath, filter, outpath):
    """
    Plot the size-luminosity relation.

    Args:
        filepath (str): The path to the file to plot.
    """
    # Open the file and extract the sizes and luminosities
    with h5py.File(filepath, "r") as hdf:
        # Get the redshift (it's the same for all galaxies)
        redshift = hdf["Galaxies/Redshift"][0]
        sizes = hdf["Galaxies/Stars/PixelHalfLightRadii/stellar_total/JWST"][filter][
            ...
        ]
        flux = hdf["Galaxies/Stars/Photometry/Fluxes/stellar_total/JWST/"][filter][...]

    # Create the plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)

    # Remove galaxies with no flux
    mask = np.logical_and(flux > 0, sizes > 0)
    flux = flux[mask]
    sizes = sizes[mask]

    # Plot the hexbin
    im = ax.hexbin(
        flux,
        sizes,
        gridsize=100,
        mincnt=1,
        cmap="viridis",
        xscale="log",
        yscale="log",
        linewidths=0.1,
    )

    ax.text(
        0.95,
        0.05,
        f"$z={redshift}$",
        bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="k", lw=1, alpha=0.8),
        transform=ax.transAxes,
        horizontalalignment="right",
        fontsize=8,
    )

    # Set the axis labels
    ax.set_xlabel(f"$F_{filter} " r"/ [\mathrm{nJy}]$")
    ax.set_ylabel(f"$R_{1/2} /" r" [\mathrm{kpc}]$")

    # Make and label the colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$N_{\mathrm{gal}}$")

    fig.savefig(
        outpath,
        bbox_inches="tight",
        dpi=300,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--filepath",
        type=str,
        help="The path to the file to plot.",
    )
    parser.add_argument(
        "--filter",
        type=str,
        help="The filter to use for the plot.",
    )
    parser.add_argument(
        "--outpath",
        type=str,
        help="The path to save the plot.",
    )
    args = parser.parse_args()

    # Check if the file exists
    if not os.path.exists(args.filepath):
        raise FileNotFoundError(f"{args.filepath} does not exist.")

    # Plot the size-luminosity relation
    plot_size_flux_hex(args.filepath, args.filter, args.outpath)
