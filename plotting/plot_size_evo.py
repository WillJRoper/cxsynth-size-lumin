"""A script for plotting the size evolution of galaxies."""

import argparse
import glob
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic


def plot_size_evolution(filepath):
    """
    Plot the size evolution of galaxies.

    This will read all snapshots in the input directory and plot their size
    evolution.
    """
    # Get all the files in the directory
    files = glob.glob(filepath + "/Synth*.hdf5")

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

    # Compute the median in each redshift bin and the standard deviation in
    # each bin so we can plot the error bars
    median_xs = np.arange(
        -0.5,
        np.max(redshifts) + 0.5,  # We want to include the last bin
        1.0,
    )
    median_ys = binned_statistic(
        redshifts,
        sizes,
        statistic="median",
        bins=median_xs,
    )[0]

    median_ys_std = binned_statistic(
        redshifts,
        sizes,
        statistic="std",
        bins=median_xs,
    )[0]

    print(f"Fitted parameters: {popt}")
    print(f"Fitted covariance: {pcovr}")

    # Plot the data
    fig, ax = plt.subplots(figsize=(3.5, 3.5))

    # Plot the binned data
    ax.errorbar(
        (median_xs[:-1] + median_xs[1:]) / 2,
        median_ys,
        yerr=median_ys_std,
        fmt="o",
        color="k",
    )

    # Plot the fit
    ax.plot(
        fit_xs,
        fit_ys,
        color="r",
        linestyle="--",
        label="Fit",
    )

    # Set the axis labels
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$R_{1/2} / [\mathrm{kpc}]$")

    ax.legend(
        loc="upper left",
        fontsize=8,
    )

    fig.savefig(
        f"{outpath}/size_evolution.png",
        dpi=300,
        bbox_inches="tight",
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
    args = parser.parse_args()

    # Define input and output paths
    run_name = args.run_name
    variant = args.variant
    path = f"../data/{run_name}/{variant}/"
    outpath = f"../plots/{run_name}/{variant}/"

    # Check if the file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist.")

    # Create the output directory if it doesn't exist
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # Plot the size evolution
    plot_size_evolution(path)
