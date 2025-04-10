"""A script for plotting the size evolution of galaxies."""

import glob

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic

# Lstar = M_to_lum(-21)
Lstar = 10**28.51


def size_evolution_func(z, r0, m):
    """Fit the size evolution."""
    return r0 * (1 + z) ** m


def plot_size_evolution_medians(
    filepath,
    fig=None,
    ax=None,
    label=None,
    lstyle="-",
    color=None,
    outpath=None,
    mass_lim=None,
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
    files = glob.glob(filepath + "/Synthesized_imgs_*_test_grid.hdf5")

    # If there's only one file don't bother
    if len(files) <= 1:
        print("Only one file found, not plotting.")
        return fig, ax

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
    masses = []
    lums = []
    for f in files:
        # Open the file and extract the sizes and redshifts
        with h5py.File(f, "r") as hdf:
            # Get the redshift (it's the same for all galaxies)
            redshift = hdf["Galaxies/Redshift"]
            sizes.extend(hdf["Galaxies/Stars/MassRadii/0p5"][...])
            masses.extend(hdf["Galaxies/Stars/StellarMass"][...])
            lums.extend(
                hdf["Galaxies/Stars/Photometry/Luminosities/stellar_total/UV1500"][...]
            )
            redshifts.extend(redshift)

    # Convert the data to numpy arrays
    redshifts = np.array(redshifts)
    sizes = np.array(sizes)
    masses = np.array(masses)
    lums = np.array(lums)

    # Apply the mass limit if it is set
    if mass_lim is not None:
        mask = masses > mass_lim
        sizes = sizes[mask]
        redshifts = redshifts[mask]
        masses = masses[mask]
        lums = lums[mask]

    # Do the luminosity cut 0.3 * Lstar < L < Lstar
    mask = np.logical_and(lums > 0.3 * Lstar, lums < Lstar)

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
    # ax.plot(
    #     fit_xs,
    #     fit_ys,
    #     label=label if label is not None else "Fit",
    #     linestyle=lstyle,
    #     color=color,
    # )

    # Compute the median and error bars
    median_xs = np.arange(
        -0.5,
        np.max(redshifts) + 1.5,
        1,
    )
    median_ys = binned_statistic(
        redshifts,
        sizes,
        statistic="median",
        bins=median_xs,
    )[0]
    low_err = binned_statistic(
        redshifts,
        sizes,
        statistic=lambda x: np.percentile(x, 16),
        bins=median_xs,
    )[0]
    high_err = binned_statistic(
        redshifts,
        sizes,
        statistic=lambda x: np.percentile(x, 84),
        bins=median_xs,
    )[0]

    # Plot the median
    ax.plot(
        (median_xs[:-1] + median_xs[1:]) / 2,
        median_ys,
        label=label if label is not None else "Median",
        linestyle=lstyle,
        color=color,
    )

    # Extract the color from the plot if not provided
    if color is None:
        color = ax.lines[-1].get_color()

    # Plot the error bars as shaded regions
    ax.fill_between(
        (median_xs[:-1] + median_xs[1:]) / 2,
        low_err,
        high_err,
        color=color,
        alpha=0.2,
    )

    # If we are saving it go ahead and save it
    if save:
        ax.legend(
            loc="best",
            fontsize=8,
        )

        fig.savefig(
            f"{outpath}size_evolution.png",
            dpi=300,
            bbox_inches="tight",
        )

    return fig, ax


def plot_size_evolution_comps(fig, ax, outpath=None):
    """Plot the size evolution from other studies."""
    # Are we writing?
    save = False
    if outpath is not None:
        save = True

    # Plot FLARES IV
    r0_flares = 21.98
    m_flares = -1.59
    flares_zs = np.linspace(5, 12, 100)
    ax.plot(
        flares_zs,
        size_evolution_func(flares_zs, r0_flares, m_flares),
        label="FLARES IV (Roper+22)",
        linestyle="dotted",
    )

    # Plot Omerod
    r0_omerod = 4.5
    m_omerod = -0.71
    omerod_zs = np.linspace(0, 8, 100)
    ax.plot(
        omerod_zs,
        size_evolution_func(omerod_zs, r0_omerod, m_omerod),
        label="Ormerod+2023",
        linestyle="dotted",
    )
    # If we are saving it go ahead and save it
    if save:
        ax.legend(
            loc="best",
            fontsize=8,
        )

        fig.savefig(
            outpath,
            dpi=300,
            bbox_inches="tight",
        )

    return fig, ax


if __name__ == "__main__":
    outpath = "../plots/thermal_vs_hybrid_size_evo.png"

    # Plot the size evolution for Thermal vs Hybrid
    fig, ax = plot_size_evolution_medians(
        filepath="../data/L050_m6/THERMAL_AGN_m6/",
        fig=None,
        ax=None,
        label="L050_m6/Thermal",
        lstyle="--",
        color="r",
        mass_lim=1e9,
    )
    fig, ax = plot_size_evolution_medians(
        filepath="../data/L050_m6/HYBRID_AGN_m6/",
        fig=fig,
        ax=ax,
        label="L050_m6/Hybrid",
        lstyle="--",
        color="b",
        mass_lim=1e9,
    )
    fig, ax = plot_size_evolution_medians(
        filepath="../data/L200_m7/THERMAL_AGN_m7/",
        fig=fig,
        ax=ax,
        label="L200_m7/Thermal",
        lstyle="-",
        color="g",
        mass_lim=1e9,
    )
    fig, ax = plot_size_evolution_medians(
        filepath="../data/L200_m7/HYBRID_AGN_m7/",
        fig=fig,
        ax=ax,
        label="L200_m7/Hybrid",
        lstyle="-",
        color="cyan",
        mass_lim=1e9,
        outpath=outpath,
    )

    plt.close(fig)

    outpath = "../plots/all_runs_size_evo.png"
    # Plot the size evolution for all snaps completed so far
    files = glob.glob("../data/*/THERMAL_AGN_m*/")
    fig = None
    ax = None
    for file in files:
        fig, ax = plot_size_evolution_medians(
            filepath=file,
            fig=fig,
            ax=ax,
            label=f"{file.split("/")[2]}/{file.split('/')[3][:-3]}",
            lstyle="-",
            # mass_lim=1e9,
        )

    # ax.legend(
    #     loc="best",
    #     fontsize=8,
    # )
    # fig.savefig(outpath, dpi=300, bbox_inches="tight")

    fig, ax = plot_size_evolution_comps(
        fig=fig,
        ax=ax,
        outpath=outpath,
    )
