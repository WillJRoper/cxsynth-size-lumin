"""A script for plotting the size-luminosity relation."""

import argparse
import glob
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
from synthesizer.instruments import FilterCollection
from unyt import angstrom

# Lstar = M_to_lum(-21)
Lstar = 10**28.51


def size_lumin_fit(lum, Lstar, r0, b):
    """
    Fit the size-luminosity relation.

    Args:
        l (float): The luminosity.
        Lstar (float): The characteristic luminosity.
        b (float): The slope of the relation.

    Returns:
        float: The size.
    """
    return r0 * (lum / Lstar) ** b


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

    # Add a grid and make sure its always at the back
    ax.grid(True)
    ax.set_axisbelow(True)

    # Remove galaxies with no flux
    mask = np.logical_and(flux > 0, sizes > 0)
    flux = flux[mask]
    sizes = sizes[mask]

    # Plot the hexbin
    im = ax.hexbin(
        flux,
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
        np.log10(np.min(flux)),
        np.log10(np.max(flux)),
        15,
    )
    median_ys = binned_statistic(
        flux,
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
    ax.set_xlabel(r"$F_{" + filter.split(".")[-1] + r"} " r"/ [\mathrm{nJy}]$")
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
        outpath + f"Obs_{filter.split('.')[-1]}_size_flux_hex.png",
        bbox_inches="tight",
        dpi=300,
    )


def plot_size_lum_hex_uv_obs(filepath, filtpath, outpath):
    """
    Plot the size-luminosity relation.

    Args:
        filepath (str): The path to the file to plot.
    """
    # Load the filters
    nircam_filters = FilterCollection(path=filtpath + "/nircam_filters.hdf5")
    miri_filters = FilterCollection(path=filtpath + "/miri_filters.hdf5")
    filters = nircam_filters + miri_filters

    # Open the file and extract the sizes and luminosities
    with h5py.File(filepath, "r") as hdf:
        # Get the redshift (it's the same for all galaxies)
        redshift = hdf["Galaxies/Redshift"][0]

        print("Plotting the size-luminosity relation at z=", redshift)

        # What filter does 1500 angstrom correspond to at this redshift?
        search_lam = 1500 * angstrom
        filter = filters.find_filter(
            search_lam,
            redshift=redshift,
            method="transmission",
        ).filter_code.split("/")[-1]

        sizes = hdf["Galaxies/Stars/PixelHalfLightRadii/stellar_total/JWST"][filter][
            ...
        ]
        flux = hdf["Galaxies/Stars/Photometry/Luminosities/stellar_total/UV1500"][...]

    # Create the plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)

    # Add a grid and make sure its always at the back
    ax.grid(True)
    ax.set_axisbelow(True)

    # Remove galaxies with no flux
    mask = np.logical_and(flux > 0, sizes > 0)
    flux = flux[mask]
    sizes = sizes[mask]

    # Plot the hexbin
    im = ax.hexbin(
        flux,
        sizes,
        gridsize=30,
        mincnt=1,
        cmap="viridis",
        xscale="log",
        yscale="log",
        linewidths=0.1,
    )

    ax.text(
        0.95,
        0.05,
        f"$z={redshift:1f}$",
        bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="k", lw=1, alpha=0.8),
        transform=ax.transAxes,
        horizontalalignment="right",
        fontsize=8,
    )

    # Set the axis labels
    ax.set_xlabel(r"$L_{1500} / [\mathrm{nJy}]$")
    ax.set_ylabel(r"$R_{1/2} / [\mathrm{kpc}]$")

    # Make and label the colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$N_{\mathrm{gal}}$")

    fig.savefig(
        outpath + "Obs_UV_size_lum_hex.png",
        bbox_inches="tight",
        dpi=300,
    )


def plot_size_lum_hex_uv(filepath, outpath, spec_type, xlim=None, ylim=None):
    """
    Plot the size-luminosity relation.

    Args:
        filepath (str): The path to the file to plot.
    """
    # Open the file and extract the sizes and luminosities
    with h5py.File(filepath, "r") as hdf:
        # Get the redshift (it's the same for all galaxies)
        redshift = hdf["Galaxies/Redshift"][0]

        print("Plotting the size-luminosity relation at z=", redshift)

        sizes = hdf[
            f"Galaxies/Stars/PixelHalfLightRadii/Luminosity/{spec_type}/UV1500"
        ][...]
        flux = hdf[f"Galaxies/Stars/Photometry/Luminosities/{spec_type}/UV1500"][...]

    # Create the plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)

    # Add a grid and make sure its always at the back
    ax.grid(True)
    ax.set_axisbelow(True)

    # Remove galaxies with no flux
    mask = np.logical_and(flux > 0, sizes > 0)
    flux = flux[mask]
    sizes = sizes[mask]

    # Plot the hexbin
    im = ax.hexbin(
        flux,
        sizes,
        gridsize=30,
        mincnt=1,
        cmap="viridis",
        xscale="log",
        yscale="log",
        linewidths=0.1,
        norm=LogNorm(),
    )

    # Plot a median line
    median_xs = np.logspace(
        np.log10(np.min(flux)),
        np.log10(np.max(flux)),
        15,
    )
    median_ys = binned_statistic(
        flux,
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

    ax.text(
        0.95,
        0.05,
        f"$z={redshift:.1f}$",
        bbox=dict(boxstyle="round,pad=0.3", fc="w", ec="k", lw=1, alpha=0.8),
        transform=ax.transAxes,
        horizontalalignment="right",
        fontsize=8,
    )

    # Set the axis labels
    ax.set_xlabel(r"$L_{1500} / [\mathrm{erg / s / Hz}]$")
    ax.set_ylabel(r"$R_{1/2} / [\mathrm{kpc}]$")

    # Make and label the colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r"$N_{\mathrm{gal}}$")

    # Set the axis limits
    if xlim is not None:
        ax.set_xlim(xlim, None)
    if ylim is not None:
        ax.set_ylim(ylim, None)

    fig.savefig(
        outpath + f"UV_size_lum_hex_{spec_type}.png",
        bbox_inches="tight",
        dpi=300,
    )


def plot_size_lum_hex_fit_multi(
    filepath,
    outpath,
    spec_type,
    xlim=None,
    ylim=None,
    fig=None,
    ax=None,
):
    """
    Plot the size-luminosity relation.

    Args:
        filepath (str): The path to the file to plot.
    """
    # Get all the files at the path
    files = glob.glob(filepath)

    # Nothing to do with no files
    if len(files) == 0:
        print("No files found at path:", filepath)
        return

    # Create the plot
    if fig is None:
        fig = plt.figure(figsize=(3.5, 3.5))
        ax = fig.add_subplot(111)

        # Add a grid and make sure its always at the back
        ax.grid(True)
        ax.set_axisbelow(True)

    # Loop over the files
    lums = {}
    sizes = {}
    zs = []
    for file in files:
        # Open the file and extract the sizes and luminosities
        with h5py.File(filepath, "r") as hdf:
            # Get the redshift (it's the same for all galaxies)
            zs.append(hdf["Galaxies/Redshift"][0])

            print("Plotting the size-luminosity relation at z=", zs[-1])

            sizes[zs[-1]] = hdf[
                f"Galaxies/Stars/PixelHalfLightRadii/Luminosity/{spec_type}/UV1500"
            ][...]
            lums[zs[-1]] = hdf[
                f"Galaxies/Stars/Photometry/Luminosities/{spec_type}/UV1500"
            ][...]

    # Construct a colormap from the redshifts
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=min(zs), vmax=max(zs))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)

    # Loop over the redshifts, fit and plot
    for z in zs:
        # Remove galaxies with no flux
        mask = np.logical_and(lums > 0, sizes > 0)
        lum = lums[z][mask]
        size = sizes[z][mask]

        # Fit the size-luminosity relation
        popt, pcov = curve_fit(
            size_lumin_fit,
            lum,
            size,
            p0=[1, 0.5],
        )
        print(f"{filepath} Fitted parameters:", popt)

        # Plot the fit
        xs = np.logspace(
            np.log10(np.min(lum)),
            np.log10(np.max(lum)),
            100,
        )
        ax.plot(
            xs,
            size_lumin_fit(xs, *popt),
            color=sm.to_rgba(z),
            linestyle="-",
        )

    # Set the axis labels
    ax.set_xlabel(r"$L_{1500} / [\mathrm{erg / s / Hz}]$")
    ax.set_ylabel(r"$R_{1/2} / [\mathrm{kpc}]$")

    # Set the axis limits
    if xlim is not None:
        ax.set_xlim(xlim, None)
    if ylim is not None:
        ax.set_ylim(ylim, None)

    # Add a colorbar
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(r"$z$")

    fig.savefig(
        outpath + f"UV_size_lum_redshift_evo_{spec_type}.png",
        bbox_inches="tight",
        dpi=300,
    )


def plot_size_lum_hex_uv_fit(
    filepath,
    outpath,
    spec_type,
    xlim=None,
    ylim=None,
    fig=None,
    ax=None,
):
    """
    Plot the size-luminosity relation.

    Args:
        filepath (str): The path to the file to plot.
    """
    # Open the file and extract the sizes and luminosities
    with h5py.File(filepath, "r") as hdf:
        # Get the redshift (it's the same for all galaxies)
        redshift = hdf["Galaxies/Redshift"][0]

        print("Plotting the size-luminosity relation at z=", redshift)

        sizes = hdf[
            f"Galaxies/Stars/PixelHalfLightRadii/Luminosity/{spec_type}/UV1500"
        ][...]
        flux = hdf[f"Galaxies/Stars/Photometry/Luminosities/{spec_type}/UV1500"][...]

    # Create the plot
    if fig is None:
        fig = plt.figure(figsize=(3.5, 3.5))
        ax = fig.add_subplot(111)

        # Add a grid and make sure its always at the back
        ax.grid(True)
        ax.set_axisbelow(True)

    # Remove galaxies with no flux
    mask = np.logical_and(flux > 0, sizes > 0)
    flux = flux[mask]
    sizes = sizes[mask]

    # Fit the size-luminosity relation
    popt, pcov = curve_fit(
        size_lumin_fit,
        flux,
        sizes,
        p0=[1, 0.5],
    )
    print(f"{filepath} Fitted parameters:", popt)

    # Plot the fit
    ax.plot(
        flux,
        size_lumin_fit(flux, *popt),
        color="r",
        linestyle="-",
        label="",
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

    # Set the axis labels
    ax.set_xlabel(r"$L_{1500} / [\mathrm{erg / s / Hz}]$")
    ax.set_ylabel(r"$R_{1/2} / [\mathrm{kpc}]$")

    # Set the axis limits
    if xlim is not None:
        ax.set_xlim(xlim, None)
    if ylim is not None:
        ax.set_ylim(ylim, None)

    fig.savefig(
        outpath + f"UV_size_lum_hex_{spec_type}.png",
        bbox_inches="tight",
        dpi=300,
    )


def plot_size_flux_comp(filepath, filter, outpath):
    """
    Plot the size-luminosity relation.

    Args:
        filepath (str): The path to the file to plot.
    """
    # Open the file and extract the sizes and luminosities
    with h5py.File(filepath, "r") as hdf:
        # Get the redshift (it's the same for all galaxies)
        redshift = hdf["Galaxies/Redshift"][0]
        obs_sizes = hdf["Galaxies/Stars/PixelHalfLightRadii/stellar_total/JWST"][
            filter
        ][...]
        int_sizes = hdf["Galaxies/Stars/PixelHalfLightRadii/reprocessed/JWST"][filter][
            ...
        ]
        obs_flux = hdf["Galaxies/Stars/Photometry/Fluxes/stellar_total/JWST/"][filter][
            ...
        ]
        int_flux = hdf["Galaxies/Stars/Photometry/Fluxes/reprocessed/JWST/"][filter][
            ...
        ]

    # Create the plot
    fig = plt.figure(figsize=(3.5, 3.5))
    ax = fig.add_subplot(111)

    # Add a grid and make sure its always at the back
    ax.grid(True)
    ax.set_axisbelow(True)

    # Remove galaxies with no flux
    mask = np.logical_and(int_flux > 0, int_sizes > 0)
    int_sizes = int_sizes[mask]
    obs_sizes = obs_sizes[mask]
    int_flux = int_flux[mask]
    obs_flux = obs_flux[mask]

    # Fit the size-luminosity relations
    obs_popt, obs_pcov = curve_fit(
        size_lumin_fit,
        obs_flux,
        obs_sizes,
        p0=[1, 0.5],
    )
    int_popt, int_pcov = curve_fit(
        size_lumin_fit,
        int_flux,
        int_sizes,
        p0=[1, 0.5],
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

    # Set the axis labels
    ax.set_xlabel(r"$L_{1500} / [\mathrm{nJy}]$")
    ax.set_ylabel(r"$R_{1/2} / [\mathrm{kpc}]$")

    fig.savefig(
        outpath + f"Obs_{filter.split('.')[-1]}_size_lum_comp.png",
        bbox_inches="tight",
        dpi=300,
    )


def plot_size_lumin_relation_uv_fit(
    filepath,
    fig=None,
    ax=None,
    xlim=None,
    ylim=None,
):
    pass


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Plot galaxy images in different filters."
    )
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
    parser.add_argument(
        "--part-limit",
        type=int,
        help="The lower mass limit for galaxies.",
        default=100,
    )
    parser.add_argument(
        "--fof-only",
        action="store_true",
        help="If true, only load the FOF groups.",
    )
    parser.add_argument(
        "--grid",
        type=str,
        help="The path to the grid.",
        default="test_gird.hdf5",
    )
    parser.add_argument(
        "--filter",
        type=str,
        default=None,
        help="The filter to use for the plot.",
    )
    parser.add_argument(
        "--filtpath",
        type=str,
        default="../data",
        help="The path to the filter data.",
    )
    parser.add_argument(
        "--xlim",
        type=float,
        default=None,
        help="The x-axis lower limit.",
    )
    parser.add_argument(
        "--ylim",
        type=float,
        default=None,
        help="The y-axis lower limit.",
    )
    parser.add_argument(
        "--spec-type",
        type=str,
        default="stellar_total",
        help="The type of spectrum to use.",
    )

    args = parser.parse_args()

    # Define input and output paths
    run_name = args.run_name
    variant = args.variant
    part_limit = args.part_limit
    fof_only = args.fof_only
    grid_name = args.grid
    grid_name_no_ext = grid_name.split("/")[-1].split(".")[0]
    snap = str(args.snap).zfill(4)
    path = f"../data/{run_name}/{variant}/Synthesized_imgs_{args.snap:04d}.hdf5"
    outpath = f"../plots/{run_name}/{variant}/images/"

    # Define the output path, for special particle limits we all include that
    # info
    path = f"../data/{run_name}/{variant}/Synthesized_imgs_{snap}_{grid_name_no_ext}"
    outpath = f"../plots/{run_name}/{variant}/"
    if part_limit != 100:
        path += f"_part_limit_{part_limit}"
        outpath += f"/part_limit_{part_limit}"
    if fof_only:
        path += "_FOFGroups"
        outpath += "/FOFGroups"
    path += ".hdf5"

    # Check if the input file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file {path} does not exist.")

    # Create the plot directory if it doesn't exist
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # Plot the size-luminosity relation
    # if args.filter is not None:
    #     plot_size_flux_hex(path, args.filter, outpath)
    #     plot_size_flux_comp(path, args.filter, outpath)
    # else:
    #     plot_size_lum_hex_uv(path, args.filtpath, outpath)
    # plot_size_lum_hex_uv(path, outpath, args.spec_type, xlim=args.xlim, ylim=args.ylim)
    plot_size_lum_hex_fit_multi(
        path,
        outpath,
        args.spec_type,
        xlim=args.xlim,
        ylim=args.ylim,
    )
