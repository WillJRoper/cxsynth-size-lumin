"""A script for plotting galaxy images in different filters."""

import argparse
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
from synthesizer.imaging import Image, ImageCollection
from unyt import kpc


def plot_cutout_grid(path, outpath, run_name, variant, snap, gal_ind=None):
    """
    Plot the images of a galaxy in different filters.

    Args:
        path (str): The path to the HDF5 file.
        outpath (str): The path to save the output images.
        gal_ind (int, optional): The index of the galaxy to plot. If None,
            the brightest galaxy is used.
    """
    # Define the keys we'll need
    img_key = "Galaxies/Stars/PSFImages/Flux/stellar_total/JWST"
    phot_key = "Galaxies/Stars/Photometry/Fluxes/stellar_total/JWST/NIRCam.F200W"

    # Create a dict to hold each image
    img_dict = {}

    # Open the HDF5 file
    with h5py.File(path, "r") as hdf:
        # If the gal_ind is None we just want to find the brightest galaxy
        # and plot its image in each filter
        if gal_ind is None:
            gal_ind = np.argmax(hdf[phot_key][:])

        # Read each image for this galaxy into the dictionary
        for key in hdf[img_key].keys():
            img_dict[key] = Image(
                resolution=1 * kpc,
                fov=100 * kpc,
                img=hdf[img_key][key][gal_ind, :, :],
            )

        # Get the redshift of this galaxy
        redshift = hdf["Galaxies/Redshift"][gal_ind][0]

        # Get the stellar mass of this galaxy
        stellar_mass = hdf["Galaxies/Stars/StellarMass"][gal_ind][0]

    # Create the image collection
    img_coll = ImageCollection(
        resolution=1 * kpc,
        fov=100 * kpc,
        imgs=img_dict,
    )

    # Compute the 99.9 percentile of each image and take the maximum as
    # the normalization factor
    vmax = np.max(
        [np.percentile(img_coll.imgs[key].arr, 99.9) for key in img_dict.keys()],
    )
    vmin = 0.0

    print(f"Max value: {vmax}")
    print(f"Min value: {vmin}")
    print(f"Redshift: {redshift}/Snap: {snap}")
    print(
        f"Making Galaxy {gal_ind} from"
        f" {run_name}/{variant} at z={redshift:.2f} (snap {snap})",
    )

    # Plot the images
    fig, ax = img_coll.plot_images(
        show=False,
        vmin=vmin,
        vmax=vmax,
        filters=(
            "NIRCam.F090W",
            "NIRCam.F115W",
            "NIRCam.F150W",
            "NIRCam.F200W",
            "NIRCam.F277W",
            "NIRCam.F356W",
            "NIRCam.F444W",
            "MIRI.F560W",
            "MIRI.F770W",
            "MIRI.F1000W",
        ),
        ncols=5,
    )

    # Include the redshift in the title
    fig.suptitle(
        f"Galaxy {gal_ind} from {run_name}/{variant} at z={redshift:.2f} "
        f"(snap {snap}) with stellar mass {stellar_mass:.2e} Msun",
        fontsize=16,
    )

    fig.savefig(
        outpath + f"cutout_{gal_ind}.png",
        dpi=300,
        bbox_inches="tight",
    )

    # Close the figure
    plt.close(fig)


def plot_rgb_image(path, outpath, run_name, variant, snap, gal_ind=None):
    """
    Plot the images of a galaxy in different filters.

    Args:
        path (str): The path to the HDF5 file.
        outpath (str): The path to save the output images.
        gal_ind (int, optional): The index of the galaxy to plot. If None,
            the brightest galaxy is used.
    """
    # Define the keys we'll need
    img_key = "Galaxies/Stars/PSFImages/Flux/stellar_total/JWST"
    phot_key = "Galaxies/Stars/Photometry/Fluxes/stellar_total/JWST/NIRCam.F200W"

    # Create a dict to hold each image
    img_dict = {}

    # Open the HDF5 file
    with h5py.File(path, "r") as hdf:
        # If the gal_ind is None we just want to find the brightest galaxy
        # and plot its image in each filter
        if gal_ind is None:
            gal_ind = np.argmax(hdf[phot_key][:])

        # Read each image for this galaxy into the dictionary
        for key in hdf[img_key].keys():
            img_dict[key] = Image(
                resolution=1 * kpc,
                fov=100 * kpc,
                img=hdf[img_key][key][gal_ind, :, :],
            )

        # Get the redshift of this galaxy
        redshift = hdf["Galaxies/Redshift"][gal_ind][0]

        # Get the stellar mass of this galaxy
        stellar_mass = hdf["Galaxies/Stars/StellarMass"][gal_ind][0]

    # Create the image collection
    img_coll = ImageCollection(
        resolution=1 * kpc,
        fov=100 * kpc,
        imgs=img_dict,
    )

    # Compute the 99.9 percentile of each image and take the maximum as
    # the normalization factor
    vmax = np.max(
        [np.percentile(img_coll.imgs[key].arr, 99.9) for key in img_dict.keys()],
    )
    vmin = 0.0

    print(f"Max value: {vmax}")
    print(f"Min value: {vmin}")
    print(f"Redshift: {redshift}/Snap: {snap}")
    print(
        f"Making Galaxy {gal_ind} from"
        f" {run_name}/{variant} at z={redshift:.2f} (snap {snap})",
    )

    # Plot the images
    img_coll.make_rgb_image(
        rgb_filters={
            "R": ["JWST/NIRCam.F444W", "JWST/NIRCam.F356W"],
            "G": ["JWST/NIRCam.F277W", "JWST/NIRCam.F200W"],
            "B": ["JWST/NIRCam.F115W", "JWST/NIRCam.F150W"],
        }
    )
    fig, ax = img_coll.plot_rgb_image(
        show=False,
        vmin=vmin,
        vmax=vmax,
    )

    # Include the redshift in the title
    fig.suptitle(
        f"Galaxy {gal_ind} from {run_name}/{variant} at z={redshift:.2f} "
        f"(snap {snap}) with stellar mass {stellar_mass:.2e} Msun",
        fontsize=16,
    )

    fig.savefig(
        outpath + f"rgb_cutout_{gal_ind}.png",
        dpi=300,
        bbox_inches="tight",
    )

    # Close the figure
    plt.close(fig)


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
        "--gal-ind",
        type=int,
        help="The index of the galaxy to plot.",
        default=None,
    )

    args = parser.parse_args()

    # Define input and output paths
    run_name = args.run_name
    variant = args.variant
    path = f"../data/{run_name}/{variant}/Synthesized_imgs_{args.snap:04d}.hdf5"
    outpath = f"../plots/{run_name}/{variant}/images/"

    # Check if the input file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file {path} does not exist.")

    # Create the plot directory if it doesn't exist
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # Call the function to plot the images
    plot_cutout_grid(
        path,
        outpath,
        run_name,
        variant,
        args.snap,
        args.gal_ind,
    )
    plot_rgb_image(
        path,
        outpath,
        run_name,
        variant,
        args.snap,
        args.gal_ind,
    )
