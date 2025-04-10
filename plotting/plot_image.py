"""A script for plotting galaxy images in different filters."""

# import argparse
import glob

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
        redshift = hdf["Galaxies/Redshift"][gal_ind]

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
        f"(snap {snap})",
        fontsize=16,
    )

    fig.savefig(
        outpath + f"cutout_{gal_ind}.png",
        dpi=1000,
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
            gal_inds = np.arange(hdf[phot_key][:].size)
        else:
            gal_inds = [
                gal_ind,
            ]

    # Loop over the galaxies
    for gal_ind in gal_inds:
        # Open the HDF5 file
        with h5py.File(path, "r") as hdf:
            # Read each image for this galaxy into the dictionary
            for key in hdf[img_key].keys():
                img_dict[key] = Image(
                    resolution=1 * kpc,
                    fov=100 * kpc,
                    img=hdf[img_key][key][gal_ind, :, :],
                )

            # Get the redshift of this galaxy
            redshift = hdf["Galaxies/Redshift"][gal_ind]

        # Create the image collection
        img_coll = ImageCollection(
            resolution=1 * kpc,
            fov=100 * kpc,
            imgs=img_dict,
        )

        # Correct the shape
        img_coll.npix = (
            img_coll.imgs["NIRCam.F200W"].arr.shape[0],
            img_coll.imgs["NIRCam.F200W"].arr.shape[1],
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
                "R": ["NIRCam.F444W", "NIRCam.F356W"],
                "G": ["NIRCam.F277W", "NIRCam.F200W"],
                "B": ["NIRCam.F115W", "NIRCam.F150W"],
            }
        )
        fig, ax, _ = img_coll.plot_rgb_image(
            show=False,
            # vmax=np.percentile(img_coll.rgb_img, 99.99),
            # vmin=np.percentile(img_coll.rgb_img, 16),
        )

        # Include the redshift in the title
        fig.suptitle(
            f"Galaxy {gal_ind} from {run_name}/{variant} at z={redshift:.2f} "
            f"(snap {snap})",
            fontsize=5,
        )

        fig.savefig(
            outpath + f"rgb_cutout_{gal_ind}.png",
            dpi=1000,
            bbox_inches="tight",
        )

        # Close the figure
        plt.close(fig)


def mega_rgb_image(res=1080):
    """Create a mega RGB image from the images in all HDF5 files."""
    # We need to extract all the HDF5 files in the directory
    # and then loop over them
    files = glob.glob("../data/*/*/Synthesized_imgs_*")

    # Create the 4K RGB images for each band
    f115W = np.zeros((res, res), dtype=np.float32)
    f150W = np.zeros((res, res), dtype=np.float32)
    f200W = np.zeros((res, res), dtype=np.float32)
    f277W = np.zeros((res, res), dtype=np.float32)
    f356W = np.zeros((res, res), dtype=np.float32)
    f444W = np.zeros((res, res), dtype=np.float32)
    imgs = {
        "JWST/NIRCam.F115W": f115W,
        "JWST/NIRCam.F150W": f150W,
        "JWST/NIRCam.F200W": f200W,
        "JWST/NIRCam.F277W": f277W,
        "JWST/NIRCam.F356W": f356W,
        "JWST/NIRCam.F444W": f444W,
    }

    # Loop over the files
    for ifile, f in enumerate(files):
        # Open the file and extract the images (if they are there)
        with h5py.File(f, "r") as hdf:
            # Check if we have the images
            if "PSFImages" not in hdf["Galaxies/Stars"].keys():
                print(f"No images in {f}")
                continue

            # Get the shape of the images
            shape = hdf[
                "Galaxies/Stars/PSFImages/Flux/stellar_total/JWST/NIRCam.F115W"
            ].shape[1:]
            ngals = hdf[
                "Galaxies/Stars/PSFImages/Flux/stellar_total/JWST/NIRCam.F115W"
            ].shape[0]

            print(f"{f} has shape {shape}")

            # Loop over the galaxies placing each randomly in the image
            for i in range(ngals):
                print(f"Placing galaxy {i+1}/{ngals} for file {ifile+1}/{len(files)}")
                # Get a random position in the image
                x = np.random.randint(0, res - shape[0])
                y = np.random.randint(0, res - shape[1])

                # Ready the images
                for filt in [
                    "JWST/NIRCam.F115W",
                    "JWST/NIRCam.F150W",
                    "JWST/NIRCam.F200W",
                    "JWST/NIRCam.F277W",
                    "JWST/NIRCam.F356W",
                    "JWST/NIRCam.F444W",
                ]:
                    this_img = hdf[
                        "Galaxies/Stars/PSFImages/Flux/stellar_total/" + filt
                    ][i, :, :]

                    # Add the image to the mega image
                    imgs[filt][x : x + shape[0], y : y + shape[1]] += this_img

    # Write the images we've made to a HDF5 file
    with h5py.File("mega_rgb_images.hdf5", "w") as hdf:
        # Create the groups
        group = hdf.create_group("JWST")
        for key in imgs.keys():
            group.create_dataset(key, data=imgs[key])


if __name__ == "__main__":
    # # Parse command line arguments
    # parser = argparse.ArgumentParser(
    #     description="Plot galaxy images in different filters."
    # )
    # parser.add_argument(
    #     "--run-name",
    #     type=str,
    #     help="The name of the simulation (the directory in run-dir).",
    #     default="L025_m7",
    # )
    # parser.add_argument(
    #     "--variant",
    #     type=str,
    #     help="The variant of the simulation (e.g. THERMAL_AGN_m6/HYBRID_AGN_m7).",
    #     default="THERMAL_AGN_m7",
    # )
    # parser.add_argument(
    #     "--snap",
    #     type=int,
    #     help="The snapshot number to plot.",
    #     default=0,
    # )
    # parser.add_argument(
    #     "--gal-ind",
    #     type=int,
    #     help="The index of the galaxy to plot.",
    #     default=None,
    # )
    # parser.add_argument(
    #     "--part-limit",
    #     type=int,
    #     help="The lower mass limit for galaxies.",
    #     default=100,
    # )
    # parser.add_argument(
    #     "--fof-only",
    #     action="store_true",
    #     help="If true, only load the FOF groups.",
    # )
    # parser.add_argument(
    #     "--grid",
    #     type=str,
    #     help="The path to the grid.",
    # )
    #
    # args = parser.parse_args()
    #
    # # Define input and output paths
    # run_name = args.run_name
    # variant = args.variant
    # part_limit = args.part_limit
    # fof_only = args.fof_only
    # grid_name = args.grid
    # grid_name_no_ext = grid_name.split("/")[-1].split(".")[0]
    # snap = str(args.snap).zfill(4)
    # path = f"../data/{run_name}/{variant}/Synthesized_imgs_{args.snap:04d}.hdf5"
    # outpath = f"../plots/{run_name}/{variant}/images/"
    #
    # # Define the output path, for special particle limits we all include that
    # # info
    # path = f"../data/{run_name}/{variant}/Synthesized_imgs_{snap}_{grid_name_no_ext}"
    # outpath = f"../plots/{run_name}/{variant}/"
    # if part_limit != 100:
    #     path += f"_part_limit_{part_limit}"
    #     outpath += f"/part_limit_{part_limit}"
    # if fof_only:
    #     path += "_FOFGroups"
    #     outpath += "/FOFGroups"
    # path += ".hdf5"
    # outpath += "/images/"
    #
    # # Check if the input file exists
    # if not os.path.exists(path):
    #     raise FileNotFoundError(f"Input file {path} does not exist.")
    #
    # # Create the plot directory if it doesn't exist
    # if not os.path.exists(outpath):
    #     os.makedirs(outpath)
    #
    # # Call the function to plot the images
    # plot_cutout_grid(
    #     path,
    #     outpath,
    #     run_name,
    #     variant,
    #     args.snap,
    #     args.gal_ind,
    # )
    # plot_rgb_image(
    #     path,
    #     outpath,
    #     run_name,
    #     variant,
    #     args.snap,
    #     args.gal_ind,
    # )
    mega_rgb_image()
