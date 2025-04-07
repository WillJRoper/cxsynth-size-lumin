"""Script for generating the instrument files for the COLIBRE analysis."""

import argparse
import os

import h5py
import webbpsf
from astropy.cosmology import Planck18 as cosmo
from synthesizer.instruments import FilterCollection
from synthesizer.instruments.instrument import Instrument
from unyt import Mpc, angstrom, arcsecond, kpc


def angular_to_physical(angular_res_arcsec, distance):
    """
    Convert angular resolution in arcseconds to physical resolution in kpc.

    This is to be used for fixed distance observations.

    Args:
        angular_res_arcsec (float): The angular resolution in arcseconds.
        distance (float): The distance to the object in kiloparsecs.

    Returns:
        physical_resolution_kpc (float): The physical resolution in kiloparsecs.
    """
    # Convert angular resolution from arcseconds to radians
    # unyt provides unit conversion:
    angular_res = angular_res_arcsec
    angular_res_rad = angular_res.to("radian")

    # Apply the small-angle formula:
    # physical resolution = distance * angular resolution (in radians)
    physical_resolution = distance * angular_res_rad.value

    # Convert the result to kiloparsecs:
    physical_resolution_kpc = physical_resolution.to("kpc")

    return physical_resolution_kpc


def make_filters(filt_path):
    """
    Generate the filter collections for the COLIBRE analysis.

    These are always the same regardless of the snapshot, so we can
    just generate them once and save them to a file.

    Args:
        filt_path (str): The path to save the filters.
    """
    # Define the filters
    nircam_fs = FilterCollection(
        filter_codes=[
            "JWST/NIRCam.F090W",
            "JWST/NIRCam.F115W",
            "JWST/NIRCam.F150W",
            "JWST/NIRCam.F200W",
            "JWST/NIRCam.F277W",
            "JWST/NIRCam.F356W",
            "JWST/NIRCam.F444W",
        ]
    )
    nircam_fs.write_filters(path=filt_path + "/nircam_filters.hdf5")

    miri_fs = FilterCollection(
        filter_codes=[
            "JWST/MIRI.F560W",
            "JWST/MIRI.F770W",
            "JWST/MIRI.F1000W",
            # "JWST/MIRI.F1130W",
            # "JWST/MIRI.F1280W",
            # "JWST/MIRI.F1500W",
            # "JWST/MIRI.F1800W",
        ]
    )
    miri_fs.write_filters(path=filt_path + "/miri_filters.hdf5")


def make_psfs(filt_path):
    """
    Generate the PSFs for the COLIBRE analysis.

    These are always the same regardless of the snapshot, so we can
    just generate them once and return them. (In truth, they would differ
    based on location in the FOV and all sorts else... but for simplicities
    sake we ignore there second order effects.)

    Args:
        filt_path (str): The path to save the filters.

    Returns:
        nircam_psfs (dict): The PSFs for the NIRCam filters.
        miri_psfs (dict): The PSFs for the MIRI filters.
    """
    # Do the PSFs already exist?
    if os.path.exists(filt_path + "/instrument_psfs.hdf5"):
        # Read them into dictionaries
        nircam_psfs = {}
        miri_psfs = {}
        with h5py.File(filt_path + "/instrument_psfs.hdf5", "r") as hf:
            for key in hf["NIRCam/JWST"].keys():
                nircam_psfs[f"JWST/{key}"] = hf["NIRCam/JWST"][key][...]
            for key in hf["MIRI/JWST"].keys():
                miri_psfs[f"JWST/{key}"] = hf["MIRI/JWST"][key][...]

        return nircam_psfs, miri_psfs

    nircam_fs = FilterCollection(path=filt_path + "/nircam_filters.hdf5")
    miri_fs = FilterCollection(path=filt_path + "/miri_filters.hdf5")

    # Set up the PSF dictionaries and webbpsf objects
    nircam_psfs = {}
    miri_psfs = {}
    nc = webbpsf.NIRCam()
    miri = webbpsf.MIRI()

    # Get nircam PSFs
    for nc_filt in nircam_fs.filter_codes:
        nc.filter = nc_filt.split(".")[-1]
        psf = nc.calc_psf(oversample=2)
        nircam_psfs[nc_filt] = psf[0].data

    # Get miri psfs
    for miri_filt in miri_fs.filter_codes:
        miri.filter = miri_filt.split(".")[-1]
        psf = miri.calc_psf(oversample=2)
        miri_psfs[miri_filt] = psf[0].data

    # Write the PSFs to files
    with h5py.File(filt_path + "/instrument_psfs.hdf5", "w") as hf:
        # NIRCam PSFs
        group = hf.create_group("NIRCam")
        for key, value in nircam_psfs.items():
            group.create_dataset(key, data=value)

        # MIRI PSFs
        group = hf.create_group("MIRI")
        for key, value in miri_psfs.items():
            group.create_dataset(key, data=value)

    return nircam_psfs, miri_psfs


def make_instruments(inst_path, filt_path, z, nircam_psfs, miri_psfs):
    """
    Generate the instrument files for the COLIBRE analysis.

    Args:
        inst_path (str): The path to save the instruments.
        z (float): The redshift of the galaxies.
    """
    # Define the filters
    nircam_fs = FilterCollection(path=filt_path + "/nircam_filters.hdf5")
    miri_fs = FilterCollection(path=filt_path + "/miri_filters.hdf5")
    top_hat = FilterCollection(
        tophat_dict={
            "UV1500": {"lam_eff": 1500 * angstrom, "lam_fwhm": 300 * angstrom},
        },
    )

    # Define the angular resoltions
    nircam_res = 0.031 * arcsecond
    miri_res = 0.111 * arcsecond

    # Convert the angular resolutions to physical kpc.
    # NOTE: When the luminosity distance is less than 10 Mpc we just take the
    # resolution at 10 Mpc.
    arcsec_to_kpc = (
        cosmo.kpc_proper_per_arcmin(z).to("kpc/arcsec").value * kpc / arcsecond
    )

    # Get the luminosity distance
    d_lum = cosmo.luminosity_distance(z).to("Mpc").value
    if d_lum >= 10:
        nircam_res_spatial = arcsec_to_kpc * nircam_res
        miri_res_spatial = arcsec_to_kpc * miri_res
    else:
        nircam_res_spatial = angular_to_physical(
            angular_res_arcsec=nircam_res,
            distance=10 * Mpc,
        )
        miri_res_spatial = angular_to_physical(
            angular_res_arcsec=miri_res,
            distance=10 * Mpc,
        )

    # Set up the instruments
    nircam = Instrument(
        "JWST.NIRCam",
        filters=nircam_fs,
        psfs=nircam_psfs,
        resolution=nircam_res_spatial,
    )
    miri = Instrument(
        "JWST.MIRI",
        filters=miri_fs,
        psfs=miri_psfs,
        resolution=miri_res_spatial,
    )
    uv = Instrument(
        "UV1500",
        filters=top_hat,
        resolution=2.66 / (1 + z) * kpc,
    )

    # Combine them
    instruments = nircam + miri + uv

    # Save the instruments
    instruments.write_instruments(inst_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Which simulation?
    parser.add_argument(
        "--run-dir",
        type=str,
        help="The directory of the simulation.",
        default="/cosma8/data/dp004/colibre/Runs/",
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
        "--aperture",
        type=int,
        help="The aperture to use for the observations.",
        default=100,
    )
    args = parser.parse_args()

    run_folder = args.run_dir
    run_name = args.run_name
    variant = args.variant
    aperture = args.aperture

    # Define the whole path to the data
    path = f"{run_folder}/{run_name}/{variant}/"

    # Define the path to the filter data
    # (this is the same for all snapshots)
    filt_path = "../data"

    # Make the filters if they don't exist
    if not os.path.exists("../data/nircam_filters.hdf5") or not os.path.exists(
        "../data/miri_filters.hdf5"
    ):
        make_filters(filt_path)

    # Create the directory if it doesn't exist
    if not os.path.exists(f"../data/{run_name}/{variant}"):
        os.makedirs(f"../data/{run_name}/{variant}")

    # Make the PSFs
    nircam_psfs, miri_psfs = make_psfs(filt_path)

    # Loop over possible snapshots
    for snap_ind in range(0, 128):
        snap = str(snap_ind).zfill(4)

        # Read in the redshift and while we do it make sure we actually have
        # SOAP data for this snap
        try:
            with h5py.File(f"{path}/SOAP/halo_properties_{snap}.hdf5") as hf:
                redshift = hf["Cosmology"].attrs["Redshift"][0]
        except FileNotFoundError:
            print(f"No SOAP data for snapshot {snap}.")
            continue

        # Check that we have the aperture
        with h5py.File(f"{path}/SOAP/halo_properties_{snap}.hdf5") as hf:
            if f"{aperture}kpc" not in hf["ExclusiveSphere"]:
                print(f"No {aperture}kpc aperture for snapshot {snap}.")
                continue

        # Define the instrument path
        inst_path = f"../data/{run_name}/{variant}/instruments_{snap}.hdf5"

        print(f"Making instruments for snapshot {snap} at redshift {redshift}.")

        make_instruments(inst_path, filt_path, redshift, nircam_psfs, miri_psfs)
