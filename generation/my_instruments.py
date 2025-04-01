"""Script for generating the instrument files for the COLIBRE analysis."""

import webbpsf
from astropy.cosmology import Planck15 as cosmo
from synthesizer.instruments import FilterCollection
from synthesizer.instruments.instrument import Instrument
from unyt import angstrom, arcsecond, kpc

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

miri_fs = FilterCollection(
    filter_codes=[
        "JWST/MIRI.F560W",
        "JWST/MIRI.F770W",
        "JWST/MIRI.F1000W",
        "JWST/MIRI.F1130W",
        "JWST/MIRI.F1280W",
        "JWST/MIRI.F1500W",
        "JWST/MIRI.F1800W",
    ]
)
top_hat = FilterCollection(
    tophat_dict={
        "UV1500": {"lam_eff": 1500 * angstrom, "lam_fwhm": 300 * angstrom},
    },
)


def make_instruments(inst_path, z):
    """
    Generate the instrument files for the COLIBRE analysis.

    Args:
        inst_path (str): The path to save the instruments.
        z (float): The redshift of the galaxies.
    """
    # Set up the PSF dictionaries and webbpsf objects
    nircam_psfs = {}
    miri_psfs = {}
    nc = webbpsf.NIRCam()
    miri = webbpsf.MIRI()

    # Get nircam PSFs
    for nc_filt in nircam_fs.filter_codes:
        nc.filter = nc_filt.split(".")[-1]
        psf = nc.calc_psf(oversample=2)
        nircam_psfs[nc_filt.replace("/", "_")] = psf[0].data

    # Get miri psfs
    for miri_filt in miri_fs.filter_codes:
        miri.filter = miri_filt.split(".")[-1]
        psf = miri.calc_psf(oversample=2)
        miri_psfs[miri_filt.replace("/", "_")] = psf[0].data

    # Define the angular resoltions
    nircam_res = 0.031 * arcsecond
    miri_res = 0.111 * arcsecond

    # Convert the angular resolutions to physical kpc
    arcsec_to_kpc = (
        cosmo.kpc_proper_per_arcmin(z).to("kpc/arcsec").value * kpc / arcsecond
    )

    # Set up the instruments
    nircam = Instrument(
        "JWST.NIRCam",
        filters=nircam_fs,
        psfs=nircam_psfs,
        resolution=arcsec_to_kpc * nircam_res,
    )
    miri = Instrument(
        "JWST.MIRI",
        filters=miri_fs,
        psfs=miri_psfs,
        resolution=arcsec_to_kpc * miri_res,
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
