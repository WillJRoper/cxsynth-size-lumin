"""An module containing extra analysis functions for the pipeline."""

import numpy as np
from scipy.interpolate import interp1d
from synthesizer.conversions import angular_to_spatial_at_z
from unyt import Msun, angstrom, arcsecond, unyt_array, yr

# Use galaxtic Msun
Msun = Msun.in_base("galactic")


def get_stars_1d_velocity_dispersion(gal):
    """Get the 1D velocity dispersion of the stars in the galaxy."""
    return (
        np.array(
            [
                np.std(gal.stars._velocities[:, 0], ddof=0),
                np.std(gal.stars._velocities[:, 1], ddof=0),
                np.std(gal.stars._velocities[:, 2], ddof=0),
            ]
        )
        * gal.stars.velocities.units
    )


def get_gas_1d_velocity_dispersion(gal):
    """Get the 1D velocity dispersion of the gas in the galaxy."""
    return (
        np.array(
            [
                np.std(gal.gas._velocities[:, 0], ddof=0),
                np.std(gal.gas._velocities[:, 1], ddof=0),
                np.std(gal.gas._velocities[:, 2], ddof=0),
            ]
        )
        * gal.gas.velocities.units
    )


def get_stars_3d_velocity_dispersion(gal):
    """Get the 3D velocity dispersion of the stars in the galaxy."""
    return (
        np.std(np.sqrt(np.sum(gal.stars._velocities**2, axis=1)), ddof=0)
        * gal.stars.velocities.units
    )


def get_gas_3d_velocity_dispersion(gal):
    """Get the 3D velocity dispersion of the gas in the galaxy."""
    return (
        np.std(np.sqrt(np.sum(gal.gas._velocities**2, axis=1)), ddof=0)
        * gal.gas.velocities.units
    )


def _pixel_hlr(img, pix_area):
    """
    Calculate the half-light radius of an image using the pixel technique.

    Args:
        img (Image): The image to calculate the half-light radius for.
        pix_area (float): The area of a pixel in the image.

    Returns:
        float: The half-light radius in kpc.
    """
    # Sort pixel values from brightest to faintest
    pixels = np.sort(img.arr.flatten())[::-1]

    # Calculate the cumulative sum of the pixels
    cumsum = np.cumsum(pixels)
    half_light = cumsum[-1] / 2

    # Create an array of pixel areas
    pixel_areas = np.arange(len(pixels)) * pix_area

    # Find the pixel that corresponds to the half-light radius
    hlr_ind = np.argmin(np.abs(cumsum - half_light))

    # Interpolate around this pixel to get a more accurate value
    low_ind = max(0, hlr_ind - 1)
    high_ind = min(len(pixels) - 1, hlr_ind + 1)
    if low_ind == 0:
        y = np.array(
            [pixel_areas[hlr_ind], pixel_areas[high_ind], pixel_areas[high_ind + 1]]
        )
        x = np.array([cumsum[hlr_ind], cumsum[high_ind], cumsum[high_ind + 1]])
    elif high_ind == len(pixels) - 1:
        y = np.array(
            [pixel_areas[low_ind], pixel_areas[hlr_ind], pixel_areas[low_ind - 1]]
        )
        x = np.array([cumsum[low_ind], cumsum[hlr_ind], cumsum[low_ind - 1]])
    else:
        y = np.array(
            [pixel_areas[low_ind], pixel_areas[hlr_ind], pixel_areas[high_ind]]
        )
        x = np.array([cumsum[low_ind], cumsum[hlr_ind], cumsum[high_ind]])

    # Interpolate to find the half-light radius
    interp_func = interp1d(x, y, kind="linear", fill_value="extrapolate")

    # Get the area of pixels containing half the light
    hlr_area = interp_func(half_light)

    # Get the half-light radius
    hlr = np.sqrt(hlr_area / np.pi)

    return hlr


def get_pixel_based_hlr(obj):
    """
    Get the half-light radius of the galaxy using the pixel technique.

    Args:
        obj (Galaxy/Stars): The galaxy or component to get the half-light radius for.

    Returns:
        dict: A dictionary containing the half-light radius for each instrument.
    """
    results = {}

    # Loop over PSFed images
    for inst_name, d in obj.images_psf_fnu.items():
        # Loop over spectrum types
        for spec_type, imgs in d.items():
            # Loop over filters
            for filt, img in imgs.items():
                # Get the image
                img = obj.images_psf_fnu[inst_name][spec_type][filt]
                pix_area = img._resolution * img._resolution

                # Get the half-light radius
                hlr = _pixel_hlr(img, pix_area)

                # Store the results
                results.setdefault(f"Flux/{spec_type}", {})[filt] = (
                    hlr * img.resolution.units
                ).to("kpc")

    # Also do the Rest frame luminosity
    for inst_name, d in obj.images_lnu.items():
        # Loop over spectrum types
        for spec_type, imgs in d.items():
            # Loop over filters
            for filt, img in imgs.items():
                # Get the image
                img = obj.images_lnu[inst_name][spec_type][filt]
                pix_area = img._resolution * img._resolution

                # Get the half-light radius
                hlr = _pixel_hlr(img, pix_area)

                # Store the results
                results.setdefault(f"Luminosity/{spec_type}", {})[filt] = (
                    hlr * img.resolution.units
                ).to("kpc")

    return results


def get_colors_and_lrd_flags(gal, cosmo, nthreads):
    """
    Get the colors and LRD flags for the galaxy and it's components.

    The LRD status is defined by a set of colors and a compactification metric
    drived from a 0.2" and 0.4" aperture in the F444W filter. The colors are
    defined as:

        F115W - F150W < 0.8
        F200W - F277W > 0.7
        F200W - F356W > 1.0

        or

        F150W - F200W < 0.8
        F277W - F356W > 0.6
        F277W - F444W > 0.7

    Args:
        gal (Galaxy): The galaxy to get the colors and LRD flags for.
    """
    # Define a dictionary to store the results in (galaxy level results are
    # stored in the "Galaxy" key)
    results = {"Stars": {}, "BlackHoles": {}}

    # Compute spatial apertures
    ang_apertures = np.array([0.2, 0.4]) * arcsecond
    kpc_apertures = angular_to_spatial_at_z(ang_apertures, cosmo, gal.redshift)

    # Compute aperture photometry
    results["AperturePhotometry"] = {}
    results["Stars"]["AperturePhotometry"] = {}
    results["BlackHoles"]["AperturePhotometry"] = {}
    for inst, d in gal.images_psf_fnu.items():
        for spec_type, imgs in d.items():
            results["AperturePhotometry"][spec_type] = {}
            for filt, img in imgs.items():
                results["AperturePhotometry"][spec_type][filt] = {}
                for i, ang_ap in enumerate(ang_apertures):
                    results["AperturePhotometry"][spec_type][filt][
                        f"Aperture_{ang_apertures[i].value}".replace(".", "p")
                    ] = img.get_signal_in_aperture(
                        kpc_apertures[i].to("Mpc"),
                        nthreads=nthreads,
                    )
    for inst, d in gal.stars.images_psf_fnu.items():
        for spec_type, imgs in d.items():
            results["Stars"]["AperturePhotometry"][spec_type] = {}
            for filt, img in imgs.items():
                results["Stars"]["AperturePhotometry"][spec_type][filt] = {}
                for i, ang_ap in enumerate(ang_apertures):
                    results["Stars"]["AperturePhotometry"][spec_type][filt][
                        f"Aperture_{ang_apertures[i].value}".replace(".", "p")
                    ] = img.get_signal_in_aperture(
                        kpc_apertures[i].to("Mpc"),
                        nthreads=nthreads,
                    )
    for inst, d in gal.black_holes.images_psf_fnu.items():
        for spec_type, imgs in d.items():
            results["BlackHoles"]["AperturePhotometry"][spec_type] = {}
            for filt, img in imgs.items():
                results["BlackHoles"]["AperturePhotometry"][spec_type][filt] = {}
                for i, ang_ap in enumerate(ang_apertures):
                    results["BlackHoles"]["AperturePhotometry"][spec_type][filt][
                        f"Aperture_{ang_apertures[i].value}".replace(".", "p")
                    ] = img.get_signal_in_aperture(
                        kpc_apertures[i].to("Mpc"),
                        nthreads=nthreads,
                    )

    # Compute the compactness criterion
    results["Compactness"] = {}
    results["Stars"]["Compactness"] = {}
    results["BlackHoles"]["Compactness"] = {}
    for spec_type in results["AperturePhotometry"].keys():
        if spec_type == "Stars" or spec_type == "Blackholes":
            continue
        results["Compactness"][spec_type] = (
            results["AperturePhotometry"][spec_type]["JWST_NIRCam.F444W"][
                "Aperture_0p4"
            ]
            / results["AperturePhotometry"][spec_type]["JWST_NIRCam.F444W"][
                "Aperture_0p2"
            ]
        )
    for spec_type in results["Stars"]["AperturePhotometry"].keys():
        results["Stars"]["Compactness"][spec_type] = (
            results["Stars"]["AperturePhotometry"][spec_type]["JWST_NIRCam.F444W"][
                "Aperture_0p4"
            ]
            / results["Stars"]["AperturePhotometry"][spec_type]["JWST_NIRCam.F444W"][
                "Aperture_0p2"
            ]
        )
    for spec_type in results["BlackHoles"]["AperturePhotometry"].keys():
        results["BlackHoles"]["Compactness"][spec_type] = (
            results["BlackHoles"]["AperturePhotometry"][spec_type]["JWST_NIRCam.F444W"][
                "Aperture_0p4"
            ]
            / results["BlackHoles"]["AperturePhotometry"][spec_type][
                "JWST_NIRCam.F444W"
            ]["Aperture_0p2"]
        )

    # Do the galaxy level colors
    results["Colors"] = {spec_type: {} for spec_type in results["Compactness"].keys()}
    results["Stars"]["Colors"] = {
        spec_type: {} for spec_type in results["Stars"]["Compactness"].keys()
    }
    results["BlackHoles"]["Colors"] = {
        spec_type: {} for spec_type in results["BlackHoles"]["Compactness"].keys()
    }
    for spec_type, phot in gal.photo_fnu.items():
        results["Colors"][spec_type]["F115W_F150W"] = -2.5 * np.log10(
            phot["JWST_NIRCam.F115W"] / phot["JWST_NIRCam.F150W"]
        )
        results["Colors"][spec_type]["F150W_F200W"] = -2.5 * np.log10(
            phot["JWST_NIRCam.F150W"] / phot["JWST_NIRCam.F200W"]
        )
        results["Colors"][spec_type]["F200W_F277W"] = -2.5 * np.log10(
            phot["JWST_NIRCam.F200W"] / phot["JWST_NIRCam.F277W"]
        )
        results["Colors"][spec_type]["F200W_F356W"] = -2.5 * np.log10(
            phot["JWST_NIRCam.F200W"] / phot["JWST_NIRCam.F356W"]
        )
        results["Colors"][spec_type]["F277W_F356W"] = -2.5 * np.log10(
            phot["JWST_NIRCam.F277W"] / phot["JWST_NIRCam.F356W"]
        )
        results["Colors"][spec_type]["F277W_F444W"] = -2.5 * np.log10(
            phot["JWST_NIRCam.F277W"] / phot["JWST_NIRCam.F444W"]
        )
    for spec_type, phot in gal.stars.photo_fnu.items():
        results["Stars"]["Colors"][spec_type]["F115W_F150W"] = -2.5 * np.log10(
            phot["JWST_NIRCam.F115W"] / phot["JWST_NIRCam.F150W"]
        )
        results["Stars"]["Colors"][spec_type]["F150W_F200W"] = -2.5 * np.log10(
            phot["JWST_NIRCam.F150W"] / phot["JWST_NIRCam.F200W"]
        )
        results["Stars"]["Colors"][spec_type]["F200W_F277W"] = -2.5 * np.log10(
            phot["JWST_NIRCam.F200W"] / phot["JWST_NIRCam.F277W"]
        )
        results["Stars"]["Colors"][spec_type]["F200W_F356W"] = -2.5 * np.log10(
            phot["JWST_NIRCam.F200W"] / phot["JWST_NIRCam.F356W"]
        )
        results["Stars"]["Colors"][spec_type]["F277W_F356W"] = -2.5 * np.log10(
            phot["JWST_NIRCam.F277W"] / phot["JWST_NIRCam.F356W"]
        )
        results["Stars"]["Colors"][spec_type]["F277W_F444W"] = -2.5 * np.log10(
            phot["JWST_NIRCam.F277W"] / phot["JWST_NIRCam.F444W"]
        )
    for spec_type, phot in gal.black_holes.photo_fnu.items():
        results["BlackHoles"]["Colors"][spec_type]["F115W_F150W"] = -2.5 * np.log10(
            phot["JWST_NIRCam.F115W"] / phot["JWST_NIRCam.F150W"]
        )
        results["BlackHoles"]["Colors"][spec_type]["F150W_F200W"] = -2.5 * np.log10(
            phot["JWST_NIRCam.F150W"] / phot["JWST_NIRCam.F200W"]
        )
        results["BlackHoles"]["Colors"][spec_type]["F200W_F277W"] = -2.5 * np.log10(
            phot["JWST_NIRCam.F200W"] / phot["JWST_NIRCam.F277W"]
        )
        results["BlackHoles"]["Colors"][spec_type]["F200W_F356W"] = -2.5 * np.log10(
            phot["JWST_NIRCam.F200W"] / phot["JWST_NIRCam.F356W"]
        )
        results["BlackHoles"]["Colors"][spec_type]["F277W_F356W"] = -2.5 * np.log10(
            phot["JWST_NIRCam.F277W"] / phot["JWST_NIRCam.F356W"]
        )
        results["BlackHoles"]["Colors"][spec_type]["F277W_F444W"] = -2.5 * np.log10(
            phot["JWST_NIRCam.F277W"] / phot["JWST_NIRCam.F444W"]
        )

    # Define the LRD flags
    results["LRDFlag"] = {}
    results["Stars"]["LRDFlag"] = {}
    results["BlackHoles"]["LRDFlag"] = {}
    results["CompactnessFlag"] = {}
    results["Stars"]["CompactnessFlag"] = {}
    results["BlackHoles"]["CompactnessFlag"] = {}
    results["ColorFlag"] = {}
    results["Stars"]["ColorFlag"] = {}
    results["BlackHoles"]["ColorFlag"] = {}
    for spec_type in results["Compactness"].keys():
        comp_mask = results["Compactness"][spec_type] < 1.7
        mask1 = np.logical_and(
            results["Colors"][spec_type]["F115W_F150W"] < 0.8,
            results["Colors"][spec_type]["F200W_F277W"] > 0.7,
        )
        mask1 = np.logical_and(mask1, results["Colors"][spec_type]["F200W_F356W"] > 1.0)
        mask2 = np.logical_and(
            results["Colors"][spec_type]["F150W_F200W"] < 0.8,
            results["Colors"][spec_type]["F277W_F356W"] > 0.6,
        )
        mask2 = np.logical_and(mask2, results["Colors"][spec_type]["F277W_F444W"] > 0.7)
        results["CompactnessFlag"][spec_type] = comp_mask
        results["ColorFlag"][spec_type] = np.logical_or(mask1, mask2)
        results["LRDFlag"][spec_type] = np.logical_and(
            comp_mask, np.logical_or(mask1, mask2)
        )
    for spec_type in results["Stars"]["Compactness"].keys():
        comp_mask = results["Stars"]["Compactness"][spec_type] < 1.7
        mask1 = np.logical_and(
            results["Stars"]["Colors"][spec_type]["F115W_F150W"] < 0.8,
            results["Stars"]["Colors"][spec_type]["F200W_F277W"] > 0.7,
        )
        mask1 = np.logical_and(
            mask1, results["Stars"]["Colors"][spec_type]["F200W_F356W"] > 1.0
        )
        mask2 = np.logical_and(
            results["Stars"]["Colors"][spec_type]["F150W_F200W"] < 0.8,
            results["Stars"]["Colors"][spec_type]["F277W_F356W"] > 0.6,
        )
        mask2 = np.logical_and(
            mask2, results["Stars"]["Colors"][spec_type]["F277W_F444W"] > 0.7
        )
        results["Stars"]["CompactnessFlag"][spec_type] = comp_mask
        results["Stars"]["ColorFlag"][spec_type] = np.logical_or(mask1, mask2)
        results["Stars"]["LRDFlag"][spec_type] = np.logical_and(
            comp_mask, np.logical_or(mask1, mask2)
        )
    for spec_type in results["BlackHoles"]["Compactness"].keys():
        comp_mask = results["BlackHoles"]["Compactness"][spec_type] < 1.7
        mask1 = np.logical_and(
            results["BlackHoles"]["Colors"][spec_type]["F115W_F150W"] < 0.8,
            results["BlackHoles"]["Colors"][spec_type]["F200W_F277W"] > 0.7,
        )
        mask1 = np.logical_and(
            mask1, results["BlackHoles"]["Colors"][spec_type]["F200W_F356W"] > 1.0
        )
        mask2 = np.logical_and(
            results["BlackHoles"]["Colors"][spec_type]["F150W_F200W"] < 0.8,
            results["BlackHoles"]["Colors"][spec_type]["F277W_F356W"] > 0.6,
        )
        mask2 = np.logical_and(
            mask2, results["BlackHoles"]["Colors"][spec_type]["F277W_F444W"] > 0.7
        )
        results["BlackHoles"]["CompactnessFlag"][spec_type] = comp_mask
        results["BlackHoles"]["ColorFlag"][spec_type] = np.logical_or(mask1, mask2)
        results["BlackHoles"]["LRDFlag"][spec_type] = np.logical_and(
            comp_mask, np.logical_or(mask1, mask2)
        )

    return results


def get_bh_total_mass(gal):
    """
    Return the total mass of the black holes in the galaxy.

    Args:
        gal (Galaxy): The galaxy to get the black hole data for.
    """
    if gal.black_holes is None or gal.black_holes.nbh == 0:
        return 0.0 * Msun

    return np.sum(gal.black_holes._masses) * Msun


def get_bh_average_accretion_rate(gal):
    """
    Return the average accretion rate of the black holes in the galaxy.

    Args:
        gal (Galaxy): The galaxy to get the black hole data for.
    """
    if gal.black_holes is None or gal.black_holes.nbh == 0:
        return 0.0 * Msun / yr

    return np.mean(gal.black_holes._accretion_rate) * Msun / yr


def get_bh_central_mass(gal):
    """
    Return the mass of the central black hole in the galaxy.

    Args:
        gal (Galaxy): The galaxy to get the black hole data for.
    """
    if gal.black_holes is None or gal.black_holes.nbh == 0:
        return unyt_array(0.0, Msun)

    central_bh = np.argmax(gal.black_holes._masses)

    return unyt_array(gal.black_holes._masses[central_bh].item(), Msun)


def get_bh_central_accretion_rate(gal):
    """
    Return the accretion rate of the central black hole in the galaxy.

    Args:
        gal (Galaxy): The galaxy to get the black hole data for.
    """
    if gal.black_holes is None or gal.black_holes.nbh == 0:
        return unyt_array(0.0, Msun / yr)

    central_bh = np.argmax(gal.black_holes._masses)

    return unyt_array(gal.black_holes._accretion_rate[central_bh].item(), Msun / yr)


def get_bh_number(gal):
    """
    Return the number of black holes in the galaxy.

    Args:
        gal (Galaxy): The galaxy to get the black hole data for.
    """
    if gal.black_holes is None or gal.black_holes.nbh == 0:
        return 0

    return gal.black_holes.nbh


def get_UV_slopes(obj):
    """
    Get the UV slope of the galaxy.

    Args:
        obj (Galaxy/Stars/BlackHoles): The object to get the UV slope for.

    Returns:
        float: The UV slope.
    """
    # Dictionary to hold the slopes
    slopes = {}

    # Loop over the spectra
    for spec_type, d in obj.spectra.items():
        slopes[spec_type] = obj.spectra[spec_type].measure_beta(
            window=(1500 * angstrom, 3000 * angstrom)
        )

    return slopes


def get_IR_slopes(obj):
    """
    Get the IR slopes of the galaxy.

    Args:
        obj (Galaxy/Stars/BlackHoles): The object to get the IR slopes for.

    Returns:
        float: The IR slopes.
    """
    # Dictionary to hold the slopes
    slopes = {}

    # Loop over the spectra
    for spec_type, d in obj.spectra.items():
        slopes[spec_type] = obj.spectra[spec_type].measure_beta(
            window=(4400 * angstrom, 7500 * angstrom)
        )

    return slopes


def get_optical_depth(obj):
    """
    Return the average optical for the object.

    Args:
        obj (Galaxy/Stars/BlackHoles): The object to get the optical depth for.
    """
    # Check we have an optical depth
    if obj.tau_v is None:
        print("No optical depth data available.", type(obj))
        return 0.0

    # Return the average optical depth
    return np.mean(obj.tau_v)


def get_dust_to_metal_ratio(gal):
    """
    Return the dust-to-metal ratio for the object.

    Args:
        obj (Galaxy/Stars/BlackHoles): The object to get the dust-to-metal ratio for.
    """
    # Check we have a dust-to-metal ratio
    if gal.gas.dust_to_metals is None:
        return 0.3

    # Return the dust-to-metal ratio
    return np.mean(gal.gas.dust_to_metal_ratio)
