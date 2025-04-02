"""A set of helper functions for partitioning and loading COLIBRE data."""

import h5py
import numpy as np
import swiftsimio
from mpi4py import MPI as mpi
from swiftgalaxy import SOAP, SWIFTGalaxies
from synthesizer.particle import Galaxy, Gas, Stars
from unyt import Gyr, Myr, kpc, unyt_array


def partition_galaxies(location, snap, lower_mass_lim, aperture):
    """Partition the galaxies between the MPI processes."""
    # Get the number of processes and this rank
    nranks = mpi.COMM_WORLD.Get_size()
    this_rank = mpi.COMM_WORLD.Get_rank()

    # Read in the masses so we can make a cut based on the stellar mass
    cat = swiftsimio.load(f"{location}/SOAP/halo_properties_{snap}.hdf5")
    nstars = getattr(
        getattr(cat, f"exclusive_sphere_{int(aperture)}kpc"),
        "number_of_star_particles",
    ).to_value()

    # Create an array of galaxy indices
    gal_inds = np.arange(len(nstars))

    # Sanitise away galaxies below the threshold
    gal_inds = gal_inds[nstars >= 100]

    # Split the galaxies between the processes
    indices = np.array_split(gal_inds, nranks)

    return indices[this_rank]


def _set_up_swift_galaxy(
    location,
    snap,
    chunk_inds,
):
    # Read in some useful metadata
    with h5py.File(f"{location}/SOAP/halo_properties_{snap}.hdf5") as hf:
        aexp = hf["Cosmology"].attrs["Scale-factor"]
        redshift = hf["Cosmology"].attrs["Redshift"]

    soap = SOAP(
        f"{location}/SOAP/halo_properties_{snap}.hdf5",
        soap_index=chunk_inds,
        extra_mask="bound_only",
    )

    # By predefining these attributes we can speed up the loading of the data
    # later on
    preload = {
        "stars.birth_scale_factors",
        "stars.masses",
        "stars.initial_masses",
        "stars.metal_mass_fractions",
        "stars.coordinates",
        "gas.masses",
        "gas.metal_mass_fractions",
        "gas.dust_mass_fractions",
        "gas.coordinates",
        "gas.smoothing_lengths",
        "gas.species_fractions",
    }

    # Load the SWIFT galaxies object (this is the interface to the data)
    sgs = SWIFTGalaxies(
        f"{location}/SOAP/colibre_with_SOAP_membership_{snap}.hdf5",
        soap,
        preload=preload,
        auto_recentre=False,
    )

    return soap, sgs, aexp, redshift


def _get_galaxies(
    chunk_inds,
    location,
    snap,
    cosmo,
    aperture,
):
    """
    Get a galaxy from the master file.

    Args:
        chunk_inds (np.ndarray): The indices of the galaxies to load.
        location (str): The location of the data.
        snap (str): The snapshot to load.
        cosmo (astropy.cosmology): The cosmology to use.
        aperture (float): The aperture to use.

    Returns:
        np.ndarray: An array of galaxy objects.
    """
    # We'll need these later to get all the dust species
    dcols = [
        "GraphiteLarge",
        "MgSilicatesLarge",
        "FeSilicatesLarge",
        "GraphiteSmall",
        "MgSilicatesSmall",
        "FeSilicatesSmall",
    ]

    # How many galaxies are on this chunk?
    ngals = len(chunk_inds)

    # First up, get out I/O helpers and some metadata from SWIFTGalaxy
    soap, sgs, aexp, redshift = _set_up_swift_galaxy(
        location,
        snap,
        chunk_inds,
    )

    # Initialise a container for the galaxies we are about to load
    gals = np.empty(ngals, dtype=object)

    # Get centres in physical coordinates
    centre = soap.centre.to_physical().to("Mpc")
    centre = centre[sgs.iteration_order]

    # swiftgalaxy picks its own efficient iteration order
    for gal_ind, swift_gal in enumerate(sgs):
        # Get the centre
        cent = np.mean(
            swift_gal.stars.coordinates.to_physical().to("Mpc"),
            axis=0,
        )
        # Derive the radii for star and gas particles
        star_coords = swift_gal.stars.coordinates.to_physical().to("Mpc")
        star_radii = np.linalg.norm(cent - star_coords, axis=1).to("kpc")
        gas_coords = swift_gal.gas.coordinates.to_physical().to("Mpc")
        gas_radii = np.linalg.norm(cent - gas_coords, axis=1).to("kpc")

        # Define masks for the particles within the aperture
        star_mask = star_radii <= (aperture * kpc)
        gas_mask = gas_radii <= (aperture * kpc)

        print(
            "Centre from SOAP:",
            centre[gal_ind],
            "Geometric Mean Centre",
            cent,
        )

        # Derive the ages from the scale_factors
        scale_factors = swift_gal.stars.birth_scale_factors.to_value()
        scale_factors[scale_factors > aexp] = aexp
        scale_factors = (1.0 / scale_factors) - 1.0
        ages = (
            cosmo.age((1.0 / aexp) - 1.0).value - cosmo.age(scale_factors).value
        ) * Gyr
        ages = ages.to("Myr")

        # NOTE: Some ages can be broken by compression if they are close to
        # zero. We'll set these to 0.1 Myr
        ages[ages <= 0] = 0.1 * Myr

        # Compute the birth cloud optical depth for young stars from the
        # stellar metallicity. We approximate tau_v_bc = Z / 0.01
        star_metals = swift_gal.stars.metal_mass_fractions
        young_tau_v = star_metals.to_value() / 0.01

        # Derive the dust masses by summing the different dust species
        # fractions
        dust_mass_fracs = np.sum(
            np.vstack(
                [getattr(swift_gal.gas.dust_mass_fractions, kk) for kk in dcols]
            ).T,
            axis=1,
        )

        # Convert the dust fractions to masses
        gas_masses = swift_gal.gas.masses.to("Msun")
        dmasses = dust_mass_fracs * gas_masses

        # Ensure all arrays are contiguous (we need this for the C extension
        # to not produce garbage)
        star_ini_masses = unyt_array(
            np.ascontiguousarray(
                swift_gal.stars.initial_masses.to("Msun").to_value()[star_mask],
            ),
            "Msun",
        )
        star_current_masses = unyt_array(
            np.ascontiguousarray(
                swift_gal.stars.masses.to("Msun").to_value()[star_mask]
            ),
            "Msun",
        )
        star_ages = unyt_array(
            np.ascontiguousarray(ages[star_mask].to_value()),
            "Myr",
        )
        star_metals = unyt_array(
            np.ascontiguousarray(star_metals[star_mask].to_value()),
            "dimensionless",
        )
        star_coords = unyt_array(
            np.ascontiguousarray(star_coords.to("Mpc").to_value()[star_mask]),
            "Mpc",
        )
        star_smls = unyt_array(
            np.ascontiguousarray(
                swift_gal.stars.smoothing_lengths.to_physical()
                .to("Mpc")
                .to_value()[star_mask]
            ),
            "Mpc",
        )
        young_star_tau_v = unyt_array(
            np.ascontiguousarray(young_tau_v[star_mask]),
            "dimensionless",
        )
        star_radii = unyt_array(
            np.ascontiguousarray(star_radii[star_mask].to_value()),
            "kpc",
        )
        gas_masses = unyt_array(
            np.ascontiguousarray(
                swift_gal.gas.masses.to("Msun").to_value()[gas_mask],
            ),
            "Msun",
        )
        gas_dust_masses = unyt_array(
            np.ascontiguousarray(dmasses.to("Msun").to_value()[gas_mask]),
            "Msun",
        )
        gas_metals = unyt_array(
            np.ascontiguousarray(
                swift_gal.gas.metal_mass_fractions.to_value()[gas_mask]
            ),
            "dimensionless",
        )
        gas_coords = unyt_array(
            np.ascontiguousarray(gas_coords.to("Mpc").to_value()[gas_mask]),
            "Mpc",
        )
        gas_smls = unyt_array(
            np.ascontiguousarray(
                swift_gal.gas.smoothing_lengths.to_physical()
                .to("Mpc")
                .to_value()[gas_mask]
            ),
            "Mpc",
        )
        gas_radii = unyt_array(
            np.ascontiguousarray(gas_radii.to("Mpc").to_value()[gas_mask]),
            "Mpc",
        )

        # Create the galaxy object
        gal = Galaxy(
            stars=Stars(
                initial_masses=star_ini_masses,
                current_masses=star_current_masses,
                ages=star_ages,
                metallicities=star_metals,
                coordinates=star_coords,
                smoothing_lengths=star_smls,
                young_tau_v=young_star_tau_v,
                radii=star_radii,
                redshift=redshift,
            ),
            gas=Gas(
                masses=gas_masses,
                metallicities=gas_metals,
                coordinates=gas_coords,
                smoothing_lengths=gas_smls,
                dust_masses=gas_dust_masses,
                radii=gas_radii,
                redshift=redshift,
            ),
            redshift=redshift,
            centre=cent,
        )

        gals[gal_ind] = gal

    return gals
