"""A set of helper functions for partitioning and loading COLIBRE data."""

import h5py
import numpy as np
import swiftsimio
from mpi4py import MPI as mpi
from swiftgalaxy import SOAP, SWIFTGalaxies
from swiftsimio import cosmo_array
from synthesizer.particle import Galaxy, Gas, Stars
from unyt import Gyr, Mpc, Myr, kpc, unyt_array


def partition_galaxies(location, snap, part_limit, aperture, fof_only):
    """Partition the galaxies between the MPI processes."""
    # Get the number of processes and this rank
    nranks = mpi.COMM_WORLD.Get_size()
    this_rank = mpi.COMM_WORLD.Get_rank()

    # Load the SWIFT galaxy catalogue
    cat = swiftsimio.load(f"{location}/SOAP/halo_properties_{snap}.hdf5")

    # Get the number of star particles in the exclusive sphere
    nstars = getattr(
        getattr(cat, f"exclusive_sphere_{int(aperture)}kpc"),
        "number_of_star_particles",
    ).to_value()

    # Create an array of galaxy indices
    gal_inds = np.arange(len(nstars))

    # Create the mask for galaxies above the particle limit
    mask = nstars >= part_limit

    # If we are only doing the FOF then we want to ensure we only have centrals
    if fof_only:
        is_central = cat.input_halos.is_central.to_value()
        mask = np.logical_and(mask, is_central)

    # Apply the mask to the galaxy indices
    gal_inds = gal_inds[mask]

    # If we have no galaxies, we can't do anything
    if len(gal_inds) == 0:
        return []

    # If we have only one rank, we can just return all the galaxies
    if nranks == 1:
        return np.array(gal_inds, dtype=int)

    # Now divide the galaxies between processes keeping the galaxies in
    # contiguous chunks but balancing the number of stars on each process
    stars_per_rank = np.zeros(nranks, dtype=int)
    galaxies_on_rank = np.zeros(nranks, dtype=object)
    target_stars = np.sum(nstars[gal_inds]) // nranks
    for i in range(nranks):
        galaxies_on_rank[i] = []
    igal = 0
    current_rank = 0
    while igal < len(gal_inds):
        this_gal_ind = gal_inds[igal]
        if stars_per_rank[current_rank] + nstars[this_gal_ind] > target_stars:
            current_rank += 1
            if current_rank > nranks - 1:
                current_rank = nranks - 1
        stars_per_rank[current_rank] += nstars[this_gal_ind]
        galaxies_on_rank[current_rank].append(this_gal_ind)
        igal += 1

    return np.array(galaxies_on_rank[this_rank], dtype=int)


def _set_up_swift_galaxy(
    location,
    snap,
    chunk_inds,
    fof_only=False,
):
    """
    Set up the SWIFT galaxy object.

    Args:
        location (str): The location of the data.
        snap (str): The snapshot to load.
        chunk_inds (np.ndarray): The indices of the galaxies to load.
        fof_only (bool): Whether to only load the FOF groups.

    Returns:
        tuple: A tuple containing the SOAP object, the SWIFT galaxies object,
            the scale factor, and the redshift.
    """
    # Read in some useful metadata
    with h5py.File(f"{location}/SOAP/halo_properties_{snap}.hdf5") as hf:
        aexp = hf["Cosmology"].attrs["Scale-factor"]
        redshift = hf["Cosmology"].attrs["Redshift"]
        comoving_soft = (
            float(hf["SWIFT/Parameters"].attrs["Gravity:comoving_baryon_softening"])
            * Mpc
        )
        max_phys_soft = (
            float(hf["SWIFT/Parameters"].attrs["Gravity:max_physical_baryon_softening"])
            * Mpc
        )
        print([comoving_soft / (1 + redshift[0]), max_phys_soft])
        soft = np.max([comoving_soft / (1 + redshift[0]), max_phys_soft])

    # If we have an empty chunk, we can't do anything
    if len(chunk_inds) == 0:
        return None, None, aexp, redshift

    # Define the custom spatial offset if we need one
    custom_spatial_offsets = None
    if fof_only:
        cat = swiftsimio.load(f"{location}/SOAP/halo_properties_{snap}.hdf5")
        # Get the FOF group centre
        custom_spatial_offsets = cosmo_array(
            [[-1, 1], [-1, 1], [-1, 1]],
            Mpc,
            comoving=True,
            scale_factor=cat.metadata.a,
            scale_exponent=1,
        )

    soap = SOAP(
        f"{location}/SOAP/halo_properties_{snap}.hdf5",
        soap_index=chunk_inds,
        extra_mask="bound_only" if not fof_only else "fof",
        custom_spatial_offsets=custom_spatial_offsets,
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
        auto_recentre=True,
    )

    return soap, sgs, aexp, redshift, soft


def _get_galaxies(
    chunk_inds,
    location,
    snap,
    cosmo,
    aperture,
    fof_only,
):
    """
    Get a galaxy from the master file.

    Args:
        chunk_inds (np.ndarray): The indices of the galaxies to load.
        location (str): The location of the data.
        snap (str): The snapshot to load.
        cosmo (astropy.cosmology): The cosmology to use.
        aperture (float): The aperture to use.
        fof_only (bool): Whether to only load the FOF groups.

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

    # If we have no galaxies, we can't do anything
    if ngals == 0:
        return []

    # First up, get out I/O helpers and some metadata from SWIFTGalaxy
    soap, sgs, aexp, redshift, soft = _set_up_swift_galaxy(
        location,
        snap,
        chunk_inds,
        fof_only,
    )

    # Initialise a container for the galaxies we are about to load
    gals = np.empty(ngals, dtype=object)

    # swiftgalaxy picks its own efficient iteration order
    for gal_ind, swift_gal in enumerate(sgs):
        # Get the centre
        star_coords = swift_gal.stars.coordinates.to_physical().to("Mpc")
        cent = np.array([0.0, 0.0, 0.0], dtype=float) * Mpc

        # Derive the radii for star and gas particles
        star_radii = np.linalg.norm(star_coords, axis=1).to("kpc")
        gas_coords = swift_gal.gas.coordinates.to_physical().to("Mpc")
        if gas_coords.size > 1:
            gas_radii = np.linalg.norm(gas_coords, axis=1).to("kpc")
        elif gas_coords.size == 1:
            gas_radii = unyt_array(
                np.array([np.linalg.norm(gas_coords).to("kpc").value]), "kpc"
            )
        else:
            gas_radii = unyt_array(np.array([]), "kpc")

        # Define masks for the particles within the aperture
        star_mask = star_radii <= (aperture * kpc)
        gas_mask = gas_radii <= (aperture * kpc)

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
                swift_gal.stars.initial_masses.to_value("Msun")[star_mask],
                dtype=np.float64,
            ),
            "Msun",
        )
        star_current_masses = unyt_array(
            np.ascontiguousarray(
                swift_gal.stars.masses.to_value("Msun")[star_mask],
                dtype=np.float64,
            ),
            "Msun",
        )
        star_ages = unyt_array(
            np.ascontiguousarray(
                ages[star_mask].to_value(),
                dtype=np.float64,
            ),
            "Myr",
        )
        star_metals = unyt_array(
            np.ascontiguousarray(
                star_metals[star_mask].to_value(),
                dtype=np.float64,
            ),
            "dimensionless",
        )
        star_coords = unyt_array(
            np.ascontiguousarray(
                star_coords.to_value("Mpc")[star_mask],
                dtype=np.float64,
            ),
            "Mpc",
        )
        star_smls = unyt_array(
            np.ascontiguousarray(
                swift_gal.stars.smoothing_lengths.to_physical().to_value("Mpc")[
                    star_mask
                ],
                dtype=np.float64,
            ),
            "Mpc",
        )
        young_star_tau_v = unyt_array(
            np.ascontiguousarray(
                young_tau_v[star_mask],
                dtype=np.float64,
            ),
            "dimensionless",
        )
        star_radii = unyt_array(
            np.ascontiguousarray(
                star_radii[star_mask].to_value(),
                dtype=np.float64,
            ),
            "kpc",
        )
        gas_masses = unyt_array(
            np.ascontiguousarray(
                swift_gal.gas.masses.to_value("Msun")[gas_mask],
                dtype=np.float64,
            ),
            "Msun",
        )
        gas_dust_masses = unyt_array(
            np.ascontiguousarray(
                dmasses.to_value("Msun")[gas_mask],
                dtype=np.float64,
            ),
            "Msun",
        )
        gas_metals = unyt_array(
            np.ascontiguousarray(
                swift_gal.gas.metal_mass_fractions.to_value()[gas_mask],
                dtype=np.float64,
            ),
            "dimensionless",
        )
        gas_coords = unyt_array(
            np.ascontiguousarray(
                gas_coords.to_value("Mpc")[gas_mask],
                dtype=np.float64,
            ),
            "Mpc",
        )
        gas_smls = unyt_array(
            np.ascontiguousarray(
                swift_gal.gas.smoothing_lengths.to_physical().to_value("Mpc")[gas_mask],
                dtype=np.float64,
            ),
            "Mpc",
        )
        gas_radii = unyt_array(
            np.ascontiguousarray(
                gas_radii.to_value("Mpc")[gas_mask],
                dtype=np.float64,
            ),
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
            redshift=redshift[0],
            centre=cent,
            physical_softening=soft,
        )

        gals[gal_ind] = gal

    return gals
