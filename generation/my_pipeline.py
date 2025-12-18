"""A pipeline for generating synthetic observations from COLIBRE."""

import argparse
import concurrent.futures
import os
import time
import warnings
from functools import partial

import h5py
import numpy as np
from astropy.cosmology import w0waCDM
from colibre_data_loader import _get_galaxies, partition_galaxies
from mpi4py import MPI as mpi
from my_emission_models import ColibreLOSEmission
from my_extra_analysis import get_curve_of_growth_hlr, get_pixel_based_hlr
from synthesizer.grid import Grid
from synthesizer.instruments import InstrumentCollection
from synthesizer.kernel_functions import Kernel
from synthesizer.pipeline import Pipeline
from unyt import Msun, angstrom, kpc

# Silence warnings (only because we now what we're doing)
warnings.filterwarnings("ignore")

# Msun needs to be respected
Msun = Msun.in_base("galactic")


def load_galaxies(
    partition,
    location,
    snap,
    cosmo,
    aperture,
    nthreads=1,
    fof_only=False,
    pah_fraction=0.1,
):
    """
    Load the galaxies into memory.

    Args:
        partition (list): The list of galaxies to load.
        location (str): The location of the data.
        snap (str): The snapshot to load.
        cosmo (w0waCDM): The cosmology to use.
        aperture (float): The aperture to use.
        nthreads (int): The number of threads to use.
        fof_only (bool): If True, only load the FOF groups.
        pah_fraction (float): Fraction of small graphite to allocate to PAHs.

    Returns:
        list: The list of galaxies.
    """
    # If we have no galaxies exist, exit we'll deal with it later
    if len(partition) < 2:
        return []

    # If we aren't multithreaded then just load the galaxies
    if nthreads == 1 or nthreads == 0 or partition.size < nthreads:
        galaxies = _get_galaxies(
            partition, location, snap, cosmo, aperture, fof_only, pah_fraction
        )

    # Otherwise, distribute the read
    else:
        # Create a partial function that binds extra1 and extra2
        _get_galaxies_with_args = partial(
            _get_galaxies,
            location=location,
            snap=snap,
            cosmo=cosmo,
            aperture=aperture,
            fof_only=fof_only,
            pah_fraction=pah_fraction,
        )

        # Equally split the partition into nthreads chunks
        chunks = np.array_split(partition, nthreads)

        with concurrent.futures.ProcessPoolExecutor(max_workers=nthreads) as executor:
            futures = [
                executor.submit(_get_galaxies_with_args, chunk) for chunk in chunks
            ]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        # Flatten the resulting list of lists of galaxies
        galaxies = np.concatenate(results)

    # Get the SPH kernel
    sph_kernel = Kernel()
    kernel_data = sph_kernel.get_kernel()

    # Loop over galaxies and calculate the column densities
    for gal in galaxies:
        if gal.gas.nparticles > 0:
            # Calculate column densities for each dust grain type and hydrogen
            # Small graphite (0.01 µm)
            gal.stars.get_los_column_density(
                other_parts=gal.gas,
                density_attr="graphite_small_masses",
                kernel=kernel_data,
                column_density_attr="sigmalos_graphite_a0p01um",
                nthreads=nthreads,
            )
            # Large graphite (0.1 µm)
            gal.stars.get_los_column_density(
                other_parts=gal.gas,
                density_attr="graphite_large_masses",
                kernel=kernel_data,
                column_density_attr="sigmalos_graphite_a0p1um",
                nthreads=nthreads,
            )
            # Small silicates (0.01 µm)
            gal.stars.get_los_column_density(
                other_parts=gal.gas,
                density_attr="silicate_small_masses",
                kernel=kernel_data,
                column_density_attr="sigmalos_silicate_a0p01um",
                nthreads=nthreads,
            )
            # Large silicates (0.1 µm)
            gal.stars.get_los_column_density(
                other_parts=gal.gas,
                density_attr="silicate_large_masses",
                kernel=kernel_data,
                column_density_attr="sigmalos_silicate_a0p1um",
                nthreads=nthreads,
            )
            # Ionised PAH (0.005 µm)
            gal.stars.get_los_column_density(
                other_parts=gal.gas,
                density_attr="pah_ionised_masses",
                kernel=kernel_data,
                column_density_attr="sigmalos_pahionised_a0p005um",
                nthreads=nthreads,
            )
            # Neutral PAH (0.005 µm)
            gal.stars.get_los_column_density(
                other_parts=gal.gas,
                density_attr="pah_neutral_masses",
                kernel=kernel_data,
                column_density_attr="sigmalos_pahneutral_a0p005um",
                nthreads=nthreads,
            )
            # Hydrogen
            gal.stars.get_los_column_density(
                other_parts=gal.gas,
                density_attr="h_mass",
                kernel=kernel_data,
                column_density_attr="sigmalos_H",
                nthreads=nthreads,
            )
        else:
            # If no gas particles, set all column densities to zero
            gal.stars.sigmalos_graphite_a0p01um = np.zeros(gal.stars.nparticles)
            gal.stars.sigmalos_graphite_a0p1um = np.zeros(gal.stars.nparticles)
            gal.stars.sigmalos_silicate_a0p01um = np.zeros(gal.stars.nparticles)
            gal.stars.sigmalos_silicate_a0p1um = np.zeros(gal.stars.nparticles)
            gal.stars.sigmalos_pahionised_a0p005um = np.zeros(gal.stars.nparticles)
            gal.stars.sigmalos_pahneutral_a0p005um = np.zeros(gal.stars.nparticles)
            gal.stars.sigmalos_H = np.zeros(gal.stars.nparticles)

    return galaxies


def get_emission_model(
    grid_name,
    grid_dir,
    fesc=0.0,
    fesc_ly_alpha=1.0,
):
    """Get the emission model to use for the observations."""
    grid = Grid(
        grid_name,
        grid_dir,
        lam_lims=(900 * angstrom, 6 * 10**5 * angstrom),
    )
    model = ColibreLOSEmission(grid)

    # # Limit the spectra to be saved
    # model.save_spectra("attenuated", "reprocessed")

    return model


if __name__ == "__main__":
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Derive synthetic observations for FLARES."
    )

    # Add the general Synthesizer arguments
    parser.add_argument(
        "--grid",
        type=str,
        help="The path to the grid.",
    )
    parser.add_argument(
        "--grid-dir",
        type=str,
        default=None,
        help="The directory to save the grid (default: None, uses Synthesizer default).",
    )

    # What snapshot are we using?
    parser.add_argument(
        "--snap-ind",
        type=int,
        help="The snapshot to use.",
    )

    # How many threads are we using?
    parser.add_argument(
        "--nthreads",
        type=int,
        help="The number of threads to use.",
    )

    # Cosmology arguments

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

    # Lower mass limit?
    parser.add_argument(
        "--part-limit",
        type=int,
        help="The lower mass limit for galaxies.",
        default=100,
    )

    # Aperture?
    parser.add_argument(
        "--aperture",
        type=float,
        help="The aperture to use for the observations.",
        default=100,
    )

    # Galaxies or FOF groups?
    parser.add_argument(
        "--fof-only",
        action="store_true",
        help="If true, only load the FOF groups.",
    )

    # PAH fraction
    parser.add_argument(
        "--pah-fraction",
        type=float,
        help="Fraction of small graphite to allocate to PAHs (default: 0.1).",
        default=0.1,
    )

    # Path to the instruments file
    parser.add_argument(
        "--inst-path",
        type=str,
        help="The path to the instruments file. Default is './instruments.hdf5'.",
        default="./instruments.hdf5",
    )

    # Get MPI info
    comm = mpi.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Parse the arguments
    args = parser.parse_args()
    grid_name = args.grid
    grid_dir = args.grid_dir
    snap = str(args.snap_ind).zfill(4)
    nthreads = args.nthreads

    run_folder = args.run_dir
    run_name = args.run_name
    variant = args.variant
    inst_path = args.inst_path

    # Switches for behaviour
    part_limit = args.part_limit
    fof_only = args.fof_only
    pah_fraction = args.pah_fraction

    # Some metadata we'll use
    aperture = args.aperture

    # Define the whole path to the data
    path = f"{run_folder}/{run_name}/{variant}/"

    # Get a version of the grid name with an extension for labelling
    grid_name_no_ext = grid_name.split("/")[-1].split(".")[0]

    # Read in the redshift and while we do it make sure we actually have
    # SOAP data for this snap
    try:
        with h5py.File(f"{path}/SOAP-HBT/halo_properties_{snap}.hdf5") as hf:
            redshift = hf["Cosmology"].attrs["Redshift"][0]
    except FileNotFoundError:
        print(f"No SOAP data for snapshot {snap}.")
        exit(0)

    # Set up the cosmology
    H0 = 68.0999
    Om0 = 0.304611
    Ob0 = 0.0486
    Ode0 = 0.693922
    boxl = float(run_name.split("_")[0].split("N")[0].replace("L", ""))
    vol = boxl**3  # /Mpc^3
    h = 0.681
    w0 = -1
    wa = 0
    Neff = 3.044
    Ode0 = 0.693922
    T_CMB_0 = 2.7255

    cosmo = w0waCDM(
        H0=H0,
        Om0=Om0,
        Ob0=Ob0,
        wa=wa,
        w0=w0,
        Ode0=Ode0,
        Tcmb0=T_CMB_0,
        Neff=Neff,
    )

    # Get the redshift of this snapshot
    redshifts = np.genfromtxt(
        f"{run_folder}/{run_name}/{variant}/output_list.txt",
        usecols=(0),
        dtype=np.float32,
        skip_header=1,
        delimiter=",",
    )
    redshift = redshifts[args.snap_ind]

    # Define the output path, for special particle limits we all include that
    # info
    outpath = f"../data/{run_name}/{variant}/Synthesized_imgs_{snap}_{grid_name_no_ext}"
    if part_limit != 100:
        outpath += f"_part_limit_{part_limit}"
    if fof_only:
        outpath += "_FOFGroups"
    outpath += ".hdf5"

    # Create the directory if it doesn't exist
    if rank == 0 and not os.path.exists(f"../data/{run_name}/{variant}"):
        os.makedirs(f"../data/{run_name}/{variant}")

    # Get the SPH kernel
    kernel_data = Kernel().get_kernel()

    # Make sure we have the instrument collection
    if not os.path.exists(inst_path):
        raise FileNotFoundError(
            f"Instrument collection not found at {inst_path}."
            " Please run the instrument generation script first."
        )

    # Can't move on until we have the instruments file made
    comm.Barrier()

    # Partition the galaxies
    read_start = time.perf_counter()
    indices = partition_galaxies(
        location=path,
        snap=snap,
        part_limit=part_limit,
        aperture=aperture,
        fof_only=fof_only,
    )

    # Do we have any galaxies anywhere? If not, we can't do anything
    n_gals_all = comm.allreduce(len(indices), op=mpi.SUM)
    n_gals_per_rank = comm.gather(len(indices), root=0)
    if rank == 0:
        print(f"Total galaxies: {n_gals_all}")
        print(f"Galaxies per rank: {n_gals_per_rank}")
    if n_gals_all == 0:
        print("No galaxies found.")
        comm.Abort()

    # Load the galaxies
    galaxies = load_galaxies(
        indices,
        location=path,
        snap=snap,
        cosmo=cosmo,
        aperture=aperture,
        nthreads=nthreads,
        fof_only=fof_only,
        pah_fraction=pah_fraction,
    )
    print(f"Reading took {time.perf_counter() - read_start:.2f} seconds.")

    # If any rank has 0 galaxies we must abort
    if len(galaxies) == 0:
        print("No galaxies found.")
        comm.Abort()

    # Load instruments and split them up by purpose
    all_insts = InstrumentCollection(filepath=inst_path)

    # Set the resolution of the UV1500 instrument
    all_insts["UV1500"].resolution = galaxies[0].physical_softening.to("kpc")

    # Split instruments for different purposes
    # NIRCam and MIRI for photometry and PSF flux images
    jwst_insts = all_insts["JWST.NIRCam"] + all_insts["JWST.MIRI"]

    # UV1500 only for luminosity images (no flux needed)
    uv_inst = all_insts["UV1500"]

    # Set up the pipeline (instruments now passed to each operation independently)
    pipeline = Pipeline(
        emission_model=get_emission_model(grid_name, grid_dir),
        nthreads=nthreads,
        comm=comm,
    )

    # Add the extra analysis functions we want
    for frac in [0.2, 0.5, 0.8]:
        frac_key = f"{frac}".replace(".", "p")
        pipeline.add_analysis_func(
            lambda gal, frac=frac: gal.stars.get_attr_radius(
                "current_masses",
                frac=frac,
            ),
            f"Stars/MassRadii/{frac_key}",
        )
    pipeline.add_analysis_func(
        lambda gal: get_pixel_based_hlr(gal.stars),
        "Stars/PixelHalfLightRadii",
    )
    pipeline.add_analysis_func(
        lambda gal: get_curve_of_growth_hlr(gal.stars),
        "Stars/CurveOfGrowthHalfLightRadii",
    )
    pipeline.add_analysis_func(lambda gal: gal.redshift, "Redshift")
    pipeline.add_analysis_func(
        lambda gal: gal.stars.total_mass,
        "Stars/StellarMass",
    )

    # Add them to the pipeline
    pipeline.add_galaxies(list(galaxies))

    # Rest frame photometry with UV top hats, observed with JWST instruments
    pipeline.get_photometry_luminosities(uv_inst)
    pipeline.get_photometry_fluxes(jwst_insts, cosmo=cosmo)

    # Luminosity images with UV1500 only
    pipeline.get_images_luminosity(
        uv_inst,
        fov=61 * kpc if not fof_only else 1000 * kpc,
        kernel=kernel_data,
        labels=["reprocessed", "stellar_total"],
        write=False if part_limit < 1000 else True,
    )

    # PSF flux images with JWST instruments (NIRCam + MIRI)
    pipeline.get_images_flux(
        jwst_insts,
        fov=61 * kpc if not fof_only else 1000 * kpc,
        kernel=kernel_data,
        cosmo=cosmo,
        labels=["reprocessed", "stellar_total"],
        write=False if part_limit < 1000 else True,
    )

    # Run the pipeline
    pipeline.run()

    # Save the pipeline and combine the files into a single virtual file
    pipeline.write(outpath)
    pipeline.combine_files_virtual()
