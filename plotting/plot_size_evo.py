"""A script for plotting the size evolution of galaxies."""

import argparse
import glob
import os


def plot_size_evolution(filepath):
    """
    Plot the size evolution of galaxies.

    This will read all snapshots in the input directory and plot their size
    evolution.
    """
    # Get all the files in the directory
    files = glob.glob(filepath + "/Synth*.hdf5")

    # We need to make sure we only have the combined files, luckily these all
    # have 1 extra element delineated by an underscore so we can use that to
    # filter out the other files
    lengths = [len(f.split("_")) for f in files]
    combined_lengths = min(lengths)
    files = [f for f in files if len(f.split("_")) == combined_lengths]
    print(f"Found {len(files)} files.")
    print(f"Files: {files}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()

    # Define input and output paths
    run_name = args.run_name
    variant = args.variant
    path = f"../data/{run_name}/{variant}/"
    outpath = f"../plots/{run_name}/{variant}/"

    # Check if the file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} does not exist.")

    # Create the output directory if it doesn't exist
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    # Plot the size evolution
    plot_size_evolution(path)
