#!/bin/bash
# This wrapper script is called with named parameters.
# It creates a temporary sbatch submission script with the appropriate SBATCH directives,
# then submits that script via sbatch.

# Default values (adjust as needed)
RUN_NAME="L100_m7"
VARIANT="THERMAL_AGN_m7"
PART_LIMIT=""
NTASKS=8
NTHREADS=16
PARTITION="cosma8"
FOFONLY=0

# Parse named arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
    --run-name)
        RUN_NAME="$2"
        shift 2
        ;;
    --variant)
        VARIANT="$2"
        shift 2
        ;;
    --part-limit)
        PART_LIMIT="$2"
        shift 2
        ;;
    --ntasks)
        NTASKS="$2"
        shift 2
        ;;
    --nthreads)
        NTHREADS="$2"
        shift 2
        ;;
    --partition)
        PARTITION="$2"
        shift 2
        ;;
    --fof-only)
        FOFONLY=1
        shift 2
        ;;
    *)
        echo "Unknown parameter passed: $1"
        exit 1
        ;;
    esac
done

# Create a temporary file for the SBATCH submission script
TMPFILE=$(mktemp /tmp/submit_job.XXXXXX.sh)

# Write the SBATCH script header with variable expansion.
cat >"$TMPFILE" <<EOF
#!/bin/bash -l
#SBATCH --ntasks=${NTASKS}
#SBATCH -J SynthXCOLIBRE_${RUN_NAME}/${VARIANT}
#SBATCH --output=../logs/SynthXCOLIBRE_${RUN_NAME}_${VARIANT}_%A_%a.out
#SBATCH -p ${PARTITION}
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH --cpus-per-task=${NTHREADS}
#SBATCH --time=24:00:00
#SBATCH --array=0-127

module purge
module load gnu_comp/11.1.0 openmpi

# Calculate snap based on task ID
snap=\$((SLURM_ARRAY_TASK_ID))
EOF

# If a part limit was provided, add it as a variable assignment in the submission script.
if [ -n "$PART_LIMIT" ]; then
    echo "PART_LIMIT=\"--part-limit ${PART_LIMIT}\"" >>"$TMPFILE"
else
    echo "PART_LIMIT=\"\"" >>"$TMPFILE"
fi

# Create the fof argument depending on the flag
if [ "$FOFONLY" -eq 1 ]; then
    FOF_ARG="--fof-only"
else
    FOF_ARG=""
fi

# Append the main execution block.
# We use a here-document with unquoted EOF so that wrapper variables are expanded,
# but we escape the SLURM variables so they remain literal in the SBATCH script.
cat >>"$TMPFILE" <<EOF

# Execute the pipeline with mpirun.
mpirun -np \$SLURM_NTASKS python my_pipeline.py \\
    --grid test_grid.hdf5 \\
    --grid-dir /snap8/scratch/dp004/dc-rope1/flares_test \\
    --snap \$snap \\
    --nthreads \$SLURM_CPUS_PER_TASK \\
    --run-name ${RUN_NAME} \\
    --variant ${VARIANT} \$PART_LIMIT ${FOF_ARG}

echo "Job done, info follows..."
sacct -j \$SLURM_JOBID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode
EOF

# Optionally, display the temporary script for debugging
# cat "$TMPFILE"

# Submit the job using sbatch
sbatch "$TMPFILE"

# Optionally remove the temporary file after submission.
# rm "$TMPFILE"
