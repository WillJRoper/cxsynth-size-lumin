#!/bin/bash
# This wrapper script is called with named parameters.
# It creates a temporary sbatch submission script with the appropriate SBATCH directives,
# then submits that script via sbatch.

# Default values (adjust as needed)
RUN_NAME="L100_m7"
VARIANT="THERMAL_AGN_m7"
PART_LIMIT=""
PARTITION="cosma8"

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
    --partition)
        PARTITION="$2"
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

# Write the SBATCH script to the temporary file.
# Note that we use variable expansion (via <<EOF) for parameters that need to be injected.
cat >"$TMPFILE" <<EOF
#!/bin/bash -l
#SBATCH --ntasks=8
#SBATCH -J SynthXCOLIBRE_${RUN_NAME}/${VARIANT}
#SBATCH --output=../logs/survey_log_%A_%a_${RUN_NAME}_${VARIANT}.txt
#SBATCH -p ${PARTITION}
#SBATCH -A dp004
#SBATCH --exclusive
#SBATCH --cpus-per-task=16
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

# Append the main execution block.
# Use a here-doc with no variable expansion to keep the embedded commands intact.
cat >>"$TMPFILE" <<'EOF'
# Execute the pipeline with mpirun.
mpirun -np $SLURM_NTASKS python my_pipeline.py \
    --grid test_grid.hdf5 \
    --grid-dir /snap8/scratch/dp004/dc-rope1/flares_test \
    --snap $snap \
    --nthreads $SLURM_CPUS_PER_TASK \
    --run-name '"${RUN_NAME}"' \
    --variant '"${VARIANT}"' $PART_LIMIT

echo "Job done, info follows..."
sacct -j $SLURM_JOBID --format=JobID,JobName,Partition,AveRSS,MaxRSS,AveVMSize,MaxVMSize,Elapsed,ExitCode
EOF

# Optionally, display the temporary script for debugging
cat "$TMPFILE"

# Submit the job using sbatch
#sbatch "$TMPFILE"

# Optionally remove the temporary file after submission.
# rm "$TMPFILE"
