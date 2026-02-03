#!/usr/bin/env bash
# ------------------------------------------------------------------
# Create the folder layout expected by datasets/mot.py
# Adjust ROOT if your parent directory is different.
# ------------------------------------------------------------------

# Define base directory for MOT datasets
ROOT=DATASET_ROOT           # == --mot_path

# Create symbolic link for MOT17_all pointing to MOT17
ln -s "${ROOT}/MOT/MOT17" "${ROOT}/MOT/MOT17_all"

# Set up custom train/val split
cd "${ROOT}/MOT/MOT17"

# Remove existing val directory if it exists
rm -rf val
mkdir -p val

# Define which sequences should be in validation set (customize this list)
VAL_SEQUENCES=("MOT17-02" "MOT17-04" "MOT17-09")

# Create symbolic links for validation sequences
for seq in "${VAL_SEQUENCES[@]}"; do
    if [ -d "train/$seq" ]; then
        echo "Adding $seq to validation set"
        ln -s "../train/$seq" "val/$seq"
    else
        echo "Warning: $seq not found in train directory"
    fi
done

# Verify the setup
echo "Checking MOT17_all directory structure:"
ls -R "${ROOT}/MOT/MOT17_all/train" | head

echo -e "\nChecking validation directory:"
ls "${ROOT}/MOT/MOT17/val"

echo -e "\nChecking training directory:"
ls "${ROOT}/MOT/MOT17/train"


# #!/usr/bin/env bash
# # ------------------------------------------------------------------
# # Create the folder layout expected by datasets/mot.py
# # Adjust ROOT if your parent directory is different.
# # ------------------------------------------------------------------

# # Define base directory for MOT datasets
# ROOT=DATASET_ROOT           # == --mot_path

# # Create symbolic link for MOT17_all pointing to MOT17
# ln -s "${ROOT}/MOT/MOT17" "${ROOT}/MOT/MOT17_all"

# # Set up validation data using training data
# cd "${ROOT}/MOT/MOT17"
# rm -rf val                    # remove any empty dir/link
# ln -s train val               # reuse the train sequences as "val"

# # Verify the setup
# echo "Checking MOT17_all directory structure:"
# ls -R "${ROOT}/MOT/MOT17_all/train" | head

# echo "\nChecking validation directory:"
# ls "${ROOT}/MOT/MOT17/val" | head   # should list MOT17-02, MOT17-04 …
