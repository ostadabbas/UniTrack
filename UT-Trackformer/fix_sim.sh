find USER_HOME/ICML/UT-MOTR/data/MOT17/ -xtype l | while read symlink; do
    # Extract the target path of the broken symlink
    target=$(readlink "$symlink")

    # Fix the incorrect "MOT20" reference in the target path
    new_target="${target/MOT20/MOT17}"

    # Check if the corrected target file exists
    if [ -f "$new_target" ]; then
        # Remove the broken symlink and create a new one
        ln -sf "$new_target" "$symlink"
        echo "Fixed symlink: $symlink -> $new_target"
    else
        echo "WARNING: Target file still missing for $symlink"
    fi
done
