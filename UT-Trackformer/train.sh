#!/bin/bash

# Get the argument passed to the script
ARGUMENT=$1

# Run the first ffmpeg command to extract frames
# ffmpeg -i data/$ARGUMENT/$ARGUMENT.mp4 -vf fps=30 data/$ARGUMENT/%06d.png
export PYTHONUNBUFFERED=1
# Run the python tracking script with the provided argument
python src/track.py with \
    dataset_name=DEMO \
    data_root_dir=USER_HOME/trackformer/data/$ARGUMENT \
    output_dir=val/$ARGUMENT \
    write_images=pretty


# Extract sequence number from the actual data location
# sequences=$(ls USER_HOME/ICML/UT-MOTR/data/MOT17/$ARGUMENT/MOT17-*-FRCNN_000001.jpg | head -n 1 | sed -n 's/.*MOT17-\([0-9]*\)-FRCNN.*/\1/p')
# if [ -z "$sequences" ]; then
#     echo "Error: Could not find sequence number"
#     exit 1
# fi
# use the following command to convert the output frames to videos.

# Create trackformer output video
# ffmpeg -y -framerate 20 -i "USER_HOME/trackformer/val/$ARGUMENT/trackformer/DEMO/$ARGUMENT/%06d.jpg" \
#     -c:v libx264 -pix_fmt yuv420p "val/$ARGUMENT/trackformer/trackformer_output.mp4"

# Create uni_trackformer output video
# ffmpeg -y -framerate 20 -i "USER_HOME/trackformer/val/$ARGUMENT/uni_trackformer/DEMO/$ARGUMENT/MOT17-${sequences}-FRCNN_%06d.jpg" \
#     -c:v libx264 -pix_fmt yuv420p "val/$ARGUMENT/uni_trackformer/uni_trackformer_output.mp4"
