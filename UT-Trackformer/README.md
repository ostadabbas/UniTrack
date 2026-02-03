# TrackFormer: Multi-Object Tracking with Transformers

This repository provides the official implementation of the [TrackFormer: Multi-Object Tracking with Transformers](https://arxiv.org/abs/2101.02702) paper by [Tim Meinhardt](https://dvl.in.tum.de/team/meinhardt/), [Alexander Kirillov](https://alexander-kirillov.github.io/), [Laura Leal-Taixe](https://dvl.in.tum.de/team/lealtaixe/) and [Christoph Feichtenhofer](https://feichtenhofer.github.io/). The codebase builds upon [DETR](https://github.com/facebookresearch/detr), [Deformable DETR](https://github.com/fundamentalvision/Deformable-DETR) and [Tracktor](https://github.com/phil-bergmann/tracking_wo_bnw).

<!-- **As the paper is still under submission this repository will continuously be updated and might at times not reflect the current state of the [arXiv paper](https://arxiv.org/abs/2012.01866).** -->

<div align="center">
    <img src="docs/MOT17-03-SDP.gif" alt="MOT17-03-SDP" width="375"/>
    <img src="docs/MOTS20-07.gif" alt="MOTS20-07" width="375"/>
</div>

## Abstract

The challenging task of multi-object tracking (MOT) requires simultaneous reasoning about track initialization, identity, and spatiotemporal trajectories.
We formulate this task as a frame-to-frame set prediction problem and introduce TrackFormer, an end-to-end MOT approach based on an encoder-decoder Transformer architecture.
Our model achieves data association between frames via attention by evolving a set of track predictions through a video sequence.
The Transformer decoder initializes new tracks from static object queries and autoregressively follows existing tracks in space and time with the new concept of identity preserving track queries.
Both decoder query types benefit from self- and encoder-decoder attention on global frame-level features, thereby omitting any additional graph optimization and matching or modeling of motion and appearance.
TrackFormer represents a new tracking-by-attention paradigm and yields state-of-the-art performance on the task of multi-object tracking (MOT17) and segmentation (MOTS20).

<div align="center">
    <img src="docs/method.png" alt="TrackFormer casts multi-object tracking as a set prediction problem performing joint detection and tracking-by-attention. The architecture consists of a CNN for image feature extraction, a Transformer encoder for image feature encoding and a Transformer decoder which applies self- and encoder-decoder attention to produce output embeddings with bounding box and class information."/>
</div>

## Unitrack Loss Integration

In this implementation, we have integrated the Unitrack loss function to enhance the tracking performance. The Unitrack loss is calculated alongside the original tracking losses, allowing for a more robust evaluation of track IDs.

### Key Features of Unitrack Loss Integration:
- **Handling of Track IDs**: The model now manages track IDs by padding them to ensure compatibility across batches with varying numbers of track IDs.
- **Weight Parameters**: The Unitrack loss is weighted with a parameter defined in the combined criterion. The current weight is set to `0.5`, which can be adjusted based on the training requirements.
- **Loss Calculation**: The Unitrack loss is computed based on the predicted track IDs, their corresponding boxes, and logits, ensuring that only valid track IDs are considered in the loss computation.

### Running the Training
To train the model with the Unitrack loss, you can use the following command:
```bash
python src/train.py mot17 deformable multi_frame --output_dir=models/mot17_deformable_multi_frame_unitrack
```
This command will initiate training on the MOT17 dataset using the deformable multi-frame tracking approach, saving the outputs to the specified directory.

## Updates to Utility Functions

We have updated the utility function for distributed mode in the `misc.py` file to ensure that the model operates in non-distributed mode. The updated function is as follows:

```python
def init_distributed_mode(args):
    # Force non-distributed mode
    print('Not using distributed mode')
    args.distributed = False
    return
```

This change ensures that the training process does not attempt to utilize distributed training, simplifying the setup for single-machine training scenarios.

## Installation

We refer to our [docs/INSTALL.md](docs/INSTALL.md) for detailed installation instructions.

## Train TrackFormer

We refer to our [docs/TRAIN.md](docs/TRAIN.md) for detailed training instructions.

## Evaluate TrackFormer

In order to evaluate TrackFormer on a multi-object tracking dataset, we provide the `src/track.py` script which supports several datasets and splits interchangle via the `dataset_name` argument (See `src/datasets/tracking/factory.py` for an overview of all datasets.) The default tracking configuration is specified in `cfgs/track.yaml`. To facilitate the reproducibility of our results, we provide evaluation metrics for both the train and test set.

### MOT17

#### Private detections

```
python src/track.py with reid
```

<center>

| MOT17     | MOTA         | IDF1           |       MT     |     ML     |     FP       |     FN              |  ID SW.      |
|  :---:    | :---:        |     :---:      |    :---:     | :---:      |    :---:     |   :---:             |  :---:       |
| **Train** |     74.2     |     71.7       |     849      | 177        |      7431    |      78057          |  1449        |
| **Test**  |     74.1     |     68.0       |    1113      | 246        |     34602    |     108777          |  2829        |

</center>

#### Public detections (DPM, FRCNN, SDP)

```
python src/track.py with \
    reid \
    tracker_cfg.public_detections=min_iou_0_5 \
    obj_detect_checkpoint_file=models/mot17_deformable_multi_frame/checkpoint_epoch_50.pth
```

<center>

| MOT17     | MOTA         | IDF1           |       MT     |     ML     |     FP       |     FN              |  ID SW.      |
|  :---:    | :---:        |     :---:      |    :---:     | :---:      |    :---:     |   :---:             |  :---:       |
| **Train** |     64.6     |     63.7       |    621       | 675        |     4827     |     111958          |  2556        |
| **Test**  |     62.3     |     57.6       |    688       | 638        |     16591    |     192123          |  4018        |

</center>

### MOT20

#### Private detections

```
python src/track.py with \
    reid \
    dataset_name=MOT20-ALL \
    obj_detect_checkpoint_file=models/mot20_crowdhuman_deformable_multi_frame/checkpoint_epoch_50.pth
```

<center>

| MOT20     | MOTA         | IDF1           |       MT     |     ML     |     FP       |     FN              |  ID SW.      |
|  :---:    | :---:        |     :---:      |    :---:     | :---:      |    :---:     |   :---:             |  :---:       |
| **Train** |     81.0     |     73.3       |    1540      | 124        |     20807    |     192665          |  1961        |
| **Test**  |     68.6     |     65.7       |     666      | 181        |     20348    |     140373          |  1532        |

</center>

### MOTS20

```
python src/track.py with \
    dataset_name=MOTS20-ALL \
    obj_detect_checkpoint_file=models/mots20_train_masks/checkpoint.pth
```

Our tracking script only applies MOT17 metrics evaluation but outputs MOTS20 mask prediction files. To evaluate these download the official [MOTChallengeEvalKit](https://github.com/dendorferpatrick/MOTChallengeEvalKit).

<center>

| MOTS20    | sMOTSA         | IDF1           |       FP     |     FN     |     IDs      |
|  :---:    | :---:          |     :---:      |    :---:     | :---:      |    :---:     |
| **Train** |     --         |     --         |    --        |   --       |     --       |
| **Test**  |     54.9       |     63.6       |    2233      | 7195       |     278      |

</center>

### Demo

To facilitate the application of TrackFormer, we provide a demo interface which allows for a quick processing of a given video sequence.

```
ffmpeg -i data/snakeboard/snakeboard.mp4 -vf fps=30 data/snakeboard/%06d.png

python src/track.py with \
    dataset_name=DEMO \
    data_root_dir=data/snakeboard \
    output_dir=data/snakeboard \
    write_images=pretty
```



## Running the Project

There are several ways to run this project depending on your needs:

### Quick Demo with Your Video

1. Place your dataset in the `data/xxx` directory  (follow instructions in the [docs/INSTALL.md](docs/INSTALL.md) and [docs/TRAIN.md](docs/TRAIN.md))
2. Run the demo script:
```bash
bash train.sh xxx
```

This will:
- Process your video through the tracking system
- Generate output frames in the `val/<video_name>` directory
- Create visualizations with tracking information

### Direct Python Execution with track.py

You can run the tracking system directly using the Python.


Note: update the weights path in the track.py file in the variables uni_trackformer_checkpoint and trackformer_checkpoint.

```bash
python src/track.py with \
    dataset_name=DEMO \
    data_root_dir=data/<your_dataset> \
    output_dir=val/<output_folder> \
    write_images=pretty
```

Key parameters:
- `dataset_name`: Type of dataset (e.g., DEMO, MOT17-ALL, MOTS20-ALL)
- `data_root_dir`: Directory containing your input data
- `output_dir`: Where to save the results
- `write_images`: Set to "pretty" for visualization output
- `reid`: Add this flag to enable ReID features

The script will:
1. Run both TrackFormer and UniTrackformer models
2. Generate tracking results in separate directories:
   - `<output_dir>/trackformer/`
   - `<output_dir>/uni_trackformer/`
3. Create log files with tracking metrics and comparisons
4. Save visualization frames if `write_images` is enabled

### Output Format

The tracking results are saved in two formats:
1. Frame-by-frame images with visualization of tracking results
2. Compiled video output (when using ffmpeg commands)

Note: Make sure you have ffmpeg installed to create video outputs from the tracked frames.


<div align="center">
    <img src="docs/snakeboard.gif" alt="Snakeboard demo" width="600"/>
</div>

## Publication
If you use this software in your research, please cite our publication:

```
@InProceedings{meinhardt2021trackformer,
    title={TrackFormer: Multi-Object Tracking with Transformers},
    author={Tim Meinhardt and Alexander Kirillov and Laura Leal-Taixe and Christoph Feichtenhofer},
    year={2022},
    month = {June},
    booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
}