# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import sys
import time
from os import path as osp
import logging
from datetime import datetime

import motmetrics as mm
import numpy as np
import sacred
import torch
import tqdm
import yaml
from torch.utils.data import DataLoader

from trackformer.datasets.tracking import TrackDatasetFactory
from trackformer.models import build_model
from trackformer.models.tracker import Tracker
from trackformer.util.misc import nested_dict_to_namespace
from trackformer.util.track_utils import (evaluate_mot_accums, get_mot_accum,
                                          interpolate_tracks, plot_sequence)

mm.lap.default_solver = 'lap'

ex = sacred.Experiment('track')
ex.add_config('cfgs/track.yaml')
ex.add_named_config('reid', 'cfgs/track_reid.yaml')


@ex.automain
def main(seed, dataset_name, obj_detect_checkpoint_file, tracker_cfg,
         write_images, output_dir, interpolate, verbose, load_results_dir,
         data_root_dir, generate_attention_maps, frame_range,
         _config, _log, _run, obj_detector_model=None):
    
    # Create logs directory
    log_dir = 'logs'
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    
    # Setup file logger
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = osp.join(log_dir, f'{output_dir.split("/")[-1]}.log')

    # Configure file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Create logger
    track_logger = logging.getLogger('tracking_comparison')
    track_logger.setLevel(logging.INFO)
    track_logger.addHandler(file_handler)
    
    # Log initial information
    track_logger.info("Starting tracking comparison")
    track_logger.info(f"Dataset: {dataset_name}")
    track_logger.info(f"Output directory: {output_dir}")
    
    if write_images:
        assert output_dir is not None

    # obj_detector_model is only provided when run as evaluation during
    # training. in that case we omit verbose outputs.
    if obj_detector_model is None:
        sacred.commands.print_config(_run)

    # set all seeds
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True

    if output_dir is not None:
        # Create separate directories for each model
        trackformer_output = osp.join(output_dir, 'trackformer')
        uni_output = osp.join(output_dir, 'uni_trackformer')
        
        for dir_path in [trackformer_output, uni_output]:
            if not osp.exists(dir_path):
                os.makedirs(dir_path)

        # Save config for both models
        yaml.dump(
            _config,
            open(osp.join(trackformer_output, 'track.yaml'), 'w'),
            default_flow_style=False)
        yaml.dump(
            _config,
            open(osp.join(uni_output, 'track.yaml'), 'w'),
            default_flow_style=False)

    ##########################
    # Initialize the modules #
    ##########################

    # Initialize both models
    trackformer_checkpoint = "PROJECT_ROOT/models/mot17_deformable_multi_frame_unitrack/checkpoint_best_BBOX_AP_IoU_0_50-0_95.pth"
    # uni_trackformer_checkpoint = "USER_HOME/ICML/UT-MOTR/models/u_full_motr/checkpoint_best_MOTA.pth"
    uni_trackformer_checkpoint = "PROJECT_ROOT/models/mot17_deformable_multi_frame_unitrack/checkpoint_best_BBOX_AP_IoU_0_50-0_95.pth"
    
    # # Initialize trackformer model
    # trackformer_config_path = os.path.join(os.path.dirname(trackformer_checkpoint), 'config.yaml')
    # trackformer_args = nested_dict_to_namespace(yaml.unsafe_load(open(trackformer_config_path)))
    # trackformer_model, _, trackformer_post = build_model(trackformer_args)
    # trackformer_state_dict = torch.load(trackformer_checkpoint, map_location=lambda storage, loc: storage)['model']
    # trackformer_state_dict = {k.replace('detr.', ''): v for k, v in trackformer_state_dict.items() if 'track_encoding' not in k}
    # trackformer_model.load_state_dict(trackformer_state_dict)
    # trackformer_model.cuda()
    # trackformer_model.tracking()
    
    # Initialize uni-trackformer model
    uni_config_path = os.path.join(os.path.dirname(uni_trackformer_checkpoint), 'config.yaml')
    uni_args = nested_dict_to_namespace(yaml.unsafe_load(open(uni_config_path)))
    uni_model, _, uni_post = build_model(uni_args)
    uni_state_dict = torch.load(uni_trackformer_checkpoint, map_location=lambda storage, loc: storage)['model']
    uni_state_dict = {k.replace('detr.', ''): v for k, v in uni_state_dict.items() if 'track_encoding' not in k}
    uni_model.load_state_dict(uni_state_dict)
    uni_model.cuda()
    uni_model.tracking()
    
    # Initialize trackers
    # trackformer_tracker = Tracker(trackformer_model, trackformer_post, tracker_cfg, generate_attention_maps, track_logger.info, verbose)
    uni_tracker = Tracker(uni_model, uni_post, tracker_cfg, generate_attention_maps, track_logger.info, verbose)

    time_total = 0
    num_frames = 0
    mot_accums = []
    
    # Define img_transform (usually None for tracking as transforms are handled in the dataset)
    img_transform = None
    
    dataset = TrackDatasetFactory(
        dataset_name, root_dir=data_root_dir, img_transform=img_transform)
    print("data_root_dir: ", data_root_dir)

    for seq in dataset:
        # trackformer_tracker.reset()
        uni_tracker.reset()

        _log.info("------------------")
        _log.info("TRACK SEQ: {}".format(seq))

        start_frame = int(frame_range['start'] * len(seq))
        end_frame = int(frame_range['end'] * len(seq))
        print("frame_range: ", frame_range)
        seq_loader = DataLoader(torch.utils.data.Subset(seq, range(start_frame, end_frame)))

        num_frames += len(seq_loader)

        results = seq.load_results(load_results_dir)

        if not results:
            start = time.time()

            trackformer_results = []
            uni_results = []

            for frame_id, frame_data in enumerate(tqdm.tqdm(seq_loader, file=sys.stdout)):
                with torch.no_grad():
                    # Get predictions from both models
                    # trackformer_tracker.step(frame_data)
                    uni_tracker.step(frame_data)
                    
                    # Get results for current frame
                    # trackformer_results = trackformer_tracker.get_results()
                    uni_results = uni_tracker.get_results()
                    
                    # Compare IDs for current frame
                    # trackformer_ids = set(track.id for track in trackformer_tracker.tracks)
                    uni_ids = set(track.id for track in uni_tracker.tracks)
                    
                    # # Check for mismatches
                    # if trackformer_ids != uni_ids:
                    #     track_logger.info(f"Frame {frame_id}:")
                    #     track_logger.info(f"  Trackformer IDs: {trackformer_ids}")
                    #     track_logger.info(f"  Uni-trackformer IDs: {uni_ids}")
                    #     track_logger.info(f"  Mismatched IDs: {trackformer_ids.symmetric_difference(uni_ids)}")
                    #     track_logger.info("---")

            time_total += time.time() - start

            # _log.info("Trackformer - NUM TRACKS: {} ReIDs: {}".format(
            #     len(trackformer_results), trackformer_tracker.num_reids))
            _log.info("Uni-trackformer - NUM TRACKS: {} ReIDs: {}".format(
                len(uni_results), uni_tracker.num_reids))
            _log.info("RUNTIME: {:.2f} s".format(time.time() - start))

            if interpolate:
                # trackformer_results = interpolate_tracks(trackformer_results)
                uni_results = interpolate_tracks(uni_results)

            if output_dir is not None:
                _log.info("WRITE RESULTS")
                # seq.write_results(trackformer_results, trackformer_output)
                seq.write_results(uni_results, uni_output)
        else:
            _log.info("LOAD RESULTS")

        if seq.no_gt:
            _log.info("NO GT AVAILBLE")
        else:
            # Create separate accumulators for each model
            # trackformer_mot_accum = get_mot_accum(trackformer_results, seq_loader)
            uni_mot_accum = get_mot_accum(uni_results, seq_loader)
            # mot_accums.append(('trackformer', trackformer_mot_accum))
            mot_accums.append(('uni', uni_mot_accum))

            if verbose:
                # Process metrics for both models
                for model_name, mot_accum in [('Uni-trackformer', uni_mot_accum)]:
                    _log.info(f"\nMetrics for {model_name}:")
                    mot_events = mot_accum.mot_events
                    reid_events = mot_events[mot_events['Type'] == 'SWITCH']
                    match_events = mot_events[mot_events['Type'] == 'MATCH']

                    switch_gaps = []
                    for index, event in reid_events.iterrows():
                        frame_id, _ = index
                        match_events_oid = match_events[match_events['OId'] == event['OId']]
                        match_events_oid_earlier = match_events_oid[
                            match_events_oid.index.get_level_values('FrameId') < frame_id]

                        if not match_events_oid_earlier.empty:
                            match_events_oid_earlier_frame_ids = \
                                match_events_oid_earlier.index.get_level_values('FrameId')
                            last_occurrence = match_events_oid_earlier_frame_ids.max()
                            switch_gap = frame_id - last_occurrence
                            switch_gaps.append(switch_gap)

                    switch_gaps_hist = None
                    if switch_gaps:
                        switch_gaps_hist, _ = np.histogram(
                            switch_gaps, bins=list(range(0, max(switch_gaps) + 10, 10)))
                        switch_gaps_hist = switch_gaps_hist.tolist()

                    _log.info('SWITCH_GAPS_HIST (bin_width=10): {}'.format(switch_gaps_hist))

        if output_dir is not None and write_images:
            _log.info("PLOT SEQ")
            # Plot sequences for each model
            # plot_sequence(
            #     trackformer_results, seq_loader, 
            #     osp.join(trackformer_output, dataset_name, str(seq)),
            #     write_images, generate_attention_maps)
            plot_sequence(
                uni_results, seq_loader, 
                osp.join(uni_output, dataset_name, str(seq)),
                write_images, generate_attention_maps)

    if time_total:
        _log.info("RUNTIME ALL SEQS (w/o EVAL or IMG WRITE): "
                  "{:.2f} s for {} frames "
                  "({:.2f} Hz)".format(
                      time_total, num_frames, num_frames / time_total))

    if obj_detector_model is None:
        _log.info("EVAL:")

        # Separate results by model
        # trackformer_accums = [acc for name, acc in mot_accums if name == 'trackformer']
        uni_accums = [acc for name, acc in mot_accums if name == 'uni']

        # Evaluate both models
        # _log.info("\nTrackformer Results:")
        # trackformer_summary, trackformer_str_summary = evaluate_mot_accums(
        #     trackformer_accums,
        #     [str(s) for s in dataset if not s.no_gt])
        # _log.info('\n{}'.format(trackformer_str_summary))

        _log.info("\nUni-trackformer Results:")
        uni_summary, uni_str_summary = evaluate_mot_accums(
            uni_accums,
            [str(s) for s in dataset if not s.no_gt])
        _log.info('\n{}'.format(uni_str_summary))

        return {'uni': uni_summary}

    return mot_accums
