import os
import configparser  # To parse ini files
from tqdm import tqdm
def convert_mot_to_fairmot(mot_dir, fairmot_dir, dataset_type='train'):
    dataset_dir = os.path.join(mot_dir, dataset_type)
    sequences = [s for s in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, s))]
    
    for seq in tqdm(sequences):
        img_dir = os.path.join(dataset_dir, seq, 'img1')
        gt_file = os.path.join(dataset_dir, seq, 'gt', 'gt.txt')
        ini_file = os.path.join(dataset_dir, seq, 'seqinfo.ini')
        fairmot_img_dir = os.path.join(fairmot_dir, 'images', dataset_type, seq, 'img1')
        fairmot_label_dir = os.path.join(fairmot_dir, 'labels_with_ids', dataset_type, seq, 'img1')

        if not os.path.exists(fairmot_img_dir):
            os.makedirs(fairmot_img_dir)
        if not os.path.exists(fairmot_label_dir):
            os.makedirs(fairmot_label_dir)

        # Read image dimensions from seqinfo.ini
        config = configparser.ConfigParser()
        config.read(ini_file)
        if 'Sequence' not in config:
            print(f"Warning: No 'Sequence' section in {ini_file}. Skipping this sequence.")
            continue
        frame_width = int(config['Sequence']['imWidth'])
        frame_height = int(config['Sequence']['imHeight'])

        images = sorted(os.listdir(img_dir))
        for img_name in images:
            img_path = os.path.join(img_dir, img_name)
            frame_id = int(img_name.split('.')[0])

            if dataset_type == 'train':
                if not os.path.isfile(gt_file):
                    print(f"Warning: Ground truth file does not exist at {gt_file}. Skipping this image.")
                    continue

                with open(gt_file, 'r') as f:
                    lines = f.readlines()

                frame_data = [line.strip().split(',') for line in lines if int(line.split(',')[0]) == frame_id]
                label_file_path = os.path.join(fairmot_label_dir, f'{img_name.replace(".jpg", ".txt")}')

                with open(label_file_path, 'w') as label_file:
                    for data in frame_data:
                        if float(data[6]) == 0:  # Skip detections with a confidence of zero
                            continue
                        x_center = (float(data[2]) + float(data[4])/2) / frame_width
                        y_center = (float(data[3]) + float(data[5])/2) / frame_height
                        width = float(data[4]) / frame_width
                        height = float(data[5]) / frame_height
                        label_file.write(f'0 {data[1]} {x_center} {y_center} {width} {height}\n')

            new_img_path = os.path.join(fairmot_img_dir, img_name)
            if not os.path.exists(new_img_path):  # Check to prevent overwriting existing files
                os.symlink(img_path, new_img_path)

# Usage for train dataset
convert_mot_to_fairmot('/work/aclab/bishoy/data/MOT17/', '/work/aclab/bishoy/post_data/MOT17/', 'train')

# Usage for test dataset
# convert_mot_to_fairmot('/work/aclab/bishoy/data/MOT15/', '/work/aclab/bishoy/post_data/MOT15_test', 'test')
