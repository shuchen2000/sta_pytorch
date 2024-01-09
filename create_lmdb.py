"""Create LMDB only for training set of MFQEv2.

GT: non-overlapping 7-frame sequences extracted from 108 videos.
LQ: HM16.5-compressed sequences.
key: assigned from 0000 to 9999.

NOTICE: MAX NFS OF LQ IS 300!!!

Sym-link MFQEv2 dataset root to ./data folder."""
import argparse
import os
import glob
import yaml
import os.path as op
from utils import make_y_lmdb_from_yuv

parser = argparse.ArgumentParser()
parser.add_argument(
    '--opt_path', type=str, default='option_R3_mfqev2_1D.yml',
    help='Path to option YAML file.'
    )
args = parser.parse_args()

yml_path = args.opt_path

# 2*2+1=5
radius = 2


def create_lmdb_for_mfqev2():
    # video info
    with open(yml_path, 'r') as fp:
        fp = yaml.load(fp, Loader=yaml.FullLoader)
        root_dir = fp['dataset']['train']['root']
        gt_folder = fp['dataset']['train']['gt_folder']
        lq_folder = fp['dataset']['train']['lq_folder']
        gt_path = fp['dataset']['train']['gt_path']
        lq_path = fp['dataset']['train']['lq_path']
        gt_valid = fp['dataset']['train']['gt_valid']
        lq_valid = fp['dataset']['train']['lq_valid']
    gt_dir = op.join(root_dir, gt_folder)
    lq_dir = op.join(root_dir, lq_folder)
    lmdb_gt_path = op.join(root_dir, gt_path)
    lmdb_lq_path = op.join(root_dir, lq_path)

    # scan all videos
    print('Scaning videos...')
    gt_video_list = sorted(glob.glob(op.join(gt_dir, '*.yuv')))
    #lq_video_list = sorted(glob.glob(op.join(lq_dir, '*.yuv')))
    lq_video_list = [op.join(
        lq_dir,
        gt_video_path.split('/')[-1]
        ) for gt_video_path in gt_video_list]
    msg = f'> {len(gt_video_list)} videos found.'
    print(msg)

    # generate LMDB for GT
    print("Scaning GT frames (only center frames of each sequence)...")
    frm_list = []
    for gt_video_path in gt_video_list:
        nfs = int(gt_video_path.split('.')[-2].split('/')[-1].split('_')[-1])
        nfs = nfs if nfs <= 300 else 300  # !!!!!!
        num_seq = nfs // (2 * radius+1)
        # frm_list.append([radius + iter_seq * (2 * radius + 1) for iter_seq in range(num_seq)])

        frm_list.append([list(range(iter_seq * (2 * radius+1), (iter_seq + 1) \
                                    * (2 * radius+1))) for iter_seq in range(num_seq)])

    num_frm_total = sum([len(frms)*(2 * radius+1) for frms in frm_list])
    msg = f'> {num_frm_total} frames found.'
    print(msg)



    key_list = []
    video_path_list = []
    video_lq_list = []
    index_frame_list = []
    for iter_vid in range(len(gt_video_list)):
        frms = frm_list[iter_vid]
        for iter_frm in range(len(frms)):
            key_list.extend(['{:03d}/{:03d}/im{:d}.png'.format(iter_vid + 1, \
                                                               iter_frm + 1, i) for i in range(1, (2 * radius+1) + 1)])
            video_path_list.extend([gt_video_list[iter_vid]]* (2 * radius+1))
            video_lq_list.extend([lq_video_list[iter_vid]] * (2 * radius+1))
            index_frame_list.extend(frms[iter_frm])

    if gt_valid:
        print("Writing LMDB for GT data...")
        make_y_lmdb_from_yuv(
            video_path_list=video_path_list,
            index_frame_list=index_frame_list,
            key_list=key_list,
            lmdb_path=lmdb_gt_path,
            multiprocessing_read=True,
            )

    if lq_valid:
        print("Writing LMDB for LQ data...")
        make_y_lmdb_from_yuv(
            video_path_list=video_lq_list,
            index_frame_list=index_frame_list,
            key_list=key_list,
            lmdb_path=lmdb_lq_path,
            multiprocessing_read=True,
            )
    print("> Finish.")


if __name__ == '__main__':
    create_lmdb_for_mfqev2()
