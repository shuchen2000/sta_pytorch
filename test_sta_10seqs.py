import torch
import numpy as np
from collections import OrderedDict
import math
from sta_network import Net
import utils
from tqdm import tqdm
import glob
import os.path as op
import numpy as np
import time
#_woSaf

# 以下需要自定义修改

# 训练完模型sd所在的路径，文件名
sd_22_path = "./model_state_dicts/STA22_state_dict.pth"
sd_27_path = "./model_state_dicts/STA27_state_dict.pth"
sd_32_path = "./model_state_dicts/STA32_state_dict.pth"
sd_37_path = "./model_state_dicts/STA37_state_dict.pth"

# gt视频 yuv8bits
gt_dir = '/home/shuchen/datasets/ScreenContentDataset/RAW/test/8bit'

# lq视频 VTM12.1 yuv10bits
lq_22_dir = "/home/shuchen/datasets/ScreenContentDataset/LDP/test/VTM_10/22"
lq_27_dir = "/home/shuchen/datasets/ScreenContentDataset/LDP/test/VTM_10/27"
lq_32_dir = "/home/shuchen/datasets/ScreenContentDataset/LDP/test/VTM_10/32"
lq_37_dir = "/home/shuchen/datasets/ScreenContentDataset/LDP/test/VTM_10/37"
# 测试log文件路
log_fp = open("./10seqs_results.log", 'w')


QPS = [22, 27, 32, 37]
SD_PATHS = [sd_22_path, sd_27_path, sd_32_path, sd_37_path]
LQ_DIRS = [lq_22_dir, lq_27_dir, lq_32_dir, lq_37_dir]
GT_DIR = gt_dir
TEST_SEQ_NAME6 = ['WebBro','FSN_12','SlideE','ChinaS','JujuKa','Basket','SlideS','Map_12','HW_128','WordEd']
TEST_SEQ_SEL = [1,1,1,1,1,1,1,1,1,1]
TEST_SEQ_NAME6_SEL = []
for index in range(0,10):
    if TEST_SEQ_SEL[index]==1:
        TEST_SEQ_NAME6_SEL.append(TEST_SEQ_NAME6[index])


# 以上需要自定义修改


def test(raw_yuv_path,lq_yuv_path,sd_path,qp):
    model = Net(channels=64)  # 创建模型
    model.load_state_dict(torch.load(sd_path))
    model = model.cuda()
    model.eval()
    msg = "Load complete!"
    print(msg)
    torch.cuda.empty_cache()
    vname = raw_yuv_path.split("/")[-1].split('.')[0]
    _, wxh, nfs = vname.split('_')
    nfs = int(nfs)
    w, h = int(wxh.split('x')[0]), int(wxh.split('x')[1])  # 视频各个参数
    print(">>> [" + vname + "]  QP: " + str(qp))
    print("        raw video path: " + raw_yuv_path)
    print("        low quality yuv path: " + lq_yuv_path)
    print("    >>> Loading yuv videos (y only)")
    if not op.exists(lq_yuv_path):
        return "error"
    raw_y = utils.import_yuv(
        seq_path=raw_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=True, only_u=False, only_v=False
    )
    raw_y = raw_y.astype(np.float32) / 255.
    lq_y = utils.import_yuv(
        seq_path=lq_yuv_path, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=True, only_u=False, only_v=False, date_type=np.uint16
    )
    lq_y = lq_y.astype(np.float32) / 4 / 255.  # VTM专用
    print("            yuv load complete")  # 加载yuv文件为np.array

    ori_psnr_counter = utils.Counter()
    enh_psnr_counter = utils.Counter()
    ori_ssim_counter = utils.Counter()
    enh_ssim_counter = utils.Counter()

    lq_y = torch.from_numpy(lq_y)
    lq_y = torch.unsqueeze(lq_y, 0).cuda()

    enhanced = torch.from_numpy(np.zeros([1, nfs, 1, h, w]))

    print("    >>> Start enhancing")
    time.sleep(0.1)
    unit = 'dB'
    pbar = tqdm(total=nfs, ncols=110)
    pbar.set_description(
        "            Enhancing"
    )
    # 开始增强
    for index in range(0, nfs):  # 对每一帧进行增强
        torch.cuda.empty_cache()
        # 取5帧，支持帧越界则重复取中间帧
        indexes = [index - 2, index - 1, index, index + 1, index + 2]
        for i in range(0, 5):
            if indexes[i] < 0:
                indexes[i] = 0
            elif indexes[i] > (nfs - 1):
                indexes[i] = nfs - 1
        f2n = lq_y[:, indexes[0], :, :].unsqueeze(1)
        f1n = lq_y[:, indexes[1], :, :].unsqueeze(1)
        f0 = lq_y[:, indexes[2], :, :].unsqueeze(1)
        f1 = lq_y[:, indexes[3], :, :].unsqueeze(1)
        f2 = lq_y[:, indexes[4], :, :].unsqueeze(1)

        # 增强
        with torch.no_grad():
            f0_enhanced = model(f0=f0.cuda(), f1=f1.cuda(), f1n=f1n.cuda(), f2=f2.cuda(), f2n=f2n.cuda())
        # 消除异常值
        f0_enhanced[f0_enhanced > 1] = 1
        f0_enhanced[f0_enhanced < 0] = 0

        # 写入
        enhanced[0, index, 0, :, :] = f0_enhanced

        pbar.update()
    pbar.close()
    enhanced = np.float32(enhanced.cpu())
    print("            enhance complete")
    lq_y = np.float32(lq_y.cpu())
    print("    >>> start calculating PSNR and SSIM")
    pbar = tqdm(total=nfs, ncols=110)
    pbar.set_description(
        "            [{:.3f}] {:s} -> [{:.3f}] {:s}"
        .format(0., unit, 0., unit)
    )

    for idx in range(0, nfs):
        batch_ori = utils.calculate_psnr(lq_y[0, idx, ...], raw_y[idx], data_range=1.0)
        batch_perf = utils.calculate_psnr(enhanced[0, idx, 0, :, :], raw_y[idx], data_range=1.0)
        ssim_ori = utils.calculate_ssim(lq_y[0, idx, ...], raw_y[idx], data_range=1.0)
        ssim_perf = utils.calculate_ssim(enhanced[0, idx, 0, :, :], raw_y[idx], data_range=1.0)

        ori_psnr_counter.accum(volume=batch_ori)
        enh_psnr_counter.accum(volume=batch_perf)
        ori_ssim_counter.accum(volume=ssim_ori)
        enh_ssim_counter.accum(volume=ssim_perf)

        # display
        pbar.set_description(
            "            [{:.3f}] {:s} -> [{:.3f}] {:s}"
            .format(batch_ori, unit, batch_perf, unit)
        )
        pbar.update()
    pbar.close()

    ori_ = ori_psnr_counter.get_ave()
    enh_ = enh_psnr_counter.get_ave()
    ori_ssim = ori_ssim_counter.get_ave()
    enh_ssim = enh_ssim_counter.get_ave()
    msg = "    >>> [ RESULT ] QP: {:d}  VideoName {:s} ave: ori [{:.5f}] {:s}, enh [{:.5f}] {:s}, delta [{:.5f}] {:s}  ave ori_ssim [{:.5f}], enh_ssim [{:.5f}], delta_ssim [{:.4f}]".format(
        qp, vname, ori_, unit, enh_, unit, (enh_ - ori_), unit, ori_ssim, enh_ssim, (enh_ssim - ori_ssim)
    )
    print(msg)
    return msg + '\n'


def main():
    for i in [0,1,2,3]:  # 0:QP22 1:QP27 2:QP32 3:QP37
        qp = QPS[i]
        sd_path = SD_PATHS[i]
        gt_dir = GT_DIR
        lq_dir = LQ_DIRS[i]  # 获取路径等信息
        print("Start testing qp " + str(qp) + " model on class F")
        print("Start loading model from [" + sd_path + "]")
        msg = "loaded model from [" + sd_path + "]"
        log_fp.write(msg + '\n')
        log_fp.flush()
        
        gt_video_list = sorted(glob.glob(op.join(gt_dir, '*.yuv')))  # yuv路径列表(gt)

        for xxxx in range(len(gt_video_list)):
            raw_yuv_path = gt_video_list[xxxx]  # lq路径以gt路径为准
            lq_yuv_path = lq_dir + "/" + raw_yuv_path.split("/")[len(raw_yuv_path.split("/")) - 1]
            name6 = raw_yuv_path.split("/")[len(raw_yuv_path.split("/")) - 1][0:6]
            if name6 not in TEST_SEQ_NAME6_SEL:
                continue
            if not op.exists(lq_yuv_path):
                continue
            msg = test(raw_yuv_path,lq_yuv_path,sd_path,qp)
            print(msg)
            log_fp.write(msg + '\n')
            log_fp.flush()
            print("\n\n")


if __name__ == '__main__':
    main()
