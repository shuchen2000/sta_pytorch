import os

import numpy as np


def bitdepth_10to8(dir_10bit, dir_8bit):
    names = os.listdir(dir_10bit)
    for name in names:
        path_10bit = dir_10bit + "/" + name
        path_8bit = dir_8bit + "/" + name
        type = name.split(".")[1]
        if type == "yuv":
            w = name.split("_")[1].split("x")[0]
            h = name.split("_")[1].split("x")[1]
            vname = name.split("_")[0]
            if vname == "BasketballDrillText":
                continue
            cmd = "/home/shuchen/ffmpeg/ffmpeg-5.1.1-amd64-static/ffmpeg -s " + w + "x"+ h +" -pix_fmt yuv420p10le -i "+path_10bit+" -pix_fmt yuv420p "+ path_8bit
            print(cmd)
            os.system(cmd)


if __name__=="__main__":
    bitdepth_10to8(dir_10bit="/home/shuchen/datasets/ScreenContent/train_72/VTM_12.1_RA_10/QP27"
                   ,dir_8bit="/home/shuchen/datasets/ScreenContent/train_72/VTM_12.1_RA_8/QP27")
