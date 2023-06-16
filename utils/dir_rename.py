import os

dir = "/home/shuchen/datasets/ScreenContent/train_72/RAW"

names = os.listdir(dir)
for name in names:
    vname = name.split(".yuv")[0]
    new_name = vname.split("_")[0] + "_" + vname.split("_")[1] + "x" + vname.split("_")[2] + "_" +vname.split("_")[3] + ".yuv"
    path = dir + "/" + name
    new_path = dir + "/" + new_name
    os.rename(path,new_path)

