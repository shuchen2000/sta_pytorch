# sta_pytorch

Official implementation of "Spatial-Temporal Adaptive Compressed Screen Content Video Quality Enhancement"

### Paper

The paper is accepted by ***IEEE Transactions on Circuits and Systems II: Express Briefs***, and you can find it on https://ieeexplore.ieee.org/document/10382566/ .

C. Shu, M. Ye, H. Guo and X. Li, "Spatial-Temporal Adaptive Compressed Screen Content Video Quality Enhancement," in IEEE Transactions on Circuits and Systems II: Express Briefs, doi: 10.1109/TCSII.2024.3350772.

### Dataset downloading

Train and test dataset (8-bits raw version and 10-bits compressed version) are available at https://pan.baidu.com/s/1Bqj_CXaThORriGsMoVaObw ,  extraction code: scve.

### Quick test:

You can use the following command to do a quick test with our provided well-trainned model parameters.

```python test_sta_10seqs.py```

> You need to set the path in `test_sta_10seqs.py` to the path where the dataset you downloaded is located.

> Remark: The model parameter files we provide are different from the checkpoint files, they only include parameters of the model, so the sizes of them are significantly smaller the checkpoint files.

Results are saved at `10seqs_results.log`.

### Dataset Preprocessing:
Before training, you need to first convert the downloaded YUV files into LMDB files. Our provided method for LMDB creating can only process the 8-bits videos, so you should firstly use function `bitdepth_10to8(dir_10bit,dir_8bit)` provided in `utils/bitdepth_10to8.py` to convert all videos in a folder `dir_10bit` containing 10-bits videos to 8-bits ones and save them as in another folder `dir_8bit`.

Then you can create LMDB files by using:

```python create_lmdb.py opt_path=create_lmdb_cfg.yml ```

> You need to set the path in yml file `create_lmdb_cfg.yml` to the path where the dataset you downloaded is located.

### Training
You can use the following command:

```python train_sta.py --opt_path=train_sta_cfg.yml ```

> You need to set the LMDB file path, checkpoints save path, result printing interval, etc. in the configuration file `train_sta_cfg.yml` according to your specific configuration and requirements.


