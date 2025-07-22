

# <p align=center>IMM-MOT</p> 

![cover](pic/cover.png)

## 📢News

+ 🎉🎉🎉🎉 2025-07-22 Source code is released.
+ 🎉🎉🎉 2025-06-17 Accepted by IROS 2025.
+ 🎉🎉 2025-02-12 Submitted to IROS 2025
+ 🎉 2025-02-12 Submitted to [Arxiv](https://arxiv.org/abs/2502.09672).

## 🔻Abstract

This work introduces an improved 3D Multi-Object Tracking (MOT) method called IMM-MOT. To address the limitation of existing approaches that rely on a single motion model and struggle to adapt to complex object motion changes, IMM-MOT integrates the **Interacting Multiple Model (IMM) filter**. The IMM filter dynamically combines multiple motion models to more accurately fit an object's movement patterns. Additionally, a **Damping Window mechanism** is incorporated into trajectory lifecycle management, utilizing an object's continuous association status to control trajectory initiation and termination, thereby reducing missed detections of low-confidence true targets. Furthermore, a **Distance-Based Score Enhancement module** is proposed, which adjusts detection confidence scores to better distinguish between true and false positives, enhancing the effectiveness of the score filter. Collectively, these three innovations in IMM-MOT overcome the limitations of traditional methods, improving tracking accuracy and robustness.



## 🥟Results

All the results are tested on the NuScenes Dataset. We didn't make a large scale experiments on Test set because we can't reproduce the published result on Test set of Poly-MOT. And the suitable parameters need batches of experiments. So we just validate the IMM Module on the Test set using the reproduced result.

### Val set

| Method    | Detector    | AMOTA⬆   | AMOTP⬇   | IDS⬇    | FP⬇       | FN⬇       |
| --------- | ----------- | -------- | -------- | ------- | --------- | --------- |
| Poly-MOT  | CenterPoint | 73.1     | 51.7     | **232** | **13051** | 17593     |
| Fast-Poly | CenterPoint | 73.7     | 53.0     | 414     | 14713     | 15900     |
| IMM-MOT   | CenterPoint | **73.8** | **51.6** | 326     | 13433     | **15658** |

### Test set for IMM Module

| Method               | Detector    | Overall(AMOTA) | Bic. | Bus  | Car  | Motor. | Ped. | Tra. | Tru. |
| -------------------- | ----------- | -------------- | ---- | ---- | ---- | ------ | ---- | ---- | ---- |
| Poly-MOT(Reproduced) | LargeKernel | 74.2           | 54.5 | 76.7 | 85.6 | 79.3   | 81.2 | 75.4 | 66.6 |
| IMM-MOT              | LargeKernel | 74.9           | 54.6 | 77   | 86   | 79.6   | 84.1 | 75.8 | 67   |

### Ablation Study

| ID   | DBSE | IMM  | DW   | AMOTA⬆    | AMOTP⬇    | MT⬆      | ML⬇      | TP⬆       | FP⬇   | FN⬇       | IDS⬇    |
| ---- | ---- | ---- | ---- | --------- | --------- | -------- | -------- | --------- | ----- | --------- | ------- |
| 1    |      |      |      | 73.08     | 51.68     | 4670     | 1444     | 84072     | 13051 | 17593     | **232** |
| 2    | ✔    |      |      | 73.21     | **51.04** | 4746     | 1331     | 84654     | 12510 | 16942     | 301     |
| 3    |      | ✔    |      | 73.59     | 51.76     | 4690     | 1433     | 84303     | 13039 | 17342     | 252     |
| 4    |      |      | ✔    | 73.26     | 52.27     | 4773     | 1292     | 85161     | 12669 | 16417     | 319     |
| 5    |      | ✔    | ✔    | 73.66     | 52.29     | 4808     | 1252     | 85722     | 13146 | 15850     | 325     |
| 6    | ✔    | ✔    |      | 73.72     | 51.05     | 4705     | 1427     | 84521     | 13384 | 17124     | 252     |
| 7    | ✔    |      | ✔    | 73.34     | 51.53     | 4783     | 1295     | 85234     | 12850 | 16343     | 320     |
| 8    | ✔    | ✔    | ✔    | **73.79** | 51.61     | **4827** | **1246** | **85913** | 13433 | **15658** | 326     |



## 📃Usage

>  **📢Notice**: The implement of IMM-MOT project is partly based on [Poly-MOT](https://github.com/lixiaoyu2000/Poly-MOT). What we did is to expand some modules. So the entire usage, including the structure and config file, is similar to [Poly-MOT](https://github.com/lixiaoyu2000/Poly-MOT).
>
> The main changes are as follows:
>
> 1. We add DBSE module in the  `IMM/pre_processing/dbss.py`, and you can find the entry in the `IMM/dataloader/nusc_loader.py`.
> 2. We add IMM module in the `IMM/motion_module/kalman_filter.py`. 
> 3. We add DW module in the `IMM/tracking/nusc_score_manage.py`. 
> 4. All the changes are presented in the configure file `IMM/config/nusc_config.yaml`.
> 5. We set different parameters for different object type. Check the code for more details.



#### 1. Create and activate virtual environment

```bash
conda env create -f environment.yaml
conda activate imm
```

#### 2. Data preparation

First download the [NuScenes Dataset](https://www.nuscenes.org/nuscenes#download).

Our work uses 3D detector (especially Lidar-only detector) and please download the detection file in the corresponding website like [CenterPoint](https://github.com/tianweiy/CenterPoint) and [Largekernel](https://github.com/dvlab-research/LargeKernel3D). The CenterPoint detection file we use is downloaded [here](https://drive.google.com/drive/folders/1oGgi2RXZWnVJeCDK9G4PKaYNMjT2V6BO). Note that CenterPoint for NuScenes Val set and Largekernel for Test set. 

Then run the following instruction to exact the first frame table:

```bash
cd IMM/data/script
python first_frame.py
```

The result will be output in `IMM/data/utils/first_token_table/{version}/nusc_first_token.json`.

Then run the following instructions to reorder the detection file with the first token table we have got above.

```bash
cd IMM/data/script
python reorder_detection.py
```

The result will be output in `IMM/data/detector`.

> 📢📢📢 Before running all the instructions please check all the paths and ensure they are correct.

#### 3. Running and Evaluation

All the parameters are presented in `config/nusc_config.yaml`. Change the value to customize the tracker.

```BASH
python test.py
```

Run the `test.py` to start the tracking process, and the evaluation stage will start directly after tracking.

## Contact

If you have any suggestions or questions, Please submit the Issue or contact us with email.

+ Xulong Zhao: apollo.xlzhao@gmail.com 
+ Xiaohong Liu: xiaohongl@stu.xidian.edu.cn

## Acknowledgement

This project is not possible without the following excellent open-source codebases ✊.

+ [PolyMOT](https://github.com/lixiaoyu2000/Poly-MOT)
+ [CenterPoint](https://github.com/tianweiy/CenterPoint)
+ [Largerkernel3D](https://github.com/dvlab-research/LargeKernel3D)

## Citation

If you find this project useful in your research, please cite us by: 

```bibtex
@article{liu2025imm,
  title={IMM-MOT: A Novel 3D Multi-object Tracking Framework with Interacting Multiple Model Filter},
  author={Liu, Xiaohong and Zhao, Xulong and Liu, Gang and Wu, Zili and Wang, Tao and Meng, Lei and Wang, Yuhan},
  journal={arXiv preprint arXiv:2502.09672},
  year={2025}
}
```

## License

IMM-MOT is released under the MIT license.