# Mutual Information Treatment Network (ICML 2023)

Estimating Heterogeneous Treatment Effects: Mutual Information Bounds and Learning Algorithms [[paper]](https://openreview.net/pdf?id=DDwSa7XDxA)

Towards estimating heterogeneous treatment effect (HTE) for **general treatment spaces**, we propose:
- Theoretical bound under **mutual information** for estimating HTE.
- Theory-guided algorithm **MitNet** for practical usage.

<p align="center">
<img src=".\fig\model.png" height = "200" alt="" align=center />
<br><br>
<b>Figure 1.</b> Overview of MitNet.
</p>

## Get Started

1. Install Python 3.8. For convenience, execute the following command.

```bash
pip install -r requirements.txt
```

2. Prepare Data. You can obtain experimental datasets from the following links.

| Dataset | Link                                                                                                 |
|---------|------------------------------------------------------------------------------------------------------|
| TCGA    | [[Google Drive]](https://drive.google.com/file/d/1xSXs9hP8RvtmgeLBBLXejwD3reyuXNlm/view?usp=sharing) |

3. Train and evaluate model.

```bash
cd TCGA
bash run_mitnet.sh
```

## Results

We experiment on three benchmark datasets. MitNet reaches remarkable performance.

<p align="center">
<img src=".\fig\main_results.png" height = "350" alt="" align=center />
<br><br>
<b>Table 1.</b> Results for HTE estimation on IHDP, News and TCGA. A lower metric indicates better performance.
</p>

## Citation

If you find this repo useful, please cite our paper. 

```
@inproceedings{guo2023MitNet,
  title={Estimating Heterogeneous Treatment Effects: Mutual Information Bounds and Learning Algorithms},
  author={Xingzhuo Guo and Yuchen Zhang and Jianmin Wang and Mingsheng Long},
  booktitle={International Conference on Machine Learning},
  year={2023}
}
```

## Contact

If you have any questions or want to use the code, please contact [gxz23@mails.tsinghua.edu.cn](mailto:gxz23@mails.tsinghua.edu.cn).

## Acknowledgement

We appreciate the following github repos a lot for their valuable code base:

https://github.com/thuml/Transfer-Learning-Library

