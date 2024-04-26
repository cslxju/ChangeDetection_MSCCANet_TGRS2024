## Citation
X. Zhang, L. Wang and S. Cheng, "A Multiscale Cascaded Cross-Attention Hierarchical Network for Change Detection on Bitemporal Remote Sensing Images," in IEEE Transactions on Geoscience and Remote Sensing, vol. 62, pp. 1-16, 2024
(https://github.com/cslxju/ChangeDetection_MSCCANet_TGRS2024)

## Requirements

- Python 3.6

- Pytorch 1.4

- torchvision 0.5.0


## Dataset

- [CDD](https://drive.google.com/file/d/1GX656JqqOyBi_Ef0w65kDGVto-nHrNs9/edit) (Change Detection Dataset)
- [LEVIR](https://justchenhao.github.io/LEVIR/), 
- [BCDD](https://study.rsgis.whu.edu.cn/pages/download/)
- Crop LEVIR and BCDD datasets into 256x256 patches. The pre-processed BCDD dataset can be obtained from [BCDD_256x256](https://drive.google.com/file/d/1VrdQ-rxoGVM_8ecA-ObO0u-O8rSTpSHA/view?usp=sharing).

- Prepare datasets into the following structure and set their path in `metadata.json`
    ```
    ├─Train
        ├─A        ...jpg/png
        ├─B        ...jpg/png
        ├─label    ...jpg/png
        └─list     ...txt
    ├─Val
        ├─A
        ├─B
        ├─label
        └─list
    ├─Test
        ├─A
        ├─B
        ├─label
        └─list
    ```

## Train from scratch

    python train_val.py
	
	

## Evaluate model performance

    python eval.py

## Visualization

    python visualization.py

