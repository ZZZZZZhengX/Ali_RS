# Ali-RS_Semantic_Segmentation

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cmx-cross-modal-fusion-for-rgb-x-semantic/semantic-segmentation-on-rsmss)](https://paperswithcode.com/sota/semantic-segmentation-on-rsmss?p=cmx-cross-modal-fusion-for-rgb-x-semantic)

![Example segmentation](segmentation.jpg?raw=true "Example segmentation")


## Usage
### Installation
1. Requirements

- Python 3.7+
- PyTorch 1.7.0 or higher
- CUDA 10.2 or higher

We have tested the following versions of OS and softwares:

- OS: Ubuntu 18.04.6 LTS
- CUDA: 10.2
- PyTorch 1.8.2
- Python 3.8.11

2. Install all dependencies.
Install pytorch, cuda and cudnn, then install other dependencies via:
```shell
pip install -r requirements.txt
```

### Datasets

Orgnize the dataset folder in the following structure:
```shell
<datasets>
|-- <DatasetName1>
    |-- <RGBFolder>
        |-- <name1>.<ImageFormat>
        |-- <name2>.<ImageFormat>
        ...
    |-- <NirFolder>
        |-- <name1>.<ModalXFormat>
        |-- <name2>.<ModalXFormat>
        ...
    |-- <LabelFolder>
        |-- <name1>.<LabelFormat>
        |-- <name2>.<LabelFormat>
        ...
    |-- train.txt
    |-- test.txt
|-- <DatasetName2>
|-- ...
```

`train.txt` contains the names of items in training set, e.g.:
```shell
<name1>
<name2>
...
```


### Train
1. Pretrain weights:

    Download the pretrained segformer here [pretrained segformer](https://drive.google.com/drive/folders/10XgSW8f7ghRs9fJ0dE-EV8G2E_guVsT5?usp=sharing).

2. Config

    Edit config file in `configs.py`, including dataset and network settings.

3. Run multi GPU distributed training:
    ```shell
    $ CUDA_VISIBLE_DEVICES="GPU IDs" python -m torch.distributed.launch --nproc_per_node="GPU numbers you want to use" train.py
    ```

- The tensorboard file is saved in `log_<datasetName>_<backboneSize>/tb/` directory.
- Checkpoints are stored in `log_<datasetName>_<backboneSize>/checkpoints/` directory.

### Evaluation
Run the evaluation by:
```shell
CUDA_VISIBLE_DEVICES="GPU IDs" python eval.py -d="Device ID" -e="epoch number or range"
```
If you want to use multi GPUs please specify multiple Device IDs (0,1,2...).


## Result
We offer the pre-trained weights on different RGBX datasets (Some weights are not avaiable yet, Due to the difference of training platforms, these weights may not be correctly loaded.):

