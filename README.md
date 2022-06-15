# <div align="center">Semantic Segmentation on AI4MARS</div>

We use BiSeNet-v1 with ResNet-18 as backbone pretrained on ImageNet.


## <div align="center">Usage</div>

<details open>
  <summary><strong>Installation</strong></summary>

* python >= 3.6
* torch >= 1.8.1
* torchvision >= 0.9.1

Then, clone the repo and install the project with:

```bash
$ git clone 
$ cd semantic-segmentation
$ pip install -e .
```

</details>

<br>
<details>
  <summary><strong>Configuration</strong> (click to expand)</summary>

Modify the configuration file  `configs` [here](configs/ai4mars.yaml). Move the msl folder from ai4mars dataset to [data folder](data) or change the field of dataset path in `config` file. Set the `DEVICE` type and the `SAVE_DIR` parameters This configuration file is needed for all of training, evaluation and inference scripts.

</details>

<br>
<details>
  <summary><strong>Training</strong> (click to expand)</summary>

To train:
 * download [weights](https://drive.google.com/drive/folders/1MXP3Qx51c91PL9P52Tv89t90SaiTYuaC) and move it to [checkpoint path](checkpoints/backbones).
 * Run:

```bash
$ python train.py
```

</details>

<br>
<details>
  <summary><strong>Fine-tuning</strong> (click to expand)</summary>

To fine-tune or resume training:
* Move the saved checkpoint to [checkpoint path](checkpoints/backbones).
* Change `TRAIN` >> `RESUME_TRAIN` in configuration file to true.
* Set `TRAIN` >> `PRETRAINED`.
* Run:

```bash
$ python train.py
```

</details>

<br>
<details>
  <summary><strong>Evaluation</strong> (click to expand)</summary>
To evaluate:
* Set `EVAL` >> `MODEL_PATH` of the configuration file to your trained model directory.
* Run:
```bash
$ python val.py
```


</details>

<br>
<details open>
  <summary><strong>Inference</strong></summary>

To make an inference, there is an image from AI4MARS and another from PANCAM with their labels in [assests folder](assests).
* Set `TEST` >> `MODEL_PATH` to pretrained weights of the testing model.
To add extra imagesfor inference:
If the image has label:
* Move the image and its label to [labeled folder](assests/labeled).
* The names of the image and label should be like name.JPG and name_merged.png respectively.
If the image is unlabeled:
* Move the image to [unlabeled folder](assests/unlabeled).
* The name of the image should be like name.JPG
The results will be in new folder `test_results`

```bash
$ python infer.py
```


</details>

