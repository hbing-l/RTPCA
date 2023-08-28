# Refined Temporal Pyramidal Compression-and-Amplification Transformer for 3D Human Pose Estimation

## Environment

The code is conducted under the following environment:

* Ubuntu 18.04
* Python 3.6.10
* PyTorch 1.8.1
* CUDA 10.2

You can create the environment as follows:

```bash
conda env create -f requirements.yaml
```

## Dataset

The Human3.6M dataset and HumanEva dataset setting follow the [VideoPose3D](https://github.com/facebookresearch/VideoPose3D).
Please refer to it to set up the Human3.6M dataset (under ./data directory).

The MPI-INF-3DHP dataset setting follows the [MMPose](https://github.com/open-mmlab/mmpose).
Please refer it to set up the MPI-INF-3DHP dataset (also under ./data directory).

# Evaluation

* Download the checkpoints from [Baidu Disk](https://pan.baidu.com/s/1pu2C7hobuA8mYRWtJ2Bgdg)(54d5);

Then run the command below (evaluate on 243 frames input):

> python run.py -k gt -c <checkpoint_path> --evaluate <checkpoint_file> -f 243 -s 243

# Training from scratch

Training on the 243 frames with two GPUs:

>  python run.py -k gt -f 243 -s 243 -l log/run -c checkpoint -gpu 0,1

## Acknowledgement

Thanks for the baselines, we construct the code based on them:

* VideoPose3D
* SimpleBaseline
