# D_Slope
Introduction
Official code repository for the paper:
D-Slope: Dual-Sample Learning for Occluded Human Pose Estimation

Abstract
This paper presents D-Slope, a dual-sample learning approach for occluded human pose estimation to address the challenges of accurately estimating keypoint coordinates in scenarios where the human body is partially or fully occluded. D-Slope comprises two parallel networks for mutual learning, namely the general network (GNet) and the coupled expert network (ENet). The general network (GNet) processes the normal images and summarizes the learned knowledge, while the coupled expert network (ENet) is tailored specifically for handling occlusion. By performing mutual learning between these two networks, D-Slope is able to effectively address the issue of occlusion in human pose estimation.
To facilitate network learning, we construct paired samples comprising both normal and occluded images, resulting in the creation of two benchmarks for occluded human pose estimation: C-COCO and C-MPII. Additionally, due to its model-agnostic nature, D-Slope is applicable to a broad range of neural network models. Experimental results from synthetic and real-world benchmarks demonstrate that D-Slope offers significant improvements in handling occlusions. We further conduct experiments on face alignment and recognition tasks to illustrate the superior performance of D-Slope.



Usage
Install
Install mmpose.
run python setup.py develop.
Training
You can follow the guideline of mmpose.

Train with multiple GPUs
./tools/dist_train.sh ${CONFIG_FILE} ${GPU_NUM} [optional arguments]
Train with multiple machines
If you can run this code on a cluster managed with slurm, you can use the script slurm_train.sh. (This script also supports single machine training.)

./tools/slurm_train.sh ${PARTITION} ${JOB_NAME} ${CONFIG_FILE} ${WORK_DIR}
Here is an example of using 16 GPUs to train POMNet on the dev partition in a slurm cluster. (Use GPUS_PER_NODE=8 to specify a single slurm cluster node with 8 GPUs, CPUS_PER_TASK=2 to use 2 cpus per task. Assume that Test is a valid ${PARTITION} name.)

GPUS=16 GPUS_PER_NODE=8 CPUS_PER_TASK=2 ./tools/slurm_train.sh Test pomnet \
  configs/mp100/pomnet/pomnet_mp100_split1_256x256_1shot.py \
  work_dirs/pomnet_mp100_split1_256x256_1shot
MP-100 Dataset


Terms of Use
The dataset is only for non-commercial research purposes.
All images of the MP-100 dataset are from existing datasets (COCO, 300W, AFLW, OneHand10K, DeepFashion, AP-10K, MacaquePose, Vinegar Fly, Desert Locust, CUB-200, CarFusion, AnimalWeb, Keypoint-5), which are not our property. We are not responsible for the content nor the meaning of these images.
We provide the annotations for training and testing. However, for legal reasons, we do not host the images. Please follow the guidance to prepare MP-100 dataset.
Citation
@article{xu2022pose,
  title={Pose for Everything: Towards Category-Agnostic Pose Estimation},
  author={Xu, Lumin and Jin, Sheng and Zeng, Wang and Liu, Wentao and Qian, Chen and Ouyang, Wanli and Luo, Ping and Wang, Xiaogang},
  booktitle={European Conference on Computer Vision (ECCV)},
  year={2022},
  month={October}
}
Acknowledgement
Thanks to:

MMPose
License
This project is released under the Apache 2.0 license.
