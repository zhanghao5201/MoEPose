# **Introduction**
Official code repository for the paper:
**D-Slope: Dual-Sample Learning for Occluded Human Pose Estimation**

# **Abstract**:

This paper presents D-Slope, a dual-sample learning approach for occluded human pose estimation to address the challenges of accurately estimating keypoint coordinates in scenarios where the human body is partially or fully occluded. D-Slope comprises two parallel networks for mutual learning, namely the general network (GNet) and the coupled expert network (ENet). The general network (GNet) processes the normal images and summarizes the learned knowledge, while the coupled expert network (ENet) is tailored specifically for handling occlusion. By performing mutual learning between these two networks, D-Slope is able to effectively address the issue of occlusion in human pose estimation.
To facilitate network learning, we construct paired samples comprising both normal and occluded images, resulting in the creation of two benchmarks for occluded human pose estimation: C-COCO and C-MPII. Additionally, due to its model-agnostic nature, D-Slope is applicable to a broad range of neural network models. Experimental results from synthetic and real-world benchmarks demonstrate that D-Slope offers significant improvements in handling occlusions. We further conduct experiments on face alignment and recognition tasks to illustrate the superior performance of D-Slope.

# **Visualization Results**:
![figure](figs/COCO.pdf)

# **Dataset**:
If you need to use the dataset we produced, please contact the author email address: zhanghao520@stu.xjtu.edu.cn. All commercial use of the dataset is prohibited, except for research work.

# **Acknowledgement**:
This project is developed based on the [HRFormer](https://github.com/HRNet/HRFormer) , [MMPOSE](https://github.com/open-mmlab/mmpose)
