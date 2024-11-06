# Gaussian Splatting论文阅读笔记

## Gaussian Splatting的整体流程解读
![image](https://github.com/user-attachments/assets/ce66fe71-31e7-4cbd-a9d5-dfa565a145b1)

**第一步**：我们需要通过colmap使用SFM算法估计出初始的3D点云sfm Points，假设初始点云有10000个点

**第二步**：我们需要将点云中的每个点膨胀成3D高斯椭球，这些椭球的初始形状：

- 是一个各向同性的高斯球，即一个圆球
- 使用knn法，找到3个邻近的点，然后将这三个点距离的平均作为高斯球的半径
- 然后每个椭球的参数包含：
  * 中心点位置： $(x, y, z)$
  * 协方差矩阵： $R, S$
  * 球谐函数系数： $16*3$
  * 透明度： $\ \alpha \$

**第三步**：

