# Gaussian Splatting论文阅读笔记

## Gaussian Splatting的整体流程解读

首先先用一句话来概括3D高斯：从已有的点云模型出发，以每个点为中心，建立可学习的3D高斯表达，并用Splatting也即抛雪球的方法进行渲染。

![image](https://github.com/user-attachments/assets/ce66fe71-31e7-4cbd-a9d5-dfa565a145b1)

**第一步**：我们需要输入多视角的图片，然后通过colmap使用SFM算法估计出初始的3D点云sfm Points，假设初始点云有10000个点。

**第二步**：我们需要将点云中的每个点膨胀成3D高斯椭球，这些椭球的初始形状：

- 是一个各向同性的高斯球，即一个圆球
- 使用knn法，找到3个邻近的点，然后将这三个点距离的平均作为高斯球的半径
- 然后每个椭球的参数包含：
  * 中心点位置： $(x, y, z)$
  * 协方差矩阵(控制椭球形状)： $R, S$
  * 球谐函数系数(控制椭球颜色)： $16*3$
  * 透明度： $\ \alpha \$

**第三步**：通过渲染(Splatting)将所有的高斯球投影到2D图像平面上。

**第四步**：通过快速可微光栅化器渲染图片。

**第五步**：将生成的图像和GT图去做loss，再反向传播回来优化每个椭球的参数。

**第六步**：作者还根据反向传播回来的梯度信息去做了一个自适应密度控制，对应pipline中的(Adaptive Control of Gaussian)

- 简单来说，就是
  * 太大的椭球进行拆分
  * 太小的椭球进行克隆
  * 存在感(透明度)太低的椭球进行删除

![image](https://github.com/user-attachments/assets/dec0c859-6aa6-4b67-b1b6-20206cc15b1f)

---

看完上面这个流程，相信大家脑海里面浮现出了许多问题❓：


为了方便探究问题的细节，我们可以将pipline划分为三个核心的部分：

- 捏雪球
- 抛雪球：从3D投影到2D，得到足迹
- 加以合成，形成最后的图像

![image](https://github.com/user-attachments/assets/c0f8229a-4329-4167-9181-fd985389bea8)

---
## 捏雪球过程我们需要解决的问题

**问题1：根据我们前面所说：我们实际输入的是一些初始的点云，这些点本质上来说是没有体积的，所以我们需要选择一个核对这些点进行膨胀。那么问题来了：这个核我们可以选择高斯/圆/正方体，这里作者为什么选择高斯？**

回答：我们先来回顾一下高斯分布：

**一维高斯分布**

对于一维高斯分布，协方差矩阵 $\Sigma$ 简化为一个标量 $\sigma$ ,均值 $\mu$ 也是一个标量，公式如下： 

$$G(x) = \frac{1}{\sqrt{2\pi \sigma^2}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}$$

其中：

- $x$ 是变量
- $\mu$ 是均值
- $\sigma^2$ 是方差

**二维高斯分布**

对于二维高斯分布，协方差矩阵 $\Sigma$ 是 $2*2$ 的矩阵，均值 $\mu$ 是一个二维向量 $(\mu_1, \mu_2)$ 。公式如下：

$$G(\mathbf{x}) = \frac{1}{2\pi |\Sigma|^{1/2}} e^{-\frac{1}{2} (\mathbf{x} - \mu)^T \Sigma^{-1} (\mathbf{x} - \mu)}$$

其中：

- $\mathbf{x}$ 是一个二维向量 $(x_1, x_2)$
- $\mu$ 是均值向量 $(\mu_1, \mu_2)$
- $\Sigma$ 是协方差矩阵

$$\Sigma = \begin{pmatrix}
\sigma_{11} & \sigma_{12} \\
\sigma_{21} & \sigma_{22}
\end{pmatrix}$$

- $|\Sigma|$是协方差矩阵的行列式

**三维高斯分布**

对于三维高斯分布，协方差矩阵 $\Sigma$ 是 $3*3$ 的矩阵，均值 $\mu$ 是一个三维向量 $(\mu_1, \mu_2, \mu3)$ 。公式如下：

$$G(\mathbf{x}) = \frac{1}{(2\pi)^{3/2} |\Sigma|^{1/2}} e^{-\frac{1}{2} (\mathbf{x} - \mu)^T \Sigma^{-1} (\mathbf{x} - \mu)}$$

其中：

- $\mathbf{x}$ 是一个三维向量 $(x_1, x_2, x_3)$
- $\mu$ 是均值向量 $(\mu_1, \mu_2, \mu_3)$
- $\Sigma$ 是协方差矩阵

$$\Sigma = \begin{pmatrix}
\sigma_{11} & \sigma_{12} & \sigma_{13} \\
\sigma_{21} & \sigma_{22} & \sigma_{23} \\
\sigma_{31} & \sigma_{32} & \sigma_{33}
\end{pmatrix}$$

- $|\Sigma|$是协方差矩阵的行列式

**对协方差矩阵的解释**

- 是一个对称矩形，决定高斯分布的形状
- 对角线上元素为x轴/y轴/z轴的方差
- 反斜对角线上的值为协方差，
  * 表示x和y，x和z，y和z的线性相关程度

作者之所以选择3D高斯作为核来对点进行膨胀是因为3D高斯具有很好的数学性质：

- 经过仿射变换后高斯核仍然闭合
- 沿着某一个轴积分，从3D降到2D后依然为高斯

所以作者选择用3D高斯作为核

**问题2：我们选择3D高斯核对点进行膨胀，最终会得到一个椭球。那么问题来了：3D高斯表示出来为什么是一个椭球呢？**

回答：我们先来复习一下椭球的相关公式

$\frac{x^2}{a^2} + \frac{y^2}{b^2} + \frac{z^2}{c^2} = 1$ 表示中心点为 $(0,0,0)$ 的标准椭球公式

$A x^2 + B y^2 + C z^2 + 2 D x y + 2 E x z + 2 F y z = 1$ 表示引入了旋转和偏移后的椭球一般公式

然后我们再结合高斯分布去看

当一维高斯分布 $G(x) = \frac{1}{\sqrt{2\pi \sigma^2}} e^{-\frac{(x - \mu)^2}{2\sigma^2}}$ 中的指数部分为常数时

即 $-\frac{(x - \mu)^2}{2\sigma^2}=constant$ 时,G(x)表示一条等值线。

![Figure_1](https://github.com/user-attachments/assets/3b1119ee-2e31-43ca-bb59-5e355a11af19)

当二维高斯分布 $G(\mathbf{x}) = \frac{1}{2\pi |\Sigma|^{1/2}} e^{-\frac{1}{2} (\mathbf{x} - \mu)^T \Sigma^{-1} (\mathbf{x} - \mu)}$ 中的指数部分为常数时，

即： $-\frac{1}{2} (\mathbf{x} - \mu)^T \Sigma^{-1} (\mathbf{x} - \mu)=constant$ 时

展开可得： $\frac{(x - \mu_1)^2}{\sigma_1^2} + \frac{(y - \mu_2)^2}{\sigma_2^2} - \frac{2 \sigma_{xy} (x - \mu_1)(y - \mu_2)}{\sigma_1 \sigma_2} = \text{constant}$

表示的是一个椭圆（即常数越大，椭圆越小），这种椭圆通常被称为等概率密度轮廓线，即同一个椭圆上的每一点都是等概率的

![Figure_2](https://github.com/user-attachments/assets/e8246202-bf62-430c-863c-9ea419115cd5)

当三维高斯分布 $G(\mathbf{x}) = \frac{1}{(2\pi)^{3/2} |\Sigma|^{1/2}} e^{-\frac{1}{2} (\mathbf{x} - \mu)^T \Sigma^{-1} (\mathbf{x} - \mu)}$ 中的指数部分为常数时：

即： $-\frac{1}{2} (\mathbf{x} - \mu)^T \Sigma^{-1} (\mathbf{x} - \mu)=constant$ 时

展开可得： 

$$\frac{(x - \mu_1)^2}{\sigma_1^2} + \frac{(y - \mu_2)^2}{\sigma_2^2} + \frac{(z - \mu_3)^2}{\sigma_3^2} - \frac{2 \sigma_{xy} (x - \mu_1)(y - \mu_2)}{\sigma_1 \sigma_2} - \frac{2 \sigma_{xz} (x - \mu_1)(z - \mu_3)}{\sigma_1 \sigma_3} - \frac{2 \sigma_{yz} (y - \mu_2)(z - \mu_3)}{\sigma_2 \sigma_3}=constant$$



表示的是一个椭球面，这种椭球面称为等概率密度曲面，同一椭球曲面上的点具有相同的概率密度值。然后多个椭球面构成了一个实心的椭球。所以3D高斯表示的是一个实心的椭球。

![Figure_3](https://github.com/user-attachments/assets/cc7e0107-7169-4513-8dea-e184c6b3c93d)

**问题3：3D高斯椭球中的各向异性和各向同性是什么意思？**

回答：

- 各向同性：
  * 在所有方向上都具有相同的扩散程度
  * 表示的是一个圆球
  * 协方差矩阵是一个对角矩阵
 
$$\Sigma = \begin{bmatrix} 
\sigma^2 & 0 & 0 \\ 
0 & \sigma^2 & 0 \\ 
0 & 0 & \sigma^2 
\end{bmatrix}$$

- 各向异性：
  * 在不同方向具有不同的扩散程度(梯度)
  * 表示的是一个椭球
  * 协方差矩阵是对角矩阵

$$\Sigma = \begin{bmatrix} 
\sigma_x^2 & \sigma_{xy} & \sigma_{xz} \\ 
\sigma_{yx} & \sigma_y^2 & \sigma_{yz} \\ 
\sigma_{zx} & \sigma_{zy} & \sigma_z^2 
\end{bmatrix}$$

**问题4：我们从前面可知：椭球的形状是通过协方差矩阵去控制的，那么问题来了：协方差矩阵为什么可以控制椭球的形状？**

回答：假设我们有一个标准的高斯分布：

- $\mathbf{x} \sim \mathcal{N}(\vec{0}, I)$
- 均值 $[0,0,0]$
- 协方差矩阵

$$I = \begin{bmatrix} 
1 & 0 & 0 \\ 
0 & 1 & 0 \\ 
0 & 0 & 1
\end{bmatrix}$$

然后我们对这个标准的高斯分布进行仿射变换

- $w=A \mathbf{x} +b$
- $\mathbf{w} \sim \mathcal{N}(A \mu + b, A \cdot I \cdot A^T)$

其中 $\Sigma=A \cdot I \cdot A^T$ 表示**任意高斯都可以看作是标准高斯通过仿射变换得到**，即任意椭球可以看作是球通过仿射变换得到的，所以说协方差矩阵控制了椭球的形状。

**问题5：协方差矩阵为什么能用旋转和缩放矩阵去表达？**

回答：由我们前面可知，对一个高斯分布进行仿射变换

- $w=A \mathbf{x} +b$
- $\mathbf{w} \sim \mathcal{N}(A \mu + b, A \cdot \Sigma \cdot A^T)$

根据仿射变换的定义可知，这个A的本质就是一个旋转矩阵乘上一个缩放矩阵，即 $A=R \cdot S$

代入到 $\Sigma=A \cdot I \cdot A^T$ 可得：

$$\Sigma = A \cdot I \cdot A^T \\
= R \cdot S \cdot I \cdot (R \cdot S)^T \\
= R \cdot S \cdot (S)^T \cdot (R)^T$$

所以协方差矩阵为什么能用旋转和缩放矩阵去表达

**好了，到目前为止，我们终于是成功把雪球捏出来了🎉🎉🎉**

下一步，我们就要去抛雪球了

---
## 抛雪球过程我们需要解决的问题

抛雪球就是从一个3D到像素的一个过程。在抛雪球的过程中，我们主要会涉及到下面这几个坐标系的变换，具体细节可以参考[GAMES101-现代计算机图形学入门-闫令琪](https://www.bilibili.com/video/BV1X7411F744/?spm_id_from=333.337.search-card.all.click&vd_source=1a02178b1644ddc9b579739c3c1616b4)的课程讲解

- 观测变换
- 投影变换
- 视口变换
- 光栅化

**观测变换**

- 从世界坐标系到相机坐标系的变换 
- 实质上是仿射变换
- $w=Ax+b$

**3D高斯中的观测变换：**

物理坐标系
- 高斯核中心 $t_k = [t_0 \quad t_1 \quad t_2]^T$
- $V''_k$ 是协方差矩阵

经过一个仿射变换 $u = \varphi(t) = Wt + d$ 从物理坐标系转换到相机坐标系

相机坐标系
- 均值 $u_k = W t_k + d$
- 高斯核中心 $u_k = [u_0 \quad u_1 \quad u_2]^T$
- 协方差矩阵 $V'_k = W V''_k W^T$

**投影变换**

- 从相机坐标系到2D像素平面
- 正交投影，与z无关(无远小近大的关系)
- 透视投影，与z相关(有远小近大关系)

![image](https://github.com/user-attachments/assets/0d9278d8-21f2-4395-83ec-5a7869709907)

*正交投影*

![image](https://github.com/user-attachments/assets/127ba6d5-d63a-466c-824a-7a75c2424690)

在正交投影中，视锥体是一个立方体，用 $[l,r] * [b,t] * [f,n]$ 去表示，我们需要将这个正方体平移到坐标系的原点，然后缩放至 $[-1, 1]^3$ 的正方体。整个过程是一个仿射变换的过程，仿射变换的矩阵如下所示：

$$M_{\text{ortho}} = \begin{bmatrix} 
\frac{2}{r - l} & 0 & 0 & 0 \\ 
0 & \frac{2}{t - b} & 0 & 0 \\ 
0 & 0 & \frac{2}{n - f} & 0 \\ 
0 & 0 & 0 & 1 
\end{bmatrix}
\begin{bmatrix} 
1 & 0 & 0 & -\frac{r + l}{2} \\ 
0 & 1 & 0 & -\frac{t + b}{2} \\ 
0 & 0 & 1 & -\frac{n + f}{2} \\ 
0 & 0 & 0 & 1 
\end{bmatrix}$$

*透视投影*

![image](https://github.com/user-attachments/assets/c6a5afc8-8caf-493a-9d0f-c82a13944970)

因为透视投影考虑了近大远小的关系，所以会比正交投影稍微复杂一点。首先我们要先把视锥体"压缩"成一个立方体，变换矩阵如下所示，然后我们再去进行正交投影。

$$M_{\text{persp} \rightarrow \text{ortho}} = \begin{bmatrix} 
n & 0 & 0 & 0 \\ 
0 & n & 0 & 0 \\ 
0 & 0 & n + f & -nf \\ 
0 & 0 & 1 & 0 
\end{bmatrix}$$

**3D高斯中的投影变换**

相机坐标系:
- 高斯核中心 $u_k=[u_0, u_1, u_2]$
- $V'_k$ 是协方差矩阵

经过一个投影变换 $x=m(t)$ 从相机坐标系转换到像素坐标系：

- 均值 $x_k=m(u_k)$
- 高斯核中心 $x_k=[x_0, x_1, x_2]^T$
- **但是在对协方差矩阵进行处理时我们就遇到了问题**

采用透视投影处理协方差矩阵时我们会遇到一个问题：透视投影是非线性的，即非仿射变换，如下图所示。非仿射变换是不可以用于处理协方差矩阵的，这个问题要怎么解决？

![image](https://github.com/user-attachments/assets/ac8f45d8-53a8-4683-9b27-6bd0704ec91e)

**作者为了解决这个问题引入了雅可比矩阵**

我们先举一个例子来了解一下什么是雅可比矩阵：

设想我们对二维空间中的某个点 $(x,y)$ 进行非线性变换，即：

$$f_1(x,y)=x+sin(y)$$
$$f_2(x,y)=y+sin(x)$$

这个变换在整个空间中是非线性的, $sin(x)$ 和 $sin(y)$ 的引入使得网格会产生不同程度的拉伸，扭曲和压缩，如下图所示：

![output](https://github.com/user-attachments/assets/6cf3bbba-7154-4fdf-b668-0d504ba18b0c)


然后我们求这个非线性变换的雅可比矩阵，雅可比矩阵中的每个元素都是对函数的一阶偏导数，即：

![image](https://github.com/user-attachments/assets/6c3ffd56-2b86-4285-acea-f548db607f9c)


假设这个点为 $(-2, 1)$ ，则雅可比矩阵的数值为:

$$\begin{bmatrix} 
1 & 0.54 \\ 
-0.42 & 1 
\end{bmatrix}$$

这个矩阵的元素告诉我们，在 $(-2, 1)$ 附近：

沿着x方向的变化对 $f_1$ 的影响比例是1，即如果我们让x增加 $\Delta x$ ,那么 $f_1$ 也会增加 $\Delta f_1=1 \cdot \Delta x$ 

沿着x方向的变化对 $f_2$ 的影响比例是-0.42，即如果我们让x增加  $\Delta x$，那么 $f_2$ 也会增加 $\Delta f_2=-0.42 \cdot \Delta x$ 

沿着y方向的变化对 $f_1$ 的影响比例是0.54，即如果我们让y增加 $\Delta y$ ,那么 $f_1$ 也会增加 $\Delta f_1=0.54 \cdot \Delta y$ 

沿着y方向的变化对 $f_2$ 的影响比例是1，即如果我们让y增加 $\Delta y$ ,那么 $f_2$ 也会增加 $\Delta f_2=1 \cdot \Delta y$ 

结论：**雅可比矩阵是对非线性变换的局部线性近似**

然后我们来看一下我们实际所使用的投影变换雅可比矩阵长什么样

已知

$$M_{\text{persp} \rightarrow \text{ortho}} = \begin{bmatrix} 
n & 0 & 0 & 0 \\ 
0 & n & 0 & 0 \\ 
0 & 0 & n + f & -nf \\ 
0 & 0 & 1 & 0 
\end{bmatrix}$$

假设视锥中有一个点 $[x, y, z, 1]^T$

这个点经过投影变换后的矩阵如下所示：

![image](https://github.com/user-attachments/assets/ff74aafe-901a-4f8d-8160-b4af84b9b550)

我们把最后一维的1删掉，则有

![image](https://github.com/user-attachments/assets/7da7d9ab-d52e-4d7d-a0c9-e6b799de65e1)

然后我们将 $f_1$ 和 $f_2$ 和 $f_3$ 分别对x, y, z求一阶偏导，即可得到雅可比矩阵：

![image](https://github.com/user-attachments/assets/b0bcbd13-de0c-4c56-acb8-02d0581104f8)


然后我们把这个雅可比矩阵应用到我们3D高斯的协方差矩阵中，通过雅可比矩阵J去做一个仿射变换，来模拟非线性的变换，即：

$V_k = J V'_k J^T$

$V_k = J V'_k J^T = J W V''_k W^T J^T$

这样我们就可以得到经过投影变换后的协方差矩阵了。

**投影变换小结：**

相机坐标系:
- 高斯核中心 $u_k=[u_0, u_1, u_2]$
- $V'_k$ 是协方差矩阵

经过一个投影变换 $x=m(t)$ 从相机坐标系转换到像素坐标系：

- 均值 $x_k=m(u_k)$
- 高斯核中心 $x_k=[x_0, x_1, x_2]^T$
- 协方差矩阵 $V_k = J V'_k J^T = J W V''_k W^T J^T$

要注意一点：此时的均值和协方差是不在同一个坐标系里面的

- 均值在NDC坐标系里面，即我们前面所说的那个 $[-1, 1]^3$ 的那个立方体
- 而协方差矩阵由于我们是通过雅可比矩阵去做近似，没有做后续的压缩和变换的过程，所以协方差矩阵是在一个未缩放的正交坐标系里面
- **所以我们需要对均值进行后面的视口变换，而协方差矩阵不需要**

**视口变换**

根据我们前面所说的，我们还需要对处于NDC坐标系中的均值进行一个视口变换

我们先来了解一下什么是视口变换

![image](https://github.com/user-attachments/assets/5f8eda53-e212-4a2e-ab41-1cfe8eafd259)


视口变换与z无关，它的作用是将 $[-1, 1]^2$ 的矩形变换至 $[0, w] * [0, h]$

变换矩阵为：

$$M_{\text{viewport}} = \begin{bmatrix}
\frac{w}{2} & 0 & 0 & \frac{w}{2} \\
0 & \frac{h}{2} & 0 & \frac{h}{2} \\
0 & 0 & 1 & 0 \\
0 & 0 & 0 & 1
\end{bmatrix}$$

**3D高斯中的视口变换**

根据我们前面所说的，我们还需要对处于NDC坐标系中的均值进行一个视口变换

在经过投影变换后，高斯核中心 $x_k=[x_0, x_1, x_2]^T$

然后我们对高斯核中心进行视口变换，将高斯核中心乘以上面的视口变换矩阵即可得到视口变换后的高斯核中心，设经过视口变换后的高斯核中心为 $\mu = [\mu_1, \mu_2, \mu_3]^T$

对于协方差矩阵，我们需要进行足迹渲染，即离散计算：

$G(\hat{x}) = \exp \left( -\frac{1}{2} (x - \mu)^{T} V_{k}^{-1} (x - \mu) \right)$

离均值 $\mu$ 越近， $G(\hat{x})$ 值越大。

**光栅化**

最后，我们呢还需要对3D高斯进行光栅化的操作

光栅化的定义：
- 简单来说，光栅化就是要把东西画在屏幕上
- 是一个连续转离散的过程
- 所使用的方法是采样

![image](https://github.com/user-attachments/assets/b65cbb6e-fe74-415a-80f2-982742258282)

到这里，我们也已经成功把雪球抛到了一个2D平面上，但此时我们的雪球还是白色的，我们需要给雪球涂上好看的颜色，所以下一部分我们就要去讨论如何给雪球上色！

## 给雪球涂上颜色 

我们在给椭球涂颜色的时候我们希望可以实现从不同的观测角度观察，椭球会根据光照等信息而呈现出不同的颜色，那么这要怎么去做呢？

回答：作者为了解决这个问题引入了**球谐函数**这个概念

在正式认识球谐函数之前，我们先来回顾一下我们小学二年级学过的傅里叶变换：任何一个函数都可以分解为正弦和余弦的线性组合，即： 

$f(x) = a_0 + \sum_{n=1}^{+\infty} a_n \cos \frac{n \pi}{l} x + b_n \sin \frac{n \pi}{l} x$

其中，我们把 $\cos \frac{n \pi}{l} x$ 和 $\sin \frac{n \pi}{l} x$ 称为基函数

同样的，任何一个球面坐标的函数可以用多个**球谐函数**来近似:

$f(t) \approx \sum_{l} \sum_{m=-l}^{l} c_{l}^{m} y_{l}^{m}(\theta, \phi)$

其中：
- $c_{l}^{m}$ 为各项的系数，有RGB三个值，是一个[1, 3]的向量。
- $y_{l}^{m}(\theta, \phi)$ 是基函数，是一个包含了方向信息的标量 ， $\theta$ 是仰角， $\phi$ 是方位角。

![image](https://github.com/user-attachments/assets/882a2111-064f-4b20-ac88-8fc0e6df8c48)

下面我们将球谐函数展开：

$f(t) \approx \sum_{l} \sum_{m=-l}^{l} c_{l}^{m} y_{l}^{m}(\theta, \phi) = c_{0}^{0} y_{0}^{0} + \quad c_{1}^{-1} y_{1}^{-1} + c_{1}^{0} y_{1}^{0} + c_{1}^{1} y_{1}^{1} + \quad c_{2}^{-2} y_{2}^{-2} + c_{2}^{-1} y_{2}^{-1} + c_{2}^{0} y_{2}^{0} + c_{2}^{1} y_{2}^{1} + c_{2}^{2} y_{2}^{2} + \quad c_{3}^{-3} y_{3}^{-3} + \dots$

其中各阶的基函数我们是可以算出来的：

$y_{0}^{0} = \sqrt{\frac{1}{4 \pi}} = 0.28$

$y_{1}^{-1} = -\sqrt{\frac{3}{4 \pi}} \frac{y}{r} = -0.49 \frac{y}{r}$

$y_{1}^{0} = \sqrt{\frac{3}{4 \pi}} \frac{z}{r} = 0.49 \frac{z}{r}$

$y_{1}^{1} = -\sqrt{\frac{3}{4 \pi}} \frac{x}{r} = -0.49 \frac{x}{r}$

然后在3D高斯的原论文中，作者实际使用的是3阶的球谐函数，故一共有16*3个球谐函数系数

所以作者通过引入了球谐函数来表达椭球的颜色，从而能够实现从不同的角度观测椭球可以显示出不同的颜色差别。

很好，到目前为止，我们雪球也捏出来了，也成功将雪球抛到了一个平面上，并且我们还给雪球加上了颜色。我们离完成只差最后一步了，那就是把图像渲染出来！

---

## 使用快速可微光栅化器进行渲染

![image](https://github.com/user-attachments/assets/60507db8-1f9b-4fd2-a686-f626f24927b3)




---

- 具体来说就是在重建不充分的区域往往会有较大的梯度，我们可以设定一个梯度阈值，对超过梯度阈值的位置我们对椭球进行分裂或者克隆：
  * 对于方差大的位置，说明椭球的形状大，我们需要对椭球进行分裂
  * 对于方差小的位置，说明椭球的形状小，我们需要对椭球进行克隆
如果我们按照上面的流程做下来，我们会发现一个问题：就是我们虽然可以优化每个椭球的形状，颜色，透明度等，但我们始终无法改变点云的数量，强烈依赖sfm生成的初始点云数量。





