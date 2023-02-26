# HPCA-Linear-Regression

Implementazione della Regressione Lineare usando le API di CUDA

## Fonti:

### Dataset:

1. [medical insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance?select=insurance.csv)

2. [Salary Skill](https://github.com/lianaling/dspc-cuda/blob/main/dspc-cuda/mock.csv)

### From Appgallery Web:

- [Linear Regression app](https://github.com/maxeler/Linear-Regression/blob/master/ORIG/LinearRegressionCPUOnly.c)

### Implementazione Linear Regression Github:

- [x] [CUDA implementation of machine learning models, such as linear regression.](https://github.com/YvoElling/CUDA-ML)
- [ ] [Implement gradient descend linear regression and optimize time of resolve it with Cuda.](https://github.com/mohammaddan/gradient-descend-cuda)
- [x] [CUDA parallelisation for multivariate linear regression](https://github.com/lianaling/dspc-cuda)
- [ ] [calculate N-variable linear regression with cuda](https://github.com/ForrestSu/linear_regression_cuda)
- [ ] [Computing linear-Multiple-Regression on GPU using CUDA](https://github.com/ironmanMA/CUDA_lm)

<details>
    <summary>Possibili altri progetti</summary>
    <ul>
        <li> <a href="https://github.com/Bwas123/HPC_CourseWork">Password Cracking, Image Processing, Linear Regression using Posix, Cuda and MPI.</a> </li>
        <li> <a href="https://github.com/TristanNagan/HPC-Project-Kmeans">A comparison of K-means serial and parallel implementations in C using CUDA and MPI</a> </li>
        <li> <a href="https://github.com/adakri/HPC_GPU_NBody_visualisation">N-body simulations implemented through GPU general computing.</a> </li>
        <li> <a href="https://github.com/Leyxargon/Strassen-GPU">Progetto per l'esame di High Performance Computing, Algoritmo di Strassen in ambiente GPU-CUDA</a> </li>
        <li> <a href="https://github.com/Dantekk/Canny-GPU-CUDA-implementation">CUDA implementation of Canny edge detector in C/C++.</a> </li>
        <li> <a href="https://github.com/Org-Placeholder/CV_CUDA">Sobel Filter Canny Edge Detection Mean Blur Gaussian Blur Noise Addition Noise Reduction Bokeh Blur</a> </li>
        <li> <a href="https://github.com/haoyuhsu/Parallel-Low-Poly-Image">Low-poly style image translation parallelized by CUDA library</a> </li>
    </ul>

</details>

### Altra roba sempre con cuda:

- [CUDA Hacker's Toolkit (tanta roba utile)](https://github.com/tensorush/CUDA-Hackers-Toolkit)
- [Matrix Mul, Vector Add](https://github.com/GitRealFan/Simple-Projects-CUDA)
- [tutorial Base](https://github.com/priteshgohil/CUDA-programming-tutorial)
- [matrix_transpose](https://github.com/Logeswaran123/CUDA-Programming/tree/main/9_matrix_transpose)

## Osservazioni:

1. il Dataset `/data/insurance.csv` dalle prove fatte non segue un andamento lineare. Usando la Multi Linear Regression si ottiene una precisione del 70% circa.

2. La regressione lineare si può risolvere in modi differenti. La repo di Maxeler usa la _least squares_
