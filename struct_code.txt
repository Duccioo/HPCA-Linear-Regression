Project for High Prfromance Computer Architecture

CPUs/GPUs Comparison of Multi Linear Regression Algorithm

Authors:
Maccantelli Francesco
Meconcelli Duccio

Professor:
Giorgi Roberto

Material present:
|
|   data
|   |   test.csv    # test dataset normalized
|   |   train.csv   # train dataset normalized
|   |   mock.csv    # origiant dataset 
|   |   generated   # folder with generated dataset
|   |   |   2D
|   |   |   | *
|   |   |   4D
|   |   |   | *
|
|   img
|   | *
|   save            # folder with saved results
|   | *
|
|   data_analysis.py
|   data_gen.py
|   norm.py
|   main_LR.cu
|   linear_regression4d.cuh
|   report.pdf
|   requirements.txt
|   README.txt


PREREQUISITES:

In order to run all the script you need the following:

for the python scripts:
> pip install -r requirements.txt

for the main_LR.cu script:
nvcc compiler
CUDA 11.5


                      _                        _           _                          _                      
                     (_)                      | |         | |                        | |                     
  _ __ ___     __ _   _   _ __              __| |   __ _  | |_    __ _   ___    ___  | |_        ___   _   _ 
 | '_ ` _ \   / _` | | | | '_ \            / _` |  / _` | | __|  / _` | / __|  / _ \ | __|      / __| | | | |
 | | | | | | | (_| | | | | | | |          | (_| | | (_| | | |_  | (_| | \__ \ |  __/ | |_   _  | (__  | |_| |
 |_| |_| |_|  \__,_| |_| |_| |_|           \__,_|  \__,_|  \__|  \__,_| |___/  \___|  \__| (_)  \___|  \__,_|
                                  ______                                                                     
                                 |______|            

The main_LR.cu is the main file that allow to run the algorithm of the linear regression.
This automatically execute the CPU and the GPU version of the algorithm. This script returns the results
of the elapsed time in file saved in /result folder of the cpu and gpu parameters.

To ececute:
> nvcc -o main_LR main_LR.cu
> ./main_LR


  _   _                                                                                          _                   _  _         _                      _     
 | | (_)                                                                                        (_)                 | || |       | |                    | |    
 | |  _   _ __     ___    __ _   _ __            _ __    ___    __ _   _ __    ___   ___   ___   _    ___    _ __   | || |_    __| |       ___   _   _  | |__  
 | | | | | '_ \   / _ \  / _` | | '__|          | '__|  / _ \  / _` | | '__|  / _ \ / __| / __| | |  / _ \  | '_ \  |__   _|  / _` |      / __| | | | | | '_ \ 
 | | | | | | | | |  __/ | (_| | | |             | |    |  __/ | (_| | | |    |  __/ \__ \ \__ \ | | | (_) | | | | |    | |   | (_| |  _  | (__  | |_| | | | | |
 |_| |_| |_| |_|  \___|  \__,_| |_|             |_|     \___|  \__, | |_|     \___| |___/ |___/ |_|  \___/  |_| |_|    |_|    \__,_| (_)  \___|  \__,_| |_| |_|
                                        ______                  __/ |                                                                                          
                                       |______|                |___/                                                                                           

The linear_regression4d.cuh file contains the kernel for the linear regresion algorithm for the GPU implementation.
This file is used in the main_dataset.cu file.

      _           _                                              _                 _                           
     | |         | |                                            | |               (_)                          
   __| |   __ _  | |_    __ _             __ _   _ __     __ _  | |  _   _   ___   _   ___       _ __    _   _ 
  / _` |  / _` | | __|  / _` |           / _` | | '_ \   / _` | | | | | | | / __| | | / __|     | '_ \  | | | |
 | (_| | | (_| | | |_  | (_| |          | (_| | | | | | | (_| | | | | |_| | \__ \ | | \__ \  _  | |_) | | |_| |
  \__,_|  \__,_|  \__|  \__,_|           \__,_| |_| |_|  \__,_| |_|  \__, | |___/ |_| |___/ (_) | .__/   \__, |
                                ______                                __/ |                     | |       __/ |
                               |______|                              |___/                      |_|      |___/ 

The data_analysis.py script allow to plot graphically the mean of the multiple run saved on the file in the
folder /result, if multiple files are present in the folder, the script automatically plots all the data in the same plot.

                                                       
                                                       
  _ __     ___    _ __   _ __ ___        _ __    _   _ 
 | '_ \   / _ \  | '__| | '_ ` _ \      | '_ \  | | | |
 | | | | | (_) | | |    | | | | | |  _  | |_) | | |_| |
 |_| |_|  \___/  |_|    |_| |_| |_| (_) | .__/   \__, |
                                        | |       __/ |
                                        |_|      |___/ 

The norm.py script allow to normalize the dataset in input.


                                      _           _                               
                                     | |         | |                              
   __ _    ___   _ __              __| |   __ _  | |_    __ _       _ __    _   _ 
  / _` |  / _ \ | '_ \            / _` |  / _` | | __|  / _` |     | '_ \  | | | |
 | (_| | |  __/ | | | |          | (_| | | (_| | | |_  | (_| |  _  | |_) | | |_| |
  \__, |  \___| |_| |_|           \__,_|  \__,_|  \__|  \__,_| (_) | .__/   \__, |
   __/ |                 ______                                    | |       __/ |
  |___/                 |______|                                   |_|      |___/ 

  This script allow to generate datasets in 4D, parameters like the Number of data must be specified in the file


  _                  _                       
 | |                | |                      
 | |_    ___   ___  | |_       _ __    _   _ 
 | __|  / _ \ / __| | __|     | '_ \  | | | |
 | |_  |  __/ \__ \ | |_   _  | |_) | | |_| |
  \__|  \___| |___/  \__| (_) | .__/   \__, |
                              | |       __/ |
                              |_|      |___/ 

This file allow to test the performance of founded parameters of the regresions.
The parameters like: intercept, slope1, slope2 and slope3 must be specified in the file.
Then it returns the Accuracy.ls                            

