# Facial Beauty prediction
Implementaion of pretrained model from:
* https://github.com/HCIILAB/SCUT-FBP5500-Database-Release

* Using flask to serve up the model
* Using Alexnet Architecture for prediction
* Correlation:0.8721      Mae:0.2724      Rmse:0.3552

Research Paper
* https://www.mdpi.com/2673-4591/56/1/125#:~:text=Facial%20beauty%20prediction%20is%20a,scores%20for%20facial%20beauty%20prediction.


### Steps to run
* Clone the repository
```
git clone <repository_url>
```
* Navigate to the Project Directory
```
cd <project_directory>
```
* Download pytorch model from the link given below
```
https://drive.google.com/file/d/1tAhZ3i4Pc_P3Fabmg62hGVHwKeSQtYaY/view
```

* copy the contents (.pth files) into the trained_models_for_pytorch/models folder

* create virtual environment and activate
```
py -m venv venv
venv\Scripts\activate
```
* install requirements
```
pip install -r requirements.txt
```
* install torch with CUDA support (NVIDIA GPU REQUIRED)
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
* run the server
```
py server.py
```

Go to http://127.0.0.1:5000 to get the interface for facial beauty prediction
## 1 Description

The SCUT-FBP5500 dataset has totally 5500 frontal faces with diverse properties
(male/female, Asian/Caucasian, ages) and diverse labels (facial landmarks, beauty scores in 5 scales, beauty score distribution), which allows different computational model with different facial beauty prediction paradigms, such as appearance-based/shape-based facial beauty classification/regression/ranking model for male/female of Asian/Caucasian. 

## 2 Database Construction

The SCUT-FBP5500 Dataset can be divided into four subsets with different races and gender, including 2000 Asian females(AF), 2000 Asian males(AM), 750 Caucasian females(CF) and 750 Caucasian males(CM). Most of the images of the SCUT-FBP5500 were collected from Internet, where some portions of Asian faces were from the DataTang, GuangZhouXiangSu and our laboratory, and some Caucasian faces were from the 10k US Adult Faces database.
![image](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release/blob/master/SCUT-FBP5500.jpg)



### Training/Testing Set Split

We use two kinds of experimental settings to evaluate the facial beauty prediction methods on SCUT-FBP5500 benchmark, which includes: 

1) 5-folds cross validation. For each validation, 80% samples (4400 images) are used for training and the rest (1100 images) are used for testing.
2) The split of 60% training and 40% testing. 60% samples (3300 images) are used for training and the rest (2200 images) are used for testing.
We have provided the training and testing files in this link.  



### Trained Models for Pytorch
The trained models for Pytorch (Size = 101MB) can be downloaded throught the following link:
* Download link: 
https://drive.google.com/file/d/1tAhZ3i4Pc_P3Fabmg62hGVHwKeSQtYaY/view

### Dataset download link:
https://drive.google.com/file/d/1w0TorBfTIqbquQVd6k3h_77ypnrvfGwf/view

#### Requirements:
blinker==1.8.2
click==8.1.7
colorama==0.4.6
filelock==3.14.0
Flask==3.0.3
Flask-Cors==4.0.1
fsspec==2024.3.1
intel-openmp==2021.4.0
itsdangerous==2.2.0
Jinja2==3.1.4
MarkupSafe==2.1.5
mkl==2021.4.0
mpmath==1.3.0
networkx==3.3
numpy==1.26.4
pillow==10.3.0
sympy==1.12
tbb==2021.12.0
torch==2.3.0
torchvision==0.18.0
typing_extensions==4.11.0
Werkzeug==3.0.3


## 3 Benchmark Evaluation

We set AlexNet, ResNet-18, and ResNeXt-50 as the benchmarks of the SCUT-FBP5500 dataset, and we evaluate the benchmark on various measurement metrics, including: Pearson correlation (PC), maximum absolute error (MAE), and root mean square error (RMSE). The evaluation results are shown in the following. Please refer to the paper for more details. 

![image](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release/blob/master/Results%20of%205-folds%20cross%20validations.png)
![image](https://github.com/HCIILAB/SCUT-FBP5500-Database-Release/blob/master/Results%20of%20the%20split%20of%2060%25%20training%20and%2040%25%20testing.png) 


## 4 Citation and Contact

Please consider to cite the paper when you use the database:
```
@article{liang2017SCUT,
  title     = {SCUT-FBP5500: A Diverse Benchmark Dataset for Multi-Paradigm Facial Beauty Prediction},
  author    = {Liang, Lingyu and Lin, Luojun and Jin, Lianwen and Xie, Duorui and Li, Mengru},
  jurnal    = {ICPR},
  year      = {2018}
}
```

Note: The SCUT-FBP5500 database can be only used for non-commercial research purpose. 

For any questions about this database please contact the authors by sending email to `lianwen.jin@gmail.com` and `lianglysky@gmail.com`.


##  Desclaimer

This AI algorithm is purely for academic research purpose. The dataset and codes are for academic research use only. We are not responsible for the objectivity and accuracy of the proposed model and algorithm.
