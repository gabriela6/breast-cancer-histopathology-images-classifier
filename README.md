# breast-cancer-histopathology-images-classifier
The goal of the project is to create a binary neural network classifier of Invasive Ductal Carcinoma (IDC) cancer in histopathology images from Kaggle database. 
## Motivation 
Learning how to use TensorFlow library to create a neural network model working on real medical data. 
## Invasive Ductal Carcinoma
Invasive Ductal Carcinoma is the most common form of invasive breast cancer. Cancer cells from the lining of the milk ducts invade breast tissue beyond the walls of the duct. One of the diagnosis methods is breast biopsy. 

“Invasive breast cancer detection is a time consuming and challenging task primarily because it involves a pathologist scanning large swathes of benign regions to ultimately identify the areas of malignancy. Precise delineation of IDC in WSI is crucial to the subsequent estimation of grading tumor aggressiveness and predicting patient outcome”  (Cruz-Roa et al., 2014).
## Dataset structure 
Dataset contains 277,524 patches of size 50 x 50 from histopathology images. Dataset is divided into 279 folders. Each folder contains 2 folders: one named "0" with patches classified as non-IDC tissue and one named "1" with patches classified as IDC tissue.
Patches have following file format: u_xX_yY_classC.png, where u is patient ID, X is x coordinate, Y is y coordinate and C is a class (0 or 1). Example: 8863_idx5_x51_y1251_class0.png
## Model Structure
Model has a structure of a convolutional neural network. ReLU was used as an activation function for every layer, except from last dense layer, where Softmax was used. An Adam Optimizer has been chosen as an optimization algorithm. Sparse categorical crossentropy was used as loss function.

| Layer (type) | Output Shape | Param | 
| --- | --- |--- |
| conv2d (Conv2D) | (None, 48, 48, 16) | 448 |   
| conv2d_1 (Conv2D) | (None, 46, 46, 16) | 2320 |      
| max_pooling2d (MaxPooling2D) | (None, 23, 23, 16) | 0 |       
| conv2d_2 (Conv2D) | (None, 21, 21, 32) | 4640 |     
| conv2d_3 (Conv2D) | (None, 19, 19, 32) | 9248 |     
| max_pooling2d_1 (MaxPooling2D) | (None, 9, 9, 32) | 0 |       
| conv2d_4 (Conv2D) | (None, 7, 7, 64) | 18496 |    
| conv2d_5 (Conv2D) | (None, 5, 5, 64) | 36928 |     
| max_pooling2d_2 (MaxPooling2D) | (None, 2, 2, 64) | 0 |       
| flatten (Flatten) | (None, 256) | 0 |     
| dense (Dense) | (None, 128) | 32896 |
| dense_1 (Dense) | (None, 2) | 258 |      

Total params: 105,234
## Model evaluation
The model achieved F-score of 75.33% and balanced accuracy of 83.14%.
## How to run a program
A program is located in this repository in a jupyter notebook named Breast_Cancer_Histopathology_Images_Classifier.ipynb. It is possible to run this program in Google Colab by clicking this button: 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](http://colab.research.google.com/github/gabriela6/breast-cancer-histopathology-images-classifier/blob/main/breast_cancer_histopathology_images_classifier.ipynb)

Kaggle API token is necessary for downloading Kaggle dataset. A link to instruction on how to download a token : https://www.analyticsvidhya.com/blog/2021/06/how-to-load-kaggle-datasets-directly-into-google-colab/

## Used libraries
scikit-learn, TensorFlow, Matplotlib, OpenCV, NumPy
## Sources
https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images/discussion

Cruz-Roa, A. et al. , "Automatic detection of invasive ductal carcinoma in whole slide images with convolutional neural networks," Proc. SPIE 9041, Medical Imaging 2014: Digital Pathology, 904103 (20 March 2014) https://doi.org/10.1117/12.2043872

https://www.hopkinsmedicine.org/health/conditions-and-diseases/breast-cancer/invasive-ductal-carcinoma-idc#:~:text=Invasive%20ductal%20carcinoma%20is%20cancer,the%20cancer%20cells%20can%20spread.
