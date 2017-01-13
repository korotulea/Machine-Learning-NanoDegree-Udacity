# Machine Learning Engineer Nanodegree
## Alexandru Korotcov
## January 10, 2017
## Specializations: Deep Learning
## Project: Recognizing digits and numbers in natural scene images

##Files:
-   1_SVHN_dataset_prep_V2.ipynb: download and preprocess datasets; split into train, validation, test datasets and create pickle files
-   2_SVHN_dataset_description_V2.ipynb: download datasets from saved pikle files; exploratory analysis visualization
-   3_SVHN_dataset_NeuralNets_V2.ipynb: download datasets from saved pikle files; convolutional neural network models for digits recognition; models training, tuning, and validation with performance visualization

##Software and libraries
Following installations are required for this project:
-   Python 2.7 (data download, data exploration)
-   Python 3.5 (used for training, but 3_SVHN_dataset_NeuralNets_V2.ipynb is compatible with Python 2.7)
-   Jupyter Notebook

**Note**
I recommend use Anaconda installation or Udacity tensorflow on Docker (the most simple approach to install tensorflow)

(https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/udacity)

Python libraries:
-   numpy
-   tensorflow
-   six
-   pandas
-   matplotlib
-   seaborn
-   scipy
-   sklearn
-   PIL
-   h5py

##Dataset
http://ufldl.stanford.edu/housenumbers/

SVHN is a real-world image dataset for developing machine learning and object recognition algorithms with minimal requirement on data preprocessing and formatting. It can be seen as similar in flavor to MNIST (e.g., the images are of small cropped digits), but incorporates an order of magnitude more labeled data (over 600,000 digit images) and comes from a significantly harder, unsolved, real world problem (recognizing digits and numbers in natural scene images). SVHN is obtained from house numbers in Google Street View images.

##Run
-   Download the iPython notebooks in your local folder.
-   1_SVHN_dataset_prep_V2.ipynb must be run to prepare train, validation, and test dataset.
-   2_SVHN_dataset_description_V2.ipynb help you to explore the datasets. 
-   I wouldn't recommend to execute cells in 3_SVHN_dataset_NeuralNets_V2.ipynb unless you are confident in computer power of your machine. All the modelling steps are described and visualized in this notebook. Also, all the performance metrics for all model are reported here.

**Note**

Running of the Convolutional Models training in 3_SVHN_dataset_NeuralNets_V2.ipynb may take a long time. All the training in this project were performed using Nvidia Tesla K20 GPU.
