# Advancing-Taxonomy-with-Machine-Learning-A-Hybrid-Ensemble-for-Species-and-Genus-Classification
Advancing Taxonomy with Machine Learning: A Hybrid Ensemble for Species and Genus Classification

For SVM we have used: https://www.csie.ntu.edu.tw/~cjlin/libsvm/
For training SVM the features were linearly normalized between 0 and 1, 
using only the training data for finding the parameters to normalize the data.

Before the fusion between SVM and CNN, the scores of each approach are normalized to mean 0 
and standard deviation 1. You can extract a validation set from the training set, and then use it 
to find the parameters needed for the normalization approach. 

The approach eDNA is here available:
https://github.com/LorisNanni/AI-powered-Biodiversity-Assessment-Species-Classification-via-DNA-Barcoding-and-Deep-Learning

The classificaton step is performed in MATLAB; the feature extraction step is performed in Python.

Matlab code is saved in Matlab.rar; the file ExampleFish_DWT.m  shows how to apply DWT for training/testing CNN

Python code is folder python. Follow the README.md for complete instructions.



Dataset:
Badirli dataset is available here https://dataworks.indianapolis.iu.edu/handle/11243/41
The two new datasets here proposed, detailed in subsections 2.1 and 2.2, are available at: https://zenodo.org/records/14277812
The Beetle and the Fish datasets are available at: https://zenodo.org/records/14728702
