# AD_RF_Analysis

The code for "Deep Learning Analysis of Retinal Structures and Risk Factors of Alzheimer’s Disease". Please contact Seowung Leem for implementation details. 

* Lab: <leem.s@ufl.edu>
* Personal: <leem.s@ufl.edu>

## Publication
Deep Learning Analysis of Retinal Structures and Risk Factors of Alzheimer’s Disease
Conference
Seowung Leem, Yunchao Yang, Adam J. Woods, Ruogu Fang
The 46th International Conference of the IEEE Engineering in Medicine and Biology Society, , Orlando, FL
Publication Date: July 15-19, 2024

Risk Factor Prediction &amp; Analysis using fundus image. Funded by NSF

## Abstract
The importance of early Alzheimer’s Disease screening is becoming more apparent, given the fact that there is no way to revert the patient’s status after the onset. However, the diagnostic procedure of Alzheimer’s Disease involves a comprehensive analysis of cognitive tests, blood sampling, and imaging, which limits the screening of a large population in a short period. Preliminary works show that rich neurological and cardiovascular information is encoded in the patient’s eye. Due to the relatively fast and easy procedure acquisition, early-stage screening of Alzheimer’s Disease patients with eye images holds great promise. In this study, we employed a deep neural network as a framework to investigate the relationship between risk factors of Alzheimer’s Disease and retinal structures. Our result shows that the model not only can predict several risk factors above the baseline but also can discover the relationship between the retinal structures and risk factors to provide insights into the retinal imaging biomarkers of Alzheimer’s disease.

## How to run the code

### Requirements
- Please refer to the *environment.yml* to meet the requirements for the environment. 

- The dataset used in this experiment is from the *UK Biobank dataset* <https://www.ukbiobank.ac.uk/>. For the data availability, please visit this site to request the access. 

### (1) Model Training & Evaluation.

First, to train the model, we have to run the *train_classification_multi.py*. However, UKB has ~10,000 image counts, and the base model *Swin Transformer* has big parameter size. Therefore, using multi GPU is essential. The multi GPU version of *train_classification_multi.py* is *train_classification_multi_mlflow.py*. This is same for the regression task. 

### (2) GradCAM visualization.

Then, trained weights are used for saliency map visualization. This provides the interpretability of the trained model, in terms of how the retinal structures contributed to the model's risk factor prediction. For this part, we utilized the code from <https://github.com/jacobgil/pytorch-grad-cam>. Please visit the link for implementation detail. The code is *GradCam_Visualization_Classification.py* for classification, and *GradCam_Visualization_Regression.py* for regression.  

### (3) Inference Calculation.

For segmentation map creation, we used the <https://github.com/rmaphoh/AutoMorph> for segmentation map generation. The image quality evaluation of UK-Biobank dataset was also performed with this repo (in specific, the M0_Preprocess Part). For implementation detail, plese visit the link. The notebook *Classification_Overlap_Calculation.ipynb* gives the way how we performed the inference calculation. 








