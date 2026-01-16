"""
Generate predictions on data using a trained model.

This script loads a trained Keras/TensorFlow model and generates predictions on
test data. Includes functions for data loading, preprocessing, and prediction
generation with visualization capabilities.

Note: Uses legacy TensorFlow/Keras code. Consider migrating to PyTorch version
for consistency with current training pipeline.
"""

# Import packages
import cv2
import os, gc, sys, glob
import pandas as pd
import numpy as np
from numpy import expand_dims
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import random
import itertools
import csv

import pandasql as ps

from sklearn import model_selection
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

from subprocess import check_output

import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model, load_model
from keras import applications
from keras.callbacks import ReduceLROnPlateau
from keras.layers.normalization import BatchNormalization
from keras.metrics import categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.applications.inception_resnet_v2 import preprocess_input

def read_img(img_path, size):
    """Read and resize a single image."""
    img = cv2.imread(str(img_path))
    img = cv2.resize(img, (size[0], size[1]))
    return img

# Function to load full data and labels
def read_data(main_path, size):
    folders = [a for a in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, a)) and 'oversample' not in a]
    dirs = [main_path + '/' + f for f in folders]

    data = []
    labels = []

    count=1
    for img_path in Path(dirs[0]).glob('**/*.png'):
        data.append(read_img(img_path, size))
        labels.append([os.path.basename(img_path), folders[0]])
        if count%1000==0:
          print('Loading data from dir ' + dirs[0] + ': ' + str(count))
        count +=1
        

    count=1
    for img_path in Path(dirs[1]).glob('**/*.png'):
        data.append(read_img(img_path, size))
        labels.append([os.path.basename(img_path), folders[1]])
        if count%1000==0:
          print('Loading data from dir ' + dirs[1] + ': ' + str(count))
        count +=1
    
    data = np.array(data, np.float32) / 255

    label_encoder = LabelEncoder()
    labels = pd.DataFrame(labels, columns = ['image','label'])
    labels['tiff'] = [name[:name.find('_')] for name in labels['image']]
    enc_labels = label_encoder.fit_transform(np.array(labels.iloc[:,1]))

    return data, labels, enc_labels

# Function to calculate predictions
# All useful information is stored in dataframe, including patient characteristics
def calculate_pred():
  # Predict on data, using cut-off of 0.5
  class_names = list(np.unique(labels.label))
  probabilities = model.predict(data)
  predictions = (probabilities[:,0] >= 0.5).astype(int)
  true_preds = np.where(enc_labels == predictions)
  wrong_preds = np.where(enc_labels != predictions)

  # Store information in dataframe
  preds_df = pd.DataFrame()
  preds_df['image'] = labels.image
  preds_df['airway'] = [(img_name.split('__')[1]).split(')')[0] + ')' for img_name in list(labels.image)]
  preds_df['tiff'] = labels.tiff
  preds_df['label'] = labels.label
  preds_df['enc_label'] = enc_labels
  preds_df['pred'] = predictions
  preds_df['pred_label'] = [class_names[i] for i in list(predictions)]
  preds_df['prob_outside'] = probabilities[:,0]
  preds_df['prob_correct'] = abs(1 - (1 * preds_df.enc_label + preds_df.prob_outside))
  preds_df['ind_correct_pred'] = (preds_df.enc_label == preds_df.pred).astype(int)
  preds_df['n_tiff'] = preds_df.groupby('tiff')['tiff'].transform('count')

  # Left join patient characteristics
  patient_chars_df = pd.read_csv(data_path + '/Patient_characteristics.csv', sep=';')
  preds_df_full = preds_df.merge(patient_chars_df, on='tiff', how='left')

  return preds_df_full


##model_path = 'N:/Werkstudenten/1. Eigen mappen/Esmee/UMCG/Results/inceptionresnetv2_ex_overs_retr_true_2_0005_06_150_co60'
##data_path = 'N:/Werkstudenten/1. Eigen mappen/Esmee/UMCG/Data/With tiff number/Patches_clean'  # Path where data is in (per class one folder)
##size = [224, 224]      # Input size of the model
##
##print('Reading data')
##data, labels, enc_labels = read_data(data_path, size)
##
##print('Loading model')
##model = load_model(model_path + '/weights_auc.hdf5') 
##
##print('Calculating predictions')
##preds_df = calculate_pred()
##
##agg_query = (" SELECT tiff, "
##            "patient, "
##            "airway, "
##            "SUM(ind_correct_pred) / (1.000 * COUNT(*))             AS acc,  "  
##            "COUNT(*)                                               AS n_patches, " 
##            "SUM(enc_label)                                         AS n_outside, " 
##            "COUNT(*) - SUM(enc_label)                              AS n_inside, " 
##            "SUM(CASE WHEN enc_label = 1 AND ind_correct_pred = 1 THEN 1 ELSE 0 END)  AS tp, " 
##            "SUM(CASE WHEN enc_label = 0 AND ind_correct_pred = 1 THEN 1 ELSE 0 END)  AS tn, " 
##            "SUM(CASE WHEN enc_label = 1 AND ind_correct_pred = 0 THEN 1 ELSE 0 END)  AS fp, " 
##            "SUM(CASE WHEN enc_label = 0 AND ind_correct_pred = 0 THEN 1 ELSE 0 END)  AS fn " 
##      "FROM preds_df "
##      "GROUP BY patient, tiff, airway")
##
##aggs = ps.sqldf(agg_query)
##try:
##  os.mkdrs(model_path + '/Visualizations/')
##except:
##  pass
##aggs.to_csv(model_path + '/Visualizations/airway_aggregates.csv')
