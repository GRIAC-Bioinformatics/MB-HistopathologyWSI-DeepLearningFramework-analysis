# Import packages
import cv2
import os, gc, sys, glob
import pandas as pd
import numpy as np
from numpy import expand_dims
from tqdm import tqdm
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

import seaborn as sns
import random
import itertools
import csv
import statistics

!pip install SQLAlchemy==1.4.46
!pip install pandasql
import pandasql as ps

from sklearn import model_selection
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy import stats
from sklearn.svm import SVC

from subprocess import check_output

import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Model, load_model
from keras import applications
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization
from keras.metrics import categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.applications.inception_resnet_v2 import preprocess_input

import tensorflow as tf

from tf_keras_vis import utils
import tempfile

import torch
import torchvision


import statsmodels.api as sm
import statsmodels.formula.api as smf

"""# 0. Function definitions

## Functions to read and predict on data
"""

# Function to read single image
def read_img(img_path, size):
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Model is trained on RGB colors, but CV2 imports image as BGR
    img = cv2.resize(img, (size[0], size[1]))
    return img

  # Function to load full data and labels
  def read_data(main_path, size):
      folders = [a for a in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, a)) and 'oversample' not in a and 'dir' not in a]
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

  map_dict = {'Inside': 'Submucosa',
              'Outside': 'Adventitia'}

  # Store information in dataframe
  preds_df = pd.DataFrame()
  preds_df['image'] = labels.image
  preds_df['airway'] = [(img_name.split('__')[1]).split(')')[0] + ')' for img_name in list(labels.image)]
  preds_df['tiff'] = labels.tiff
  preds_df['label'] = labels.label
  preds_df['official_label'] = preds_df.label.replace(map_dict)
  preds_df['enc_label'] = enc_labels
  preds_df['pred'] = predictions
  preds_df['pred_label'] = [class_names[i] for i in list(predictions)]
  preds_df['prob_outside'] = probabilities[:,0]
  preds_df['prob_correct'] = abs(1 - (1 * preds_df.enc_label + preds_df.prob_outside))
  preds_df['ind_correct_pred'] = (preds_df.enc_label == preds_df.pred).astype(int)

  # Left join patient characteristics
  patient_chars_df = pd.read_csv(data_path + '/mapping.csv', sep=';')
  preds_df_incl_patient_chars = preds_df.merge(patient_chars_df, on='tiff', how='left')

  # Create new group where COPD stages I, II, III, IV are joined
  preds_df_incl_patient_chars['copd_group_full'] = 'COPD'
  preds_df_incl_patient_chars.loc[preds_df_incl_patient_chars.copd_group == 'Normal', 'copd_group_full'] =  'Normal'

  # Calculate patient-level aggregations
  patient_aggs = preds_df_incl_patient_chars.groupby('patient').agg({
      'tiff': 'nunique',
      'airway': 'nunique',
      'image': 'count',
      'label': lambda x: sum(x == 'Inside')
  }).rename(columns={
      'tiff': 'n_tiffs',
      'airway': 'n_airways', 
      'image': 'n_images',
      'label': 'n_submucosa'
  })

  # Calculate additional metrics
  patient_aggs['n_adventitia'] = patient_aggs['n_images'] - patient_aggs['n_submucosa']
  patient_aggs['perc_submucosa'] = patient_aggs['n_submucosa'] / patient_aggs['n_images']

  # Merge aggregations back with original dataframe
  preds_df_final = preds_df_incl_patient_chars.merge(
      patient_aggs,
      left_on='patient',
      right_index=True
  )

  # Remove duplicate images ending with _2.png
  preds_df_final = preds_df_final[~preds_df_final.image.str.endswith('_2.png')]

  return preds_df_final

# Function to determine top x true positives, negatives and false positives, negatives
def determine_prototypes(top, output_path):
  tn_certain = preds_df[(preds_df.enc_label == 0) & (preds_df.ind_correct_pred == 1)].sort_values(by=['prob_outside'], ascending=True)['image'].head(top)
  tp_certain = preds_df[(preds_df.enc_label == 1) & (preds_df.ind_correct_pred == 1)].sort_values(by=['prob_outside'], ascending=False)['image'].head(top)
  fn_certain = preds_df[(preds_df.enc_label == 1) & (preds_df.ind_correct_pred == 0)].sort_values(by=['prob_outside'], ascending=True)['image'].head(top)
  fp_certain = preds_df[(preds_df.enc_label == 0) & (preds_df.ind_correct_pred == 0)].sort_values(by=['prob_outside'], ascending=False)['image'].head(top)

  file_names = list(tn_certain) + list(tp_certain) + list(fn_certain) + list(fp_certain)
  pred_categories = list(np.repeat('tn', len(list(tn_certain)))) + list(np.repeat('tp', len(list(tp_certain)))) + list(np.repeat('fn', len(list(fn_certain)))) + list(np.repeat('fp', len(list(fp_certain))))

  for i in range(len(file_names)):
    file_name = file_names[i]
    img_index = labels[labels['image']==file_name].index.values.astype(int)[0]
    img = data[img_index] * 255
    patient = preds_df.patient[img_index]
    prob = preds_df.prob_outside[img_index]
    category = pred_categories[i]

    try:
      os.makedirs(output_path + '/Visualizations/prototypes/overall/' + category + '/')
    except:
      pass

    try:
      os.makedirs(output_path + '/Visualizations/prototypes/per_patient/' + str(patient) + '/' + category + '/')
    except:
      pass

    cv2.imwrite(output_path + '/Visualizations/prototypes/overall/' + category + '/' + str(prob) + '_' + file_name, img.astype(int))
    cv2.imwrite(output_path + '/Visualizations/prototypes/per_patient/' + str(patient) + '/' + category + '/' + str(prob) + '_' + file_name, img.astype(int))

  return tn_certain, tp_certain, fn_certain, fp_certain

# Function to extract features resulting from the last pooling layer
# prior to final classification layer
def calculate_features():
  model_without_top = Model(inputs=model.inputs, outputs=model.get_layer('global_average_pooling2d_7').output)
  features = model_without_top.predict(data)

  return features

# Function to calculate predictions - avoid GPU memory error
# All useful information is stored in dataframe, including patient characteristics
def calculate_pred_and_features():
  # Predict on data, using cut-off of 0.5
  test_datagen = ImageDataGenerator(rescale=1.0/255)
  test_gen = test_datagen.flow_from_directory(os.path.join(data_path, 'full_dir'),
                                        target_size=(224, 224),
                                        batch_size=128,
                                        class_mode='binary',
                                        shuffle=False)

  filenames = test_gen.filenames
  enc_labels = test_gen.labels
  class_names = list(np.unique([f.split('/')[0] for f in filenames]))

  probabilities = model.predict_generator(test_gen, steps = None, verbose=1)
  predictions = (probabilities[:,0] >= 0.5).astype(int)
  true_preds = np.where(enc_labels == predictions)
  wrong_preds = np.where(enc_labels != predictions)

  map_dict = {'Inside': 'Submucosa',
              'Outside': 'Adventitia'}

  # Store information in dataframe
  preds_df = pd.DataFrame()
  preds_df['image'] = [os.path.basename(f) for f in filenames]
  preds_df['airway'] = [(img_name.split('__')[1]).split(')')[0] + ')' for img_name in list(preds_df['image'])]
  preds_df['tiff'] = [img_name[:img_name.find('_')] for img_name in preds_df['image']]
  preds_df['label'] = [f.split('/')[0] for f in filenames]
  preds_df['official_label'] = preds_df.label.replace(map_dict)
  preds_df['enc_label'] = enc_labels
  preds_df['pred'] = predictions
  preds_df['pred_label'] = [class_names[i] for i in list(predictions)]
  preds_df['prob_outside'] = probabilities[:,0]
  preds_df['prob_correct'] = abs(1 - (1 * preds_df.enc_label + preds_df.prob_outside))
  preds_df['ind_correct_pred'] = (preds_df.enc_label == preds_df.pred).astype(int)

  # Left join patient characteristics
  patient_chars_df = pd.read_csv(data_path + '/mapping.csv', sep=';')
  preds_df_incl_patient_chars = preds_df.merge(patient_chars_df, on='tiff', how='left')

  # Create new group where COPD stages I, II, III, IV are joined
  preds_df_incl_patient_chars['copd_group_full'] = 'COPD'
  preds_df_incl_patient_chars.loc[preds_df_incl_patient_chars.copd_group == 'Normal', 'copd_group_full'] =  'Normal'

  # Calculate patient-level aggregations
  patient_aggs = preds_df_incl_patient_chars.groupby('patient').agg({
      'tiff': 'nunique',
      'airway': 'nunique',
      'image': 'count',
      'label': lambda x: sum(x == 'Inside')
  }).rename(columns={
      'tiff': 'n_tiffs',
      'airway': 'n_airways', 
      'image': 'n_images',
      'label': 'n_submucosa'
  })

  # Calculate additional metrics
  patient_aggs['n_adventitia'] = patient_aggs['n_images'] - patient_aggs['n_submucosa']
  patient_aggs['perc_submucosa'] = patient_aggs['n_submucosa'] / patient_aggs['n_images']

  # Merge aggregations back with original dataframe
  preds_df_final = preds_df_incl_patient_chars.merge(
      patient_aggs,
      left_on='patient',
        right_index=True
    )

  # Remove duplicate images ending with _2.png
  preds_df_final = preds_df_final[~preds_df_final.image.str.endswith('_2.png')] 
  preds_df_final = preds_df_final.drop(preds_df_final[preds_df_final.image.str[-6:] == '_2.png'].index)

  model_without_top = Model(inputs=model.inputs, outputs=model.get_layer('global_average_pooling2d_1').output)
  features = model_without_top.predict_generator(test_gen, steps = None, verbose=1)
  features_df = pd.DataFrame(features, columns = ['f_' + str(i) for i in range(features.shape[1])])
  features_df['image'] = [os.path.basename(f) for f in filenames]

  preds_and_features_df = preds_df_final.merge(features_df, on='image')

  return preds_df_final, preds_and_features_df

"""## Functions to analyze (visualize) model"""

# Function to plot confusion matrix
def plot_cm(y_true, y_pred, output_path, figsize=(10,10)):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.set(font_scale=2)
    sns.heatmap(cm, cmap= "Blues", annot=annot, fmt='', ax=ax)
    fig.savefig(output_path)

# Function to plot histogram, with two vertical lines if desired
def plot_histogram(data, xlabel, range, vertical_line_1, vertical_line_2, output_path):
  fig = plt.figure()
  plt.hist(data, density = True, range=range)
  plt.xlabel(xlabel)
  plt.ylabel('Density')

  if vertical_line_1 != None:
    plt.axvline(vertical_line_1, color='k', linestyle='dashed', linewidth=1)
  if vertical_line_2 != None:
    plt.axvline(vertical_line_2, color='k', linestyle='dashed', linewidth=1)

  fig.savefig(output_path)

try:
  os.mkdir(model_path + '/Visualizations/copd_group/')
except:
  pass


# Function to get feature maps for specific layer
def visualize_filter(model, file_name, layer, output_path):
    img_index = labels[labels['image']==file_name].index.values.astype(int)[0]
    img = data[img_index]
    image = expand_dims(img, axis=0)
    sub_model = Model(inputs=model.inputs, outputs=model.layers[layer].output)
    feature_maps = sub_model.predict(image)

    fig = plt.figure()

    ix = 1
    for _ in range(4):
        for _ in range(8):
            ax = plt.subplot(4, 8, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.imshow(feature_maps[0,:,:,ix-1], cmap = 'gray')
            ix += 1

    fig.suptitle('Layer ' + str(layer))

    try:
      os.makedirs(output_path + '/Visualizations/Filters/')
    except:
      pass

    fig.savefig(output_path + '/Visualizations/Filters/Layer=' + str(layer) + '_' + file_name)

    return fig

# Function to apply grad-cam (heatmaps visualization)
def grad_cam(image, model, eps=1e-8):
    gradModel = Model(
          inputs=[model.inputs],
          outputs=[model.get_layer(layerName).output,
            model.output])

    with tf.GradientTape() as tape:
          # cast the image tensor to a float-32 data type, pass the
          # image through the gradient model, and grab the loss
          # associated with the specific class index
          (convOutputs, predictions) = gradModel(np.expand_dims(img, axis=0))
          loss = predictions[:, 0]
    # use automatic differentiation to compute the gradients
    grads = tape.gradient(loss, convOutputs)

    # compute the guided gradients
    castConvOutputs = tf.cast(convOutputs > 0, "float32")
    castGrads = tf.cast(grads > 0, "float32")
    guidedGrads = castConvOutputs * castGrads * grads
    # the convolution and guided gradients have a batch dimension
    # (which we don't need) so let's grab the volume itself and
    # discard the batch
    convOutputs = convOutputs[0]
    guidedGrads = guidedGrads[0]

    # compute the average of the gradient values, and using them
    # as weights, compute the ponderation of the filters with
    # respect to the weights
    weights = tf.reduce_mean(guidedGrads, axis=(0, 1))
    cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)

    (w, h) = (img.shape[0], img.shape[1])
    heatmap = cv2.resize(cam.numpy(), (w, h))
    # normalize the heatmap such that all values lie in the range
    # [0, 1], scale the resulting values to the range [0, 255],
    # and then convert to an unsigned 8-bit integer
    numer = heatmap - np.min(heatmap)
    denom = (heatmap.max() - heatmap.min()) + eps
    heatmap = numer / denom
    heatmap = (heatmap * 255).astype("uint8")

    heatmap = cv2.resize(heatmap, (img.shape[0], img.shape[1]))
    heatmap = cv2.applyColorMap(heatmap, colormap=cv2.COLORMAP_JET)

    return heatmap

# function to define apply_modifications
def apply_modifications(model, custom_objects=None):
  """Applies modifications to the model layers to create a new Graph. For example, simply changing
  `model.layers[idx].activation = new activation` does not change the graph. The entire graph needs to be updated
  with modified inbound and outbound tensors because of change in layer building function.
  Args:
      model: The `keras.models.Model` instance.
  Returns:
      The modified model with changes applied. Does not mutate the original `model`.
  """
  # The strategy is to save the modified model and load it back. This is done because setting the activation
  # in a Keras layer doesnt actually change the graph. We have to iterate the entire graph and change the
  # layer inbound and outbound nodes with modified tensors. This is doubly complicated in Keras 2.x since
  # multiple inbound and outbound nodes are allowed with the Graph API.
  model_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.h5')
  try:
      model.save(model_path)
      return load_model(model_path, custom_objects=custom_objects)
  finally:
      os.remove(model_path)

# Function to cluster patches using UMAP cluster algorithm
def cluster_patches(output_path, features, file_names, labels):
    index_filenames = labels.image.index[labels.image.isin(file_names)]
    features_selected = features[index_filenames]
    labels_selected = labels.label[index_filenames]

    mapping = umap.UMAP(n_neighbors=20,
                        min_dist=0.001,
                        metric='correlation').fit(features_selected)

    if len(np.unique(labels_selected)) == 2:
      color_key = ['red','blue']
    if len(np.unique(labels_selected)) == 1 and labels_selected.iloc[0] == 'Outside':
      color_key = ['blue']
    if len(np.unique(labels_selected)) == 1 and labels_selected.iloc[0] == 'Inside':
      color_key=['red']

    plt.figure()
    p = umap.plot.points(mapping, labels=labels_selected, color_key=color_key)
    fig = p.get_figure()
    fig.savefig(output_path)


# Function to determine and classify UMAP clusters
def determine_cluster(features, file_names, labels):
  index_filenames = labels.image.index[labels.image.isin(file_names)]
  features_selected = features[index_filenames]

  standard_embedding = umap.UMAP(random_state=42).fit_transform(features_selected)

  clusterable_embedding = umap.UMAP(n_neighbors=200,
                                      min_dist=0.0,
                                      metric='correlation',
                                      n_components=100,
                                      random_state=42
                                    ).fit_transform(features_selected)

  # classify based on UMAP cluster, using HDBSCAN
  labels_pred_hdbscan = hdbscan.HDBSCAN(min_samples=100,
                                        min_cluster_size=500,
                                    ).fit_predict(clusterable_embedding)

  clustered = (labels_pred_hdbscan >= 0)

  return standard_embedding, clustered, labels_pred_hdbscan

# Function to visualize clusters and quantify clusterability
# With the option to only select a specific subgroup through param file_names_selected
def visualize_and_quantify_cluster(output_path, file_names_full, file_names_selected, standard_embediing, clustered, labels_pred_hdbscan):

  # get labels corresponding to selected filenames
  index_filenames = labels.image.index[labels.image.isin(file_names_selected)]
  labels_selected = labels.label[index_filenames]

  # get embeddings and classifications corresponding to selected filenames
  index_filenames = [file_names_full.index(f) for f in file_names_selected]
  standard_embedding_selected = standard_embedding[index_filenames]
  clustered_selected = clustered[index_filenames]
  labels_pred_hdbscan_selected = labels_pred_hdbscan[index_filenames]

  cdict = {'Submucosa': 'red',
           'Adventitia': 'blue'
  }

  # Original clusters
  plt.figure()
  for compartment in np.unique(labels_selected):
    ix = (labels_selected == compartment)
    compartment = 'Submucosa' if compartment == 'Inside' else 'Adventitia'
    plt.scatter(standard_embedding_selected[ix, 0], standard_embedding_selected[ix, 1], c = cdict[compartment], label = compartment, s = 0.2)

  lgnd = plt.legend(loc = 'upper right', markerscale=9)
  plt.savefig(output_path.replace('_classified',''))

  # Classified clusters
  plt.figure()
  plt.scatter(standard_embedding_selected[~clustered_selected, 0],
            standard_embedding_selected[~clustered_selected, 1],
            c=(0.5, 0.5, 0.5),
            s=0.2,
            alpha=0.5)

  for compartment in np.unique(labels_selected):
    ix = (labels_selected == compartment)
    clustered_compartment = clustered_selected & ix
    compartment = 'Submucosa' if compartment == 'Inside' else 'Adventitia'
    plt.scatter(standard_embedding_selected[clustered_compartment, 0], standard_embedding_selected[clustered_compartment, 1], c = cdict[compartment], label = compartment, s = 0.2)

  lgnd = plt.legend(loc = 'upper right', markerscale=9)

  plt.savefig(output_path)

  # quantify goodness of cluster
  # on all data, including noise
  homogen_score_full = metrics.homogeneity_score(labels_selected, labels_pred_hdbscan_selected)
  complete_score_full = metrics.completeness_score(labels_selected, labels_pred_hdbscan_selected)
  v_score_full = metrics.v_measure_score(labels_selected, labels_pred_hdbscan_selected)

  n_noise = len(labels_selected) - len(labels_selected[clustered_selected])
  n_noise_submucosa = len([l for l in labels_selected if l == 'Inside']) - len([l for l in labels_selected[clustered_selected] if l == 'Inside'])
  n_noise_adventitia = len([l for l in labels_selected if l == 'Outside']) - len([l for l in labels_selected[clustered_selected] if l == 'Outside'])

  # on only predictions for which HDBSCAN is sure
  homogen_score_clustered = metrics.homogeneity_score(labels_selected[clustered_selected], labels_pred_hdbscan_selected[clustered_selected])
  complete_score_clustered = metrics.completeness_score(labels_selected[clustered_selected], labels_pred_hdbscan_selected[clustered_selected])
  v_score_clustered = metrics.v_measure_score(labels_selected[clustered_selected], labels_pred_hdbscan_selected[clustered_selected])

  # simple metrics
  n_patches = len(labels_selected)
  n_submucosa = len([l for l in labels_selected if l == 'Inside'])
  n_adventitia = len([l for l in labels_selected if l == 'Outside'])

  return n_patches, n_submucosa, n_adventitia, homogen_score_full, complete_score_full, v_score_full, homogen_score_clustered, complete_score_clustered, v_score_clustered, n_noise, n_noise_submucosa, n_noise_adventitia

"""# 1. Read and predict on data

"""

# Input data configurations
org_size = [120, 120]   # original size of input
size = [224,224]        # input size of model (for Inception and ResNet: 224 x 224)
data_type = 'random'
threshold = 0.2

data_path = '/content/drive/My Drive/Esmee/' + str(org_size[0]) + 'x' + str(org_size[1]) + '/Data_' + str(data_type) + '/Patches_cutoff_' + str(threshold) + '/' # Path where data is in (per class one folder)

# Train-validation-test split configurations
independent_sets = False
class_imb = 'Oversample'

# Model training configurations
tf_model = 'Inception-ResNet-V2'
full_retrain = True
freeze_till_block = 0
loss_func = 'binary_crossentropy'
lr = 0.005
decay = 0
dr = 0.6

# Create output folder
folder_name = 'threshold=' + str(threshold) + '_' + tf_model + '_' + class_imb + '_' + 'retrain=' + str(full_retrain) + '_' + str(freeze_till_block) + '_'+ loss_func + '_'+ str(lr) + '_'+ str(decay) + '_' + str(dr)
model_path = '/content/drive/My Drive/Esmee/' +  str(org_size[0]) + 'x' + str(org_size[1]) + '/Results_' + str(data_type) + '_threshold_' + str(threshold) + '/' + folder_name #+ '_full_set'
model_path = '/content/drive/My Drive/Esmee/' +  str(org_size[0]) + 'x' + str(org_size[1]) + '/Results_' + str(data_type) + '/' + folder_name #+ '_full_set'

history_df = pd.read_csv(model_path + '/history.csv')

# Visualize training per class
acc_class_0 = np.array(history_df['tn']) / (np.array(history_df['tn']) + np.array(history_df['fp'])) * 100
val_acc_class_0 = np.array(history_df['val_tn']) / (np.array(history_df['val_tn']) + np.array(history_df['val_fp'])) * 100

plt.figure()
plt.plot(acc_class_0)
plt.plot(val_acc_class_0)
plt.ylim((50,100))
plt.title('Development of classification accuracy for class 0 (Inside)')
plt.ylabel('% Correct classification')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(model_path + '/acc_class0_development.png')

acc_class_1 = np.array(history_df['tp']) / (np.array(history_df['tp']) + np.array(history_df['fn'])) * 100
val_acc_class_1 = np.array(history_df['val_tp']) / (np.array(history_df['val_tp']) + np.array(history_df['val_fn'])) * 100

plt.figure()
plt.plot(acc_class_1)
plt.plot(val_acc_class_1)
plt.ylim((50,100))
plt.title('Development of classification accuracy for class 1 (Outside)')
plt.ylabel('% Correct classification')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig(model_path + '/acc_class1_development.png')

"""### A. If predictions not yet calculated"""

print('Loading model')
model = load_model(model_path + '/weights_auc.hdf5')
model.summary()

# print('Reading data')
# data, labels, enc_labels = read_data(data_path, size)

print('Calculating predictions')
preds_df, preds_and_features_df = calculate_pred_and_features()
preds_df.to_csv(model_path + '/Visualizations/model_predictions.csv')
preds_and_features_df.to_csv(model_path + '/Visualizations/model_predictions_with_features.csv')

# print('Generate confusing matrix on full set')
# obs = preds_df.label
# preds = preds_df.pred_label
# plot_cm(obs, preds, model_path + '/Visualizations/confusion_matrix_auc_model.psvgng')

# print('Determining prototypes')
# tn_certain, tp_certain, fn_certain, fp_certain = determine_prototypes(1000, model_path)

"""### B. If predictions and features yet calculated"""

preds_df = pd.read_csv(model_path + '/Visualizations/model_predictions.csv', index_col=0)
output_df = pd.read_csv(model_path + '/Visualizations/model_predictions_with_features.csv', index_col=0)

"""# 2. Basic analyses per intersection

### Aggregates on patient, WSI and airway level
"""

agg_patient_query = (" SELECT patient, "
            "copd_group, "
            "smoking_group, "
            "gold_stage, "
            "borderline, "
            "COUNT(DISTINCT tiff)                                   AS n_tiffs,  "
            "COUNT(*)                                               AS n_patches, "
            "SUM(enc_label)                                         AS n_adventitia, "
            "COUNT(*) - SUM(enc_label)                              AS n_submucosa, "
            "(COUNT(*) - SUM(enc_label)) / (1.000 * COUNT(*))       AS perc_submucosa,"
            "SUM(CASE WHEN enc_label = 1 AND ind_correct_pred = 1 THEN 1 ELSE 0 END)  AS tp, "
            "SUM(CASE WHEN enc_label = 0 AND ind_correct_pred = 1 THEN 1 ELSE 0 END)  AS tn, "
            "SUM(CASE WHEN enc_label = 1 AND ind_correct_pred = 0 THEN 1 ELSE 0 END)  AS fp, "
            "SUM(CASE WHEN enc_label = 0 AND ind_correct_pred = 0 THEN 1 ELSE 0 END)  AS fn "
      "FROM preds_df "
      "GROUP BY patient, copd_group, smoking_group, gold_stage, borderline")

aggs_patient = ps.sqldf(agg_patient_query)
aggs_patient.to_csv(model_path + '/Visualizations/patient_aggregates.csv')

agg_wsi_query = (" SELECT patient, "
            "copd_group, "
            "smoking_group, "
            "gold_stage, "
            "borderline, "
            "tiff, "
            "SUM(ind_correct_pred) / (1.000 * COUNT(*))             AS acc,  "
            "COUNT(*)                                               AS n_patches, "
            "SUM(enc_label)                                         AS n_adventitia, "
            "COUNT(*) - SUM(enc_label)                              AS n_submucosa, "
            "(COUNT(*) - SUM(enc_label)) / (1.000 * COUNT(*))       AS perc_submucosa,"
            "SUM(CASE WHEN enc_label = 1 AND ind_correct_pred = 1 THEN 1 ELSE 0 END)  AS tp, "
            "SUM(CASE WHEN enc_label = 0 AND ind_correct_pred = 1 THEN 1 ELSE 0 END)  AS tn, "
            "SUM(CASE WHEN enc_label = 1 AND ind_correct_pred = 0 THEN 1 ELSE 0 END)  AS fp, "
            "SUM(CASE WHEN enc_label = 0 AND ind_correct_pred = 0 THEN 1 ELSE 0 END)  AS fn "
      "FROM preds_df "
      "GROUP BY patient, tiff, copd_group, smoking_group, gold_stage, borderline")

aggs_wsi = ps.sqldf(agg_wsi_query)
aggs_wsi.to_csv(model_path + '/Visualizations/wsi_aggregates.csv')


agg_airway_query = (" SELECT patient, "
            "copd_group, "
            "smoking_group, "
            "gold_stage, "
            "borderline, "
            "tiff, "
            "airway, "
            "SUM(ind_correct_pred) / (1.000 * COUNT(*))             AS acc,  "
            "COUNT(*)                                               AS n_patches, "
            "SUM(enc_label)                                         AS n_adventitia, "
            "COUNT(*) - SUM(enc_label)                              AS n_submucosa, "
            "(COUNT(*) - SUM(enc_label)) / (1.000 * COUNT(*))       AS perc_submucosa,"
            "SUM(CASE WHEN enc_label = 1 AND ind_correct_pred = 1 THEN 1 ELSE 0 END)  AS tp, "
            "SUM(CASE WHEN enc_label = 0 AND ind_correct_pred = 1 THEN 1 ELSE 0 END)  AS tn, "
            "SUM(CASE WHEN enc_label = 1 AND ind_correct_pred = 0 THEN 1 ELSE 0 END)  AS fp, "
            "SUM(CASE WHEN enc_label = 0 AND ind_correct_pred = 0 THEN 1 ELSE 0 END)  AS fn "
      "FROM preds_df "
      "GROUP BY patient, tiff, airway, copd_group, smoking_group, gold_stage, borderline")

aggs_airway = ps.sqldf(agg_airway_query)
aggs_airway.to_csv(model_path + '/Visualizations/airway_aggregates.csv')

"""### RUN! Intersection on group level (smoking history, COPD) and patient level

"""

min_patches_per_compartment = 10

for intersection in ['smoking_group', 'copd_group', 'copd_group_full']:
    # Create folder for visualizations per intersection
    try:
      os.makedirs(model_path + '/Visualizations/' + intersection + '/')
    except:
      pass

    # Filter specific groups
    if intersection == 'smoking_group':
      preds_filter =  preds_df[(preds_df.smoking_group != '?') & (preds_df.smoking_group != 'ExS < year')]
    if intersection == 'copd_group':
      preds_filter = preds_df[(preds_df.smoking_group == 'ExS >= year') & (preds_df.copd_group != 'Else')]

    # Aggregation per group: accuracy, counts, true/false positives/negatives
    agg_query = (" SELECT A." + intersection + ", "
            "SUM(ind_correct_pred) / (1.000 * COUNT(*))             AS acc,  "
            "COUNT(DISTINCT A.patient)                              AS n_patients, "
            "COUNT(DISTINCT A.tiff)                                 AS n_tiffs, "
            "COUNT(*)                                               AS n_patches, "
            "SUM(enc_label)                                         AS n_outside, "
            "COUNT(*) - SUM(enc_label)                              AS n_inside, "
            "SUM(CASE WHEN enc_label = 1 AND ind_correct_pred = 1 THEN 1 ELSE 0 END)  AS tp, "
            "SUM(CASE WHEN enc_label = 0 AND ind_correct_pred = 1 THEN 1 ELSE 0 END)  AS tn, "
            "SUM(CASE WHEN enc_label = 1 AND ind_correct_pred = 0 THEN 1 ELSE 0 END)  AS fp, "
            "SUM(CASE WHEN enc_label = 0 AND ind_correct_pred = 0 THEN 1 ELSE 0 END)  AS fn "
      "FROM preds_filter A "
      "LEFT JOIN ( "
          "SELECT " + intersection + ", "
                "Tiff, "
                "SUM(ind_correct_pred) / COUNT(*)                    AS tiff_acc "
          "FROM preds_filter "
          "GROUP BY " + intersection + ", tiff "
      ") B "
      "ON A." + intersection + " = B." + intersection + " "
        "AND A.tiff = B.tiff "
      "GROUP BY A." + intersection )

    aggs = ps.sqldf(agg_query)
    aggs.to_csv(model_path + '/Visualizations/' + intersection + '/aggregates.csv')

    # Barchart: # Correct/incorrect preds per group
    count_cor_preds_df = preds_filter.groupby([intersection, 'ind_correct_pred'])[intersection].count().unstack('ind_correct_pred')
    plot = count_cor_preds_df.plot(kind="bar")
    plot.legend(['Incorrect prediction', 'Correct prediction'])
    fig = plot.get_figure()
    fig.savefig(model_path + '/Visualizations/' + intersection + '/barchart_count_cor_preds.svg')

    # Confusion matrix:
    for group in list(np.unique(preds_filter[intersection].astype(str))):
      group_df = preds_filter[(preds_filter[intersection].astype(str) == group)]
      obs = group_df.label
      preds = group_df.pred_label

      plot_cm(obs, preds, model_path + '/Visualizations/' + intersection + '/confusion_matrix_' + str(group.replace('/','')) + '.png')
      sns.set()
      sns.set_style("whitegrid", {'axes.grid' : False})


    # Violin plots
    # Correct predictions - both lung compartments
    fig = plt.figure()
    ax = sns.violinplot(x=intersection, y='prob_correct', inner='quartile', data=preds_filter, color="silver", cut=0, order=preds_filter[intersection].sort_values(ascending=True).unique())         # violin plot with horizontal bars for median and quartiles
    #sns.stripplot(x=intersection, y='prob_correct', data=preds_filter, jitter=True, zorder=1)     # add jiter
    for l in ax.lines[1::3]:
      l.set_linestyle('-')
      l.set_linewidth(1.2)
      l.set_color('black')
      l.set_alpha(0.8)
    if intersection == 'smoking_group':
      plt.xlabel('Intersection on smoking history')
    else:
      plt.xlabel('Intersection on lung function')
    plt.ylabel('True class probability')
    ax.plot()
    fig.savefig(model_path + '/Visualizations/' + intersection + '/violinplot_correct_probs.png')

    # Correct predictions - Submucosa
    preds_filter_inside = preds_filter[preds_filter.label == 'Inside']
    fig = plt.figure()
    ax = sns.violinplot(x=intersection, y='prob_correct', inner='quartile', data=preds_filter_inside, color="salmon", cut=0, order=preds_filter_inside[intersection].sort_values(ascending=True).unique())         # violin plot with horizontal bars for median and quartiles
    #sns.stripplot(x=intersection, y='prob_correct', data=preds_filter_inside, jitter=True, zorder=1)     # add jiter
    for l in ax.lines[1::3]:
      l.set_linestyle('-')
      l.set_linewidth(1.2)
      l.set_color('black')
      l.set_alpha(0.8)
    if intersection == 'smoking_group':
      plt.xlabel('Intersection on smoking history')
    else:
      plt.xlabel('Intersection on lung function')
    plt.ylabel('True class probability')
    plt.title('Correct probability distribution - Submucosa')
    ax.plot()
    fig.savefig(model_path + '/Visualizations/' + intersection + '/violinplot_submucosa_correct_probs.png')

    # Correct predictions - Adventitia
    preds_filter_outside = preds_filter[preds_filter.label == 'Outside']
    fig = plt.figure()
    ax = sns.violinplot(x=intersection, y='prob_correct', inner='quartile', data=preds_filter_outside, color="skyblue", cut=0, order=preds_filter_outside[intersection].sort_values(ascending=True).unique())         # violin plot with horizontal bars for median and quartiles
    #sns.stripplot(x=intersection, y='prob_correct', data=preds_filter_outside, jitter=True, zorder=1)     # add jiter
    for l in ax.lines[1::3]:
      l.set_linestyle('-')
      l.set_linewidth(1.2)
      l.set_color('black')
      l.set_alpha(0.8)
    if intersection == 'smoking_group':
      plt.xlabel('Intersection on smoking history')
    else:
      plt.xlabel('Intersection on lung function')
    plt.ylabel('True class probability')
    plt.title('Correct probability distribution - Adventitia')
    ax.plot()
    fig.savefig(model_path + '/Visualizations/' + intersection + '/violinplot_adventitia_correct_probs.png')

    # Per group within intersection on patient level
    intersection = 'copd_group'
    for group in list(np.unique(preds_filter[intersection].astype(str))):
      group_df = preds_filter[(preds_filter[intersection].astype(str) == group) & (preds_filter.n_submucosa >= min_patches_per_compartment) & (preds_filter.n_adventitia >= min_patches_per_compartment)]
      group_df.patient = group_df.patient.astype(int)

      try:
        # Correct predictions - both lung compartments
        fig = plt.figure(figsize=(12, 6))
        ax = sns.violinplot(x='patient', y='prob_correct', inner='quartile', data=group_df, color="silver", cut=0, order=group_df.patient.sort_values(ascending=True).unique())         # violin plot with horizontal bars for median and quartiles
        #sns.stripplot(x='patient', y='prob_correct', data=group_df, jitter=True, zorder=1)     # add jiter
        for l in ax.lines[1::3]:
          l.set_linestyle('-')
          l.set_linewidth(1.2)
          l.set_color('black')
          l.set_alpha(0.8)
        plt.xlabel('Patient')
        plt.ylabel('True class probability')
        plt.title('Correct probability distribution')
        ax.plot()
        fig.savefig(model_path + '/Visualizations/' + intersection + '/violinplot_' + str(group.replace('/','')) + '_correct_probs.png')
      except:
        pass

      # Correct predictions - Submucosa
      try:
        group_df_inside = group_df[group_df.label == 'Inside']
        fig = plt.figure()
        ax = sns.violinplot(x='patient', y='prob_correct', inner='quartile', data=group_df_inside, color="salmon", cut=0, order=group_df.patient.sort_values(ascending=True).unique())          # violin plot with horizontal bars for median and quartiles
        #sns.stripplot(x=intersection, y='prob_correct', data=preds_filter_inside, jitter=True, zorder=1)     # add jiter
        for l in ax.lines[1::3]:
          l.set_linestyle('-')
          l.set_linewidth(1.2)
          l.set_color('black')
          l.set_alpha(0.8)
        plt.xlabel('Patient')
        plt.ylabel('True class probability')
        plt.title('Correct probability distribution - Submucosa')
        ax.plot()
        fig.savefig(model_path + '/Visualizations/' + intersection + '/violinplot_submucosa_' + str(group.replace('/','')) + '_correct_probs.png')
      except:
        pass

      # Correct predictions - Adventitia
      try:
        group_df_outside = group_df[group_df.label == 'Outside']
        fig = plt.figure()
        ax = sns.violinplot(x='patient', y='prob_correct', inner='quartile', data=group_df_outside, color="skyblue", cut=0, order=group_df.patient.sort_values(ascending=True).unique())          # violin plot with horizontal bars for median and quartiles
        #sns.stripplot(x=intersection, y='prob_correct', data=preds_filter_inside, jitter=True, zorder=1)     # add jiter
        for l in ax.lines[1::3]:
          l.set_linestyle('-')
          l.set_linewidth(1.2)
          l.set_color('black')
          l.set_alpha(0.8)
        plt.xlabel('Patient')
        plt.ylabel('True class probability')
        plt.title('Correct probability distribution - Adventitia')
        ax.plot()
        fig.savefig(model_path + '/Visualizations/' + intersection + '/violinplot_adventitia_' + str(group.replace('/','')) + '_correct_probs.png')
      except:
        pass

intersection = 'copd_group'

preds_filter = preds_df[preds_df.smoking_group == 'ExS >= year']
preds_filter['new_division'] = 'Else'
preds_filter.loc[(preds_filter.gold_stage == 4), 'new_division'] = 'COPD stage IV'
preds_filter.loc[(preds_filter.gold_stage == 3), 'new_division'] = 'COPD stage III'
preds_filter.loc[(preds_filter.copd_group == 'Normal'), 'new_division'] = 'Normal'
preds_filter.loc[(preds_filter.borderline == 1), 'new_division'] = 'Borderline'


# Violin plots
# Correct predictions - both lung compartments
fig = plt.figure()
ax = sns.violinplot(x='new_division', y='prob_correct', inner='quartile', data=preds_filter, color="silver", cut=0, order=preds_filter['new_division'].sort_values(ascending=True).unique())         # violin plot with horizontal bars for median and quartiles
#sns.stripplot(x=intersection, y='prob_correct', data=preds_filter, jitter=True, zorder=1)     # add jiter
for l in ax.lines[1::3]:
  l.set_linestyle('-')
  l.set_linewidth(1.2)
  l.set_color('black')
  l.set_alpha(0.8)
plt.xlabel('Intersection on lung function')
plt.ylabel('True class probability')
ax.plot()
fig.savefig(model_path + '/Visualizations/' + intersection + '/violinplot_correct_probs_detailed.svg')

# Correct predictions - Submucosa
preds_filter_inside = preds_filter[preds_filter.label == 'Inside']
fig = plt.figure()
ax = sns.violinplot(x='new_division', y='prob_correct', inner='quartile', data=preds_filter_inside, color="salmon", cut=0, order=preds_filter_inside['new_division'].sort_values(ascending=True).unique())         # violin plot with horizontal bars for median and quartiles
#sns.stripplot(x=intersection, y='prob_correct', data=preds_filter_inside, jitter=True, zorder=1)     # add jiter
for l in ax.lines[1::3]:
  l.set_linestyle('-')
  l.set_linewidth(1.2)
  l.set_color('black')
  l.set_alpha(0.8)
plt.xlabel('Intersection on lung function')
plt.ylabel('True class probability')
plt.title('Correct probability distribution - Submucosa')
ax.plot()
fig.savefig(model_path + '/Visualizations/' + intersection + '/violinplot_submucosa_correct_probs_detailed.svg')

# Correct predictions - Adventitia
preds_filter_outside = preds_filter[preds_filter.label == 'Outside']
fig = plt.figure()
ax = sns.violinplot(x='new_division', y='prob_correct', inner='quartile', data=preds_filter_outside, color="skyblue", cut=0, order=preds_filter_outside['new_division'].sort_values(ascending=True).unique())         # violin plot with horizontal bars for median and quartiles
#sns.stripplot(x=intersection, y='prob_correct', data=preds_filter_outside, jitter=True, zorder=1)     # add jiter
for l in ax.lines[1::3]:
  l.set_linestyle('-')
  l.set_linewidth(1.2)
  l.set_color('black')
  l.set_alpha(0.8)
plt.xlabel('Intersection on lung function')
plt.ylabel('True class probability')
plt.title('Correct probability distribution - Adventitia')
ax.plot()
fig.savefig(model_path + '/Visualizations/' + intersection + '/violinplot_adventitia_correct_probs_detailed.svg')

sns.set()
sns.set_style("whitegrid", {'axes.grid' : False})

# Density plot true class distribution per lung function group
for group in list(np.unique(aggs_patient['copd_group'].astype(str))):
    group_df = preds_df[(preds_df['copd_group'].astype(str) == group) & (preds_df['smoking_group'] == 'ExS >= year')]

    fig = plt.figure()
    ax = sns.kdeplot(data=group_df, x=group_df.prob_correct, hue=group_df.official_label, fill=True, cut=0, common_norm=False, alpha=0.3, palette = ['salmon', 'cornflowerblue'])
    plt.legend(title='Lung compartment', loc='upper left', labels=['Adventitia', 'Submucosa'])
    plt.xlabel('True class probability')
    plt.ylabel('Probability density')
    plt.ylim((0,5))
    fig.savefig(model_path + '/Visualizations/copd_group/combined_density_plot_' + group.replace('/','') + '.png')


# Density plot true class distribution per lung compartment
for lung_comp in list(np.unique(preds_df['label'].astype(str))):
    lung_comp_df = preds_df[(preds_df['label'].astype(str) == lung_comp) & (preds_df['smoking_group'] == 'ExS >= year') & (preds_df['copd_group'] != 'Else')]

    fig = plt.figure()
    ax = sns.kdeplot(data=lung_comp_df, x=lung_comp_df.prob_correct, hue=lung_comp_df.copd_group, cut=0, fill=True, common_norm=False, alpha=0.3)
    plt.legend(title='Lung function', loc='upper left', labels=['COPD stage III or IV', 'Normal'])
    plt.xlabel('True class probability')
    plt.ylabel('Probability density')
    plt.ylim((0,5))
    fig.savefig(model_path + '/Visualizations/copd_group/combined_density_plot_' + lung_comp.replace('/','') + '.png')

"""# 3. Hypothesis testing - True class probabilities

## Perform hierarchical bootstrapping

### RUN! 1. Group level (lung function) - no aggregation (data points of patches)
"""

# NAIVE TESTING: TEST FOR SIGNIFICANCE WITHOUT BOOTSTRAPPING

# Exclude patients with less than 10 patches for one compartment
min_patches_per_compartment = 10
#bootstrap_patients_copd = list(aggs_patient.patient[(aggs_patient.copd_group == 'COPD stage III or IV') & (aggs_patient.smoking_group == 'ExS >= year') & (aggs_patient.n_adventitia >= min_patches_per_compartment) & (aggs_patient.n_submucosa >= min_patches_per_compartment)])
bootstrap_patients_copd = list(aggs_patient.patient[(aggs_patient.gold_stage > 0) & (aggs_patient.copd_group != 'Normal') & (aggs_patient.smoking_group == 'ExS >= year') & (aggs_patient.n_adventitia >= min_patches_per_compartment) & (aggs_patient.n_submucosa >= min_patches_per_compartment)])
bootstrap_patients_normal = list(aggs_patient.patient[(aggs_patient.copd_group == 'Normal') & (aggs_patient.smoking_group == 'ExS >= year') & (aggs_patient.n_adventitia >= min_patches_per_compartment) & (aggs_patient.n_submucosa >= min_patches_per_compartment)])

# Check number of patients per group
print('Total number of COPD patients: ' + str(aggs_patient[(aggs_patient.copd_group == 'COPD stage III or IV') & (aggs_patient.smoking_group == 'ExS >= year')].shape[0]))
print('Total number of healthy patients: ' + str(aggs_patient[(aggs_patient.copd_group == 'Normal') & (aggs_patient.smoking_group == 'ExS >= year')].shape[0]))

preds_df_filter = preds_df[preds_df.patient.isin(bootstrap_patients_copd + bootstrap_patients_normal)]

# Test per lung compartment
for lung_compartment in ['Inside', 'Outside']:
  copd_sample = list(preds_df_filter.prob_correct[(preds_df_filter.smoking_group == 'ExS >= year') & (preds_df_filter.copd_group == 'COPD stage III or IV') & (preds_df_filter.label == lung_compartment)])
  normal_sample = list(preds_df_filter.prob_correct[(preds_df_filter.smoking_group == 'ExS >= year') & (preds_df_filter.copd_group == 'Normal') & (preds_df_filter.label == lung_compartment)])

  # print quantity
  print(lung_compartment.upper() + ' - Number of COPD patients: ' + str(len(preds_df_filter.patient[(preds_df_filter.smoking_group == 'ExS >= year') & (preds_df_filter.copd_group == 'COPD stage III or IV') & (preds_df_filter.label == lung_compartment)].unique())) +
        ' , number of patches: ' + str(len(copd_sample)))
  print(lung_compartment.upper() + ' - Number of normal patients: ' + str(len(preds_df_filter.patient[(preds_df_filter.smoking_group == 'ExS >= year') & (preds_df_filter.copd_group == 'Normal') & (preds_df_filter.label == lung_compartment)].unique())) +
        ' , number of patches: ' + str(len(normal_sample)))

  # Perform distribution testing
  u_stat, u_p = stats.mannwhitneyu(copd_sample, normal_sample)

  # Perform median testing
  median_copd = statistics.median(copd_sample)
  median_var = statistics.median(normal_sample)

  mood_stat, mood_p, mood_med, tbl = stats.median_test(copd_sample, normal_sample)

  # Perform variance testing
  var_copd = statistics.variance(copd_sample)
  var_normal = statistics.variance(normal_sample)

  # Parametric median Levene test
  levene_stat, levene_p = stats.levene(copd_sample, normal_sample, center='median')

  print(lung_compartment.upper() + '\n' +
    '- Distribution: Statistics=' + str(u_stat) + ', p=' + str(u_p) + '\n' +
    '- Median: Statistics=' + str(mood_stat) + ', p=' + str(mood_p) + '\n' +
    '- Variance: Statistics=' + str(levene_stat) + ', p=' + str(levene_p))

# Exclude patients with less than 10 patches for one compartment
min_patches_per_compartment = 10
bootstrap_patients_copd = list(aggs_patient.patient[(aggs_patient.copd_group == 'COPD stage III or IV') & (aggs_patient.smoking_group == 'ExS >= year') & (aggs_patient.n_adventitia >= min_patches_per_compartment) & (aggs_patient.n_submucosa >= min_patches_per_compartment)])
#bootstrap_patients_copd = list(aggs_patient.patient[(aggs_patient.gold_stage > 0) & (aggs_patient.copd_group != 'Normal') & (aggs_patient.smoking_group == 'ExS >= year') & (aggs_patient.n_adventitia >= min_patches_per_compartment) & (aggs_patient.n_submucosa >= min_patches_per_compartment)])
bootstrap_patients_normal = list(aggs_patient.patient[(aggs_patient.copd_group == 'Normal') & (aggs_patient.smoking_group == 'ExS >= year') & (aggs_patient.n_adventitia >= min_patches_per_compartment) & (aggs_patient.n_submucosa >= min_patches_per_compartment)])

# Bootstrap parameters
B = 10000 # number of bootstrap samples for hypothesis testing
patient_size = 5 # number of patients selected with replacement
compartment_size = 150 # number of patches from each patient/lung compartment selected with replacement

# Empty dataframe to write results of bootstrap samples to
group_tests_df = pd.DataFrame(columns=['lung_compartment', 'u_stat', 'u_p_value', 'mood_stat', 'mood_p_value', 'np_levene_stat', 'np_levene_p_value', 'levene_stat', 'levene_p_value', 'median_copd', 'median_normal', 'var_copd', 'var_normal'])

# Hierachical bootstrapping
for b in range(B):
  # Randomly select patients from copd and normal group
  copd_patients_sel = random.choices(bootstrap_patients_copd, k=patient_size)
  normal_patients_sel = random.choices(bootstrap_patients_normal, k=patient_size)

  for lung_compartment in ['Inside', 'Outside']:

    # Empty lists to fill per group
    copd_sample = []
    normal_sample = []

    for patient in copd_patients_sel:
        # Select submucosa/adventitia images from patients
        patient_compartment = list(preds_df.prob_correct[(preds_df.patient == patient) & (preds_df.label == lung_compartment)])
        patient_compartment_sel = random.choices(patient_compartment, k=compartment_size)

        copd_sample.extend(patient_compartment_sel)

    for patient in normal_patients_sel:
        # Select submucosa/adventitia images from patients
        patient_compartment = list(preds_df.prob_correct[(preds_df.patient == patient) & (preds_df.label == lung_compartment)])
        patient_compartment_sel = random.choices(patient_compartment, k=compartment_size)

        normal_sample.extend(patient_compartment_sel)

    # Perform distribution testing
    u_stat, u_p = stats.mannwhitneyu(copd_sample, normal_sample)

    # Perform median testing
    median_copd = statistics.median(copd_sample)
    median_normal = statistics.median(normal_sample)

    mood_stat, mood_p, mood_med, tbl = stats.median_test(copd_sample, normal_sample)

    # Perform variance testing
    var_copd = statistics.variance(copd_sample)
    var_normal = statistics.variance(normal_sample)

    # Parametric median Levene test
    levene_stat, levene_p = stats.levene(copd_sample, normal_sample, center='median')

    # Non-parametric Levene test (via rank transformations)
    copd_df = pd.DataFrame(['copd' for i in range(len(copd_sample))],columns = ['group'])
    copd_df['probs'] = copd_sample

    normal_df = pd.DataFrame(['normal' for i in range(len(copd_sample))],columns = ['group'])
    normal_df['probs'] = normal_sample

    combined_df = pd.concat([copd_df, normal_df], ignore_index = True)
    combined_df['rank'] = combined_df['probs'].rank(method ='min')

    #np_levene_stat, np_levene_p = stats.levene(list(combined_df[combined_df.group == 'copd'].rank), list(combined_df[combined_df.group == 'normal'].rank), center='mean')
    np_levene_stat = 0
    np_levene_p = 1

    print(lung_compartment.upper() + ' - Bootstrap ' + str(b) + '\n' +
          '- distribution: Statistics=' + str(u_stat) + ', p=' + str(u_p) + '\n' +
          '- variance:'  + '\n' +
          '   parametric: Statistics=' + str(levene_stat) + ', p=' + str(levene_p) + '\n' +
          '   non-parametric: Statistics=' + str(np_levene_stat) + ', p=' + str(np_levene_p))


    group_tests_df = group_tests_df.append({'lung_compartment': lung_compartment,
                                                        'u_stat': u_stat,
                                                        'ustat_p_value': u_p,
                                                        'mood_stat': mood_stat,
                                                        'mood_p_value': mood_p,
                                                        'np_levene_stat': np_levene_stat,
                                                        'np_levene_p_value': np_levene_p,
                                                        'levene_stat': levene_stat,
                                                        'levene_p_value': levene_p,
                                                        'median_copd': median_copd,
                                                        'median_normal': median_normal,
                                                        'var_copd': var_copd,
                                                        'var_normal': var_normal }, ignore_index=True)

# Critical value
alpha = 0.05

# Add significance indicators
group_tests_df['ind_significant_dist_diff'] = np.where(group_tests_df.ustat_p_value <= alpha, 1, 0)
group_tests_df['ind_significant_median_diff'] = np.where(group_tests_df.mood_p_value <= alpha, 1, 0)
group_tests_df['ind_significant_var_diff'] = np.where(group_tests_df.levene_p_value <= alpha, 1, 0)
group_tests_df['ind_significant_var_diff_np'] = np.where(group_tests_df.np_levene_stat <= alpha, 1, 0)

# Bootstrap results wegschrijven als CSV
group_tests_df.to_csv(model_path + '/Visualizations/group_level_test_results_COPD34_vs_normal.csv')


# Determine bootstrapped p-value
dist_significance_inside = 1 - group_tests_df[group_tests_df.lung_compartment == 'Inside'].ind_significant_dist_diff.sum() / (1.000 * group_tests_df[group_tests_df.lung_compartment == 'Inside'].ind_significant_dist_diff.count())
dist_significance_outside = 1 - group_tests_df[group_tests_df.lung_compartment == 'Outside'].ind_significant_dist_diff.sum() / (1.000 * group_tests_df[group_tests_df.lung_compartment == 'Outside'].ind_significant_dist_diff.count())
print('FOUND SIGNIFICANCE FOR DIFFERENCE IN DISTRIBUTIONS: ' + '\n' +
       '- Inside: ' + str(dist_significance_inside) + '\n' +
      '- Outside: ' + str(dist_significance_outside))

median_significance_inside = 1 - group_tests_df[group_tests_df.lung_compartment == 'Inside'].ind_significant_median_diff.sum() / (1.000 * group_tests_df[group_tests_df.lung_compartment == 'Inside'].ind_significant_median_diff.count())
median_significance_outside = 1 - group_tests_df[group_tests_df.lung_compartment == 'Outside'].ind_significant_median_diff.sum() / (1.000 * group_tests_df[group_tests_df.lung_compartment == 'Outside'].ind_significant_median_diff.count())
print('FOUND SIGNIFICANCE FOR DIFFERENCE IN MEDIANS: ' + '\n' +
       '- Inside: ' + str(median_significance_inside) + '\n' +
      '- Outside: ' + str(median_significance_outside))

var_significance_inside = 1 - group_tests_df[group_tests_df.lung_compartment == 'Inside'].ind_significant_var_diff.sum() / (1.000 * group_tests_df[group_tests_df.lung_compartment == 'Inside'].ind_significant_var_diff.count())
var_significance_outside = 1 - group_tests_df[group_tests_df.lung_compartment == 'Outside'].ind_significant_var_diff.sum() / (1.000 * group_tests_df[group_tests_df.lung_compartment == 'Outside'].ind_significant_var_diff.count())
var_significance_inside_np = 1 - group_tests_df[group_tests_df.lung_compartment == 'Inside'].ind_significant_var_diff_np.sum() / (1.000 * group_tests_df[group_tests_df.lung_compartment == 'Inside'].ind_significant_var_diff_np.count())
var_significance_outside_np = 1 - group_tests_df[group_tests_df.lung_compartment == 'Outside'].ind_significant_var_diff_np.sum() / (1.000 * group_tests_df[group_tests_df.lung_compartment == 'Outside'].ind_significant_var_diff_np.count())
print('FOUND SIGNIFICANCE FOR DIFFERENCE IN VARIANCES: ' + '\n' +
       '- Inside: ' + str(var_significance_inside) + ' - np: ' + str(var_significance_inside_np) + '\n' +
      '- Outside: ' + str(var_significance_outside) + ' - np: ' + str(var_significance_outside_np))


# 95% Confidence Interval (CI) for medians/variances using bootstrapping
# as in https://www.researchgate.net/publication/10821126_Indications_for_InterferonRibavirin_Therapy_in_Hepatitis_C_Patients_Findings_from_a_Survey_of_Canadian_Hepatologists
for lung_compartment in ['Inside', 'Outside']:

  # Plot distribution of medians for COPD/normal patients, including CI
  med_max = max(pd.concat([group_tests_df.median_copd[group_tests_df.lung_compartment == lung_compartment], group_tests_df.median_normal[group_tests_df.lung_compartment == lung_compartment]], axis=0))
  med_min = min(pd.concat([group_tests_df.median_copd[group_tests_df.lung_compartment == lung_compartment], group_tests_df.median_normal[group_tests_df.lung_compartment == lung_compartment]], axis=0))

  lower_bound_median_copd = np.percentile(group_tests_df.median_copd[group_tests_df.lung_compartment == lung_compartment], 5)
  upper_bound_median_copd = np.percentile(group_tests_df.median_copd[group_tests_df.lung_compartment == lung_compartment], 95)
  plot_histogram(group_tests_df.median_copd[group_tests_df.lung_compartment == lung_compartment], 'Median COPD patients', (med_min - 0.1, med_max + 0.1), lower_bound_median_copd, upper_bound_median_copd, model_path + '/Visualizations/distribution_median_copd_' + lung_compartment + '.png')

  lower_bound_median_normal = np.percentile(group_tests_df.median_normal[group_tests_df.lung_compartment == lung_compartment], 5)
  upper_bound_median_normal = np.percentile(group_tests_df.median_normal[group_tests_df.lung_compartment == lung_compartment], 95)
  plot_histogram(group_tests_df.median_normal[group_tests_df.lung_compartment == lung_compartment], 'Median normal patients', (med_min - 0.1, med_max + 0.1), lower_bound_median_normal, upper_bound_median_normal, model_path + '/Visualizations/distribution_median_normal_' + lung_compartment + '.png')

  # Plot distribution of variance for COPD/normal patients, including CI
  var_max = max(pd.concat([group_tests_df.var_copd[group_tests_df.lung_compartment == lung_compartment], group_tests_df.var_normal[group_tests_df.lung_compartment == lung_compartment]], axis=0))
  var_min = min(pd.concat([group_tests_df.var_copd[group_tests_df.lung_compartment == lung_compartment], group_tests_df.var_normal[group_tests_df.lung_compartment == lung_compartment]], axis=0))

  lower_bound_variance_copd = np.percentile(group_tests_df.var_copd[group_tests_df.lung_compartment == lung_compartment], 5)
  upper_bound_variance_copd = np.percentile(group_tests_df.var_copd[group_tests_df.lung_compartment == lung_compartment], 95)
  plot_histogram(group_tests_df.var_copd[group_tests_df.lung_compartment == lung_compartment], 'Variance COPD patients', (var_min - 0.01, var_max + 0.01), lower_bound_variance_copd, upper_bound_variance_copd, model_path + '/Visualizations/distribution_variance_copd_' + lung_compartment + '.png')

  lower_bound_variance_normal = np.percentile(group_tests_df.var_normal[group_tests_df.lung_compartment == lung_compartment], 5)
  upper_bound_variance_normal = np.percentile(group_tests_df.var_normal[group_tests_df.lung_compartment == lung_compartment], 95)
  plot_histogram(group_tests_df.var_normal[group_tests_df.lung_compartment == lung_compartment], 'Variance normal patients', (var_min - 0.01, var_max + 0.01), lower_bound_variance_normal, upper_bound_variance_normal, model_path + '/Visualizations/distribution_variance_normal_' + lung_compartment + '.png')

"""### 2. Group level (lung function) - aggregation on patient / lung compartment level"""

# Calculate median on patient, lung compartment level and join with aggs_per_patient
patient_medians = preds_df[['patient', 'label', 'copd_group', 'smoking_group', 'gold_stage', 'borderline', 'prob_correct']].groupby(['patient','label', 'copd_group', 'smoking_group', 'gold_stage', 'borderline']).median().reset_index()
patient_medians_2 = patient_medians.merge(aggs_patient[['patient', 'n_adventitia', 'n_submucosa']], on='patient', how='left')

# Calculate median on patient, lung compartment level and join with aggs_per_patient
patient_medians = preds_df[['patient', 'label', 'copd_group', 'smoking_group', 'gold_stage', 'borderline', 'prob_correct']].groupby(['patient','label', 'copd_group', 'smoking_group', 'gold_stage', 'borderline']).median().reset_index()
patient_medians = patient_medians.merge(aggs_patient[['patient', 'n_adventitia', 'n_submucosa']], on='patient', how='left')

# Exclude patients with less than 10 patches for one compartment
min_patches_per_compartment = 10

for lung_compartment in ['Inside', 'Outside']:

  copd_sample = list(patient_medians.prob_correct[(patient_medians.copd_group == 'COPD stage III or IV') & (patient_medians.smoking_group == 'ExS >= year') & (patient_medians.label == lung_compartment) & (patient_medians.n_adventitia >= min_patches_per_compartment) & (patient_medians.n_submucosa >= min_patches_per_compartment)])
  normal_sample = list(patient_medians.prob_correct[(patient_medians.copd_group == 'Normal') & (patient_medians.smoking_group == 'ExS >= year') & (patient_medians.label == lung_compartment) & (patient_medians.n_adventitia >= min_patches_per_compartment) & (patient_medians.n_submucosa >= min_patches_per_compartment)])

  # print quantity
  print(lung_compartment.upper() + ' - Number of COPD patients: ' + str(len(copd_sample)))
  print(lung_compartment.upper() + ' - Number of normal patients: ' + str(len(normal_sample)))

  # Perform distribution testing
  u_stat, u_p = stats.mannwhitneyu(copd_sample, normal_sample)

  # Perform median testing
  median_copd = statistics.median(copd_sample)
  median_var = statistics.median(normal_sample)

  mood_stat, mood_p, mood_med, tbl = stats.median_test(copd_sample, normal_sample)

  # Perform variance testing
  var_copd = statistics.variance(copd_sample)
  var_normal = statistics.variance(normal_sample)

  # Parametric median Levene test
  levene_stat, levene_p = stats.levene(copd_sample, normal_sample, center='median')

  print(lung_compartment.upper() + '\n' +
    '- Distribution: Statistics=' + str(u_stat) + ', p=' + str(u_p) + '\n' +
    '- Median: Statistics=' + str(mood_stat) + ', p=' + str(mood_p) + '\n' +
    '- Variance: Statistics=' + str(levene_stat) + ', p=' + str(levene_p))

"""### 3. Group level - ratio submucosa / adventitia on patient level"""

# Calculate median on patient, lung compartment level and join with aggs_per_patient
patient_medians = preds_df[['patient', 'label', 'copd_group', 'smoking_group', 'gold_stage', 'borderline', 'prob_correct']].groupby(['patient','label', 'copd_group', 'smoking_group', 'gold_stage', 'borderline']).median().reset_index()
patient_medians = patient_medians.merge(aggs_patient[['patient', 'n_adventitia', 'n_submucosa']], on='patient', how='left')

# Join lung compartment info
patient_medians_submucosa = patient_medians[patient_medians.label == 'Inside']
patient_medians_adventitia = patient_medians[patient_medians.label == 'Outside']

patient_medians = patient_medians_adventitia[['patient', 'copd_group', 'smoking_group', 'gold_stage', 'borderline', 'n_adventitia', 'n_submucosa', 'prob_correct']].merge(patient_medians_submucosa[['patient', 'prob_correct']], on='patient', how='left')
patient_medians['ratio'] = patient_medians.prob_correct_x / patient_medians.prob_correct_y

# Exclude patients with less than 10 patches for one compartment
min_patches_per_compartment = 10


copd_patients = patient_medians[(patient_medians.copd_group == 'COPD stage III or IV') & (patient_medians.smoking_group == 'ExS >= year') & (patient_medians.n_adventitia >= min_patches_per_compartment) & (patient_medians.n_submucosa >= min_patches_per_compartment)]
normal_patients = patient_medians[(patient_medians.copd_group == 'Normal') & (patient_medians.smoking_group == 'ExS >= year') & (patient_medians.n_adventitia >= min_patches_per_compartment) & (patient_medians.n_submucosa >= min_patches_per_compartment)]

copd_sample = list(copd_patients.ratio)
normal_sample = list(normal_patients.ratio)

# print quantity
print('Number of COPD patients: ' + str(len(copd_sample)))
print('Number of normal patients: ' + str(len(normal_sample)))

# Perform distribution testing
u_stat, u_p = stats.mannwhitneyu(copd_sample, normal_sample)

# Perform median testing
median_copd = statistics.median(copd_sample)
median_var = statistics.median(normal_sample)

mood_stat, mood_p, mood_med, tbl = stats.median_test(copd_sample, normal_sample)

# Perform variance testing
var_copd = statistics.variance(copd_sample)
var_normal = statistics.variance(normal_sample)

# Parametric median Levene test
levene_stat, levene_p = stats.levene(copd_sample, normal_sample, center='median')

print(
  '- Distribution: Statistics=' + str(u_stat) + ', p=' + str(u_p) + '\n' +
  '- Median: Statistics=' + str(mood_stat) + ', p=' + str(mood_p) + '\n' +
  '- Variance: Statistics=' + str(levene_stat) + ', p=' + str(levene_p))

full_selected_columns = ['patient', 'tiff', 'airway', 'label', 'copd_group', 'smoking_group', 'gold_stage', 'borderline', 'prob_correct']
full_groupby_columns = ['patient', 'tiff', 'airway', 'label', 'copd_group', 'smoking_group', 'gold_stage', 'borderline']

for intersection in ['patient', 'tiff', 'airway']:
  if intersection == 'patient':
    groupby_columns = [i for i in full_groupby_columns if i not in ['tiff', 'airway']]
    selected_columns = [i for i in full_selected_columns if i not in ['tiff', 'airway']]
  if intersection == 'tiff':
    groupby_columns = [i for i in full_groupby_columns if i not in ['airway']]
    selected_columns = [i for i in full_selected_columns if i not in ['airway']]
  else:
    groupby_columns = full_groupby_columns
    selected_columns = full_selected_columns

  # Calculate median on patient, lung compartment level and join with aggs_per_patient
  medians = preds_df[selected_columns].groupby(groupby_columns).median().reset_index()

  if intersection == 'patient':
    medians = medians.merge(aggs_patient[['patient', 'n_adventitia', 'n_submucosa']], on='patient', how='left')
  if intersection == 'tiff':
    medians = medians.merge(aggs_wsi[['tiff', 'n_adventitia', 'n_submucosa']], on='tiff', how='left')
  else:
    medians = medians.merge(aggs_wsi[['airway', 'n_adventitia', 'n_submucosa']], on='airway', how='left')

  # Join lung compartment info
  medians_submucosa = medians[medians.label == 'Inside']
  medians_adventitia = medians[medians.label == 'Outside']

  medians = medians_adventitia[[intersection, 'copd_group', 'smoking_group', 'gold_stage', 'borderline', 'n_adventitia', 'n_submucosa', 'prob_correct']].merge(medians_submucosa[[intersection, 'prob_correct']], on=intersection, how='left')
  medians['ratio'] = medians.prob_correct_x / medians.prob_correct_y

    # Exclude patients with less than 10 patches for one compartment
  min_patches_per_compartment = 10


  copd_medians = medians[(medians.copd_group == 'COPD stage III or IV') & (medians.smoking_group == 'ExS >= year') & (medians.n_adventitia >= min_patches_per_compartment) & (medians.n_submucosa >= min_patches_per_compartment)]
  normal_medians = medians[(medians.copd_group == 'Normal') & (medians.smoking_group == 'ExS >= year') & (medians.n_adventitia >= min_patches_per_compartment) & (medians.n_submucosa >= min_patches_per_compartment)]

  copd_sample = list(copd_medians.ratio)
  normal_sample = list(normal_medians.ratio)

  # print quantity
  print('Number of COPD patients: ' + str(len(copd_sample)))
  print('Number of normal patients: ' + str(len(normal_sample)))

  # Perform distribution testing
  u_stat, u_p = stats.mannwhitneyu(copd_sample, normal_sample)

  # Perform median testing
  median_copd = statistics.median(copd_sample)
  median_var = statistics.median(normal_sample)

  mood_stat, mood_p, mood_med, tbl = stats.median_test(copd_sample, normal_sample)

  # Perform variance testing
  var_copd = statistics.variance(copd_sample)
  var_normal = statistics.variance(normal_sample)

  # Parametric median Levene test
  levene_stat, levene_p = stats.levene(copd_sample, normal_sample, center='median')

  print(
    '- Distribution: Statistics=' + str(u_stat) + ', p=' + str(u_p) + '\n' +
    '- Median: Statistics=' + str(mood_stat) + ', p=' + str(mood_p) + '\n' +
    '- Variance: Statistics=' + str(levene_stat) + ', p=' + str(levene_p))

  plt.figure()
  plt.plot(copd_medians.prob_correct_x, copd_medians.prob_correct_y, 'o', color = 'black', label='COPD stage III or IV')
  plt.plot(normal_medians.prob_correct_x, normal_medians.prob_correct_y, 'o', color = 'lime', label='Normal')
  plt.xlabel('Adventitia')
  plt.ylabel('Submucosa')
  plt.legend()
  plt.savefig(model_path + '/Visualizations/copd_group/scatter_' + intersection + '_ratio_ins_out_COPD34_normal.svg')

plt.figure()
plt.plot(copd_patients.prob_correct_x, copd_patients.prob_correct_y, 'o', color = 'black', label='COPD stage III or IV')
plt.plot(normal_patients.prob_correct_x, normal_patients.prob_correct_y, 'o', color = 'lime', label='Normal')
plt.xlabel('Adventitia')
plt.ylabel('Submucosa')
plt.legend()
plt.savefig(model_path + '/Visualizations/copd_group/scatter_patient_ratio_ins_out_COPD34_normal.svg')

# Calculate median on patient, lung compartment level and join with aggs_per_patient
airway_medians = preds_df[['patient', 'airway', 'label', 'copd_group', 'smoking_group', 'gold_stage', 'borderline', 'prob_correct']].groupby(['patient', 'airway', 'label', 'copd_group', 'smoking_group', 'gold_stage', 'borderline']).median().reset_index()
airway_medians = airway_medians.merge(aggs_airway[['patient', 'airway', 'n_adventitia', 'n_submucosa']], on=['patient', 'airway'], how='left')

# Join lung compartment info
airway_medians_submucosa = airway_medians[airway_medians.label == 'Inside']
airway_medians_adventitia = airway_medians[airway_medians.label == 'Outside']

airway_medians = airway_medians_adventitia[['patient', 'airway', 'copd_group', 'smoking_group', 'gold_stage', 'borderline', 'n_adventitia', 'n_submucosa', 'prob_correct']].merge(airway_medians_submucosa[['patient', 'airway', 'prob_correct']], on=['patient', 'airway'], how='left')
airway_medians['ratio'] = airway_medians.prob_correct_x / airway_medians.prob_correct_y

# Exclude patients with less than 10 patches for one compartment
min_patches_per_compartment = 10

copd_patients = airway_medians[(airway_medians.copd_group == 'COPD stage III or IV') & (airway_medians.smoking_group == 'ExS >= year') & (airway_medians.n_adventitia >= min_patches_per_compartment) & (airway_medians.n_submucosa >= min_patches_per_compartment)]
normal_patients = airway_medians[(airway_medians.copd_group == 'Normal') & (airway_medians.smoking_group == 'ExS >= year') & (airway_medians.n_adventitia >= min_patches_per_compartment) & (airway_medians.n_submucosa >= min_patches_per_compartment)]

copd_sample = list(copd_patients.ratio)
normal_sample = list(normal_patients.ratio)

# print quantity
print('Number of COPD patients: ' + str(len(copd_sample)))
print('Number of normal patients: ' + str(len(normal_sample)))

# Perform distribution testing
u_stat, u_p = stats.mannwhitneyu(copd_sample, normal_sample)

# Perform median testing
median_copd = statistics.median(copd_sample)
median_var = statistics.median(normal_sample)

mood_stat, mood_p, mood_med, tbl = stats.median_test(copd_sample, normal_sample)

# Perform variance testing
var_copd = statistics.variance(copd_sample)
var_normal = statistics.variance(normal_sample)

# Parametric median Levene test
levene_stat, levene_p = stats.levene(copd_sample, normal_sample, center='median')

print(
  '- Distribution: Statistics=' + str(u_stat) + ', p=' + str(u_p) + '\n' +
  '- Median: Statistics=' + str(mood_stat) + ', p=' + str(mood_p) + '\n' +
  '- Variance: Statistics=' + str(levene_stat) + ', p=' + str(levene_p))

plt.figure()
plt.plot(copd_patients.prob_correct_x, copd_patients.prob_correct_y, 'o', color = 'black', label='COPD stage III or IV')
plt.plot(normal_patients.prob_correct_x, normal_patients.prob_correct_y, 'o', color = 'lime', label='Normal')
plt.xlabel('Adventitia')
plt.ylabel('Submucosa')
plt.legend()
plt.savefig(model_path + '/Visualizations/copd_group/scatter_airway_ratio_ins_out_COPD34_normal.svg')

"""### 3. Patient level"""

# Here we don't need bootstrapping, as we assume patient level is the deepest level within the nested data (we ignore airway level)
# We do need Bonferri correction to correct for multi-testing (multi-comparison of patients)

# select all patients conforming conditions
patients_copd_sel = list(aggs_patient.patient[(aggs_patient.copd_group == 'COPD stage III or IV') & (aggs_patient.smoking_group == 'ExS >= year') & (aggs_patient.n_adventitia >= compartment_size) & (aggs_patient.n_submucosa >= compartment_size)])

# Get combinations of all patients for multi-testing
patient_combinations = pd.DataFrame(list(itertools.product(patients_copd_sel, patients_copd_sel)),columns=['patient_1','patient_2'])
patient_combinations = patient_combinations[patient_combinations.patient_1 != patient_combinations.patient_2]


# Empty dataframe to write results of patient testing to
patient_tests_df = pd.DataFrame(columns=['patient_1', 'patient_2', 'lung_compartment', 'u_stat', 'u_p_value', 'mood_stat', 'mood_p_value','np_levene_stat', 'np_levene_p_value', 'levene_stat', 'levene_p_value', 'median_p1', 'median_p2', 'var_p1', 'var_p2'])

for i in range(patient_combinations.shape[0]):
  p1 = patient_combinations.iloc[i,0]
  p2 = patient_combinations.iloc[i,1]

  for lung_compartment in ['Inside', 'Outside']:
    p1_data = list(preds_df.prob_correct[(preds_df.patient == p1) & (preds_df.label == lung_compartment)])
    p2_data = list(preds_df.prob_correct[(preds_df.patient == p2) & (preds_df.label == lung_compartment)])

    # Perform distribution testing
    u_stat, u_p = stats.mannwhitneyu(p1_data, p2_data)

    # Perform median testing
    median_p1 = statistics.median(p1_data)
    median_p2 = statistics.median(p2_data)

    mood_stat, mood_p, mood_med, tbl = stats.median_test(p1_data, p2_data)

    # Perform variance testing
    var_p1 = statistics.variance(p1_data)
    var_p2 = statistics.variance(p2_data)

    # Parametric median Levene test
    levene_stat, levene_p = stats.levene(p1_data, p2_data, center='median')

    # Non-parametric Levene test (via rank transformations)
    p1_df = pd.DataFrame(['p1' for i in range(len(p1_data))],columns = ['group'])
    p1_df['probs'] = p1_data

    p2_df = pd.DataFrame(['p2' for i in range(len(p2_data))],columns = ['group'])
    p2_df['probs'] = p2_data

    combined_df = pd.concat([p1_df, p2_df], ignore_index = True)
    combined_df['rank'] = combined_df['probs'].rank(method ='min')

    #np_levene_stat, np_levene_p = stats.levene(list(combined_df.rank[combined_df.group == 'p1']), list(combined_df.rank[combined_df.group == 'p2']), center='mean')
    np_levene_stat = 0
    np_levene_p = 1

    patient_tests_df = patient_tests_df.append({'patient_1': p1,
                                                        'patient_2': p2,
                                                        'lung_compartment': lung_compartment,
                                                        'u_stat': u_stat,
                                                        'ustat_p_value': u_p,
                                                        'mood_stat': mood_stat,
                                                        'mood_p_value': mood_p,
                                                        'np_levene_stat': np_levene_stat,
                                                        'np_levene_p_value': np_levene_p,
                                                        'levene_stat': levene_stat,
                                                        'levene_p_value': levene_p,
                                                        'median_p1': median_p1,
                                                        'median_p2': median_p2,
                                                        'var_p1': var_p1,
                                                        'var_p2': var_p2 }, ignore_index=True)

    print(lung_compartment.upper() + ' - Comparing patient ' + str(p1) + ' with patient ' + str(p2)  + '\n' +
      '- Distribution: Statistics=' + str(u_stat) + ', p=' + str(u_p) + '\n' +
      '- Median: Statistics=' + str(mood_stat) + ', p=' + str(mood_p) + '\n' +
      '- Variance:'  + '\n' +
      '   parametric: Statistics=' + str(levene_stat) + ', p=' + str(levene_p) + '\n' +
      '   non-parametric: Statistics=' + str(np_levene_stat) + ', p=' + str(np_levene_p))


# Critical value
alpha = 0.05

# Individual tests
patient_tests_df['ind_significant_dist_diff_indiv'] = np.where(patient_tests_df.ustat_p_value <= alpha, 1, 0)
patient_tests_df['ind_significant_median_diff_indiv'] = np.where(patient_tests_df.mood_p_value <= alpha, 1, 0)
patient_tests_df['ind_significant_var_diff_indiv'] = np.where(patient_tests_df.levene_p_value <= alpha, 1, 0)
patient_tests_df['ind_significant_var_diff_np_indiv'] = np.where(patient_tests_df.np_levene_stat <= alpha, 1, 0)

# Write results to CSV
patient_tests_df.to_csv(model_path + '/Visualizations/patient_level_tests_results.csv')

grouped_patient_df = patient_tests_df[['lung_compartment', 'ustat_p_value', 'levene_p_value']].groupby(by=['lung_compartment']).sum().reset_index()
grouped_patient_df

"""# RUN! 4. Hypothesis testing - PCA on features

## Perform PCA on features
"""

# PCA, with n_components minimalized such that explained variance >= 80%
features = output_df.iloc[:,23:]
pca = PCA(n_components=0.80, svd_solver='full')
features_PCA = pca.fit_transform(features)

print(features_PCA.shape)

pca_df = pd.DataFrame(features_PCA, columns = ['pc_' + str(i) for i in range(features_PCA.shape[1])])
output_pca_df = output_df.iloc[:,0:23].merge(pca_df, left_index=True, right_index=True)

# Get explained variance per PC
pca_ex_var = pca.explained_variance_ratio_
np.where(pca_ex_var < 0.01)

# Select COPD and normal patients, with at least 10 patches per compartment
min_patches_per_compartment = 10
output_pca_COPD_normal = output_pca_df[(output_pca_df.smoking_group == 'ExS >= year') & (output_pca_df.copd_group != 'Else') & (output_pca_df.n_adventitia >= min_patches_per_compartment) & (output_pca_df.n_submucosa >= min_patches_per_compartment)]

"""## Perform tests on PC distance

"""

# Average pc per lung compartment / patient
pc_columns = ['pc_' + str(i) for i in range(14)]
pc_means_per_patient = output_pca_COPD_normal.groupby(['patient', 'copd_group_full', 'copd_group', 'smoking_group', 'label']).mean().reset_index()
pc_means_per_patient.columns

# Calculate Euclidean distance between inside and outside per patient
dist_ins_out = []
for p in np.unique(pc_means_per_patient.patient):
  inside_pc = pc_means_per_patient.loc[(pc_means_per_patient.patient == p) & (pc_means_per_patient.label == 'Inside'), pc_columns].to_numpy()
  outside_pc = pc_means_per_patient.loc[(pc_means_per_patient.patient == p) & (pc_means_per_patient.label == 'Outside'), pc_columns].to_numpy()

  dist = np.sqrt(np.sum((inside_pc - outside_pc)**2))

  dist_ins_out.append(dist)

dist_per_patient = pc_means_per_patient[['patient', 'copd_group']].drop_duplicates()
dist_per_patient['pc_distance_inside_outside'] = dist_ins_out

print(dist_per_patient.groupby(['copd_group']).mean())

# Perform tests
copd_sample = dist_per_patient.loc[dist_per_patient.copd_group == 'COPD stage III or IV', 'pc_distance_inside_outside']
normal_sample = dist_per_patient.loc[dist_per_patient.copd_group == 'Normal', 'pc_distance_inside_outside']

u_stat, u_p = stats.mannwhitneyu(copd_sample, normal_sample)
mood_stat, mood_p, mood_med, tbl = stats.median_test(copd_sample, normal_sample)
levene_stat, levene_p = stats.levene(copd_sample, normal_sample)

print('- Distribution: Statistics=' + str(u_stat) + ', p=' + str(u_p) + '\n' +
  '- Median: Statistics=' + str(mood_stat) + ', p=' + str(mood_p) + '\n' +
  '- Variance: Statistics=' + str(levene_stat) + ', p=' + str(levene_p))

# Plot 1st and 2nd average pc per lung compartment / patient
cdict = {'Submucosa': 'red',
          'Adventitia': 'blue'
}

# Original clusters
fig = plt.figure(figsize=(20, 12))
for compartment in np.unique(pc_means_per_patient.label):
  ix = (pc_means_per_patient.label == compartment)
  compartment = 'Submucosa' if compartment == 'Inside' else 'Adventitia'
  plt.scatter(pc_means_per_patient.loc[ix, "pc_0"], pc_means_per_patient.loc[ix, "pc_1"], c = cdict[compartment], label = compartment, s = 80)

for ix, txt in enumerate(pc_means_per_patient.patient):
  plt.annotate(txt, (pc_means_per_patient["pc_0"].iloc[ix], pc_means_per_patient["pc_1"].iloc[ix]), size=18)

plt.legend(loc = 'upper right', markerscale=1, fontsize=25)
plt.xlabel('Principal component 1', fontsize=20)
plt.ylabel('Principal component 2', fontsize=20)

fig.savefig(model_path + '/Visualizations/average_pc_per_patient_compartment.png')


# Per lung function group
for lung_function in np.unique(pc_means_per_patient.copd_group):
  print(lung_function)
  fig = plt.figure(figsize=(20, 12))
  pc_means_sel = pc_means_per_patient[pc_means_per_patient.copd_group == lung_function]

  # Save pcs
  pc_means_sel.to_csv(model_path + '/Visualizations/average_pc_per_patient_compartment_' + lung_function + '.csv')

  # Create figure from pcs
  for compartment in np.unique(pc_means_sel.label):
    ix = (pc_means_sel.label == compartment)
    compartment = 'Submucosa' if compartment == 'Inside' else 'Adventitia'
    plt.scatter(pc_means_sel.loc[ix, "pc_0"], pc_means_sel.loc[ix, "pc_1"], c = cdict[compartment], label = compartment, s = 80)

  for ix, txt in enumerate(pc_means_sel.patient):
    plt.annotate(txt, (pc_means_sel["pc_0"].iloc[ix], pc_means_sel["pc_1"].iloc[ix]), size=18)

  plt.legend(loc = 'upper right', markerscale=1, fontsize=25)
  plt.xlabel('Principal component 1', fontsize=20)
  plt.ylabel('Principal component 2', fontsize=20)

  fig.savefig(model_path + '/Visualizations/average_pc_per_patient_compartment_' + lung_function + '.png')

model_path

"""## RUN! Mixed model, with fixed and random (due to hierarchy) effects"""

# Construct mixed model, with airway as group
# Potentially add: Age, Gender, number of packyears
columns = ['enc_label','patient', 'copd_group', 'prob_correct'] + ['pc_' + str(i) for i in range(features_PCA.shape[1])]
data_for_md = output_pca_COPD_normal[columns]

formula = "prob_correct ~ C(copd_group) + enc_label"
# for i in range(3):
#   formula = formula + " + pc_" + str(i)
print(formula)

md = smf.mixedlm(formula, data_for_md, groups=data_for_md['patient'])
mdf = md.fit()
print(mdf.summary())

"""## Perform SVM on PC of features
1. Eén model voor COPD en normaal tezamen => In analyse doorsnijding op COPD/normaal
2. Twee modellen voor COPD en normaal
3. Eén model, met long functie als extra variabele

### 1. Eén model voor alle data
"""

# Fit and predict on data
# kernels = ['linear', 'poly', 'rbf']
# for kernel in kernels:
#   print(kernel)

clf_full = SVC(kernel='poly')
clf_full.fit(features_PCA, output_df.enc_label)
preds_clf_full = clf_full.predict(features_PCA)

print("Accuracy Overall:",metrics.accuracy_score(output_df.enc_label, preds_clf_full))
print("Precision Overall:",metrics.precision_score(output_df.enc_label, preds_clf_full))
print("Recall Overall:",metrics.recall_score(output_df.enc_label, preds_clf_full))

for lung_function in ['COPD stage III or IV', 'Normal']:
  inds = output_df[output_df.copd_group == lung_function].index
  print("Accuracy ", lung_function, ": ",metrics.accuracy_score(output_df.enc_label[inds], preds_clf_full[inds]))
  print("Precision ", lung_function, ": ", metrics.precision_score(output_df.enc_label[inds], preds_clf_full[inds]))
  print("Recall ", lung_function, ": ",metrics.recall_score(output_df.enc_label[inds], preds_clf_full[inds]))


# Visualize SVM
# Get support vectors themselves
support_vectors = clf_full.support_vectors_

# Visualize support vectors
plt.scatter(features_PCA[:,0], features_PCA[:,1])
plt.scatter(support_vectors[:,0], support_vectors[:,1], color='red')
plt.title('Linearly separable data with support vectors')
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

"""### 2. Aparte modellen per longfunctie"""

for lung_function in ['COPD stage III or IV', 'Normal']:
  inds = output_df[output_df.copd_group == lung_function].index
  clf_separate = SVC()
  clf_separate.fit(features_PCA[inds], output_df.enc_label[inds])
  preds_clf_separate = clf_separate.predict(features_PCA)

  print("Accuracy ", lung_function, ": ",metrics.accuracy_score(output_df.enc_label[inds], preds_clf_separate[inds]))
  print("Precision ", lung_function, ": ", metrics.precision_score(output_df.enc_label[inds], preds_clf_separate[inds]))
  print("Recall ", lung_function, ": ",metrics.recall_score(output_df.enc_label[inds], preds_clf_separate[inds]))

"""# 5. Cluster analysis: PCA and UMAP

### Visualize top PC's overall and per intersection
##### Important note: PC's are calculated for each group separately and plotted, to get an idea of the % variance captured per group for the top PC's
"""

pca = PCA(n_components=100)
features_PCA = pca.fit_transform(features)

PCA_labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
}

print(np.cumsum(pca.explained_variance_ratio_))

# TO_DO: Binnenkant punten doorzichtig
fig = px.scatter_matrix(
    features_PCA,
    labels=PCA_labels,
    dimensions=range(5),
    color=preds_df.label.map({'Inside':'Submucosa', 'Outside':'Adventitia'})
)
fig.update_traces(diagonal_visible=False)
fig.show()

fig.write_image(model_path + '/Visualizations/PCA_plot_full_data.svg', width=1980, height=1080)


# Per intersection
for intersection in ['smoking_group', 'copd_group']:
  # Filter specific groups
  if intersection == 'smoking_group':
    preds_filter =  preds_df[preds_df.smoking_group != '?']
  if intersection == 'copd_group':
    preds_filter = preds_df[preds_df.smoking_group == 'ExS >= year']

  for group in list(np.unique(preds_filter[intersection].astype(str))):
      group_df = preds_filter[(preds_filter[intersection].astype(str) == group)]
      index_filenames = labels.image.index[labels.image.isin(list(group_df.image))]

      if len(index_filenames) > 100:
        pca = PCA(n_components=100)
        features_PCA_group = pca.fit_transform(features[index_filenames])

        PCA_labels = {
            str(i): f"PC {i+1} ({var:.1f}%)"
            for i, var in enumerate(pca.explained_variance_ratio_ * 100)
        }

        fig = px.scatter_matrix(
            features_PCA_group,
            labels=PCA_labels,
            dimensions=range(5),
            color=group_df.label.map({'Inside':'Submucosa', 'Outside':'Adventitia'})
        )
        fig.update_traces(diagonal_visible=False)
        fig.show()

        fig.write_image(model_path + '/Visualizations/' + intersection + '/PCA_plot_' + group.replace('/','-') + '.svg', width=1980, height=1080)

"""### Generate, visualize and quantify clusters on overall, group, patient and airway level"""

# Create a CSV to write clusterability results to
fieldnames = ['group', 'patient', 'airway', 'n_patches', 'n_submucosa', 'n_adventitia', 'homogeneity_score_full', 'completeness_score_full', 'v_measure_full', 'homogeneity_score_clustered', 'completeness_score_clustered', 'v_measure_clustered',
              'n_noise', 'n_noise_submucosa', 'n_noise_adventitia', 'perc_noise_submucosa', 'perc_noise_adventitia']
cluster_df = pd.DataFrame(columns = fieldnames)
cluster_df.to_csv(model_path + '/Visualizations/clusterability.csv', index=False)

# Determine embedding on full dataset prior to analyzing subgroups
standard_embedding, clustered, labels_pred_hdbscan = determine_cluster(features, list(preds_df.image), labels)

# On full dataset
n_patches, n_submucosa, n_adventitia, homogen_score_full, complete_score_full, v_score_full, homogen_score_clustered, complete_score_clustered, v_score_clustered, n_noise, n_noise_submucosa, n_noise_adventitia = visualize_and_quantify_cluster(
                                                                                                                    model_path + '/Visualizations/cluster_classified_full_data.svg',
                                                                                                                    list(preds_df.image),
                                                                                                                    list(preds_df.image),
                                                                                                                    standard_embedding,
                                                                                                                    clustered,
                                                                                                                    labels_pred_hdbscan
                                                                                                                    )

values_list = ['Full',
                '-',
                '-',
                n_patches,
                n_submucosa,
                n_adventitia,
                homogen_score_full,
                complete_score_full,
                v_score_full,
                homogen_score_clustered,
                complete_score_clustered,
                v_score_clustered ,
                n_noise,
                n_noise_submucosa,
                n_noise_adventitia,
                n_noise_submucosa / n_submucosa,
                n_noise_adventitia / n_adventitia
      ]


with open(model_path + '/Visualizations/clusterability.csv', 'a', newline='') as f:
  writer = csv.writer(f)
  writer.writerow(values_list)


# Per lung compartment - only for visualization purposes
for compartment in np.unique(preds_df.label):
  visualize_and_quantify_cluster(
                                  model_path + '/Visualizations/cluster_classified_' + compartment + '.svg',
                                  list(preds_df.image),
                                  list(preds_df[preds_df.label==compartment].image),
                                  standard_embedding,
                                  clustered,
                                  labels_pred_hdbscan
                                  )




# Per intersection, patient and airway
for intersection in ['smoking_group', 'copd_group']:
  # Filter specific groups
  if intersection == 'smoking_group':
    preds_filter =  preds_df[preds_df.smoking_group != '?']
  if intersection == 'copd_group':
    preds_filter = preds_df[preds_df.smoking_group == 'ExS >= year']

  # Per intersection
  print('Quantifying clusters per ' + intersection)
  for group in list(np.unique(preds_filter[intersection].astype(str))):
      group_df = preds_filter[(preds_filter[intersection].astype(str) == group)]
      n_patches, n_submucosa, n_adventitia, homogen_score_full, complete_score_full, v_score_full, homogen_score_clustered, complete_score_clustered, v_score_clustered, n_noise, n_noise_submucosa, n_noise_adventitia = visualize_and_quantify_cluster(
                                                                                                                    model_path + '/Visualizations/' + intersection + '/cluster_classified_' + str(group.replace('/','-')) + '.svg',
                                                                                                                    list(preds_df.image),
                                                                                                                    list(group_df.image),
                                                                                                                    standard_embedding,
                                                                                                                    clustered,
                                                                                                                    labels_pred_hdbscan
                                                                                                                    )

      values_list = [str(group.replace('/','-')),
                     '-',
                     '-',
                     n_patches,
                     n_submucosa,
                     n_adventitia,
                     homogen_score_full,
                     complete_score_full,
                     v_score_full,
                     homogen_score_clustered,
                     complete_score_clustered,
                     v_score_clustered ,
                     n_noise,
                     n_noise_submucosa,
                     n_noise_adventitia,
                     n_noise_submucosa / n_submucosa,
                     n_noise_adventitia / n_adventitia
      ]


      with open(model_path + '/Visualizations/clusterability.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(values_list)

      # Per patient
      print('Quantifying clusters per patient for intersection on ' + intersection)
      for patient in list(np.unique(group_df.patient)):
        patient_df = group_df[(group_df.patient == patient)]

        try:
          os.mkdir(model_path + '/Visualizations/' + intersection +  '/per_patient/' + str(patient))
        except:
          pass
        n_patches, n_submucosa, n_adventitia, homogen_score_full, complete_score_full, v_score_full, homogen_score_clustered, complete_score_clustered, v_score_clustered, n_noise, n_noise_submucosa, n_noise_adventitia = visualize_and_quantify_cluster(
                                                                                                                model_path + '/Visualizations/' + intersection +  '/per_patient/' + str(patient) + '/cluster_classified_' + str(patient) + '.svg',
                                                                                                                list(preds_df.image),
                                                                                                                list(patient_df.image),
                                                                                                                standard_embedding,
                                                                                                                clustered,
                                                                                                                labels_pred_hdbscan
                                                                                                                )
        if min(n_submucosa, n_adventitia) > 0:
          values_list = [str(group.replace('/','-')),
                      patient,
                      '-',
                      n_patches,
                      n_submucosa,
                      n_adventitia,
                      homogen_score_full,
                      complete_score_full,
                      v_score_full,
                      homogen_score_clustered,
                      complete_score_clustered,
                      v_score_clustered,
                      n_noise,
                      n_noise_submucosa,
                      n_noise_adventitia,
                      n_noise_submucosa / n_submucosa,
                      n_noise_adventitia / n_adventitia
        ]

          with open(model_path + '/Visualizations/clusterability.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(values_list)

          # Per airway
          print('Quantifying clusters per airway for patient ' + str(patient) + ' and intersection on ' + intersection)
          for airway in np.unique(patient_df.airway):
            airway_df = patient_df[patient_df.airway == airway]
            n_patches, n_submucosa, n_adventitia, homogen_score_full, complete_score_full, v_score_full, homogen_score_clustered, complete_score_clustered, v_score_clustered, n_noise, n_noise_submucosa, n_noise_adventitia = visualize_and_quantify_cluster(
                                                                                                                    model_path + '/Visualizations/' + intersection +  '/per_patient/' + str(patient) + '/cluster_classified_' + str(patient) + '_' + str(airway) + '.svg',
                                                                                                                    list(preds_df.image),
                                                                                                                    list(airway_df.image),
                                                                                                                    standard_embedding,
                                                                                                                    clustered,
                                                                                                                    labels_pred_hdbscan
                                                                                                                    )

            if min(n_submucosa, n_adventitia) > 0:
              values_list = [str(group.replace('/','-')),
                          patient,
                          str(airway),
                          n_patches,
                          n_submucosa,
                          n_adventitia,
                          homogen_score_full,
                          complete_score_full,
                          v_score_full,
                          homogen_score_clustered,
                          complete_score_clustered,
                          v_score_clustered,
                          n_noise,
                          n_noise_submucosa,
                          n_noise_adventitia,
                          n_noise_submucosa / n_submucosa,
                          n_noise_adventitia / n_adventitia
            ]

              with open(model_path + '/Visualizations/clusterability.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(values_list)

"""
Make UMAP plot with aggregate per patient for inside and outside
"""

# Enrich df with embedding information before aggregation
preds_df_enr = preds_df.copy()
preds_df_enr['embedding_0'], preds_df_enr['embedding_1'] = standard_embedding.T
preds_df_enr['embedding_clustered'] = clustered
preds_df_enr['embedding_labels_pred_hdbscan'] = labels_pred_hdbscan

df_patient_grouped = preds_df_enr.groupby(["patient", "label"], as_index=False)["embedding_0", "embedding_1"].mean()

cdict = {'Submucosa': 'red',
          'Adventitia': 'blue'
}

# Original clusters
fig = plt.figure(figsize=(20, 12))
for compartment in np.unique(df_patient_grouped.label):
  ix = (df_patient_grouped.label == compartment)
  compartment = 'Submucosa' if compartment == 'Inside' else 'Adventitia'
  plt.scatter(df_patient_grouped.loc[ix, "embedding_0"], df_patient_grouped.loc[ix, "embedding_1"], c = cdict[compartment], label = compartment, s = 20)

## TO DO: add labels to annotate patients
for ix, txt in enumerate(df_patient_grouped.patient):
  plt.annotate(txt, (df_patient_grouped.loc[ix, "embedding_0"], df_patient_grouped.loc[ix, "embedding_1"]), size=8)

plt.legend(loc = 'upper right', markerscale=1.5)

fig.savefig(model_path + '/Visualizations/cluster_average_embedding_per_patient.svg')

"""### Interactive UMAP to visualize patches per cluster

"""

from bokeh.plotting import show, save, output_notebook, output_file

print('Umap - per prototype - tp')
dirs = [path + '/Visualizations/overall/tp/']
file_names = []
for directory in dirs:
  for img_path in Path(directory).glob('*.png'):
    if '_occ' not in str(img_path):
      file_name = os.path.basename(img_path)
      file_name = file_name.split('_',1)[-1]
      file_names.append(file_name)

index_filenames = labels.Image.index[labels.Image.isin(file_names)]
features_selected = features[index_filenames]
labels_selected = labels.Label[index_filenames]


mapping = umap.UMAP(n_neighbors=20,
                    min_dist=0.001,
                    metric='correlation').fit(features_selected)

cluster_patches(path + '/Visualizations/Umap_tp.png', features, list(file_names), labels)

hover_data = pd.DataFrame({'index': index_filenames})

p = umap.plot.interactive(mapping, labels=labels_selected, hover_data=hover_data, point_size=4)

output_notebook()
show(p)

# indexes left bottom cluster
inds_tp_cluster_bottom_left = list(np.unique([10129, 11892, 5624, 9440, 5522, 11301, 3890, 10394, 10919, 10975, 4183, 11863, 12107, 5403, 10755, 8887, 7801, 8914, 3863, 4959, 10546, 6191, 8914, 5094, 10065, 12646, 11301, 8278]))
inds_tp_cluster_top_left = list(np.unique([10706, 9607, 4179, 4153, 4127, 7560, 6359, 9108, 4414, 5012, 8757, 4645, 9829, 7844, 5629, 6217, 10969, 11424, 7650, 4197, 4198, 9385, 9437, 4376, 10034, 9330, 11325, 10165, 10670, 11142, 8757, 4645, 4210, 9503, 9574, 8123, 7844, 9829, 8126, 11139, 5952, 6359]))

# Swap softmax activation of last layer with linear (for gradient computation)
layer_idx = -1
model.layers[layer_idx].activation = keras.activations.linear
model = utils.apply_modifications(model)

data_path = '/content/drive/My Drive/Esmee/Data/Patches/'

for index in inds_tp_cluster_bottom_left:
  img = data[index]
  img = img*255
  file_name = labels.Image[index]
  cv2.imwrite(path + '/Visualizations/clusters_outside/cluster_bottom_left/' + file_name, img.astype("uint8"))

  img = read_img(str(data_path + '/Outside/' + file_name), size)
  img = np.array(img, np.float32) / 255

  layerName = 'conv_7b_ac'

  heatmap = grad_cam(img, model)
  cv2.imwrite(path + '/Visualizations/clusters_outside/cluster_bottom_left/' + file_name.replace('.png','_occ.png'), heatmap)

for index in inds_tp_cluster_top_left:
  img = data[index]
  img = img*255
  file_name = labels.Image[index]
  cv2.imwrite(path + '/Visualizations/clusters_outside/cluster_top_left/' + file_name, img.astype("uint8"))

  img = read_img(str(data_path + '/Outside/' + file_name), size)
  img = np.array(img, np.float32) / 255

  layerName = 'conv_7b_ac'

  heatmap = grad_cam(img, model)
  cv2.imwrite(path + '/Visualizations/clusters_outside/cluster_top_left/' + file_name.replace('.png','_occ.png'), heatmap)

"""# 6. Relationship between key metrics and data variations

"""

"""
Plot of relationship between % submucosa and V-measure
"""


result_df = pd.DataFrame(
                         columns=['Dataset',
              'Intersection',
              'n_patches',
              'n_submucosa',
              'n_adventitia',
              'homogen_score_full',
              'complete_score_full',
              'v_score_full',
              'homogen_score_clustered',
              'complete_score_clustered',
              'v_score_clustered' ,
              'n_noise',
              'n_noise_submucosa',
              'n_noise_adventitia',
              'n_noise_submucosa div n_submucosa',
              'n_noise_adventitia div n_adventitia'])

intersection = "no_intersection"
aggregate_level = 'airway'

try:
  os.mkdir(model_path + '/Visualizations/' + intersection)
except:
  pass

preds_filter = preds_df.copy()
values_list_total=[]
for ix in list(np.unique(preds_filter[aggregate_level])):

  filter_df = preds_filter[(preds_filter[aggregate_level] == ix)]

  try:
    os.mkdir(model_path + '/Visualizations/' + intersection +  '/per_' + aggregate_level + '/')
  except:
    pass

  # Don't perform for airways with less than 30 patches
  if filter_df.shape[0]>=30:
    n_patches, n_submucosa, n_adventitia, homogen_score_full, complete_score_full, v_score_full, homogen_score_clustered, complete_score_clustered, v_score_clustered, n_noise, n_noise_submucosa, n_noise_adventitia = visualize_and_quantify_cluster(
                                                                                                          model_path + '/Visualizations/' + intersection +  '/per_' + aggregate_level + '/cluster_' + str(ix) + '.svg',
                                                                                                          list(preds_df.image),
                                                                                                          list(filter_df.image),
                                                                                                          standard_embedding,
                                                                                                          clustered,
                                                                                                          labels_pred_hdbscan
                                                                                                          )

  # Bit crude, but for now, exclude when one of the classes is 0
  if  n_submucosa == 0 or n_adventitia == 0:
    pass
  else:
    values_list = [ix,
                str(group.replace('/','-')),
                '-',
                n_patches,
                n_submucosa,
                n_adventitia,
                homogen_score_full,
                complete_score_full,
                v_score_full,
                homogen_score_clustered,
                complete_score_clustered,
                v_score_clustered,
                n_noise,
                n_noise_submucosa,
                n_noise_adventitia,
                n_noise_submucosa / n_submucosa,
                n_noise_adventitia / n_adventitia
    ]

    values_list_total.append(values_list)


result_df = pd.DataFrame.from_records(values_list_total,
                         columns=[
                                  aggregate_level,
                                  'Dataset',
                                  'Intersection',
                                  'n_patches',
                                  'n_submucosa',
                                  'n_adventitia',
                                  'homogen_score_full',
                                  'complete_score_full',
                                  'v_score_full',
                                  'homogen_score_clustered',
                                  'complete_score_clustered',
                                  'v_score_clustered' ,
                                  'n_noise',
                                  'n_noise_submucosa',
                                  'n_noise_adventitia',
                                  'n_noise_submucosa div n_submucosa',
                                  'n_noise_adventitia div n_adventitia'])



# Make plt with regression line and significance
data_plot = result_df.copy()
data_plot['perc_submucosa'] = data_plot['n_submucosa']/data_plot['n_patches']
data_plot.dropna(inplace=True)

g = sns.lmplot(x='perc_submucosa', y='v_score_clustered', data=data_plot)

def annotate(data, **kws):
    r, p = stats.pearsonr(data['perc_submucosa'], data['v_score_clustered'])
    ax = plt.gca()
    ax.text(.05, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
            transform=ax.transAxes)

g.map_dataframe(annotate)
plt.show()

fig = plt.figure()
ax = sns.jointplot(x=result_df["n_submucosa"]/result_df['n_patches'], y=result_df["v_score_clustered"], kind="reg")
ax.set_axis_labels('% Submucosa', 'V-measure')
fig.savefig(model_path + '/Visualizations/jointplot_%submucosa_vmeasure.svg')

"""
Plot of relationship between spread in prob preds (median) and v-measure
"""

# Determine embedding on full dataset prior to analyzing subgroups
standard_embedding, clustered, labels_pred_hdbscan = determine_cluster(features, list(preds_df.image), labels)

result_df = pd.DataFrame(columns=['Dataset',
              'Intersection',
              'n_patches',
              'n_submucosa',
              'n_adventitia',
              'homogen_score_full',
              'complete_score_full',
              'v_score_full',
              'homogen_score_clustered',
              'complete_score_clustered',
              'v_score_clustered' ,
              'n_noise',
              'n_noise_submucosa',
              'n_noise_adventitia',
              'n_noise_submucosa div n_submucosa',
              'n_noise_adventitia div n_adventitia'])

intersection = "no_intersection"
aggregate_level = 'airway'

preds_filter = preds_df.copy()
values_list_total=[]
for ix in list(np.unique(preds_filter[aggregate_level])):

  filter_df = preds_filter[(preds_filter[aggregate_level] == ix)]

  # Don't perform for airways with less than 50 patches
  if filter_df.shape[0]>=30:
    n_patches, n_submucosa, n_adventitia, homogen_score_full, complete_score_full, v_score_full, homogen_score_clustered, complete_score_clustered, v_score_clustered, n_noise, n_noise_submucosa, n_noise_adventitia = visualize_and_quantify_cluster(
                                                                                                          model_path + '/Visualizations/' + intersection +  '/per_' + aggregate_level + '/cluster_' + str(ix) + '.svg',
                                                                                                          list(preds_df.image),
                                                                                                          list(filter_df.image),
                                                                                                          standard_embedding,
                                                                                                          clustered,
                                                                                                          labels_pred_hdbscan
                                                                                                          )

    #spread_preds_prob = filter_df['prob_outside'].var()
    spread_preds_prob = filter_df['prob_correct'].median()

    # Bit crude, but for now, exclude when one of the classes is 0
    if  n_submucosa == 0 or n_adventitia == 0:
      pass
    else:
      values_list = [ix,
                  str(group.replace('/','-')),
                  '-',
                  n_patches,
                  n_submucosa,
                  n_adventitia,
                  homogen_score_full,
                  complete_score_full,
                  v_score_full,
                  homogen_score_clustered,
                  complete_score_clustered,
                  v_score_clustered,
                  n_noise,
                  n_noise_submucosa,
                  n_noise_adventitia,
                  n_noise_submucosa / n_submucosa,
                  n_noise_adventitia / n_adventitia,
                  spread_preds_prob
      ]

      values_list_total.append(values_list)


result_df = pd.DataFrame.from_records(values_list_total,
                         columns=[
                                  aggregate_level,
                                  'Dataset',
                                  'Intersection',
                                  'n_patches',
                                  'n_submucosa',
                                  'n_adventitia',
                                  'homogen_score_full',
                                  'complete_score_full',
                                  'v_score_full',
                                  'homogen_score_clustered',
                                  'complete_score_clustered',
                                  'v_score_clustered' ,
                                  'n_noise',
                                  'n_noise_submucosa',
                                  'n_noise_adventitia',
                                  'n_noise_submucosa div n_submucosa',
                                  'n_noise_adventitia div n_adventitia',
                                  'spread_preds_prob'])


# https://stackoverflow.com/questions/52118245/python-seaborn-jointplot-does-not-show-the-correlation-coefficient-and-p-value-o

# Make plot of relationship %submucosa versus patient/airway/else
fig = plt.figure()
plt.scatter(result_df["spread_preds_prob"],
            result_df["v_score_clustered"], s = 2)
plt.title("Relationship between variance in pred prob vs. v-score on " + aggregate_level + " level")
plt.xlabel("Median(predicted true class probability)")
plt.ylabel("V-measure")
lgnd = plt.legend(loc = 'upper right', markerscale=9)
fig.savefig(model_path + '/Visualizations/scatter_probspread_vmeasure.svg')

# Make plt with regression line and significance
data_plot = result_df.copy()
data_plot.dropna(inplace=True)

g = sns.lmplot(x='spread_preds_prob', y='v_score_clustered', data=data_plot)

def annotate(data, **kws):
    r, p = stats.pearsonr(data['spread_preds_prob'], data['v_score_clustered'])
    ax = plt.gca()
    ax.text(.05, .8, 'r={:.2f}, p={:.2g}'.format(r, p),
            transform=ax.transAxes)

g.map_dataframe(annotate)
plt.show()

def r2(x, y):
    return stats.pearsonr(x, y)[0] ** 2
fig = plt.figure()
ax = sns.jointplot(x=result_df["spread_preds_prob"], y=result_df["v_score_clustered"], kind="reg")
ax.set_axis_labels('Median(predicted true class probability)', 'V-measure')
fig.savefig(model_path + '/Visualizations/jointplot_probspread_vmeasure.svg')

"""# RUN! 7. GRAD-CAM for true/false positives/negatives"""

try:
  os.mkdir(model_path + '/Visualizations/heatmaps/')
  os.mkdir(model_path + '/Visualizations/heatmaps/Outside/')
  os.mkdir(model_path + '/Visualizations/heatmaps/Inside/')
except:
  pass

#output_df = pd.read_csv(model_path + '/Visualizations/model_predictions.csv', index_col=0)

print('Loading model')
model = load_model(model_path + '/weights_auc.hdf5')

# Swap softmax activation of last layer with linear (for gradient computation)
layer_idx = -1
model.layers[layer_idx].activation = keras.activations.linear
model = utils.apply_modifications(model)

for directory in [ f.path for f in os.scandir(data_path) if f.is_dir() and 'oversample' not in str(f.path) and 'dir' not in str(f.path) and 'Outside' in str(f.path)]:
  count = 0
  for img_path in Path(directory + '/').glob('*.png'):
    try:
      img = read_img(str(img_path), size)
      img = np.array(img, np.float32) / 255

      file_name = os.path.basename(img_path)
      # pred_prob = float(output_df[output_df.image == file_name].prob_outside)
      # file_name = str(pred_prob) + '_' + file_name

      # skip if heatmap already exists
      main_path = os.path.dirname(img_path)
      if '/Outside/' in str(img_path):
        target_path = model_path + '/Visualizations/heatmaps/Outside/'
      else:
        target_path = model_path + '/Visualizations/heatmaps/Inside/'

      count += 1
      print(directory + '- ' + str(count))

      layerName = 'conv_7b_ac'

      heatmap = grad_cam(img, model)
      cv2.imwrite(target_path + file_name.replace('.png','_occ.png'), heatmap)
    except:
      pass

print('Loading model')
model = load_model(model_path + '/weights_auc.hdf5')

# Swap softmax activation of last layer with linear (for gradient computation)
layer_idx = -1
model.layers[layer_idx].activation = keras.activations.linear
model = utils.apply_modifications(model)

for directory in [ f.path for f in os.scandir(model_path + '/Visualizations/prototypes/overall/') if f.is_dir() ]:
  count = 0
  for img_path in Path(directory + '/').glob('*.png'):

    file_name = os.path.basename(img_path)
    file_name = file_name.split('_', 1)[1]

    # skip if heatmap already exists
    main_path = os.path.dirname(img_path)
    if os.path.isfile(str(img_path).replace('.png','_occ.png')) or '_occ' in file_name:
      continue

    count += 1
    print(directory + '- ' + str(count))

    if '/tp/' in str(img_path) or '/fn/' in str(img_path):
      img = read_img(str(data_path + '/Outside/' + file_name), size)
    else:
      img = read_img(str(data_path + '/Inside/' + file_name), size)
    img = np.array(img, np.float32) / 255

    layerName = 'conv_7b_ac'

    heatmap = grad_cam(img, model)
    cv2.imwrite(str(img_path).replace('.png','_occ.png'), heatmap)

# Boxplot per tiff
preds_df_sel = preds_df[(preds_df.Tiff =='ef8b207b58e14f25af09ac974d2ce7bf')| (preds_df.Tiff == '9584cf8ebea14007a9bf10757382dd4f')]
preds_df_sel = preds_df[preds_df.n_tiff >= 150]
preds_df_sel['Tiff'] = preds_df_sel['Tiff'].astype('category')
preds_df_sel['Tiff_cat'] = preds_df_sel['Tiff'].cat.codes

fig = plt.figure()
dd=pd.melt(preds_df_sel,id_vars=['Tiff_cat'],value_vars=['prob_correct'])
sns.boxplot(x='Tiff_cat',y='value',data=dd, color='silver')
fig.savefig(path + '/Visualizations/boxplot_prob_per_tiff_100.png')

map_tiff_cat = preds_df_sel.groupby("Tiff")["Tiff_cat"].max().reset_index()
map_tiff_cat.to_csv(path + '/Visualizations/boxplot_map_tiff_to_number.csv')

"""# 8. Correlation randomized pixels and probability"""

# Join noise percentage per image
base_data_path = '/content/drive/My Drive/Esmee/' + str(org_size[0]) + 'x' + str(org_size[1])
perc_black_per_img_df = pd.read_csv(base_data_path + '/threshold=' + str(threshold) + '-perc_black_per_img.csv', index_col=0)
predictions_features_noise_df = output_df.merge(perc_black_per_img_df, on='image', how='left')
predictions_features_noise_df['image_path'] = predictions_features_noise_df['label'] + '/' + predictions_features_noise_df['image']

# Select top 1000 most dense patches from each compartment to check for correlation
correct_preds_outside_df = predictions_features_noise_df[(predictions_features_noise_df.ind_correct_pred == 1) & (predictions_features_noise_df.enc_label == 1)]
correct_preds_outside_df['image_path'] = correct_preds_outside_df['label'] + '/' + correct_preds_outside_df['image']
patches_outside_for_calculation_df = list(correct_preds_outside_df.sort_values(by=['perc_black'], ascending=True)['image_path'][:100])

correct_preds_inside_df = predictions_features_noise_df[(predictions_features_noise_df.ind_correct_pred == 1) & (predictions_features_noise_df.enc_label == 0)]
patches_inside_for_calculation_df = list(correct_preds_inside_df.sort_values(by=['perc_black'], ascending=True)['image_path'][:100])

img_paths = list(patches_outside_for_calculation_df) + list(patches_inside_for_calculation_df)

print('Reading data')
data, labels, enc_labels = read_data(data_path, size)

print('Loading model')
model = load_model(model_path + '/weights_auc.hdf5')

# Function to get all RGB codes from patches
def get_palette():
  colors = set()
  for img_2 in data[:1000]:
    palette = img_2.reshape(img_2.shape[0]*img_2.shape[1], img_2.shape[2])
    palette = np.unique(palette, axis=0)
    for color in palette:
      if list(color) != [0,0,0]:
        colors.add(tuple(list(color)))
  return list(colors)

print('Getting color palette')
colors = get_palette()

df = pd.DataFrame()

perc_pixels_masked = range(0, 102, 2)
df['perc_pixels_masked'] = perc_pixels_masked
totals_outside = np.zeros(len(perc_pixels_masked))
count = 0
for img_path in patches_outside_for_calculation_df:
  print('Outside: ' + str(count))

  img = cv2.imread(data_path + '/' + img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = np.array(img, np.float32) / 255

  average_probs = []
  for i in perc_pixels_masked:
    n = int(i * img.shape[0] * img.shape[1] / 100)
    all_combs = list(itertools.product(range(img.shape[0]), range(img.shape[1])))
    sum_probs = 0
    n_iter = 100
    for iter in range(n_iter):
      rnd_inds = random.sample(range(len(all_combs)), n)
      rnd_sample = [all_combs[i] for i in rnd_inds]
      rnd_colors = random.choices(colors, k=n)

      img_new = img.copy()
      for c in range(len(rnd_sample)):
        px = rnd_sample[c]
        img_new[px[0], px[1]] = rnd_colors[c]

      img_new = cv2.resize(img_new, (size[0], size[1]))

      prob = model.predict(np.expand_dims(img_new, axis=0))
      sum_probs += prob[0][0]

    average_probs.append(sum_probs/n_iter)

  totals_outside += np.array(average_probs)
  count+=1


df['out'] = totals_outside / 25

totals_inside = np.zeros(len(perc_pixels_masked))
count = 0
for img_path in patches_inside_for_calculation_df:
  print('Inside: ' + str(count))

  img = cv2.imread(data_path + '/' + img_path)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = np.array(img, np.float32) / 255

  average_probs = []
  for i in perc_pixels_masked:
    n = int(i * img.shape[0] * img.shape[1] / 100)
    all_combs = list(itertools.product(range(img.shape[0]), range(img.shape[1])))
    sum_probs = 0
    n_iter = 100
    for iter in range(n_iter):
      rnd_inds = random.sample(range(len(all_combs)), n)
      rnd_sample = [all_combs[i] for i in rnd_inds]
      rnd_colors = random.choices(colors, k=n)

      img_new = img.copy()
      for c in range(len(rnd_sample)):
        px = rnd_sample[c]
        img_new[px[0], px[1]] = rnd_colors[c]

      img_new = cv2.resize(img_new, (size[0], size[1]))

      prob = model.predict(np.expand_dims(img_new, axis=0))
      sum_probs += prob[0][0]

    average_probs.append(sum_probs/n_iter)

  totals_inside += np.array(average_probs)
  count+=1

df['in'] = totals_inside / 25

fig = plt.figure()
plt.plot(perc_pixels_masked, df.iloc[:,1], label = 'Adventitia')
plt.plot(perc_pixels_masked, df.iloc[:,2], label = 'Submucosa')
plt.xlabel('Percentage of randomized pixels')
plt.ylabel('Probability of adventitia')
plt.ylim(0,1)
plt.legend()
plt.show()
fig.savefig(model_path + '/Visualizations/random_check_multiple_step02.svg')

df.to_csv(model_path + 'Visualizations/random_probs_multiple_step02.csv')


plt.ylim(0,1)
plt.legend()
plt.show()
fig.savefig(model_path + '/Visualizations/random_check_multiple_step02.svg')

df.to_csv(model_path + 'Visualizations/random_probs_multiple_step02.csv')

