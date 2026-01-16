import sys
import os

# Add this near the top of your script, after the imports
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush() # If you want the output to be visible immediately
    def flush(self):
        for f in self.files:
            f.flush()

# Redirect stdout to both terminal and file
f = open('/workspace/terminal_output.txt', 'w')
original_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, f)

# Import necessary libraries
import cv2
import os, gc, sys, glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import itertools
import random
import datetime
import time
import shutil
import logging

# Import libraries for handling imbalanced datasets
import imblearn
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler

# Import scikit-learn libraries for data preprocessing and evaluation
from sklearn import model_selection
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, GroupShuffleSplit, StratifiedKFold
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import shuffle

# Import image processing libraries
from skimage.transform import rotate
import scipy.ndimage as ndi

from subprocess import check_output

# Import TensorFlow and Keras libraries
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import applications
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint, CSVLogger
from tensorflow.keras.metrics import categorical_accuracy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers.schedules import ExponentialDecay

# Other configs
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Hide TensorFlow warnings from XLA that are not relevant
import absl.logging
import logging

# Adjust logging levels
absl.logging.set_verbosity(absl.logging.INFO)
logging.getLogger('tensorflow').setLevel(logging.INFO)

# Adjust PTXAS settings
os.environ['TF_XLA_FLAGS'] = '--tf_xla_cpu_global_jit'
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

# Add this near the top of your script, after the imports
VERBOSITY_GLOBAL = 2  # Set this to 0 for minimal output, 1 for moderate, 2 for detailed

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Changed from '3' to '2'
logger = tf.get_logger()
logger.setLevel(logging.WARNING)  # Changed from ERROR to WARNING

print(tf.__version__)

# Print available devices (CPU/GPU)
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Add garbage collection at the beginning of the script
import gc
gc.collect()
tf.keras.backend.clear_session()

# Function to build and compile the model
def build_model(tf_model, full_retrain, freeze_till_block, loss_func, lr, decay, dr):
    # Select base model and set up layer freezing
    img_rows, img_cols, img_channel = 224, 224, 3
    if tf_model == 'InceptionV3':
        base_model = applications.inception_v3.InceptionV3(include_top=False,
                                                   weights='/content/drive/My Drive/Esmee/Network_weights/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5', #'imagenet',
                                                   pooling='avg',
                                                   input_shape=(img_rows, img_cols, img_channel))

        lay_num_freeze = int(inception_blocksV3[freeze_till_block,2])

    if tf_model == 'Inception-ResNet-V2':
        base_model = tf.keras.applications.InceptionResNetV2(include_top=False,
                                                   weights='imagenet',
                                                   pooling='avg',
                                                   input_shape=(img_rows, img_cols, img_channel))

        lay_num_freeze = int(inception_resnet_V2_blocks[freeze_till_block,2])

    if tf_model == 'EfficientNetB6':
        base_model = tf.keras.applications.EfficientNetB6(include_top=False,
                                                   weights='imagenet',
                                                   pooling='avg',
                                                   input_shape=(img_rows, img_cols, img_channel))

        lay_num_freeze = int(efficientnetb6_blocks[freeze_till_block,2]) # Freeze a percentage of layers

    if tf_model == 'ResNet':
        base_model = applications.resnet.ResNet50(include_top=False,
                                                  weights='/content/drive/My Drive/Esmee/Network_weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                                  pooling='avg',
                                                  input_shape=(img_rows, img_cols, img_channel))

        lay_num_freeze = int(resnet_blocks[freeze_till_block,2])

    print(f"Freezing layers: {lay_num_freeze}")

    # Add top layers to the base model
    add_model = Sequential()
    add_model.add(Dense(1024, activation='relu', input_shape=base_model.output_shape[1:],
                        kernel_initializer='he_uniform',
                        bias_initializer='zeros'))
    add_model.add(Dropout(dr))
    add_model.add(Dense(1, activation='sigmoid'))

    # Create final model
    model = Model(inputs=base_model.input, outputs=add_model(base_model.output))

    # Set up layer freezing based on full_retrain flag
    if full_retrain:
        base_model.trainable = True
    else:
        for layer in base_model.layers[:lay_num_freeze]:
            layer.trainable = False
        for layer in base_model.layers[lay_num_freeze:]:
            layer.trainable = True

    # Always make the top layers trainable
    for layer in add_model.layers:
        layer.trainable = True

    # Create a learning rate schedule
    lr_schedule = ExponentialDecay(
        initial_learning_rate=lr,
        decay_steps=1000,  # Adjust this value based on your needs
        decay_rate=decay
    )

    # In build_model function
    opt = tf.keras.mixed_precision.LossScaleOptimizer(optimizers.Nadam(learning_rate=lr_schedule))


    metrics = [
        tensorflow.keras.metrics.TruePositives(name='tp'),
        tensorflow.keras.metrics.FalsePositives(name='fp'),
        tensorflow.keras.metrics.TrueNegatives(name='tn'),
        tensorflow.keras.metrics.FalseNegatives(name='fn'),
        tensorflow.keras.metrics.BinaryAccuracy(name='accuracy'),
        tensorflow.keras.metrics.Precision(name='precision'),
        tensorflow.keras.metrics.Recall(name='recall'),
        tensorflow.keras.metrics.AUC(name='auc'),
    ]

    model.compile(loss=loss_func, optimizer=opt, metrics=metrics)

    return model

# Function to fit the model with data augmentation
def fit_model(data_path, model, train_steps, batch_size, epochs, output_path, augmentation_level='medium'):
    # Define augmentation parameters based on the level
    augmentation_params = {
        'none': {'rescale': 1.0/255},
        'light': {
            'rescale': 1.0/255,
            'rotation_range': 10,
            'width_shift_range': 0.1,
            'height_shift_range': 0.1,
            'zoom_range': 0.1,
            'horizontal_flip': True
        },
        'medium': {
            'rescale': 1.0/255,
            'rotation_range': 20,
            'width_shift_range': 0.2,
            'height_shift_range': 0.2,
            'shear_range': 0.2,
            'zoom_range': 0.2,
            'horizontal_flip': True,
            'fill_mode': 'nearest'
        },
        'heavy': {
            'rescale': 1.0/255,
            'rotation_range': 30,
            'width_shift_range': 0.3,
            'height_shift_range': 0.3,
            'shear_range': 0.3,
            'zoom_range': 0.3,
            'horizontal_flip': True,
            'vertical_flip': True,
            'fill_mode': 'nearest'
        }
    }

    # Select augmentation parameters based on the specified level
    aug_params = augmentation_params.get(augmentation_level.lower(), augmentation_params['medium'])

    # Set up data generators for training and validation
    train_datagen = ImageDataGenerator(**aug_params)
    val_datagen = ImageDataGenerator(rescale=1.0/255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_path, 'train_dir'),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary'
    )

    val_gen = val_datagen.flow_from_directory(
        os.path.join(data_path, 'val_dir'),
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary'
    )

    AUTOTUNE = tf.data.AUTOTUNE

    # Convert generators to tf.data.Dataset
    def gen_to_dataset(generator):
        return tf.data.Dataset.from_generator(
            lambda: generator,
            output_types=(tf.float32, tf.float32),
            output_shapes=([None, 224, 224, 3], [None, ])
        ).prefetch(AUTOTUNE)

    train_ds = gen_to_dataset(train_gen)
    val_ds = gen_to_dataset(val_gen)

    # Apply additional optimizations
    train_ds = train_ds.cache().shuffle(buffer_size=100).prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    # Set up callbacks
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=25, 
                                  verbose=4, cooldown=0, min_delta=0.0001, min_lr=0.00001)

    # Adjust steps_per_epoch for multi-GPU setup if applicable
    try:
        strategy = tf.distribute.get_strategy()
        steps_per_epoch = train_steps // strategy.num_replicas_in_sync
    except:
        steps_per_epoch = train_steps

    cp_weights_auc = ModelCheckpoint(
        os.path.join(output_path, "weights_auc.weights.h5"),  # Changed from .weights.h5 to .keras
        monitor='val_auc',
        save_best_only=True,
        save_weights_only=True,
        mode='max',
        save_freq = 5 * steps_per_epoch #'epoch'
    )
    cp_weights_loss = ModelCheckpoint(
        os.path.join(output_path, "weights_loss.keras"),  # Changed from .hdf5 to .keras
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        mode='min',
        save_freq= 5 * steps_per_epoch #'epoch'
    )

    # Add custom callback for epoch logging
    class EpochLogger(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            if VERBOSITY_GLOBAL > 0:
                logging.info(f"Starting epoch {epoch + 1}/{epochs}")
        
        def on_epoch_end(self, epoch, logs=None):
            if VERBOSITY_GLOBAL > 0:
                logging.info(f"Epoch {epoch + 1}/{epochs} completed. "
                             f"Loss: {logs['loss']:.4f}, Val Loss: {logs['val_loss']:.4f}, "
                             f"AUC: {logs['auc']:.4f}, Val AUC: {logs['val_auc']:.4f}")

    # Add the EpochLogger to your callbacks list
    callbacks = [reduce_lr, cp_weights_auc, cp_weights_loss, EpochLogger()]

    # Calculate validation steps
    val_steps = len(val_gen)  # Assuming val_gen is your validation generator

    # Fit the model
    if VERBOSITY_GLOBAL > 0:
        logging.info("Starting model training")

    history = model.fit(
        train_ds,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        verbose=1 if VERBOSITY_GLOBAL > 1 else 0,  # Use Keras' built-in verbosity for detailed output
        validation_data=val_ds,
        validation_steps=val_steps,  # Add this line
        callbacks=callbacks
    )

    if VERBOSITY_GLOBAL > 0:
        logging.info("Model training completed")

    # Save training history
    pd.DataFrame(history.history).to_csv(os.path.join(output_path, "history.csv"))

    return model, history

# Add this new function
def count_images_in_train_dir(data_path):
    train_dir = os.path.join(data_path, 'train_dir')
    total_images = 0
    for root, dirs, files in os.walk(train_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                total_images += 1
    return total_images

# Function to summarize model performance and create visualizations
def summarize_model(data_path, model, history):
    # Visualize training history for AUC
    plt.figure()
    plt.plot(history.history['auc'])
    plt.plot(history.history['val_auc'])
    plt.title('model AUC')
    plt.ylabel('AUC')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(output_path, 'auc_development.png'))

    # Visualize training for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(output_path, 'loss_development.png'))

    # Visualize training per class
    acc_class_0 = np.array(history.history['tn']) / (np.array(history.history['tn']) + np.array(history.history['fp'])) * 100
    val_acc_class_0 = np.array(history.history['val_tn']) / (np.array(history.history['val_tn']) + np.array(history.history['val_fp'])) * 100

    plt.figure()
    plt.plot(acc_class_0)
    plt.plot(val_acc_class_0)
    plt.title('Development of classification accuracy for class 0 (Inside)')
    plt.ylabel('% Correct classification')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(output_path, 'acc_class0_development.png'))

    acc_class_1 = np.array(history.history['tp']) / (np.array(history.history['tp']) + np.array(history.history['fn'])) * 100
    val_acc_class_1 = np.array(history.history['val_tp']) / (np.array(history.history['val_tp']) + np.array(history.history['val_fn'])) * 100

    plt.figure()
    plt.plot(acc_class_1)
    plt.plot(val_acc_class_1)
    plt.title('Development of classification accuracy for class 1 (Outside)')
    plt.ylabel('% Correct classification')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(output_path, 'acc_class1_development.png'))

    test_datagen = ImageDataGenerator(rescale=1.0/255)
    test_gen = test_datagen.flow_from_directory(os.path.join(data_path, 'test_dir'),
                                            target_size=(224, 224),
                                            batch_size=1,
                                            class_mode='binary',
                                            shuffle = False)

    # Get test evaluation
    # On best model in terms of validation AUC
    best_auc_model =  load_model(os.path.join(output_path, 'weights_auc.keras'))
    test_evaluation = best_auc_model.evaluate(test_gen)
    test_evaluation_df = pd.DataFrame([test_evaluation], columns = ['loss', 'tp', 'fp', 'tn', 'fn', 'accuracy', 'precision', 'recall', 'auc'])
    test_evaluation_df.to_csv(os.path.join(output_path, 'test_evaluation_best_auc_model.csv'))

    # On best model in terms of validation loss
    best_loss_model =  load_model(os.path.join(output_path, 'weights_loss.keras'))
    test_evaluation = best_loss_model.evaluate(test_gen)
    test_evaluation_df = pd.DataFrame([test_evaluation], columns = ['loss', 'tp', 'fp', 'tn', 'fn', 'accuracy', 'precision', 'recall', 'auc'])
    test_evaluation_df.to_csv(os.path.join(output_path, 'test_evaluation_best_loss_model.csv'))

# Define layer start indices for different model architectures
resnet_blocks = np.matrix([
    ['ResNet', 0, 0],
    ['ResNet', 1, 7],
    ['ResNet', 2, 39],
    ['ResNet', 3, 81],
    ['ResNet', 4, 143]
])

inceptionV3_blocks = np.matrix([
    ['Inception', 0, 42],
    ['Inception', 1, 65],
    ['Inception', 2, 88],
    ['Inception', 3, 102],
    ['Inception', 4, 134],
    ['Inception', 5, 166],
    ['Inception', 6, 198],
    ['Inception', 7, 230],
    ['Inception', 8, 250],
])

inception_resnet_V2_blocks = np.matrix([
    ['Inception', 0, 1],
    ['Inception', 1, 60],  # Inception-ResNet-A block
    ['Inception', 2, 288], # Inception-ResNet-B block
    ['Inception', 3, 595],
    ['Inception', 4, 631], # Inception-ResNet-C block
    ['Inception', 5, 777]
])

efficientnetb6_blocks = np.matrix([
    ['EfficientNetB6', 0, 0],    # Input and initial convolution
    ['EfficientNetB6', 1, 3],    # First set of MBConv1 blocks
    ['EfficientNetB6', 2, 7],    # MBConv6 blocks (112x112)
    ['EfficientNetB6', 3, 17],   # MBConv6 blocks (56x56)
    ['EfficientNetB6', 4, 34],   # MBConv6 blocks (28x28)
    ['EfficientNetB6', 5, 55],   # MBConv6 blocks (14x14, first part)
    ['EfficientNetB6', 6, 99],   # MBConv6 blocks (14x14, second part)
    ['EfficientNetB6', 7, 175],  # MBConv6 blocks (7x7, first part)
    ['EfficientNetB6', 8, 323],  # MBConv6 blocks (7x7, second part)
    ['EfficientNetB6', 9, 528]   # Final convolution and pooling
])

# Main execution block
def main():
    # Configuration
    config = {
        'org_size': [120, 120],
        'size': [224, 224],
        'data_imputation_type': 'random_dataset',
        'threshold': 0.8,
        'tf_model': 'Inception-ResNet-V2',
        'full_retrain': False,
        'freeze_till_block': 5,
        'loss_func': 'binary_crossentropy',
        'lr': 0.005,
        'decay': 0,
        'dr': 0.6,
        'batch_size': 32,
        'epochs': 100,
        'train_images_count': 30000, # To be dynamically updated
        'independent_sets': False,
        'class_imb': 'Oversample'
    }

    # Set up paths
    config['data_path'] = f'/workspace/ImageRecognition/1_data/patches_{config["org_size"][0]}x{config["org_size"][1]}/patches_cutoff_{config["data_imputation_type"]}_imputed_{config["threshold"]}/'
    config['output_path'] = create_output_folder(config)

    # Calculate train steps
    config['train_images_count'] = count_images_in_train_dir(config['data_path'])
    config['train_steps'] = int(np.ceil(config['train_images_count'] / config['batch_size']))

    # Check available GPUs
    gpus = tf.config.list_physical_devices('GPU')
    num_gpus = len(gpus)
    print(f"Number of GPUs available: {num_gpus}")

    if num_gpus > 1:
        # Set up multi-GPU strategy
        strategy = tf.distribute.MirroredStrategy()
        print(f"Using multi-GPU strategy with {strategy.num_replicas_in_sync} devices")
        # Adjust batch size based on number of GPUs
        config['batch_size'] *= strategy.num_replicas_in_sync
        # Build and train model using the strategy
        with strategy.scope():
            model = build_and_train_model(config)
    else:
        print("Using single GPU or CPU")
        model = build_and_train_model(config)

    # Summarize model performance
    summarize_model(config['data_path'], model, model.history)

def create_output_folder(config):
    folder_name = f'threshold={config["threshold"]}_{config["tf_model"]}_{config["class_imb"]}_retrain={config["full_retrain"]}_{config["freeze_till_block"]}_{config["loss_func"]}_{config["lr"]}_{config["decay"]}_{config["dr"]}'
    output_path = f'/workspace/ImageRecognition/5_results/patches_{config["org_size"][0]}x{config["org_size"][1]}/{config["data_imputation_type"]}/{folder_name}'

    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

    return output_path

def build_and_train_model(config):
    print('Building model')
    model = build_model(config['tf_model'], config['full_retrain'], config['freeze_till_block'], 
                        config['loss_func'], config['lr'], config['decay'], config['dr'])
    
    print(f'Start fitting {config["tf_model"]} model with parameters: '
          f'train base = {config["full_retrain"]}, '
          f'freeze till block = {config["freeze_till_block"]}, '
          f'learning rate = {config["lr"]}, '
          f'decay rate = {config["decay"]}, '
          f'dropout rate = {config["dr"]}')

    trainable_count = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)
    print(f"Number of trainable parameters: {trainable_count}")

    # Fit the model and get training history
    model, history = fit_model(config['data_path'], model, config['train_steps'], 
                               config['batch_size'], config['epochs'], config['output_path'])
    
    return model

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}", exc_info=True)
    finally:
        # Ensure we close the file and restore stdout even if an exception occurs
        f.close()
        sys.stdout = original_stdout
