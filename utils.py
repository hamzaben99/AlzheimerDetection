import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from nibabel.testing import data_path
import nibabel as nib

from pathlib import Path
from sklearn.model_selection import train_test_split

def apply_mask(img_n_mmni, img_mask):
    """
        Taking a n_mmni and apply the correspondant mask
        param:
            img_n_mmi   : image n_mmi
            img_mask    : mask
    """
    mmni_m = img_n_mmni.get_fdata()
    mask_m = img_mask.get_fdata().astype(bool)
    mask_bg = np.logical_not(mask_m)
    mmni_m[mask_bg] = 0
    return mmni_m

def process_irm_data():
    """
        Create a new directory and process all images from tha ADNI1 directory
    """
    path = str(Path().resolve())
    path_res = path + "\\ADNI_PROCESSED"
    Path(path_res).mkdir(parents=True, exist_ok=True) # Create a directory for data processed
    path = path + "\\ADNI1"
    for filename in os.listdir(path):
        if filename.startswith("n_mmni"):
            n_mmni_filename = os.path.join(path, filename)
            mask_filename = os.path.join(path, "mask_" + filename)
            img_n_mmni = nib.load(n_mmni_filename)
            img_mask = nib.load(mask_filename)
            n_mmni_mask = apply_mask(img_n_mmni, img_mask)
            img = nib.Nifti1Image(n_mmni_mask, np.eye(4))
            nib.save(img, os.path.join(path_res, filename))


def load_data(path):
    data = pd.read_csv(path, names= ['Subject ID', 'Rooster ID', 'Age', 'Sexe', 'Group', 'Conversion', 'MMSE', 'RAVLT', 'FAQ', 'CDR-SB', 'ADAS11'], usecols = ['Subject ID', 'Rooster ID', 'Group'])
    data.index = data['Subject ID']
    data = data.drop(['Subject ID'], axis=1)
    data = data[(data.Group == 'CN') | (data.Group == 'AD')]
    return data

# Plot the validation and training data separately
def plot_loss_curves(history):
      """
      Returns separate loss curves for training and validation metrics.
      """ 
      loss = history.history['loss']
      val_loss = history.history['val_loss']

      accuracy = history.history['accuracy']
      val_accuracy = history.history['val_accuracy']

      epochs = range(len(history.history['loss']))

      # Plot loss
      plt.plot(epochs, loss, label='training_loss')
      plt.plot(epochs, val_loss, label='val_loss')
      plt.title('Loss')
      plt.xlabel('Epochs')
      plt.legend()

      # Plot accuracy
      plt.figure()
      plt.plot(epochs, accuracy, label='training_accuracy')
      plt.plot(epochs, val_accuracy, label='val_accuracy')
      plt.title('Accuracy')
      plt.xlabel('Epochs')
      plt.legend()
      plt.savefig('accuracy.png')