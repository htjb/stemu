"""from globalemu.downloads import download

download().model() # Redshift-Temperature Network"""

# Download the 21cmGEM data from Zenodo
import requests
import os
import numpy as np

data_dir = 'downloaded_data/'
if not os.path.exists(data_dir):
  os.mkdir(data_dir)

files = ['Par_test_21cmGEM.txt', 'Par_train_21cmGEM.txt', 'T21_test_21cmGEM.txt', 'T21_train_21cmGEM.txt']
saves = ['test_data.txt', 'train_data.txt', 'test_labels.txt', 'train_labels.txt']

for i in range(len(files)):
  url = 'https://zenodo.org/record/4541500/files/' + files[i]
  with open(data_dir + saves[i], 'wb') as f:
      f.write(requests.get(url).content)