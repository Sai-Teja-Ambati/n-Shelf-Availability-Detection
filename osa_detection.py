# -*- coding: utf-8 -*-
"""OSA Detection.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/181VdlsHrVU9buya_qrtT96allRJSGkUC
"""

!git clone https://github.com/ultralytics/yolov5

# Commented out IPython magic to ensure Python compatibility.
# %cd yolov5
!pip install -r requirements.txt
!pip install ultralytics -q
!pip install pyyaml -q

from ultralytics import YOLO
import yaml

!pip install roboflow

# Commented out IPython magic to ensure Python compatibility.
from roboflow import Roboflow
# %cd yolov5
rf = Roboflow(api_key="jz5RDy8NpSO3kiw0DiAP")
project = rf.workspace("srm-university-52jqz").project("retail-shelf-avialability")
dataset = project.version(2).download("yolov5")

!python /content/yolov5/train.py --data /content/yolov5/retail-shelf-avialability-2/data.yaml --cfg yolov5s.yaml --batch-size 8 --epochs 100 --name Model

# install dependencies as necessary
!pip install -qr /content/yolov5/requirements.txt  # install dependencies (ignore errors)
import torch
from IPython.display import Image, clear_output  # to display images

# clear_output()
print('Setup complete. Using torch %s %s' % (torch.__version__, torch.cuda.get_device_properties(0) if torch.cuda.is_available() else 'CPU'))

# we can also output some older school graphs if the tensor board isn't working for whatever reason...
import matplotlib as pyplot
Image(filename='/content/yolov5/runs/train/Model/results.png', width=1000)  # view results.png

print("GROUND TRUTH TRAINING DATA:")
Image(filename='/content/yolov5/runs/train/Model/train_batch0.jpg', width=900)

print("GROUND TRUTH AUGMENTED TRAINING DATA:")
Image(filename='/content/yolov5/runs/train/Model/val_batch0_labels.jpg', width=900)

# Commented out IPython magic to ensure Python compatibility.
# %cd /content/yolov5/
!python detect.py --weights /content/yolov5/runs/train/Model//weights/best.pt --img 416 --conf 0.4 --source /content/yolov5/retail-shelf-avialability-2/test/images --save-txt

# !zip -r output.zip /content/yolov5/runs/detect/exp
# This line of code can be used to zip the detected results

import numpy as np
import pandas as pd
import glob
from glob import glob

all_file_path=glob('/content/yolov5/runs/detect/exp/labels/*')
print(all_file_path)

data_list = []

# List of column names
columns = ['img_name', 'no_of_voids', 'area']

# Create an empty DataFrame with specified columns
df = pd.DataFrame(columns=columns)

# Display the empty DataFrame
print(df)

for f in range(len(all_file_path)):
    filepath = all_file_path[f]
    with open(filepath, 'r') as file:
            # Read the entire file as one space-separated string and split it into columns
        lines = file.readlines()
        ar=0
        no_of_voids=len(lines)
        for line in lines:
          data=list(map(float,line.split()))
          ar+=data[-1]*data[-2]
        new_row_data = {'img_name': filepath, 'no_of_voids': no_of_voids, 'area': ar}
        new_row_df = pd.DataFrame(new_row_data, index=[0])
        # Concatenate the new row DataFrame with the original DataFrame
        df = pd.concat([df, new_row_df], ignore_index=True)
print(df)

max_value = df['area'].max()
print(max_value)