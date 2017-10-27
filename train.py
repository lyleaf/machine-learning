import google.datalab.storage as storage
from google.datalab import Context
import random, string
from os.path import join
import csv

## Bring to you by Yiling with collie
## Output should be local_eval_file and local_train_file
# import pandas as pd
# import numpy as np
# csv_path = '/content/labelcsv/all.csv'

# df_csv = pd.read_csv(csv_path,header=1,names=['img_url','isCancer'])
# # Randomize the data before selecting train / validation splits.
# df_csv = df_csv.reindex(np.random.permutation(
#   df_csv.index))
# print df_csv.columns
# print df_csv.describe()

# # Choose the first 7500 (out of 9307) examples for training.
# TRAIN_NUM =  24000            #7500
# VALIDATION_NUM = 6000        #1807

# training_examples = df_csv.head(TRAIN_NUM)

# # Choose the last 1807 (out of 9307) examples for validation.
# validation_examples = df_csv.tail(VALIDATION_NUM)

# training_examples.to_csv('/content/labelcsv/train.csv',index=False, header=False)     ## i stripped the headers
# validation_examples.to_csv('/content/labelcsv/eval.csv',index=False, header=False)    ## i stripped the headers

# local_train_file = '/content/labelcsv/train.csv'
# local_eval_file = '/content/labelcsv/eval.csv'

# !head '/content/labelcsv/train.csv'
# !head '/content/labelcsv/eval.csv'

import mltoolbox.image.classification as model
from google.datalab.ml import *
import logging
import datetime
import mltoolbox.image.classification as model
from google.datalab.ml import *
import logging
import datetime

#lets get some logging going!
logging.getLogger().setLevel(logging.INFO)
print("logging set")

#setting up preprocess start timestamp
pp_start_datetime = datetime.datetime.now()
print("start: " + pp_start_datetime.strftime("%H:%M:%S"))

#do the preprocessing
preprocessed_dir = '/content/mltrain/preprocessed_dir'
train_set = CsvDataSet(local_train_file, schema='image_url:STRING,label:STRING')  ## changed schema from img_url to image_url
model.preprocess(train_set, preprocessed_dir)

#setting up preprocess end timestamp
pp_end_datetime = datetime.datetime.now()

#print out the times
print("start: " + pp_start_datetime.strftime("%H:%M:%S"))
print("end: " + pp_end_datetime.strftime("%H:%M:%S"))
print("duration: " + str((pp_end_datetime-pp_start_datetime).total_seconds()/60) + " minutes")

batchsize = 30
iterations = 30000
import mltoolbox.image.classification as model
from google.datalab.ml import *
import logging
import datetime
model_dir = '/content/mltrain/modelV2_dir' + "_" +str(batchsize)+ "_" + str (iterations)
preprocessed_dir = '/content/mltrain/preprocessed_dir'

#setting up preprocess start timestamp
pp_start_datetime = datetime.datetime.now()
print("start: " + pp_start_datetime.strftime("%H:%M:%S"))

import logging
logging.getLogger().setLevel(logging.INFO)
model.train(preprocessed_dir, batchsize, iterations, model_dir)
logging.getLogger().setLevel(logging.WARNING)

#setting up preprocess end timestamp
pp_end_datetime = datetime.datetime.now()

#print out the times
print("start: " + pp_start_datetime.strftime("%H:%M:%S"))
print("end: " + pp_end_datetime.strftime("%H:%M:%S"))
print("duration: " + str((pp_end_datetime-pp_start_datetime).total_seconds()/60) + " minutes")
