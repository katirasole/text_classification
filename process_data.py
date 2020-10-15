#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import csv
from sklearn.model_selection import train_test_split
import os

# read the dataset and then split into train and test data
filename = ('./text_classification/data/sample_complaint_data_90k.csv')
data = pd.read_csv(filename)

#=================================Split train/test/dev ============================
xTrain, xTest = train_test_split(data, test_size = 0.1, random_state = 0)

#--------------------write train/test/dev---------------------------------

def write (data, path):
    print("---Writing starts---") 
    text = data.text.values
    category = data.category.values
    row_data = {'text':text, 'category':category}
    df = pd.DataFrame(row_data, columns = ['text', 'category'])
    df.to_csv(path)
    print("---Writing ends---") 
    return 

# Write the files if there are not exist 
if not os.path.exists('./text_classification/data/train.csv' and './text_classification/data/test.csv'):
    writeFileTrain = ('./text_classification/data/train.csv')
    writeFileTest = ('./text_classification/data/test.csv')
    write(xTrain, writeFileTrain)
    write(xTest, writeFileTest)


# In[ ]:




