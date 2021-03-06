#!/usr/bin/env python3
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import sys
from os.path import isfile

from sklearn.model_selection import train_test_split
import warnings
from tensorflow.python.framework import ops
from keras.callbacks import ModelCheckpoint
from sklearn.metrics.classification import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from keras.models import model_from_json
import pandas
from sklearn import preprocessing

import csv
import re
from os.path import abspath
import xml.etree.ElementTree as et

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
ops.reset_default_graph()


def dataset_generator(FILE_PATH):

    raw_data = pd.read_csv(FILE_PATH)					    			# Open raw .csv

    le = preprocessing.LabelEncoder()
    encoded_data = raw_data.apply(le.fit_transform)

    Y_LABEL = 'numD'                                				    # Name of the variable to be predicted
    KEYS = [i for i in encoded_data.keys().tolist() if i != Y_LABEL]	    # Name of predictors
    N_INSTANCES = encoded_data.shape[0]                     			    # Number of instances
    N_INPUT = encoded_data.shape[1] - 1                     			    # Input size
    N_CLASSES = raw_data[Y_LABEL].unique().shape[0]     			    # Number of classes (output size)
    TEST_SIZE = 0.40                                    			    # Test set size (% of dataset)
    TRAIN_SIZE = int(N_INSTANCES * (1 - TEST_SIZE))     			    # Train size
    STDDEV = 0.1                                        			    # Standard deviation (for weights random init)
    RANDOM_STATE = 12345						                        # Random state for train_test_split

    data = encoded_data[KEYS].get_values()                  				# X data
    labels = encoded_data[Y_LABEL].get_values()             				# y data

    labels_ = np.zeros((N_INSTANCES, N_CLASSES))
    labels_[np.arange(N_INSTANCES), labels] = 1

    X_train, X_test, y_train, y_test = train_test_split(data, labels_, test_size=0.20, random_state=100)
    X_train, X_valid, y_train, y_valid = train_test_split(X_test,y_test,test_size = 0.20,random_state = RANDOM_STATE)

    X_train = X_train.reshape(X_train.shape[0], N_INPUT).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], N_INPUT).astype('float32')
    X_valid = X_valid.reshape(X_valid.shape[0], N_INPUT).astype('float32')
    print("	Training data has been generated for your model! \n")

    return X_train, X_test, X_valid, y_train, y_valid, y_test, N_CLASSES


def construct_network(shape, N_CLASSES):
    model = Sequential()
    model.add(Dense(32, activation='relu', input_dim=shape, kernel_initializer="uniform"))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu', kernel_initializer="uniform"))
    model.add(Dropout(0.5))
    model.add(Dense(N_CLASSES, activation='softmax'))

    return model


def model_restorer(model, model_path):
    # Initialize weights using checkpoint if it exists. (Checkpointing requires h5py)
    load_checkpoint = True
    checkpoint_filepath = modelPath + '/DNN_weight_DIC.h5'

    if load_checkpoint:
        print("	Looking for previous weights ...")
        if isfile(checkpoint_filepath):
            print('	A previous checkpoint file has been detected!')
            print('	Loading weights from checkpoint... ')
            model.load_weights(checkpoint_filepath, by_name=True)
        else:
            print('	No checkpoint file detected so starting from scratch ...')
    else:
        print('	Starting from scratch (no checkpoint) ... \n')

    return ModelCheckpoint(filepath=checkpoint_filepath, verbose=1, save_best_only=True)


def train_network_with_pretrained_model(X_train, y_train, X_valid, y_valid, model, checkpointer):
    rmsprop = keras.optimizers.SGD(lr=1e-4, momentum=0.0, decay=0.0, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])

    model.fit(X_train, y_train, validation_data=(X_valid, y_valid), callbacks=[checkpointer], batch_size=32, epochs=100)


def networkTrainer(X_train, y_train, X_valid, y_valid, model):
    rmsprop = keras.optimizers.SGD(lr=1e-4, momentum=0.0, decay=0.0, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop, metrics=['accuracy'])

    model.fit(X_train, y_train, validation_data=(X_valid, y_valid), batch_size=32, epochs=100)

#print(model.summary())

def modelSaver(model, modelPath):
    # serialize weights to HDF5
    model.save_weights(modelPath+'/DNN_weight_DIC.h5')

def modelEvaluator(X_test, y_test, model):
    score = model.evaluate(X_test, y_test, batch_size=32)
    y_prob = model.predict(X_test)
    y_pred = y_prob.argmax(axis=-1)
    y_true = np.argmax(y_test, 1)

    roc = roc_auc_score(y_test, y_prob)

    # evaluate the model
    score, accuracy = model.evaluate(X_test, y_test, batch_size=32)

    # the F-score gives a similiar value to the accuracy score, but useful for cross-checking
    p,r,f,s = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    print("	Now printing performence metrices ...")
    print("	ROC:",  round(roc,3))
    print("	Accuracy = {:.2f}".format(accuracy))
    print("	F-Score:", round(f,2))
    print("	Precision:", round(p,2))
    print("	Recall:", round(r,2))
    print("	F-Score:", round(f,2))


def main():
    modelPath = '/pht_model'
    file_path = '/data/data.csv'
    # Generate the data

    X_train, X_test, X_valid, y_train, y_valid, y_test, N_CLASSES = dataset_generator(FILE_PATH=file_path)

    model = construct_network(shape=X_train.shape[1], N_CLASSES=N_CLASSES)

    print("Training has been started: ")
    try:
        checkpointer = modelRestorer(model, modelPath)
        networkTrainerWithPretrainedModel(X_train, y_train, X_valid, y_valid, model, checkpointer)

    except:
        print("	Pretrained model had different structure so skipping it .... ")
        networkTrainer(X_train, y_train, X_valid, y_valid, model)

    print("	Training has been completed!")

    print("	Saving model's weight to disk ...")
    modelSaver(model, modelPath)
    print("	Model's weight saved to disk! \n")

    # print("Model evaluation has been started: ")
    # modelEvaluator(X_test, y_test, model)
    # print("	Model evaluation completed!")


if __name__ == "__main__":
    main()
