from numpy import loadtxt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from typing import OrderedDict
dataset=loadtxt("books.csv",delimiter=",")
x=dataset[:,4:11]
y=dataset[:,3]
model = Sequential()
model.add(Dense(128, input_dim=8, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(56, activation='relu'))
model.add(Dense(1, activation='sigmoid'))