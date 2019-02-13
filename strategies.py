import numpy as np
from keras.models import Model, model_from_json
from keras.layers import TimeDistributed,Conv1D,Dense,Embedding,Input,Dropout,LSTM,Bidirectional,MaxPooling1D,Flatten,concatenat0e
from helper import get_model


def leastConfidense(dataset, model):
    model = 
    activeDataset = []
    datasetProb = {}
    for i, data in enumerate(dataset):
        tokens, casing,char, labels = data
        tokens = np.asarray([tokens])     
        casing = np.asarray([casing])
        char = np.asarray([char])
        predProbabilities = model.predict([tokens, casing,char], verbose=False)[0]
        predProbabilities = pred.argmax(axis=-1)
        maxProb = max(predProbabilities)
        datasetProb[i] = 1 - maxProb
    sortedProb = sorted(datasetProb.values(), reverse = True)
    # for i in range(len(sortedProb)/10):
    #     activeDataset.append(dataset[])

