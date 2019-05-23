import numpy as np 
from keras.models import Model
from keras.layers import TimeDistributed,Conv1D,Dense,Embedding,Input,Dropout,LSTM,Bidirectional,MaxPooling1D,Flatten,concatenate
from keras.utils import Progbar
from leastConfidence import leastconfidence
from entropySampling import entropysampling
from deepfool import DeepFool


def active_learn(model, learn_batch, flagged, method, package):
    confidence = {}

    # print(len(learn_batch))
    a = Progbar(len(learn_batch))

    if(method == 'deepfool'):
        active = DeepFool(package)
        data = DeepFool.generate(learn_batch, flagged)
        return data


    for i,data in enumerate(learn_batch):
        if flagged[i] == 1:
            continue    
        tokens, casing,char, labels = data
        tokens = np.asarray([tokens])     
        casing = np.asarray([casing])
        char = np.asarray([char])
        if package.modelName == "LSTM_word":  
            pred = model.predict([tokens], verbose=False)[0]
        elif package.modelName == "LSTM_word_char":
            pred = model.predict([tokens, casing,char], verbose=False)[0]
        #print(pred)
        pred = pred.max(axis=-1) #Predict the classes  
        #print(pred)    
        if method == "leastconfidence":      
            confidence[i] = leastconfidence(pred)
        elif method == "entropySampling":
            confidence[i] = entropysampling(pred)
        elif method == "deepfool":
            confidence[i] = active.genrate(data)


    data = []
    count = 0
    #print(confidence)
    for key, value in sorted(confidence.items(), key=lambda kv:(kv[1], kv[0]),reverse = True):
        data.append(learn_batch[key])
        count = count + 1
        flagged[key] = 1
        if count > len(learn_batch)/40:
            break
    # print(data)
    return data, flagged
