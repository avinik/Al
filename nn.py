#!/usr/bin/env python
# coding: utf-8

import io
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '5'
import tensorflow as tf
import numpy as np 
from validation import compute_f1
from keras.models import Model
from keras.layers import TimeDistributed,Conv1D,Dense,Embedding,Input,Dropout,LSTM,Bidirectional,MaxPooling1D,Flatten,concatenate
from prepro import readfileTwitter,createBatches,createMatrices,iterate_minibatches,addCharInformatioin,padding, readfile
from keras.utils import Progbar
from keras.preprocessing.sequence import pad_sequences
from keras.initializers import RandomUniform
from helper import get_model, save_model, tag_dataset, train, test, pseudoRead
from models import LSTM_word, LSTM_word_char, createModel
from query_strategies.active_learning import active_learn
from ModelPackage import ModelPackage
from sklearn.model_selection import KFold


epochs = 1
samplingMethod = sys.argv[2] #"entropySampling"]

#Name of the model... See models.py for details
modelName = "LSTM_word_char"
datasetName = sys.argv[1] #"Cadec"

print(datasetName + " " +  samplingMethod)


#Loading The dataset

if datasetName == "Twitter":
    trainSentences = readfileTwitter("twitter/TwitterTrainBIO.tsv")
    learnSentences = trainSentences[int(len(trainSentences)/10):]
    trainSentences = trainSentences[:int(len(trainSentences)/10)]
    testSentences = readfile("twitter/TwitterTestBIO.tsv")

elif datasetName == "Medline":
    trainSentences = readfileTwitter("twitter/MedlineBIO.tsv")
    learnSentences = []
    testSentences = []

elif datasetName == "Cadec":
    trainSentences = readfileTwitter("twitter/CadecBIO.tsv")
    learnSentences = []
    testSentences = []


trainSentences = addCharInformatioin(trainSentences)
learnSentences = addCharInformatioin(learnSentences)
testSentences = addCharInformatioin(testSentences)


labelSet = set()
words = {}

for dataset in [trainSentences, learnSentences, testSentences]:
    for sentence in dataset:
        for token,char, label in sentence:
            labelSet.add(label)
            words[token.lower()] = True

# :: Create a mapping for the labels ::
label2Idx = {}
for label in labelSet:
    label2Idx[label] = len(label2Idx)

# :: Hard coded case lookup ::
case2Idx = {'numeric': 0, 'allLower':1, 'allUpper':2, 'initialUpper':3, 'other':4, 'mainly_numeric':5, 'contains_digit': 6, 'PADDING_TOKEN':7}
caseEmbeddings = np.identity(len(case2Idx), dtype='float32')


# :: Read in word embeddings ::
word2Idx = {}
wordEmbeddings = []

fEmbeddings = io.open("embeddings/glove.6B.100d.txt", encoding="utf-8")

for line in fEmbeddings:
    split = line.strip().split(" ")
    word = split[0]
    
    if len(word2Idx) == 0: #Add padding+unknown
        word2Idx["PADDING_TOKEN"] = len(word2Idx)
        vector = np.zeros(len(split)-1) #Zero vector vor 'PADDING' word
        wordEmbeddings.append(vector)
        
        word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
        vector = np.random.uniform(-0.25, 0.25, len(split)-1)
        wordEmbeddings.append(vector)

    if split[0].lower() in words:
        vector = np.array([float(num) for num in split[1:]])
        wordEmbeddings.append(vector)
        word2Idx[split[0]] = len(word2Idx)
        
wordEmbeddings = np.array(wordEmbeddings)

char2Idx = {"PADDING":0, "UNKNOWN":1}
s = " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|<>"
s = s + '\t' + '\n' + '\x97' + '\x92'+'\x93' + '\x94' + '\xc2'
for c in s:
    char2Idx[c] = len(char2Idx)


train_set = padding(createMatrices(trainSentences,word2Idx,  label2Idx, case2Idx, char2Idx))
learn_set = padding(createMatrices(learnSentences,word2Idx, label2Idx, case2Idx,char2Idx))
test_set = padding(createMatrices(testSentences, word2Idx, label2Idx, case2Idx,char2Idx))

idx2Label = {v: k for k, v in label2Idx.items()}

train_batch,train_batch_len = createBatches(train_set)
learn_batch,learn_batch_len = createBatches(learn_set)
test_batch,test_batch_len = createBatches(test_set)


#modelPackage conatains Deatiled information 
modelPackage = ModelPackage(wordEmbeddings, caseEmbeddings, word2Idx, label2Idx, char2Idx, modelName, datasetName)

print(modelPackage.modelName)

model = createModel(modelPackage)
modelPackage.model = model
# plot_model(model, to_file='model.png')

precision = 0
recoil = 0
f1 = 0
file = open("Results/"+str(datasetName)+str(samplingMethod) + str(modelName)+ "_Results.txt", "w+")

#ctive Learning Starts here

if datasetName == "Twitter":
    #Initial Training The Model
    model = train(model, train_batch, epochs, modelPackage)
    pre_test, rec_test, f1_test = test(model, test_batch, idx2Label, modelPackage)
    file.write("Initial Precision: " + str(pre_test) + " Initial Recoil : "+str(rec_test) + " Initial F1_Score : " + str(f1_test)+ " Data size : " + str(len(train_batch))+ "\n\n")
    file.flush()
    l = len(learn_batch)
    flagged = []
    last = []
    for i in range(3):
        last.append(0)
    for i in range(l):
        flagged.append(0)
    iter = 0
    while(True):
        modelPackage.model = model

        #Finding the actice data
        active_data, flagged = active_learn(model, learn_batch, flagged, samplingMethod, modelPackage)
        train_batch = train_batch + active_data

        #Retraining of the model
        model = train(model, train_batch, epochs, modelPackage)
        pre_test, rec_test, f1_test = test(model, test_batch, idx2Label,modelPackage)
        precision = precision + pre_test
        recoil = recoil + rec_test
        f1 = f1 + f1_test
        file.write("Iteration : " + str(iter)+"\n")
        file.write("Precision: " + str(pre_test) + " Recoil : "+str(rec_test) + " F1_Score : " + str(f1_test)+ " Data size : " + str(len(train_batch))+ "\n\n")
        file.flush()
        f1_sum = (last[0] + last[1] + f1_test) - 3*last[2]
        last.insert(0, f1_test)
        last.pop()
        #Termination Condition
        if f1_sum < 0 and iter > 10:
            break
        iter = iter + 1
    
    file.write("Avg Precision: " + str(precision/10) + " Avg Recoil : "+str(recoil/10) + " Avg F1_Score : " + str(f1/10)+"\n")
    model = train(model, train_batch, epochs, modelPackage)
    pre_test, rec_test, f1_test = test(model, test_batch, idx2Label,modelPackage)
    file.write("Final Precision: " + str(pre_test) + " Final Recoil : "+str(rec_test) + " Final F1_Score : " + str(f1_test)+"\n\n\n")
    file.flush()
    


#Active Learning using 5 fold

else:
    kf = KFold(5)
    np.random.shuffle(train_batch)
    for train_index, test_index in kf.split(train_batch):
        precision = 0
        recoil = 0
        f1 = 0
        model = createModel(modelPackage)
        train_data = [train_batch[i] for i in train_index]
        test_data = [train_batch[i] for i in test_index]
        l = len(train_data)
        learn_data = train_data[int(l/5):]
        train_data = train_data[:int(l/5)]
        model = train(model, train_data, epochs, modelPackage)
        pre_test, rec_test, f1_test = test(model, test_data, idx2Label, modelPackage)
        file.write("Initial Precision: " + str(pre_test) + " Initial Recoil : "+str(rec_test) + " Initial F1_Score : " + str(f1_test)+ " Data size : " + str(len(train_batch))+ "\n\n")
        file.flush()
        l = len(learn_data)
        flagged = []
        last = []
        for i in range(3):
            last.append(0)
        for i in range(l):
            flagged.append(0)
        iter = 0
        while(True):
            modelPackage.model = model
            #Finding the actice data
            active_data, flagged = active_learn(model, learn_data, flagged, samplingMethod, modelPackage)
            if(len(active_data) == 0):
                break
            train_data = train_data + active_data

            #Retraining of the model
            model = train(model, train_data, epochs, modelPackage)
            pre_test, rec_test, f1_test = test(model, test_data, idx2Label,modelPackage)
            precision = precision + pre_test
            recoil = recoil + rec_test
            f1 = f1 + f1_test
            file.write("Iteration : " + str(iter+1)+"\n")
            file.write("Precision: " + str(pre_test) + " Recoil : "+str(rec_test) + " F1_Score : " + str(f1_test)+ " Data size : " + str(len(train_data))+ "\n\n")
            file.flush()
            f1_sum = (last[0] + last[1] + f1_test) - 3*last[2]
            last.insert(0, f1_test)
            last.pop()

            #Tremination Condition
            if f1_sum < 0 and iter > 10:
                break
            iter = iter + 1
        file.write("Avg Precision: " + str(precision/10) + " Avg Recoil : "+str(recoil/10) + " Avg F1_Score : " + str(f1/10)+"\n")
        file.flush()
        
        model = train(model, train_data, epochs, modelPackage)
        pre_test, rec_test, f1_test = test(model, test_data, idx2Label,modelPackage)
        file.write("Final Precision: " + str(pre_test) + " Final Recoil : "+str(rec_test) + " Final F1_Score : " + str(f1_test)+"\n\n\n")
        file.flush()
        
save_model(model)
