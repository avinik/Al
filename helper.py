from keras.models import Model, model_from_json
import numpy as np
from keras.utils import Progbar
from prepro import iterate_minibatches, createBatches
from validation import compute_f1


def save_model(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights("model.h5")
    print("Saved model to disk")



def get_model(json_file):
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model

def tag_dataset(dataset, model, package):
    correctLabels = []
    predLabels = []
    b = Progbar(len(dataset))
    for i,data in enumerate(dataset):    
        tokens, casing,char, labels = data
        tokens = np.asarray([tokens])     
        casing = np.asarray([casing])
        char = np.asarray([char])

        if package.modelName == "LSTM_word":  
            pred = model.predict([tokens], verbose=False)[0]
        elif package.modelName == "LSTM_word_char":
            pred = model.predict([tokens, casing,char], verbose=False)[0] 

        pred = pred.argmax(axis=-1) #Predict the classes            
        correctLabels.append(labels)
        predLabels.append(pred)
        b.update(i)
    return predLabels, correctLabels

def train(model, train_set, epochs, package):
    train_batch,train_batch_len = createBatches(train_set)

    for epoch in range(epochs):    
        print("Epoch %d/%d"%(epoch,epochs))
        a = Progbar(len(train_batch_len))
        for i,batch in enumerate(iterate_minibatches(train_batch,train_batch_len)):
            labels, tokens, casing, char = batch
            if package.modelName == "LSTM_word":  
                model.fit([tokens], labels, verbose=0)
            elif package.modelName == "LSTM_word_char":
                model.fit([tokens, casing, char], labels, verbose=0)
            a.update(i)
        print(' ')
    return model


def test(model, test_set, idx2Label, package):
    test_batch,test_batch_len = createBatches(test_set)
    predLabels, correctLabels = tag_dataset(test_batch, model, package)        
    pre_test, rec_test, f1_test= compute_f1(predLabels, correctLabels, idx2Label)
    print("Test-Data: Prec: %.3f, Rec: %.3f, F1: %.3f" % (pre_test, rec_test, f1_test))