import numpy as np
import scipy
from contextlib import closing
import pickle as pkl
import os
from keras.models import Model
import keras.backend as K

class DeepFool(object):

    def __init__(self, modelPackage):
        self.model = modelPackage.model
        self.no_of_tags = len(modelPackage.label2Idx)
        last_dense = self.model.layers[-2].output
        shape = self.model.get_input_shape_at(0)
        print(shape)
        n_channels, img_nrows, img_ncols = shape[1:]
        self.nb_class = self.model.get_output_shape_at(0)[-1]
        last_dense = self.model.layers[-2].output
        second_model = Model(self.model.input, last_dense)
        # second_model.summary()



        adversarial_image = K.placeholder((1, n_channels, img_nrows, img_ncols))
        adverserial_target = K.placeholder((1, nb_class))
        adv_noise = K.placeholder((1, n_channels, img_nrows, img_ncols))

        self.adversarial_image = adversarial_image
        self.adverserial_target = adverserial_target

        loss_classif = K.mean(second_model.call(self.adversarial_image)[0, K.argmax(self.adverserial_target)])
        grad_adverserial = K.gradients(loss_classif, self.adversarial_image)
        self.f_loss = K.function([K.learning_phase(), self.adversarial_image, self.adverserial_target], loss_classif)
        self.f_grad = K.function([K.learning_phase(), self.adversarial_image, self.adverserial_target], grad_adverserial)
        
        def eval_loss(x,y):
            y_vec = np.zeros((1, self.nb_class))
            y_vec[:,y] +=1
            return self.f_loss([0., x, y_vec])
        
        def eval_grad(x,y):
            y_vec = np.zeros((1, self.nb_class))
            y_vec[:,y] +=1
            return self.f_grad([0., x, y_vec]) 
        
        self.eval_loss = eval_loss
        self.eval_grad = eval_grad



    def predict(self, data):
        tokens, casing, char, labels = data
        tokens = np.asarray([tokens])     
        casing = np.asarray([casing])
        char = np.asarray([char])
        pred = self.model.predict([tokens, casing,char], verbose=False)[0] 

        pred = pred.argmax(axis=-1) #Predict the classes 
        return pred       

    def generate(self, data, flagged):
        perturbations = []
        adv_attacks = []
        for i in range(len(data)):
            if flagged[i] == 1:
                continue
            r_i, x_i = self.generate_sample(data[i:i+1])
            perturbations.append(r_i)
            adv_attacks.append(x_i[0])
        
        index_perturbation = np.argsort(perturbations)
        tmp = np.array(adv_attacks)
        return index_perturbation, tmp[index_perturbation]

    def match(sample1, sample2):
        if len(sample1) != len(sample2):
            return 0
        for i in range(len(sample1)):
            if(sample1[i] != sample2[i]):
                return 0
        return 1

    def generate_other_labels(self, true_tag):
        other_tags = []
        for i in range(len(true_tag)):
            curr_tag = true_tag
            for j in range(self.no_of_tags):
                if true_tag[i] != j:
                    curr_tag[i] = j
                    other_tags.append(curr_tag)

        return other_tags


    def generate_sample(self, sample):
        true_tag = self.predict(sample)

        x_i = np.copy(true_tag); i=0
        predicted = self.predict(x_i)
        while self.match(predicted, true_tag) == 1 and i < 10:
            other_tags = self.generate_other_labels(true_tag)
            w_labels=[]; f_labels=[]
            for k in other_tags:
                w_k = (self.eval_grad(x_i,k).flatten() - self.eval_grad(x_i, true_tag).flatten())
                f_k = np.abs(self.eval_loss(x_i, k).flatten() - self.eval_loss(x_i, true_tag).flatten())
                w_labels.append(w_k); f_labels.append(f_k)
            result = [f_k/(sum(np.abs(w_k))) for f_k, w_k in zip(f_labels, w_labels)]
            label_adv = np.argmin(result)
            
            r_i = (f_labels[label_adv]/(np.sum(np.abs(w_labels[label_adv]))) )*np.sign(w_labels[label_adv])
            #print(self.predict(x_i), f_labels[label_adv], np.mean(x_i), np.mean(r_i))
            if np.max(np.isnan(r_i))==True:
                return 0, true_tag
            x_i += r_i.reshape(true_tag.shape)
            #x_i = np.clip(x_i, self.mean - self.std, self.mean+self.std)
            i+=1

            predicted = self.predict(x_i)
        
        adv_tag = x_i
        adv_label = self.predict(adv_tag)
        if adv_label == true_tag:
            return np.inf, x_i
        else:
            perturbation = (x_i - true_tag).flatten()
            #return np.linalg.norm(perturbation)
            return np.max(np.abs(perturbation)), x_i

    



