import numpy as np
import scipy.stats as stats
import math

class NaiveBayes: 
    def __init__(self, training_data):
        self.data = training_data

        self.sick_data = [data for data in training_data if data[-1]==2.0]
        self.healthy_data = [data for data in training_data if data[-1]==1.0]

        self.sick_data_features = self.seperate_features(self.sick_data)
        self.healthy_data_features = self.seperate_features(self.healthy_data)

        self.sick_pdfs = self.train_model(self.sick_data_features)
        self.healthy_pdfs = self.train_model(self.healthy_data_features)

        self.prior_p_sick = len(self.sick_data)/len(self.data)
        self.prior_p_healthy = len(self.healthy_data)/len(self.data)


    #transposing the training_data will create a separate list for each feature
    def seperate_features(self, data):
        features = list(map(list, zip(*data)))

        del(features[-1])
        return features

    def train_model(self, data_features):
        pdfs = []
        for feature in data_features:
            mean= np.mean(feature)
            std = np.std(feature)
            pdfs.append(stats.norm(mean, std))

        return pdfs

    def test_entry(self, entry):
        smallest_number = math.nextafter(0.,1.)
        #using ln to prevent underflow
        healthy_p = math.log(self.prior_p_healthy)
        sick_p = math.log(self.prior_p_sick)
        for i in range(len(entry)-1):
            healthy_likelihood = self.healthy_pdfs[i].pdf(entry[i])
            sick_likelihood = self.sick_pdfs[i].pdf(entry[i])
            #likelihood is never really zero just a very small number
            if healthy_likelihood ==0:
                healthy_likelihood = smallest_number
            if sick_likelihood ==0:
                sick_likelihood = smallest_number
            
            healthy_p += math.log(healthy_likelihood)
            sick_p += math.log(sick_likelihood)

        if healthy_p>sick_p:
            if int(entry[-1])==1:
                return "TN" #True Negative
            else:
                return "FN" #False Negative
        else:
            if int(entry[-1])==2:
                return "TP" #True Positive
            else:
                return "FP" #False Positive

    def test_data_set(self, data):
        FP = 0
        TP = 0
        FN = 0
        TN = 0
                
        for entry in data:
            result = self.test_entry(entry)
            if result == "TP":
                TP+=1
            elif result == "FP":
                FP+=1
            elif result == "FN":
                FN+=1
            else:
                TN+=1

        accuracy = (TP+TN)/(TP+TN+FP+FN)
        sensitivity = TP/(TP+FN)
        specificity = TN/(TN+FP)
        geometric_mean = math.sqrt(sensitivity*specificity)
        
        return accuracy, sensitivity, specificity, geometric_mean
