import numpy as np
import scipy.stats as stats
import math
from sklearn.svm import LinearSVC
from sklearn.svm import SVC

class SupportVectorMachine: 
    def __init__(self, c, g = None):
        if g==None:
            self.SVM = LinearSVC(C=c, max_iter=70000)
        else:
            self.SVM = SVC(kernel='rbf', C=c, gamma=g, max_iter=70000)

    def train_model(self, training_data):
        training_array = np.array(training_data)

        training_x = training_array[:,:-1]
        training_y = training_array[:,-1]
        self.SVM.fit(training_x,training_y)

    def test_data_set(self, test_data):

        testing_array = np.array(test_data)
        testing_x = testing_array[:,:-1]
        testing_y = testing_array[:,-1]

        predicted_y = self.SVM.predict(testing_x)

        FP = 0
        TP = 0
        FN = 0
        TN = 0
                
        for i in range(len(predicted_y)):
            if int(predicted_y[i])==2 and int(testing_y[i])==2:
                TP+=1
            elif int(predicted_y[i])==1 and int(testing_y[i])==1:
                TN+=1
            elif int(predicted_y[i])==2 and int(testing_y[i])==1:
                FP+=1
            elif int(predicted_y[i])==1 and int(testing_y[i])==2:
                FN+=1

        accuracy = (TP+TN)/(TP+TN+FP+FN)
        sensitivity = TP/(TP+FN)
        specificity = TN/(TN+FP)
        geometric_mean = math.sqrt(sensitivity*specificity)
        return accuracy, sensitivity, specificity, geometric_mean

