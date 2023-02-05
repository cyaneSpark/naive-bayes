import numpy as np
import scipy.stats as stats
import math

class KNearestNeighbors: 
    def __init__(self, training_data):
        self.data = training_data

    def manhattan_distance(self, point_A, point_B):
        total = 0
        for i in range(len(point_A)-1):
            total += abs(point_A[i]-point_B[i])   
        return total

    def euclidean_distance(self, point_A, point_B):
        squares_total = 0
        for i in range(len(point_A)-1):
            squares_total += (point_A[i]-point_B[i])**2   
        return np.sqrt(squares_total)
    
    def test_entry(self, entry, k):

        neighbors = []

        for patient in self.data:
            neighbors.append((self.manhattan_distance(entry, patient),patient[-1]))

        neighbors.sort()

        healthy = 0
        sick = 0 

        neighbors = neighbors[:k]

        for neighbor in neighbors:
            if int(neighbor[-1])==1:
                healthy+=1
            elif int(neighbor[-1])==2:
                sick+=1

        if healthy>sick:
            if int(entry[-1])==1:
                return "TN" #True Negative
            else:
                return "FN" #False Negative
        else:
            if int(entry[-1])==2:
                return "TP" #True Positive
            else:
                return "FP" #False Positive

    def test_data_set(self, data, k):
        FP = 0
        TP = 0
        FN = 0
        TN = 0
                
        for entry in data:
            result = self.test_entry(entry, k)
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
