from pre import *
from k_fold_cross_validation import *
from naive_bayes import *
from TrainedModel import *
from matplotlib import pyplot as plt
    
data = pre_process()
training_data, testing_data = k_fold_cross_validation(5,data)

trained_models = []

for i in range(5):
    trained_models.append(TrainedModel(training_data[i]))


