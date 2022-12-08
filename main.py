from pre import *
from k_fold_cross_validation import *
from naive_bayes import *
from matplotlib import pyplot as plt
    
data = pre_process()
training_data, testing_data = k_fold_cross_validation(5,data)

for i in range(5):
    train_naive_bayes(training_data[i])
