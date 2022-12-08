from pre import *
from k_fold_cross_validation import *
from naive_bayes import *
    
data = pre_process()
training_data, testing_data = k_fold_cross_validation(5,data)

train_naive_bayes(training_data[0])
