from pre import *
from k_fold_cross_validation import *
from naive_bayes import *
    
data = pre_process()
training_data, testing_data = k_fold_cross_validation(5,data)

features = []

#transposing the training_data will create a seperate list for each feature
for data in training_data:
    features.append(list(map(list, zip(*data))))

train_naive_bayes(features[0])
