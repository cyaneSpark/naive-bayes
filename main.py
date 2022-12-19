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

accuracy = [0]*5
sensitivity = [0]*5
specificity = [0]*5
geometric_mean = [0]*5

accuracy[0], sensitivity[0], specificity[0], geometric_mean[0] = trained_models[0].test_data_set(testing_data[0])
accuracy[1], sensitivity[1], specificity[1], geometric_mean[1] = trained_models[1].test_data_set(testing_data[1])
accuracy[2], sensitivity[2], specificity[2], geometric_mean[2] = trained_models[2].test_data_set(testing_data[2])
accuracy[3], sensitivity[3], specificity[3], geometric_mean[3] = trained_models[3].test_data_set(testing_data[3])
accuracy[4], sensitivity[4], specificity[4], geometric_mean[4] = trained_models[4].test_data_set(testing_data[4])

mean_accuracy = sum(accuracy)/5
mean_sensitivity = sum(sensitivity)/5
mean_specificity = sum(specificity)/5
mean_geometric_mean = sum(geometric_mean)/5

print(mean_accuracy)
print(mean_sensitivity)
print(mean_specificity)
print(mean_geometric_mean)
