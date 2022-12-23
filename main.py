from pre_process import *
from k_fold_cross_validation import *
from NaiveBayes import *
   
data = pre_process()

k = 5
training_folds, testing_folds = k_fold_cross_validation(k,data)

trained_naive_bayes = []

for i in range(k):
    trained_naive_bayes.append(NaiveBayes(training_folds[i]))

accuracy = [0]*k
sensitivity = [0]*k
specificity = [0]*k
geometric_mean = [0]*k

for i in range(k):
    print("-----Fold " + str(i+1) + "-----")
    accuracy[i], sensitivity[i], specificity[i], geometric_mean[i] = trained_naive_bayes[i].test_data_set(testing_folds[i])

print("-----Model Means-----")
mean_accuracy = sum(accuracy)/k
print("Mean Accuracy: " + str(mean_accuracy))
mean_sensitivity = sum(sensitivity)/k
print("Mean Sensitivity: " + str(mean_sensitivity))
mean_specificity = sum(specificity)/k
print("Mean Specificity: " + str(mean_specificity))
mean_geometric_mean = sum(geometric_mean)/k
print("Mean Geometric Mean: " + str(mean_geometric_mean))
