import numpy as np
from pre_process import *
from k_fold_cross_validation import *
from NaiveBayes import *
from KNearestNeighbors import *
from SupportVectorMachine import *

def print_pretty_model_metrics(accuracy, sensitivity, specificity, geometric_mean, k):
    mean_accuracy = sum(accuracy)/k
    mean_sensitivity = sum(sensitivity)/k
    mean_specificity = sum(specificity)/k
    mean_geometric_mean = sum(geometric_mean)/k

    print("--------------------------------------------------------------------------------------------")
    print("       |      Accuracy      |     Sensitivity    |     Specificity    |   Geometric Mean   |")
    print("--------------------------------------------------------------------------------------------")
    print("FOLD 1 | " +'{0:.16f}'.format(accuracy[0]) +" | "+'{0:.16f}'.format(sensitivity[0])+" | "+'{0:.16f}'.format(specificity[0])+" | "+'{0:.16f}'.format(geometric_mean[0])+" |")
    print("FOLD 2 | " +'{0:.16f}'.format(accuracy[1]) +" | "+'{0:.16f}'.format(sensitivity[1])+" | "+'{0:.16f}'.format(specificity[1])+" | "+'{0:.16f}'.format(geometric_mean[1])+" |")
    print("FOLD 3 | " +'{0:.16f}'.format(accuracy[2]) +" | "+'{0:.16f}'.format(sensitivity[2])+" | "+'{0:.16f}'.format(specificity[2])+" | "+'{0:.16f}'.format(geometric_mean[2])+" |")
    print("FOLD 4 | " +'{0:.16f}'.format(accuracy[3]) +" | "+'{0:.16f}'.format(sensitivity[3])+" | "+'{0:.16f}'.format(specificity[3])+" | "+'{0:.16f}'.format(geometric_mean[3])+" |")
    print("FOLD 5 | " +'{0:.16f}'.format(accuracy[4]) +" | "+'{0:.16f}'.format(sensitivity[4])+" | "+'{0:.16f}'.format(specificity[4])+" | "+'{0:.16f}'.format(geometric_mean[4])+" |")
    print("--------------------------------------------------------------------------------------------")
    print("MEAN   | " +'{0:.16f}'.format(mean_accuracy) +" | "+'{0:.16f}'.format(mean_sensitivity)+" | "+'{0:.16f}'.format(mean_specificity)+" | "+'{0:.16f}'.format(mean_geometric_mean)+" |")
    print("--------------------------------------------------------------------------------------------")
    print("!!----!!----!!----!!----!!----!!----!!----!!----!!----!!----!!----!!----!!----!!----!!----!!")

    print()

def print_pretty_model_means_only(accuracy, sensitivity, specificity, geometric_mean, k, c):
    mean_accuracy = sum(accuracy)/k
    mean_sensitivity = sum(sensitivity)/k
    mean_specificity = sum(specificity)/k
    mean_geometric_mean = sum(geometric_mean)/k

    print("   "+"%0{}d".format(3)%c+"  | " +'{0:.16f}'.format(mean_accuracy) +" | "+'{0:.16f}'.format(mean_sensitivity)+" | "+'{0:.16f}'.format(mean_specificity)+" | "+'{0:.16f}'.format(mean_geometric_mean)+" |")

def print_pretty_model_means_only_float(accuracy, sensitivity, specificity, geometric_mean, k, gamma):
    mean_accuracy = sum(accuracy)/k
    mean_sensitivity = sum(sensitivity)/k
    mean_specificity = sum(specificity)/k
    mean_geometric_mean = sum(geometric_mean)/k

    print("   "+'{0:.1f}'.format(gamma)+"  | " +'{0:.16f}'.format(mean_accuracy) +" | "+'{0:.16f}'.format(mean_sensitivity)+" | "+'{0:.16f}'.format(mean_specificity)+" | "+'{0:.16f}'.format(mean_geometric_mean)+" |")
    
def naive_bayes():
    trained_naive_bayes = []

    for i in range(k):
        trained_naive_bayes.append(NaiveBayes(training_folds[i]))

    accuracy = [0]*k
    sensitivity = [0]*k
    specificity = [0]*k
    geometric_mean = [0]*k

    for i in range(k):
        accuracy[i], sensitivity[i], specificity[i], geometric_mean[i] = trained_naive_bayes[i].test_data_set(testing_folds[i])

    print("!!-----!!----!!----!!----!!----!!        NAIVE BAYES       !!----!!----!!----!!----!!-----!!")
    print_pretty_model_metrics(accuracy, sensitivity, specificity, geometric_mean, k)


def svm():
    print("!!-----!!----!!----!!----!!----!!  SUPPORT VECTOR MACHINE   !!----!!----!!----!!----!!-----!!")
    geometric_means = []
    specificities = []
    sensitivities = []
    accuracies = []
    print("!!-----!!----!!----!!----!!----!!          LINEAR           !!----!!----!!----!!----!!-----!!")
    print("--------------------------------------------------------------------------------------------")
    print("    C   |      Accuracy      |     Sensitivity    |     Specificity    |   Geometric Mean   |")
    print("--------------------------------------------------------------------------------------------")
    for c in range(1,201,5):
        svm = SupportVectorMachine(c)
        accuracy = [0]*k
        sensitivity = [0]*k
        specificity = [0]*k
        geometric_mean = [0]*k
        for i in range(k):
            svm.train_model(training_folds[i])
            accuracy[i], sensitivity[i], specificity[i], geometric_mean[i] = svm.test_data_set(testing_folds[i])

        accuracies.append((sum(accuracy)/k,c))
        sensitivities.append((sum(sensitivity)/k,c))
        specificities.append((sum(specificity)/k,c))
        geometric_means.append((sum(geometric_mean)/k,c))
        print_pretty_model_means_only(accuracy, sensitivity, specificity, geometric_mean, k, c)

    print("--------------------------------------------------------------------------------------------")
    print("!!-----!!----!!----!!----!!----!!----!!----!!----!!----!!----!!----!!----!!----!!----!!-----!!")
    print()

    accuracies.sort(reverse=True)
    print("Based on the heighest Accuracy Score("+str(accuracies[0][0])+") the best value for C is: "+str(accuracies[0][1]))
    sensitivities.sort(reverse=True)
    print("Based on the heighest Sensitivity Score("+str(sensitivities[0][0])+") the best value for C is: "+str(sensitivities[0][1]))
    specificities.sort(reverse=True)
    print("Based on the heighest Specificity Score("+str(specificities[0][0])+") the best value for C is: "+str(specificities[0][1]))
    geometric_means.sort(reverse=True)
    print("Based on the heighest Geometric Mean Score("+str(geometric_means[0][0])+") the best value for C is: "+str(geometric_means[0][1]))
    print()

    C = geometric_means[0][1]

    geometric_means = []
    specificities = []
    sensitivities = []
    accuracies = []
    print("!!-----!!----!!----!!----!!----!!          RBFSVM           !!----!!----!!----!!----!!-----!!")
    print("--------------------------------------------------------------------------------------------")
    print("    γ   |      Accuracy      |     Sensitivity    |     Specificity    |   Geometric Mean   |")
    print("--------------------------------------------------------------------------------------------")
    for gamma in np.arange(0,10,0.5):
        svm = SupportVectorMachine(C, gamma)
        accuracy = [0]*k
        sensitivity = [0]*k
        specificity = [0]*k
        geometric_mean = [0]*k
        for i in range(k):
            svm.train_model(training_folds[i])
            accuracy[i], sensitivity[i], specificity[i], geometric_mean[i] = svm.test_data_set(testing_folds[i])

        accuracies.append((sum(accuracy)/k,gamma))
        sensitivities.append((sum(sensitivity)/k,gamma))
        specificities.append((sum(specificity)/k,gamma))
        geometric_means.append((sum(geometric_mean)/k,gamma))
        print_pretty_model_means_only_float(accuracy, sensitivity, specificity, geometric_mean, k, gamma)

    print("--------------------------------------------------------------------------------------------")
    print("!!-----!!----!!----!!----!!----!!----!!----!!----!!----!!----!!----!!----!!----!!----!!-----!!")
    print()

    accuracies.sort(reverse=True)
    print("Based on the heighest Accuracy Score("+str(accuracies[0][0])+") the best value for γ is: "+str(accuracies[0][1]))
    sensitivities.sort(reverse=True)
    print("Based on the heighest Sensitivity Score("+str(sensitivities[0][0])+") the best value for γ is: "+str(sensitivities[0][1]))
    specificities.sort(reverse=True)
    print("Based on the heighest Specificity Score("+str(specificities[0][0])+") the best value for γ is: "+str(specificities[0][1]))
    geometric_means.sort(reverse=True)
    print("Based on the heighest Geometric Mean Score("+str(geometric_means[0][0])+") the best value for γ is: "+str(geometric_means[0][1]))
    print()

def knn():
    print("!!-----!!----!!----!!----!!----!!    K-NEAREST NEIGHBORS   !!----!!----!!----!!----!!-----!!")
    geometric_means = []
    specificities = []
    sensitivities = []
    accuracies = []
    for nnk in range(3,16,2): #only odds to avoid ties
        accuracy = [0]*k
        sensitivity = [0]*k
        specificity = [0]*k
        geometric_mean = [0]*k
        print("             ---------------------         k = " + str(nnk)+"     -------------------                ")
        for i in range(k):
            trained_k_nearest = KNearestNeighbors(training_folds[i])
            accuracy[i], sensitivity[i], specificity[i], geometric_mean[i] = trained_k_nearest.test_data_set(testing_folds[i],nnk)

        accuracies.append((sum(accuracy)/k,nnk))
        sensitivities.append((sum(sensitivity)/k,nnk))
        specificities.append((sum(specificity)/k,nnk))
        geometric_means.append((sum(geometric_mean)/k,nnk))
        print_pretty_model_metrics(accuracy, sensitivity, specificity, geometric_mean, k)

    accuracies.sort(reverse=True)
    print("Based on the heighest Accuracy Score("+str(accuracies[0][0])+") the best value for k is: "+str(accuracies[0][1]))
    sensitivities.sort(reverse=True)
    print("Based on the heighest Sensitivity Score("+str(sensitivities[0][0])+") the best value for k is: "+str(sensitivities[0][1]))
    specificities.sort(reverse=True)
    print("Based on the heighest Specificity Score("+str(specificities[0][0])+") the best value for k is: "+str(specificities[0][1]))
    geometric_means.sort(reverse=True)
    print("Based on the heighest Geometric Mean Score("+str(geometric_means[0][0])+") the best value for k is: "+str(geometric_means[0][1]))    
    
# PRE RPOCESS

data = pre_process()

# 5-FOLD CROSS VALIDATION

k = 5
training_folds, testing_folds = k_fold_cross_validation(k,data)

# NAIVE BAYES

naive_bayes()

# SUPPORT VECTOR MACHINE

svm()

# K NEAREST NEIGHBORS

knn()

