import csv
import scipy
import numpy as nps

def pre_process(filename='Indian Liver Patient Dataset (ILPD).csv'):

    data = []
        
    with open(filename) as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            data.append(row);
    file.close()

    features = len(data[0])
    data_min = [1000]*features
    data_max = [-1]*features
    total = [0]*features
    count = [0]*features
    
    for entry in data:
        if entry[1]=='Female':
            entry[1] = 1
        else:
            if entry[1]=='Male':
                entry[1] = 0
        for i in range(features):
            if entry[i]=='':
                continue;
                
            entry[i] = float(entry[i])
            total[i] += entry[i]
            count[i] += 1
    
            if data_min[i] > entry[i]:
                data_min[i] = entry[i]
            if data_max[i] < entry[i]:
                data_max[i] = entry[i]
                    
    for entry in data:
        for i in range(features):
            if entry[i]=='':
                entry[i]=1.0*total[i]/count[i]
                
    data_range=[]
    
    for i in range(features):
        data_range.append(data_max[i]-data_min[i])  

    for entry in data:
        for i in range(features-1):
            entry[i] = -1 +(((entry[i]-data_min[i])*(2))/data_range[i])
    return data

#transposing the training_data will create a separate list for each feature
def seperate_features(data):
    features = list(map(list, zip(*data)))

    del(features[-1])
    return features

def student_t_test(dataset):
    sick_data = [data for data in dataset if data[-1]==2.0]
    healthy_data = [data for data in dataset if data[-1]==1.0]

    sick_data_features = seperate_features(sick_data)
    healthy_data_features = seperate_features(healthy_data)

    test_results =[]
    
    for i in range(10):
        (t,p) = scipy.stats.ttest_ind(sick_data_features[i], healthy_data_features[i])
        test_results.append([i,abs(t),p])

    sorted_list = sorted(test_results, key=lambda x: x[2])
    columns_to_remove = []
    for i in range(5,10):
        columns_to_remove.append(sorted_list[i][0])
        
    dataset = [[row[i] for i in range(len(row)) if i not in columns_to_remove] for row in dataset]
    return dataset
