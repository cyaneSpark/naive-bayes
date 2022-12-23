import csv

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
