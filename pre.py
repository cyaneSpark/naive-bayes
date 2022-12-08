import csv

def pre_process(filename='Indian Liver Patient Dataset (ILPD).csv'):

    data = []
        
    with open(filename) as file:
        csv_reader = csv.reader(file, delimiter=',')
        for row in csv_reader:
            data.append(row);

    file.close()

    categories = len(data[0])
    
    data_min = [1000]*categories
    data_max = [-1]*categories

    for entry in data:
        if entry[1]=='Female':
            entry[1] = 1
        else:
            if entry[1]=='Male':
                entry[1] = 0
        for i in range(categories):
            if entry[i]=='':
                entry[i] = 0
            entry[i] = float(entry[i])
            if data_min[i] > entry[i]:
                data_min[i] = entry[i]
            if data_max[i] < entry[i]:
                data_max[i] = entry[i]

    data_range=[]
    for i in range(categories):
        data_range.append(data_max[i]-data_min[i])  

    for entry in data:
        for i in range(categories-1):
            entry[i] = -1 +(((entry[i]-data_min[i])*(2))/data_range[i])

    return data;      
