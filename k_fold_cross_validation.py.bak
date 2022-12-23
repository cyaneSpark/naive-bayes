def k_fold_cross_validation(k, data):
    
    data_size = len(data)
    testing_size = int(data_size/k)
    training_size = data_size - testing_size

    testing_data=[]
    training_data=[]
    limit_low = 0
    limit_high = testing_size
    
    for i in range(k):
        testing_data.append(data[limit_low:limit_high])
        training_data.append(data[:limit_low] + data[limit_high:])

        limit_low = limit_high
        limit_high += testing_size
        if(limit_high>data_size):
            limit_high = data_size

    return training_data, testing_data
