import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats

def train_naive_bayes(training_data):

    sick_data = [data for data in training_data if data[10]==2.0]
    healthy_data = [data for data in training_data if data[10]==1.0]

    sick_data_features = seperate_features(sick_data)
    healthy_data_features = seperate_features(healthy_data)

    generate_pdfs(sick_data_features)

    
    


    
    #x = np.linspace(mean[0]-4*std[0],mean[0]+4*std[0],100)
    #plt.plot(x, pdfs[0].pdf(x), linewidth=2, color='r')
    #plt.axis([mean[0]-4*std[0],mean[0]+4*std[0], 0, 1.5])
    #plt.scatter(features_data[0],len(features_data[0])*[0], color='k')
    #plt.show()

    return 5

#transposing the training_data will create a seperate list for each feature    
def seperate_features(data):
    features = list(map(list, zip(*data)))

    del(features[-1])
    return features

def generate_pdfs(data_features):
    pdfs = []

    for feature in data_features:
        mean= np.mean(feature)
        std = np.std(feature)
        pdfs.append(stats.norm(mean, std))

    return pdfs
