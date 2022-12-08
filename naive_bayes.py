import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats

def train_naive_bayes(training_data):

    sick_data = [data for data in training_data if data[10]==2.0]
    healthy_data = [data for data in training_data if data[10]==1.0]

    sick_data_features = seperate_features(sick_data)
    healthy_data_features = seperate_features(healthy_data)

    sick_pdfs = generate_pdfs(sick_data_features)
    healthy_pdfs = generate_pdfs(healthy_data_features)

    figure, axis = plt.subplots(1,len(healthy_pdfs))
    
    for i in range(len(sick_pdfs)):
        x1 = np.linspace(sick_pdfs[i].mean()-4*sick_pdfs[i].std(),sick_pdfs[i].mean()+4*sick_pdfs[i].std(),100)
        axis[i].plot(x1, sick_pdfs[i].pdf(x1), linewidth=2, color='r')
        
        x2 = np.linspace(healthy_pdfs[i].mean()-4*healthy_pdfs[i].std(),healthy_pdfs[i].mean()+4*healthy_pdfs[i].std(),100)
        axis[i].plot(x1, healthy_pdfs[i].pdf(x2), linewidth=1, color='b')

        axis[i].set_title("Feature " + str(i))
    

    plt.show()

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
        print(pdfs[-1].pdf(0))

    return pdfs
