import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats

def train_naive_bayes(features_data):
    
    mean = [0]* len(features_data)
    std = [0]* len(features_data)
    curves = [0]* len(features_data)
    for i in range(len(features_data)):
        mean[i]= np.mean(features_data[i])
        std[i] = np.std(features_data[i])
        curves[i] = stats.norm(mean[i], std[i])
    #plt.plot(curves[0])
    #count, bins, ignored = plt.hist(curves[0], 30, density=True)
        #plt.plot(bins, 1/(std[i]*np.sqrt(2*np.pi))*np.exp(-(bins-mean[i])**2/(2*std[i]**2)), linewidth=2, color='r')

    print(mean[0])
    print(std[0])
    print(features_data[0])

    x = np.linspace(mean[0]-4*std[0],mean[0]+4*std[0],100)
    plt.plot(x, curves[0].pdf(x), linewidth=2, color='r')
    #plt.hist(features_data[0],100)
    plt.axis([mean[0]-4*std[0],mean[0]+4*std[0], 0, 1.5])
    plt.scatter(features_data[0],467*[0], color='k', )
    plt.show()

    return 5
        
