import numpy as np
from matplotlib import pyplot as plt
import scipy.stats as stats


class TrainedModel:
    def __init__(self, training_data):
        self.data = training_data

        self.sick_data = [data for data in training_data if data[10]==2.0]
        self.healthy_data = [data for data in training_data if data[10]==1.0]

        self.sick_data_features = self.seperate_features(self.sick_data)
        self.healthy_data_features = self.seperate_features(self.healthy_data)

        self.sick_pdfs = self.train_model(self.sick_data_features)
        self.healthy_pdfs = self.train_model(self.healthy_data_features)

        print("done")

    #transposing the training_data will create a seperate list for each feature
    def seperate_features(self, data):
        features = list(map(list, zip(*data)))

        del(features[-1])
        return features


    def train_model(self, data_features):
        pdfs = []
        for feature in data_features:
            mean= np.mean(feature)
            std = np.std(feature)
            pdfs.append(stats.norm(mean, std))

        return pdfs

