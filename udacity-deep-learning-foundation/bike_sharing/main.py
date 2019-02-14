#! usr/bin/python 
# -*-encoding:utf-8 -*-
# import package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import unittest

from NeuralNetwork import NeuralNetwork
from TestMethods import TestMethods

def MSE(y, Y):
    return np.mean((y-Y)**2)

def process_data(rides):
    # data preprocess

    return train_features, train_targets, val_features, val_targets
    

if __name__ == '__main__':
    data_path = 'datasets/hour.csv'
    rides = pd.read_csv(data_path)
    print("########## load data success ##########")

    dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
    for each in dummy_fields:
        dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
        rides = pd.concat([rides, dummies], axis=1)

    fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 
                      'weekday', 'atemp', 'mnth', 'workingday', 'hr']
    data = rides.drop(fields_to_drop, axis=1)
    print("##########   process success   ##########")


    # show data
    rides[:24*10].plot(x='dteday', y='cnt')
    plt.show()



    quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
    # Store scalings in a dictionary so we can convert back later
    scaled_features = {}
    for each in quant_features:
        mean, std = data[each].mean(), data[each].std()
        scaled_features[each] = [mean, std]
        data.loc[:, each] = (data[each] - mean)/std


    # Save the last 21 days 
    test_data = data[-21*24:]
    data = data[:-21*24]

    # Separate the data into features and targets
    target_fields = ['cnt', 'casual', 'registered']
    features, targets = data.drop(target_fields, axis=1), data[target_fields]
    test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

    n_records = features.shape[0]
    split = np.random.choice(features.index, 
                             size=int(n_records*0.8), 
                             replace=False)
    train_features, train_targets = features.ix[split], targets.ix[split]
    val_features, val_targets = features.drop(split), targets.drop(split)

    ### Set the hyperparameters here ###
    epochs = 15000
    learning_rate = 0.008
    hidden_nodes = 25
    output_nodes = 1


    N_i = train_features.shape[1]
    network = NeuralNetwork(N_i, hidden_nodes, output_nodes, learning_rate)

    losses = {'train':[], 'validation':[]}
    for e in range(epochs):
        # Go through a random batch of 128 records from the training data set
        batch = np.random.choice(train_features.index, size=128)
        for record, target in zip(train_features.ix[batch].values, 
                                  train_targets.ix[batch]['cnt']):
            network.train(record, target)
            
        if e%(epochs/1000) == 0:
            # Calculate losses for the training and test sets
            train_loss = MSE(network.run(train_features), train_targets['cnt'].values)
            val_loss = MSE(network.run(val_features), val_targets['cnt'].values)
            losses['train'].append(train_loss)
            losses['validation'].append(val_loss)

        if e%(epochs/10) == 0:        
            # Print out the losses as the network is training
            print('Training loss: {:.4f}'.format(train_loss))
            print('Validation loss: {:.4f}'.format(val_loss))
      
    plt.plot(losses['train'], label='Training loss')
    plt.plot(losses['validation'], label='Validation loss')
    plt.legend() 
    plt.show()


    fig, ax = plt.subplots(figsize=(8,4))

    mean, std = scaled_features['cnt']
    predictions = network.run(test_features)*std + mean
    ax.plot(predictions[0], label='Prediction')
    ax.plot((test_targets['cnt']*std + mean).values, label='Data')
    ax.set_xlim(right=len(predictions))
    ax.legend()

    dates = pd.to_datetime(rides.ix[test_data.index]['dteday'])
    dates = dates.apply(lambda d: d.strftime('%b %d'))
    ax.set_xticks(np.arange(len(dates))[12::24])
    # _ = ax.set_xticklabels(dates[12::24], rotation=45)
    plt.show()



    

