from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import outer
import tensorflow_decision_forests as tfdf
import pandas as pd

class Net(nn.Module):
    '''
    Regression Task
    '''
    def __init__(self, num_features=227):
        super(Net, self).__init__()

        #self.dropout = nn.Dropout(0.45,  training=self.training)
        #Input (Layer 1)
        self.fc1 = nn.Linear(num_features, 64)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        
        #Hidden (Layer 2)
        self.fc2 = nn.Linear(64, 256)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)
        
        #Hidden (Layer 3)
        self.fc3 = nn.Linear(256,512)
        nn.init.xavier_uniform_(self.fc3.weight)
        self.fc3.bias.data.fill_(0.01)
       
        #Hidden (Layer 4)
        self.fc4 = nn.Linear(512,64)
        nn.init.xavier_uniform_(self.fc4.weight)
        self.fc4.bias.data.fill_(0.01)
        
        #Output (Layer 5)
        self.fc5 = nn.Linear(64, 1)
        nn.init.xavier_uniform_(self.fc5.weight)
        self.fc5.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x)
        x = self.fc2(x)
        x = F.leaky_relu(x)
        x = self.fc3(x)
        x = F.leaky_relu(x)
        x = self.fc4(x)
        x = F.leaky_relu(x)
        x = self.fc5(x)
        output = x
        return output


class Net2(nn.Module):
    '''
    Classification Task
    '''
    def __init__(self, num_features=32, num_classes=24):
        super(Net2, self).__init__()

        #self.dropout = nn.Dropout(0.45, training=self.training)
        #Input (Layer 1)
        self.fc1 = nn.Linear(num_features, 64)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc1.bias.data.fill_(0.01)
        
        #Hidden (Layer 2)
        self.fc2 = nn.Linear(64, 128)
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc2.bias.data.fill_(0.01)
        
        #Hidden (Layer 3)
        self.fc3 = nn.Linear(128,256)
        nn.init.xavier_uniform_(self.fc3.weight)
        self.fc3.bias.data.fill_(0.01)
        
        #Hidden (Layer 4)
        self.fc4 = nn.Linear(256,64)
        nn.init.xavier_uniform_(self.fc4.weight)
        self.fc4.bias.data.fill_(0.01)
        
        #Output (Layer 5)
        self.fc5 = nn.Linear(64, num_classes)
        nn.init.xavier_uniform_(self.fc5.weight)
        self.fc5.bias.data.fill_(0.01)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        # using nn.CrossEntropyLoss will take care of this
        # output = F.log_softmax(x, dim=1)
        output = x
        return output


# Generalized form of Classifier ----------------------------------------------

def get_fforward_block(input_dim:int, output_dim:int):
    '''
    FeedForward block
    Parameters:
        input_dim: the dimension of the input vector, a scalar
        output_dim: the dimension of the output vector, a scalar
    Returns:
        a neural network layer with a linear transformation
        followed by a LeakyReLU activation function.
    '''
    return nn.Sequential(
        nn.Linear(input_dim, output_dim),
        nn.BatchNorm1d(output_dim),
        nn.LeakyReLU(negative_slope=0.2, inplace=True)
    )


class Classifier(nn.Module):
    '''
    Classifier Class
    values:
        input_dim: dimension of input, a scalar
        hidden_dim: dimension of hidden layer, a scalar
    '''
    def __init__(self, input_dim:int, hidden_dim:int, output_dim:int):
        super(Classifier, self).__init__()
        self.disc = nn.Sequential(
            get_fforward_block(input_dim, hidden_dim),
            get_fforward_block(hidden_dim, hidden_dim * 2),
            get_fforward_block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, data):
        '''
        This function completes the forward pass for the discriminator: 
        Given an input tensor, returns a 6-dimensions tensor depicting the classes.
        '''
        return self.disc(data)

    def get_disc(self):
        '''
        Returns: the sequential model
        '''
        return self.disc


class Classifier2(nn.Module):
    '''
    Classifier Class
    values:
        input_dim: dimension of input, a scalar
        hidden_dim: dimension of hidden layer, a scalar
    '''
    def __init__(self, layers_dim:List[int]):
        super(Classifier2, self).__init__()

        layers = [get_fforward_block(layers_dim[i-1], layers_dim[i])
                  if i < len(layers_dim) - 1
                  else nn.linear(layers_dim[i-1], layers_dim[i])
                  for i in range(1, len(layers_dim)-1)
                  ]

        self.disc = nn.Sequential(
            *layers
        )

    def forward(self, data):
        '''
        This function completes the forward pass for the discriminator: 
        Given an input tensor, returns a class-dimensions tensor depicting the classes.
        '''
        return self.disc(data)

    def get_disc(self):
        '''
        Returns: the equational model
        '''
        return self.disc


def random_forests(train_path, test_path, save_model_path):
    
    # load dataset into pandas dataframe
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # convert the dataset into a TensorFlow dataset
    train_ds = tfdf.keras.pd_dataframe_to_tf_dataset(train_df, label="my_label")
    test_ds = tfdf.keras.pd_dataframe_to_tf_dataset(test_df, label="my_label")

    # train a Random Forest model
    model = tfdf.keras.RandomForestModel()
    model.fit(train_ds)

    # summary of the model structure
    model.summary()

    # evaluate the model
    model.evaluate(test_ds)

    # export the model to a saved model
    model.save(save_model_path)

