from sklearn.ensemble import RandomForestRegressor
   
from sklearn.multioutput import RegressorChain
from sklearn.svm import SVR

import torch.nn as nn


class RandomForest():
    def __init__(self):
        self.model = RandomForestRegressor()

    def fit(self,x,y):
        self.model.fit(x,y)
        return self.model
    
    def predict(self,x):
        #Never called
        return self.model.predict(x)
    

class SupportVector():
    def __init__(self):
        self.model = RegressorChain(SVR(kernel="rbf"))

    def fit(self,x,y):
        self.model.fit(x,y)
        return self.model
    
    def predict(self,x):
        #Never called
        return self.model.predict(x)


class Model500GELU(nn.Module):
    def __init__(self, parameter_count=2, output_count=2):
        super(Model500GELU, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(parameter_count, 200),
            nn.ReLU(),
            nn.Linear(200, 300),
            nn.ReLU(),
            nn.Linear(300, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 300),
            nn.ReLU(),
            nn.Linear(300, 200),
            nn.ReLU(),
            nn.Linear(200, output_count)
        )

    def forward(self, x):
        return self.network(x)