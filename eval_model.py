from sklearn.ensemble import RandomForestRegressor
   
from sklearn.multioutput import RegressorChain, MultiOutputRegressor
from sklearn.svm import SVR

from sklearn.neighbors import KNeighborsRegressor

import torch.nn as nn

from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math


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
        # self.model = RegressorChain(SVR(kernel="rbf"))
        self.model = MultiOutputRegressor(SVR(kernel="rbf"))

    def fit(self,x,y):
        self.model.fit(x,y)
        return self.model
    
    def predict(self,x):
        #Never called
        return self.model.predict(x)


class KNeighbors():
    def __init__(self):
        self.model = KNeighborsRegressor()()

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
    

class Transformer(nn.Module):

    def __init__(
        self,
        parameter_count=2, 
        output_count=2,
        dim_model=200,
        num_heads=2,
        num_encoder_layers=6,
        dim_hidden=200,
        dropout_p=0.1,
    ):
        super(Transformer, self).__init__()

        self.model_type = "Transformer"
        self.dim_model = dim_model

        self.embedding = nn.Linear(parameter_count, dim_model)

        encoder_layers = TransformerEncoderLayer(dim_model, num_heads, dim_hidden, dropout_p)
        self.transformer = TransformerEncoder(encoder_layers, num_encoder_layers)

        self.out = nn.Linear(dim_model, output_count)


    def forward(self, src, src_mask=None):

        src = self.embedding(src) * math.sqrt(self.dim_model)

        if src_mask is None:
            src_mask = nn.Transformer.generate_square_subsequent_mask(len(src))

        transformer_out = self.transformer(src, src_mask)
        out = self.out(transformer_out)

        return out