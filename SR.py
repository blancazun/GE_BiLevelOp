import imp
import pandas as pd
import numpy as np

class SR:
    def __init__(self, file=None):
        info = pd.read_csv(file)
        self.data = info.values
        del info
    
    def evaluate(self, string):
        x = self.data
        if string == None:
            return np.inf
        
        try:
            with np.errstate(divide='ignore', invalid='ignore'):
                """
                y = eval(string)
                tt = np.mean(x[:,1])
                yt = np.mean(y)
                b = (np.sum((x[:,-1]-tt)*(y-yt)))/(np.sum((y-yt)**2))
                a = tt - b*yt
                result = 1/(1+(np.sqrt(np.sum((x[:,-1]-(a+b*y))**2))/x.shape[0]))
                """
                #result = np.sqrt(np.sum((x[:,-1]-eval(string))**2)/x.shape[0])
                result = np.sum((x[:,-1]-eval(string))**2)/x.shape[0]
            return result if not np.isnan(result) else np.inf
        except:
            return np.inf
