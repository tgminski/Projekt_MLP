''' utilites file TG '''

#------------------------------------------------------------
# import libraries
#------------------------------------------------------------

import numpy as np
import pandas as pd




#------------------------------------------------------------
# clases
#------------------------------------------------------------

#------------------------------------------------------------
# functions
#------------------------------------------------------------

def load_data_from_csv(file):
    data = pd.read_csv(file)
    return data

def save_df_data_to_csv(data, file):
    data.to_csv(file, index=False)
    return []


