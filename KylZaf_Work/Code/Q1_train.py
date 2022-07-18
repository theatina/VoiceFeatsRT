import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

from sklearn.model_selection import train_test_split

# %% load prepared data
df = pd.read_pickle(f'..{os.sep}Data{os.sep}prepared_dataframe.pickle')

# %% male - female

act01 = df[ df['female'] == True ]
act02 = df[ df['female'] == False ]

# %% isolate features and labels

act01_features = np.vstack( act01['mfcc_profile'].to_numpy() )
act02_features = np.vstack( act02['mfcc_profile'].to_numpy() )
all_features = np.vstack((act01_features, act02_features))

act01_labels = 0*np.ones( ( act01_features.shape[0] , 1 ) )
act02_labels = 1*np.ones( ( act02_features.shape[0] , 1 ) )
all_labels = np.r_[ act01_labels , act02_labels ]

# %% train - test split

train_set , test_set = train_test_split( np.c_[ all_features , all_labels] , test_size=0.2 , random_state=42 )

train_input = train_set[:, :-1]
train_label = train_set[:, -1]
test_input = test_set[:, :-1]
test_label = test_set[:, -1]

# %% linear regression

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit( train_input , train_label )

# %% save model
filename = 'lin_reg_gender.sav'
pickle.dump(lin_reg, open(filename, 'wb'))