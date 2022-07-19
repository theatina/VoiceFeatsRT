import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import functions as funs

# %% load prepared data
data_path = f'..{os.sep}Data{os.sep}prepared_dataframe.pickle'
data_csv_path = f"..{os.sep}Data{os.sep}prepared_dataframe.csv" 
if not os.path.exists(data_csv_path):
    df = pd.read_pickle(data_path)
    df["mfcc_profile"] = [ mfcc_prof.tolist() for mfcc_prof in df["mfcc_profile"].values ]
    df.to_csv(f"{data_csv_path}", header=True, index=False)
else:
    df = pd.read_csv(data_csv_path)
    df["mfcc_profile"]= [ eval(mfcc_prof) for mfcc_prof in df["mfcc_profile"].values ]


# %% calm - angry

class0_calm = df[ df['emotion'] == "calm" ]
class1_angry = df[ df['emotion'] == "angry" ]

# %% isolate features and labels

title = "MFCC Features"
# title = "MFCC Features + Centroid + Bandwidth"
print(f"\n{title}")
subtitle = "Calm vs Angry"
# isolate features and labels
act01_features = np.vstack( class0_calm['mfcc_profile'].to_numpy() )
act02_features = np.vstack( class1_angry['mfcc_profile'].to_numpy() )
all_features = np.vstack((act01_features, act02_features))

act01_labels = 0*np.ones( ( act01_features.shape[0] , 1 ) )
act02_labels = 1*np.ones( ( act02_features.shape[0] , 1 ) )
all_labels = np.r_[ act01_labels , act02_labels ]

# %% train - test split
train_set , test_set = train_test_split( np.c_[ all_features , all_labels] , test_size=0.25 , random_state=99 )

train_input = train_set[:, :-1]
train_label = train_set[:, -1]
test_input = test_set[:, :-1]
test_label = test_set[:, -1]

# %% Classification algorithms

# Data Scaling
HistGB_scaler = StandardScaler()
SVM_scaler = MinMaxScaler()
RF_scaler = StandardScaler()
LR_scaler = MinMaxScaler()

# Classification algorithms and their parameters after manual fine-tuning and research
class_algo_dict={
    "HistGB_scaler":make_pipeline(HistGB_scaler, HistGradientBoostingClassifier()),
    "HistGB": HistGradientBoostingClassifier(),
    "SVM": make_pipeline(SVM_scaler, SVC(C=1.0, kernel="rbf")),
    "RF": make_pipeline(RF_scaler, RandomForestClassifier(n_estimators=150, warm_start=True)),
    "LR": LinearRegression(),
    "KNN": KNeighborsClassifier(n_neighbors=9, weights="distance"),
    "LogReg": LogisticRegression(C=1000.0, solver="liblinear", penalty="l2", max_iter=1000)
}

# algo="HistGB"
# algo="SVM"
# algo="RF"
# algo="LR"
# algo="KNN"
algo="LogReg"
model = class_algo_dict[algo]

model.fit( train_input , train_label )
# make predictions from training data
preds_binary = model.predict( test_input )
comparison_check = np.c_[ preds_binary , test_label ]
accuracy = np.sum( test_label == preds_binary ) / preds_binary.size

# %% cross validation - custom accuracy metric
my_scorer = make_scorer(funs.binary_accuracy, greater_is_better=True)

# %% cross validation
scores = cross_val_score( model, all_features, all_labels.ravel(), scoring=my_scorer, cv=10 )

funs.present_scores( scores , algorithm=algo )
# %% save model
filename = f'..{os.sep}Models{os.sep}{algo}_CalmAngry.model'
pickle.dump(model, open(filename, 'wb'))
