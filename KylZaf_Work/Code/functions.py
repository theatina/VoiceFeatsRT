import numpy as np
import os
import pandas as pd
import random
import shutil
import csv

from sklearn.metrics import classification_report

def present_scores( s , algorithm='method' ):
        print(30*'-')
        print( algorithm + ' accuracy in 10-fold cross validation:' )
        print('mean: ' + str( np.mean(s) ))
        print('std: ' + str( np.std(s) ))
        print('median: ' + str( np.median(s) ))

def binary_accuracy( y_true , y_pred ):
        bin_pred = np.array( y_pred >= 0.5 ).astype(int)
        return np.sum( y_true == bin_pred ) / y_true.size


def move_N_files_forEval(N):
        src = f"..{os.sep}Data{os.sep}AudioFiles"
        dst = os.path.join(f"..{os.sep}Data{os.sep}AudioFiles", "Temp_Evaluation")

        # count files in audio files directory
        file_list = [f for f in os.listdir(src) if ".wav" in f and os.path.isfile(  os.path.join(src,f)  )]
        tot_files = len(file_list)

        if tot_files < N:
                ind = [ i for i in range(tot_files)]
        else:
                # select N random files
                ind = random.sample(range(tot_files), N+1)
        
        for f in [file_list[i] for i in ind]:
                path_src = os.path.join(src,f)
                path_dest = os.path.join(dst,f)
                shutil.move(path_src, path_dest)


def moveBack_files_forEval():
        dst = f"..{os.sep}Data{os.sep}AudioFiles"
        src = os.path.join(f"..{os.sep}Data{os.sep}AudioFiles", "Temp_Evaluation")
        
        # files to be moved back to audio files directory
        file_list = [f for f in os.listdir(src) if ".wav" in f and os.path.isfile(  os.path.join(src,f)  )]
  
        for f in file_list:
                path_src = os.path.join(src,f)
                path_dest = os.path.join(dst,f)
                shutil.move(path_src, path_dest)
        

def create_csv_scores():
        df = pd.DataFrame([],columns=["Prediction", "Label", "Filename"])
        df.to_csv(f"..{os.sep}Results{os.sep}Q2{os.sep}Scores.csv", index=False, header=True)


def evaluation(n_parts, preds, true, emotion, filename, scores_csv=f"..{os.sep}Results{os.sep}Q2{os.sep}Scores.csv"):
        logfiledir = f"..{os.sep}Results{os.sep}Q2{os.sep}Logfiles"
        with open(scores_csv, "a", newline="") as csv_scores:
                csv_writer = csv.writer(csv_scores)
                for pred,label in zip(preds,true):
                        csv_writer.writerow([pred,label,filename], )

        # logfilepath = os.path.join(logfiledir,filename.split(".")[-2]+".txt") 
        logfilepath = os.path.join(logfiledir,"logs.txt") 
        with open(logfilepath, "a", encoding="utf-8") as writer:
                part_acc = sum( [ 1 if l1==l2 else 0 for l1,l2 in zip(preds,true) ] ) / n_parts
                logfile_str = f"> {filename} \nEmotion: {emotion} | Window accuracy: {part_acc*100:.2f}%\n\n\n"
                writer.write(logfile_str)


def classif_report():
        scores_csv=f"..{os.sep}Results{os.sep}Q2{os.sep}Scores.csv"
        scores_df=pd.read_csv(scores_csv)
        
        predictions=scores_df["Prediction"].values
        labels=scores_df["Label"].values
        file_list=set(scores_df["Filename"].values)
        counter= len(file_list)

        logfiledir = f"..{os.sep}Results{os.sep}Q2"
        logfilepath = os.path.join(logfiledir,f"classificationReport_{counter}Files.txt") 
        with open(logfilepath, "w", encoding="utf-8") as writer:
                
                logfile_str = f"File Number: {counter}\n\n" + classification_report(labels,predictions, target_names=["Calm", "Angry"], zero_division=1) + "\n\n"
                writer.write(logfile_str)



# create_csv_scores()
