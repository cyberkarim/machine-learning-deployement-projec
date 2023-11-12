# import useful librairies
from sklearn.preprocessing import OrdinalEncoder,StandardScaler,MinMaxScaler
from sklearn.naive_bayes import GaussianNB

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from sklearn.metrics import precision_recall_curve, balanced_accuracy_score

from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import os
import cv2

def load(file_name):
    # file_name: sting of the file name to be opened in the working directory
    data=pd.read_csv(file_name,sep=";")
    # print(data.shape)
    return data

def normalize(data,threshold):
    # threshold: percentage of original data with different values 
    # Useful to evaluate if it is a continuous or categorial variable
    unique_value_threshold=int(len(data)*threshold)
    for columns_name in data.columns:
        column=data[columns_name]
        df= pd.DataFrame({columns_name: column})
        if column.nunique()>unique_value_threshold:        
            scaler = StandardScaler()
            column=scaler.fit_transform(df[[columns_name]])
            # print("normalize a continious variable")
        else:
            scaler = scaler = MinMaxScaler()
            column=scaler.fit_transform(df[[columns_name]])
            # print("normalize a categorial variable") 
    return data


def model_train(data,model):
    # model: list containing the model and the sampling option: None:"", Oversampling:"SMOTE" and Undersampling:"TomekLinks"

   
    y=data.iloc[:,-1]
    X=data.iloc[:, :-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,stratify=y,random_state=26)


    if model[1]=="":
        model[0].fit(X_train,y_train)
    elif model[1]=="TomekLinks":
        tl=TomekLinks(sampling_strategy="auto")
        X_resampled, y_resampled = tl.fit_resample(X_train, y_train)
        model[0].fit(X_resampled,y_resampled)
    elif model[1]=="SMOTE":
        smote=SMOTE(sampling_strategy="auto")
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        model[0].fit(X_resampled,y_resampled)
    return X_test,y_test,model

def evaluate(data,models_dict):
    # models_dict: dictionnary of models whose values are 2 element lists with the model
    # and the sampling option

    class_counts=data.iloc[:,-1].value_counts()
    positive_rate=class_counts[1]/(class_counts[0]+class_counts[1])

    for model_name,model in models_dict.items():
        X_test,y_test,model=model_train(data,model)
        if model_name=="isolation_forest":
            y_score=model[0].decision_function(X_test)
            y_prob=1-(1/(1 + np.exp(-y_score)))

            # cross validation on test set to determine optimal threshold
            optimal_threshol=0
            optimal_balanced_accuracy=0
            for t in np.linspace(0,1,50):
                y_pred=(y_prob>t).astype(int) 
                balanced_accuracy=balanced_accuracy_score(y_test,y_pred)
                if balanced_accuracy>optimal_balanced_accuracy:
                    optimal_balanced_accuracy=balanced_accuracy
                    optimal_threshol=t
            y_pred=(y_prob>optimal_threshol).astype(int)

        else :
            y_prob = model[0].predict_proba(X_test)[:, 1]
            y_pred= model[0].predict(X_test)

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        balanced_accuracy=balanced_accuracy_score(y_test,y_pred)

        print(f"Balanced accuracy de {model_name}: {balanced_accuracy:.2f} \n")
        plt.plot(recall, precision, label=f"{model_name}")
    
    
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall")
    plt.legend(loc="upper right")
    plt.show()

# The following part is used to estimate the best model on our data  
"""

if __name__=="__main__":

    script_path = os.path.dirname(os.path.abspath(__file__))
    ml_project_path = os.path.dirname(script_path)
    data_path = os.path.join(ml_project_path, 'data')

    # Loading the file 
    df=load(data_path+rf"/ref_data.csv")

    # Normalize the data
    data=normalize(df,0.001)

    # Contamination estimation for isolation forest and local factor outlier
    class_count=data.iloc[:,-1].value_counts()
    contamination=class_count[1]/(class_count[0]+class_count[1])

    models_dict={
        "gnb":[GaussianNB(),""],
        "gnb_SMOTE":[GaussianNB(),"SMOTE"],
        "gnb_TomekLinks":[GaussianNB(),"TomekLinks"],
        "Random_Forest":[RandomForestClassifier(n_estimators=5),"SMOTE"],
        "isolation_forest":[IsolationForest(contamination=contamination),""],
    }

    # Evaluate different models to select the best one 
    evaluate(data,models_dict)
"""