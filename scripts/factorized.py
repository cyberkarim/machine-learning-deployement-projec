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
import xgboost as xgb
import pickle
import os
import cv2

def data_preprocessing():
    # Set the paths to your dataset
    script_path = os.path.dirname(os.path.abspath(__file__))
    ml_project_path = os.path.dirname(script_path)
    data_path = os.path.join(ml_project_path, 'data')
    train_data_path = os.path.join(data_path, 'bottle/train/good')
    test_data_path_1 = os.path.join(data_path,'bottle/test/broken_large')
    test_data_path_2 = os.path.join(data_path,'bottle/test/contamination')
    test_data_path_3 = os.path.join(data_path,'bottle/test/broken_small')
    test_data_path_4 = os.path.join(data_path, 'bottle/test/good')
    # Load and preprocess your data
    def load_images_from_folder(folders):
        images = []
        for folder in folders:
            for filename in os.listdir(folder):
                img = cv2.imread(os.path.join(folder, filename))
                if img is not None:
                    images.append(img)
        return images

    X_train = np.array(load_images_from_folder([train_data_path]))
    X_test = np.array(load_images_from_folder([test_data_path_1, test_data_path_2, test_data_path_3, test_data_path_4]))
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    y_train = np.ones(len(X_train))
    y_test = np.ones(len(X_test))
    anomalous_indices = [i for i in range(64)]
    y_test[anomalous_indices] = -1  
    X_test = X_test.reshape(X_test.shape[0], -1)
    X = np.append(X_train, X_test, axis = 0)
    y = np.append(y_train, y_test, axis = 0)
    df = pd.DataFrame(data=np.c_[X, y], columns=[f"feature_{i}" for i in range(X.shape[1])] + ['target'])
    return df
def load(file_name):
    # file_name: sting of the file name to be opened in the working directory
    data=pd.read_csv(file_name)
    return data

def ordinal_encoder(data,categorial_variables):
    # data: data frame with some categorial variables to be encoded with an ordinal encoder

    for cv in categorial_variables:
        df = pd.DataFrame({cv: data[cv]}) # suppose it is the last column
        encoder = OrdinalEncoder(categories=[data[cv].unique()])
        data[cv]=encoder.fit_transform(df[[cv]])
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
        model_filename = os.path.join('artifacts', 'scaler.pkl') 
    return data

def filtration(data,threshold):
    # threshold: instance of classes whose population is lower than the threshold are removed
    class_count=data.iloc[:,-1].value_counts()
    remove_class=class_count[class_count<threshold].index
    data = data[~data.iloc[:,-1].isin(remove_class)]
    print(data.iloc[:,-1].value_counts())
    return(data)

def model_train(data,model):
    # model: list containing the model and the sampling option: None:"", Oversampling:"over" and Undersampling:"under"

   
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
    # plt.show()

if __name__=="__main__":

    # Loading the file 
    file_name="creditcard.csv"
    data=load(file_name)

    if file_name=="creditcard.csv":
        data=data.drop("Time",axis=1)

    # Encoding categorial variables
    # categorial_variables=["protocol_type","service","flag","label"] # for KDDCup99
    categorial_variables=["Class"] # for creditcard   
    data=ordinal_encoder(data,categorial_variables)

    # Normalize the data
    data=normalize(data,0.001)

    # Set a minimum number of instance into a class or remove it
    # data=filtration(data,10)

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
   