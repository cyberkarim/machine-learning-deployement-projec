from factorized import load, model_train
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pickle
import os


script_path = os.path.dirname(os.path.abspath(__file__))
ml_project_path = os.path.dirname(script_path)
data_path = os.path.join(ml_project_path, 'data')
artifacts_path = os.path.join(ml_project_path, 'artifacts')

df=load(data_path+rf"/ref_data.csv")

def normalize_image(df):
    df.iloc[:,:-1]=df.iloc[:,:-1].div(255)
    return df

df=normalize_image(df)
X_test, y_test, model = model_train(df, [RandomForestClassifier(n_estimators=5),"SMOTE"])


# Save Scaler
path_to_scaler_pickle=artifacts_path+rf"/scaler.pkl"

with open(path_to_scaler_pickle, 'wb') as scaler_file:
    pickle.dump(normalize_image, scaler_file)


# Save Model
path_to_model_pickle=artifacts_path+rf"/model.pkl"

with open(path_to_model_pickle, 'wb') as model_file:
    pickle.dump(model, model_file)

