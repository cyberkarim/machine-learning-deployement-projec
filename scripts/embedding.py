from matplotlib import pyplot as plt
import pickle
import numpy as np
import pandas as pd
import csv
import os
import cv2

script_path = os.path.dirname(os.path.abspath(__file__))
ml_project_path = os.path.dirname(script_path)
data_path = os.path.join(ml_project_path, 'data')

def embedding(img):
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,(8,8))
    img=img.reshape(-1) # flatten the matrix
    # print(type(img))
    return img

path_to_embedding_pickle=ml_project_path+rf"/artifacts/embedding.pkl"

with open(path_to_embedding_pickle, 'wb') as file_pickle:
    pickle.dump(embedding, file_pickle)


if __name__=="__main__":

    train_data_good = os.path.join(data_path, 'bottle/train/good')
    test_data_broken_large = os.path.join(data_path,'bottle/test/broken_large')
    test_data_broken_small = os.path.join(data_path,'bottle/test/broken_small')
    test_data_good = os.path.join(data_path, 'bottle/test/good')

    # Load data to data frame format from the database
    def load_images_from_folder(folders):
        images = []
        for folder in folders:
            for filename in os.listdir(folder):
                img = cv2.imread(os.path.join(folder, filename))
                if img is not None:
                    images.append(img)
        return images

    X_train = np.array(list(map(embedding,load_images_from_folder([train_data_good]))))
    X_test = np.array(list(map(embedding,load_images_from_folder([test_data_broken_large, test_data_broken_small,test_data_good]))))

    y_train = np.zeros(len(X_train))
    y_test = np.zeros(len(X_test))
    anomalous_indices = [i for i in range(42)] # indicate manually anomalies
    y_test[anomalous_indices] = 1
    X = np.append(X_train, X_test, axis = 0)
    y = np.append(y_train, y_test, axis = 0)

    df = pd.DataFrame(data=np.c_[X, y], columns=[f"pixel_{i+1}" for i in range(X.shape[1])] + ["target"])

    with open(data_path+rf"/ref_data.csv", 'w', newline='') as file:
        df.to_csv(file,index=False,sep=";")
