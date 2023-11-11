from factorized import *

df = data_preprocessing()
X_test, y_test, model = model_train(df, [RandomForestClassifier, 'over'])
# Save Model
script_path = os.path.dirname(os.path.abspath(__file__))
ml_project_path = os.path.dirname(script_path)
artifacts_path = os.path.join(ml_project_path, 'artifacts')
model_filename = 'model.pkl'
with open(os.path.join(artifacts_path, model_filename), 'wb') as model_file:
    pickle.dump(model, model_file)
# Save Scaling Operation
scaling_info = {
    'operation': 'divide_by_255',
    'scaling_factor': 255.0
}

scaling_filename = 'scaling_info.pkl'
with open(os.path.join(artifacts_path, scaling_filename), 'wb') as scaling_file:
    pickle.dump(scaling_info, scaling_file)
