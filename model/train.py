from  bundle import build_bundle, save_bundle
import pickle
import pandas as pd

def get_pipeline(path):
    with open(path, 'rb') as file:
        return pickle.load(file)

def load_data(path):
    df = pd.read_csv(path)
    X = df.drop(columns=['claim_number', 'gender', 'zip_code','fraud'])
    y = df['fraud']
    return X, y

def run_training(pipeline_path='model/artifacts/model_v1.sav', 
                 data_path='data/train_2025.csv'):
    pipeline = get_pipeline(pipeline_path)
    X, y = load_data(data_path)
    pipeline.fit(X, y)
    with open(pipeline_path, 'wb') as file:
        pickle.dump(pipeline, file)
    bundle = build_bundle(pipeline, X)
    save_bundle(bundle, 'model/artifacts/bundle_v1.sav')

if __name__ == "__main__":
    run_training()
