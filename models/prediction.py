import pickle
import pandas as pd
import pathlib


def predict_rf(data):
    with open(pathlib.Path(__file__).parent / "model_rf.pkl", "rb") as f:
        model = pickle.load(f)

    with open(pathlib.Path(__file__).parent / "train_columns.pkl", "rb") as f:
        train_columns = pickle.load(f)

    df_app = pd.DataFrame([data])
    df_app = pd.get_dummies(df_app)

    df_app = df_app.reindex(columns=train_columns, fill_value=0)
    print(model.predict(df_app)[0])
    return "Rejected" if model.predict(df_app)[0] else "Approved"
