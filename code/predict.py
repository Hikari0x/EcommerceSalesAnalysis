import joblib
from pip._internal.req.req_file import preprocess


def load_bundle(path):
    return joblib.load(path)

def predict_new(df_new, bundle):
    model = bundle['model']
    scaler = bundle['scaler']
    encoders = bundle['encoders']

    # 和训练时一模一样的处理流程
    df_proc = preprocess(df_new, scaler, encoders)

    return model.predict(df_proc)