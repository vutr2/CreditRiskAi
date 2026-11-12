from pathlib import Path
import pandas as pd
RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"


#Function for loading training set
def load_train(path = None):
    if path is None:
        path = RAW_DIR / 'cs-training.csv'

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}")
    df = pd.read_csv(path, index_col=0)
    df.index.name = "id"
    return df

#Fucntion for loading test set

def load_test(path = None):
    if path is None:
        path = RAW_DIR / 'cs-test.csv'


    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data not found: {path}")
    df = pd.read_csv(path, index_col=0)
    df.index.name = "id"
    df = df.drop(columns="SeriousDlqin2yrs")
    return df
