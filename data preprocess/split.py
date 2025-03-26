import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.io import arff
import arff as liac_arff

data, meta = arff.loadarff('data.arff')
df = pd.DataFrame(data)

def decode_bytes(df):
    for col in df.select_dtypes([object]):
        df[col] = df[col].apply(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    return df

df = decode_bytes(df)  

X = df.iloc[:, :-1]  
y = df.iloc[:, -1] 

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

train_set = pd.concat([y_train, X_train], axis=1)
val_set = pd.concat([y_val, X_val], axis=1)
test_set = pd.concat([y_test, X_test], axis=1)

def save_to_arff(df, filename, meta):
    arff_data = {
        'description': '',
        'relation': 'dataset',
        'attributes': [(name, list(df[name].unique())) if df[name].dtype == 'object' else (name, 'REAL') for name in df.columns],
        'data': df.values.tolist()  
    }
    with open(filename, 'w') as f:
        liac_arff.dump(arff_data, f)

save_to_arff(train_set, 'train_set.arff', meta)
save_to_arff(val_set, 'val_set.arff', meta)
save_to_arff(test_set, 'test_set.arff', meta)