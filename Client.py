import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

class GPModel:
    def __init__(self):
        # 1. SINIRLARI GENİŞLETİYORUZ
        # Constant (Varyans): 1e-3 (0.001) yerine 1e-4 (0.0001) ve 1e4 (10000) yapıyoruz.
        # RBF (Length Scale): 1e-2 yerine 1e-5 (çok ince detay) ve 1e5 (çok genel) yapıyoruz.
        kernel = C(1.0, (1e-4, 1e4)) * RBF(1.0, (1e-5, 1e5))

        self.model = GaussianProcessRegressor(
            kernel=kernel,
            n_restarts_optimizer=10,  # 2. Pes etmemesi için deneme sayısını 5'ten 10'a çıkardık
            normalize_y=True,         # Hedef değişkeni normalize etmeye devam et
            alpha=1e-2,               # 3. KRİTİK NOKTA: Gürültü toleransı (Noise)
            random_state=42
        )
    
    def fit(self,X,y):
        self.model.fit(X,y)
    
    def predict(self,X):
        return self.model.predict(X, return_std=True)
    
    def get_params(self):
        return self.model.kernel_.theta

    def set_params(self,params):
        self.model.kernel_.theta = params


class Client:
    def __init__(self, client_id,table):
        self.id = client_id
        df = pd.read_csv(table)

        if "Outcome" in df.columns:
            df = df.rename(columns={"Outcome": "diabetes"})
        elif "outcome" in df.columns:
            df = df.rename(columns={"outcome": "diabetes"})
        elif "Diabetes_012" in df.columns: 
            df = df.rename(columns={"Diabetes_012": "diabetes"})
        elif "diabetes012" in df.columns:
            df = df.rename(columns={"diabetes012": "diabetes"})

        if "diabetes" in df.columns:
             df["diabetes"] = df["diabetes"].apply(lambda x: 1.0 if x >= 1 else 0.0)
            
        if len(df) > 2000:
            df = df.sample(n=2000, random_state=42)
        
        y = df["diabetes"]
        X = df.drop(columns=["diabetes"])
        X = pd.get_dummies(X, drop_first=True)
        column_names = X.columns
        original_index = X.index
        
        scaler = StandardScaler()
        X_array = scaler.fit_transform(X)
        X = pd.DataFrame(X_array, columns=column_names, index=original_index)

        data = pd.concat([X, y], axis=1).dropna()
        
        self.y = data["diabetes"]
        self.X = data.drop(columns=["diabetes"])
        self.X = self.X.astype(float)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X,y, 
                                                                                test_size=0.2,
                                                                                random_state=42)

        self.learner = GPModel()

    def train_local(self):
        self.learner.fit(self.X_train, self.y_train)
    
    def set_params(self, new_params):
        self.learner.set_params(new_params)
        
    def send_params(self):
        params =  self.learner.get_params()
        print(f"Sending params for client {self.id} to server --> {params}")
        return params
        
    def test_global_model(self):
        mu, sigma = self.learner.predict(self.X_test)
        mse = np.mean((mu - self.y_test)**2)
        print(f"Client {self.id} Test Sonucu (MSE): {mse:.5f}")
        return mse
    