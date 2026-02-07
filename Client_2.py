import torch
import gpytorch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Bu sınıf artık bir PyTorch Modeli gibi davranacak
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
    
class Client:
    def __init__(self, client_id, csv_path):
        self.id = client_id
        
        df = pd.read_csv(csv_path)
        if "Outcome" in df.columns:
            df = df.rename(columns={"Outcome": "diabetes"})
        elif "outcome" in df.columns:
            df = df.rename(columns={"outcome": "diabetes"})
        elif "Diabetes_012" in df.columns: 
            df = df.rename(columns={"Diabetes_012": "diabetes"})
        elif "diabetes012" in df.columns:
            df = df.rename(columns={"diabetes012": "diabetes"})
        y = df["diabetes"]
        X = df.drop(columns=["diabetes"])
        
        scaler = StandardScaler()
        X = pd.get_dummies(X, drop_first=True)
        X_scaled = scaler.fit_transform(X)
        
        self.X_tensor = torch.tensor(X_scaled).float()
        self.y_tensor = torch.tensor(y.values).float()

        limit = 1000
        self.X_tensor = self.X_tensor[:limit]
        self.y_tensor = self.y_tensor[:limit]

        
        train_size = int(0.8 * len(self.X_tensor))
        self.train_x = self.X_tensor[:train_size]
        self.train_y = self.y_tensor[:train_size]
        self.test_x = self.X_tensor[train_size:]
        self.test_y = self.y_tensor[train_size:]
        
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = GPModel(self.train_x, self.train_y, self.likelihood)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

    def train_local(self, training_iter=50):
        self.model.train()
        self.likelihood.train()
        
        print(f"Client {self.id} GPyTorch eğitimi başladı...")
        
        for i in range(training_iter):
            self.optimizer.zero_grad()
            
            output = self.model(self.train_x)
            
            loss = -self.mll(output, self.train_y)
            
            loss.backward()
            
            self.optimizer.step()
            
            if (i+1) % 10 == 0:
                print(f"Iter {i+1}/{training_iter} - Loss: {loss.item():.3f}")

    def send_params(self):
        
        output_scale = self.model.covar_module.outputscale.item()

        length_scale = self.model.covar_module.base_kernel.lengthscale.item()

        params = np.array([np.log(output_scale), np.log(length_scale)])
        
        return params

    def get_learner(self):
        return self

    def predict(self, X_test_tensor=None):
        if X_test_tensor is None:
            X_test_tensor = self.test_x
            
        self.model.eval()
        self.likelihood.eval()
        
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(X_test_tensor))
            
            y_pred = observed_pred.mean.numpy()
            y_var = observed_pred.variance.numpy()
            
        return y_pred, y_var

    def set_params(self, params):
        
        new_output_scale = np.exp(params[0])
        new_length_scale = np.exp(params[1])
        
        self.model.covar_module.outputscale = torch.tensor(new_output_scale)
        self.model.covar_module.base_kernel.lengthscale = torch.tensor([[new_length_scale]])

    def get_params(self):
        return self.model.covar_module.outputscale,self.model.covar_module.base_kernel.lengthscale
    
    def test_global_model(self):
        mu, sigma = self.predict()
        
        if torch.is_tensor(self.test_y):
            y_test_numpy = self.test_y.cpu().numpy()
        else:
            y_test_numpy = self.test_y
            
        mu = mu.flatten()
        y_test_numpy = y_test_numpy.flatten()
        
        mse = np.mean((mu - y_test_numpy)**2)
        
        print(f"Client {self.id} Test Sonucu (MSE): {mse:.5f}")
        return mse