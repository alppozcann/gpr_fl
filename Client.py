import torch
import gpytorch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def build_kernel(kernel_type, num_features):
    """
    Build a GPyTorch kernel.

    kernel_type options:
        "rbf"        - RBF, single lengthscale
        "matern"     - Matérn 5/2, single lengthscale
        "rbf_ard"    - RBF, one lengthscale per feature (ARD)
        "matern_ard" - Matérn 5/2, one lengthscale per feature (ARD) [default]
    """
    kt = kernel_type.lower()
    if kt == "rbf":
        base = gpytorch.kernels.RBFKernel()
    elif kt == "matern":
        base = gpytorch.kernels.MaternKernel(nu=2.5)
    elif kt == "rbf_ard":
        base = gpytorch.kernels.RBFKernel(ard_num_dims=num_features)
    else:  # matern_ard
        base = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=num_features)
    return gpytorch.kernels.ScaleKernel(base)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SparseGPModel(gpytorch.models.ApproximateGP):
    """
    Sparse GP using Variational Inference with inducing points.
    Reduces complexity from O(n³) to O(nm²) where m = number of inducing points.
    """
    def __init__(self, inducing_points, kernel):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(SparseGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


MIN_CLUSTER_SIZE = 20


class Client:
    def __init__(self, client_id, csv_path, gp_type="exact", num_inducing_points=100,
                 expected_columns=None, kernel_type="matern_ard"):
        """
        Initialize a GP client.

        Args:
            client_id: Unique identifier for this client
            csv_path: Path to the CSV data file
            gp_type: "exact" for ExactGP or "sparse" for SparseGP with inducing points
            num_inducing_points: Number of inducing points (only used if gp_type="sparse")
            expected_columns: List of expected column names after get_dummies for consistent dimensions
            kernel_type: "rbf", "matern", "rbf_ard", or "matern_ard" (default: "matern_ard")
        """
        self.id = client_id
        self.csv_path = csv_path
        self.has_data = False
        self.gp_type = gp_type.lower()
        self.kernel_type = kernel_type.lower()
        self.num_inducing_points = num_inducing_points
        self.expected_columns = expected_columns

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"❌ Client {self.id}: Dosya okunamadı! Hata: {e}")
            return

        target_map = {
            "Outcome": "diabetes", "outcome": "diabetes",
            "Diabetes_012": "diabetes", "diabetes012": "diabetes",
            "Diabetes_binary": "diabetes"
        }
        df = df.rename(columns=target_map)

        if "diabetes" not in df.columns:
            print(f"⚠️ Client {self.id}: 'diabetes' sütunu yok.")
            return

        y = df["diabetes"]
        X = df.drop(columns=["diabetes"])

        if len(df) < MIN_CLUSTER_SIZE:
            print(f"⚠️ Client {self.id}: Veri sayısı çok az ({len(df)} < {MIN_CLUSTER_SIZE}), atlanıyor.")
            return

        scaler = StandardScaler()
        X = pd.get_dummies(X, drop_first=True)

        if self.expected_columns is not None:
            for col in self.expected_columns:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.expected_columns]

        X_scaled = scaler.fit_transform(X)

        train_size = int(0.8 * len(X_scaled))
        self.train_x = torch.tensor(X_scaled[:train_size]).float()
        self.train_y = torch.tensor(y.values[:train_size]).float()
        self.test_x = torch.tensor(X_scaled[train_size:]).float()
        self.test_y = torch.tensor(y.values[train_size:]).float()
        self.num_features = self.train_x.shape[1]

        # Build kernel
        kernel = build_kernel(self.kernel_type, self.num_features)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        if self.gp_type == "sparse":
            actual_num_inducing = min(self.num_inducing_points, len(self.train_x))
            inducing_indices = torch.randperm(len(self.train_x))[:actual_num_inducing]
            inducing_points = self.train_x[inducing_indices].clone()

            print(f"🔧 Client {self.id}: Sparse GP | {actual_num_inducing} inducing pts | kernel={self.kernel_type}")

            self.model = SparseGPModel(inducing_points, kernel)
            self.optimizer = torch.optim.Adam([
                {'params': self.model.parameters()},
                {'params': self.likelihood.parameters()},
            ], lr=0.1)
            self.mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=len(self.train_y))
        else:
            print(f"🔧 Client {self.id}: Exact GP | kernel={self.kernel_type}")
            self.model = ExactGPModel(self.train_x, self.train_y, self.likelihood, kernel)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
            self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        self.has_data = True

    def train_local(self, training_iter=50):
        if not self.has_data:
            return

        self.model.train()
        self.likelihood.train()

        print(f"Client {self.id} training... (n={len(self.train_x)}, iters={training_iter})")

        for i in range(training_iter):
            self.optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -self.mll(output, self.train_y)
            loss.backward()
            self.optimizer.step()

            if torch.isnan(loss):
                print(f"⚠️ Client {self.id}: Loss NaN, stopping early.")
                break

    def send_params(self):
        """
        Serialize kernel hyperparameters for federated aggregation.
        Format: [log(output_scale), log(ls_1), ..., log(ls_d), mean_constant]
        Works for both single-lengthscale and ARD kernels.
        """
        if not self.has_data:
            return None
        output_scale = self.model.covar_module.outputscale.item()
        ls = self.model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().flatten()
        mean_constant = self.model.mean_module.constant.item()
        return np.concatenate([[np.log(output_scale)], np.log(ls), [mean_constant]])

    def set_params(self, params):
        """
        Deserialize and apply aggregated kernel hyperparameters.
        Format: [log(output_scale), log(ls_1..d), mean_constant]
        """
        if not self.has_data:
            return
        new_output_scale = float(np.exp(params[0]))
        mean_constant = float(params[-1])
        new_ls = np.exp(params[1:-1])

        self.model.covar_module.outputscale = torch.tensor(new_output_scale).float()
        ls_tensor = torch.tensor(new_ls, dtype=torch.float32).unsqueeze(0)  # shape [1, d]
        self.model.covar_module.base_kernel.lengthscale = ls_tensor
        self.model.mean_module.constant.data = torch.tensor(mean_constant).float()

        if self.gp_type == "sparse":
            self.optimizer = torch.optim.Adam([
                {'params': self.model.parameters()},
                {'params': self.likelihood.parameters()},
            ], lr=0.1)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)

    def predict(self, X_test_tensor=None):
        if not self.has_data:
            return None, None

        if X_test_tensor is None:
            X_test_tensor = self.test_x

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = self.likelihood(self.model(X_test_tensor))
            y_pred = observed_pred.mean.detach().cpu().numpy()
            y_var = observed_pred.variance.detach().cpu().numpy()

        return y_pred, y_var

    def get_learner(self):
        return self

    def test_global_model(self):
        mu, _ = self.predict()

        if torch.is_tensor(self.test_y):
            y_test_numpy = self.test_y.cpu().numpy()
        else:
            y_test_numpy = self.test_y

        mu = mu.flatten()
        y_test_numpy = y_test_numpy.flatten()
        mse = np.mean((mu - y_test_numpy) ** 2)
        print(f"Client {self.id} Test MSE: {mse:.5f}")
        return mse

    def get_params(self):
        if not self.has_data:
            return None
        output_scale = self.model.covar_module.outputscale.item()
        length_scale = self.model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().flatten()
        mean_constant = self.model.mean_module.constant.item()
        return output_scale, length_scale, mean_constant
