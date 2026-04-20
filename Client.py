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


def _choose_num_inducing(train_size):
    if train_size < 1000:
        return min(100, train_size)
    elif train_size < 50000:
        return min(300, train_size)
    else:
        return min(500, train_size)


class SVGPClassificationModel(gpytorch.models.ApproximateGP):
    """SVGP with BernoulliLikelihood for binary classification."""

    def __init__(self, inducing_points, kernel):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


MIN_CLUSTER_SIZE = 20


class Client:
    def __init__(self, client_id, csv_path, gp_type="sparse", num_inducing_points=None,
                 expected_columns=None, kernel_type="matern_ard"):
        """
        Initialize a GP classification client (always SVGP + BernoulliLikelihood).

        Args:
            client_id: Unique identifier for this client
            csv_path: Path to the CSV data file
            gp_type: Ignored — always uses SVGP classification
            num_inducing_points: Override inducing point count; auto-scaled by dataset size if None
            expected_columns: List of expected column names after get_dummies for consistent dimensions
            kernel_type: "rbf", "matern", "rbf_ard", or "matern_ard" (default: "matern_ard")
        """
        self.id = client_id
        self.csv_path = csv_path
        self.has_data = False
        self.kernel_type = kernel_type.lower()
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

        device = self.train_x.device

        n_ip = num_inducing_points if num_inducing_points is not None else _choose_num_inducing(train_size)
        n_ip = min(n_ip, train_size)

        inducing_indices = torch.randperm(train_size)[:n_ip]
        inducing_points = self.train_x[inducing_indices].clone().to(device)

        kernel = build_kernel(self.kernel_type, self.num_features)
        self.model = SVGPClassificationModel(inducing_points, kernel)
        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood()

        print(f"🔧 Client {self.id}: SVGP Classification | {n_ip} inducing pts | kernel={self.kernel_type}")

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.likelihood.parameters()),
            lr=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=50, gamma=0.5
        )
        self.mll = gpytorch.mlls.VariationalELBO(
            self.likelihood, self.model, num_data=train_size
        )

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

            pos = (self.train_y == 1).sum().float()
            neg = (self.train_y == 0).sum().float()
            pos_weight = (neg / pos).clamp(min=1.0, max=20.0)

            mll_val = self.mll(output, self.train_y)
            sample_weights = torch.where(self.train_y == 1, pos_weight, torch.ones_like(self.train_y))
            sample_weights = sample_weights / sample_weights.mean()
            loss = -(mll_val * sample_weights.mean())

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if torch.isnan(loss):
                print(f"⚠️ Client {self.id}: Loss NaN, stopping early.")
                break

    def send_params(self):
        """
        Serialize kernel hyperparameters for federated aggregation.
        Format: [log(output_scale), log(ls_1), ..., log(ls_d)]
        BernoulliLikelihood has no learnable parameters to aggregate.
        """
        if not self.has_data:
            return None
        output_scale = self.model.covar_module.outputscale.item()
        ls = self.model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().flatten()
        return np.concatenate([[np.log(output_scale)], np.log(ls)])

    def set_params(self, params):
        """
        Deserialize and apply aggregated kernel hyperparameters.
        Format: [log(output_scale), log(ls_1..d)]
        """
        if not self.has_data:
            return
        new_output_scale = float(np.exp(params[0]))
        new_ls = np.exp(params[1:])

        self.model.covar_module.outputscale = torch.tensor(new_output_scale).float()
        ls_tensor = torch.tensor(new_ls, dtype=torch.float32).unsqueeze(0)
        self.model.covar_module.base_kernel.lengthscale = ls_tensor

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.likelihood.parameters()),
            lr=0.01
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=50, gamma=0.5
        )

    def predict(self, X_test_tensor=None):
        """
        Returns (probs, hard_labels) where probs are class-1 probabilities in [0,1]
        and hard_labels are {0,1} thresholded at fixed 0.5.
        """
        if not self.has_data:
            return None, None

        if X_test_tensor is None:
            X_test_tensor = self.test_x

        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred_dist = self.likelihood(self.model(X_test_tensor))
            y_probs = pred_dist.probs.detach().cpu().numpy()
            threshold = 0.5
            y_hard = (y_probs > threshold).astype(float)

        return y_probs, y_hard

    def get_learner(self):
        return self

    def get_params(self):
        if not self.has_data:
            return None
        output_scale = self.model.covar_module.outputscale.item()
        length_scale = self.model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().flatten()
        return output_scale, length_scale
