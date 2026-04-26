import torch
import gpytorch
from gpytorch.optim import NGD
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
        variational_distribution = gpytorch.variational.NaturalVariationalDistribution(
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

        from sklearn.model_selection import train_test_split

        scaler = StandardScaler()
        X = pd.get_dummies(X, drop_first=True)

        if self.expected_columns is not None:
            for col in self.expected_columns:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.expected_columns]

        X_scaled = scaler.fit_transform(X)
        y_arr = y.values

        # Stratified split so both halves have positives
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_arr, test_size=0.2, stratify=y_arr, random_state=42
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_arr, test_size=0.2, random_state=42
            )

        # Balance train: oversample minority class
        pos_idx = np.where(y_train == 1)[0]
        neg_idx = np.where(y_train == 0)[0]
        if len(pos_idx) > 0 and len(pos_idx) < len(neg_idx):
            rng = np.random.default_rng(42)
            extra = rng.choice(pos_idx, size=len(neg_idx) - len(pos_idx), replace=True)
            aug_idx = np.concatenate([np.arange(len(y_train)), extra])
            rng.shuffle(aug_idx)
            X_train = X_train[aug_idx]
            y_train = y_train[aug_idx]

        # Balance test: undersample majority class
        pos_test = np.where(y_test == 1)[0]
        neg_test = np.where(y_test == 0)[0]
        if len(pos_test) > 0 and len(neg_test) > len(pos_test):
            rng2 = np.random.default_rng(42)
            neg_keep = rng2.choice(neg_test, size=len(pos_test), replace=False)
            keep_idx = np.concatenate([pos_test, neg_keep])
            rng2.shuffle(keep_idx)
            X_test = X_test[keep_idx]
            y_test = y_test[keep_idx]

        self.train_x = torch.tensor(X_train).float()
        self.train_y = torch.tensor(y_train).float()
        self.test_x = torch.tensor(X_test).float()
        self.test_y = torch.tensor(y_test).float()
        self.num_features = self.train_x.shape[1]
        train_size = len(X_train)

        device = self.train_x.device

        n_ip = num_inducing_points if num_inducing_points is not None else _choose_num_inducing(train_size)
        n_ip = min(n_ip, train_size)

        # Stratified inducing points: 50% from positives, 50% from negatives
        pos_ip_pool = (self.train_y == 1).nonzero(as_tuple=True)[0]
        neg_ip_pool = (self.train_y == 0).nonzero(as_tuple=True)[0]
        n_pos_ip = min(n_ip // 2, len(pos_ip_pool))
        n_neg_ip = min(n_ip - n_pos_ip, len(neg_ip_pool))
        pos_ip_idx = pos_ip_pool[torch.randperm(len(pos_ip_pool))[:n_pos_ip]]
        neg_ip_idx = neg_ip_pool[torch.randperm(len(neg_ip_pool))[:n_neg_ip]]
        inducing_indices = torch.cat([pos_ip_idx, neg_ip_idx])
        inducing_points = self.train_x[inducing_indices].clone().to(device)

        kernel = build_kernel(self.kernel_type, self.num_features)
        self.model = SVGPClassificationModel(inducing_points, kernel)
        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood()

        print(f"🔧 Client {self.id}: SVGP Classification | {n_ip} inducing pts | kernel={self.kernel_type}")

        # NGD for variational parameters (mean/covariance in natural space)
        # Adam for kernel hyperparameters, likelihood, and inducing locations
        self.variational_optimizer = NGD(
            self.model.variational_parameters(), num_data=train_size, lr=0.1
        )
        self.hyperparameter_optimizer = torch.optim.Adam([
            {'params': self.model.hyperparameters()},
            {'params': self.likelihood.parameters()},
        ], lr=0.01)
        self.mll = gpytorch.mlls.VariationalELBO(
            self.likelihood, self.model, num_data=train_size
        )

        self.has_data = True

    def train_local(self, training_iter=50, batch_size=256):
        if not self.has_data:
            return

        self.model.train()
        self.likelihood.train()

        n = len(self.train_x)
        effective_batch = min(batch_size, n)
        dataset = torch.utils.data.TensorDataset(self.train_x, self.train_y)
        loader = torch.utils.data.DataLoader(dataset, batch_size=effective_batch, shuffle=True)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.hyperparameter_optimizer, T_max=training_iter, eta_min=1e-4
        )

        print(f"Client {self.id} training... (n={n}, epochs={training_iter}, batch={effective_batch})")

        with gpytorch.settings.cholesky_jitter(1e-3):
            for epoch in range(training_iter):
                for x_batch, y_batch in loader:
                    self.variational_optimizer.zero_grad()
                    self.hyperparameter_optimizer.zero_grad()
                    output = self.model(x_batch)
                    loss = -self.mll(output, y_batch)
                    loss.backward()
                    self.variational_optimizer.step()
                    self.hyperparameter_optimizer.step()

                    if torch.isnan(loss):
                        print(f"⚠️ Client {self.id}: Loss NaN at epoch {epoch}, stopping early.")
                        return

                scheduler.step()

    def send_params(self):
        """
        Serialize kernel hyperparameters for federated aggregation.
        Format: [log(output_scale), log(ls_1), ..., log(ls_d), mean_constant]
        """
        if not self.has_data:
            return None
        output_scale = self.model.covar_module.outputscale.item()
        ls = self.model.covar_module.base_kernel.lengthscale.detach().cpu().numpy().flatten()
        mean_const = self.model.mean_module.constant.item()
        return np.concatenate([[np.log(output_scale)], np.log(ls), [mean_const]])

    def set_params(self, params):
        """
        Deserialize and apply aggregated kernel hyperparameters.
        Format: [log(output_scale), log(ls_1..d), mean_constant]
        Preserves Adam momentum buffers across FL rounds for faster convergence.
        """
        if not self.has_data:
            return
        new_output_scale = float(np.exp(params[0]))
        new_ls = np.exp(params[1:-1])
        new_mean = float(params[-1])

        self.model.covar_module.outputscale = torch.tensor(new_output_scale).float()
        ls_tensor = torch.tensor(new_ls, dtype=torch.float32).unsqueeze(0)
        self.model.covar_module.base_kernel.lengthscale = ls_tensor
        self.model.mean_module.constant.data.fill_(new_mean)

        # Reset LR to initial value but keep Adam momentum for warm-starting
        for pg in self.hyperparameter_optimizer.param_groups:
            pg['lr'] = 0.01

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

        with torch.no_grad(), gpytorch.settings.fast_pred_var(), gpytorch.settings.cholesky_jitter(1e-3):
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
