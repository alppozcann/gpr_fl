import torch
import gpytorch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix


class SmallDatasetGPModel(gpytorch.models.ExactGP):
    """ExactGP + GaussianLikelihood for small datasets (n < 2000).

    Training: ExactMarginalLogLikelihood on {-1,+1} targets (signed GP surrogate).
    Inference: Laplace probit approximation — P(y=1|x*) ≈ σ(μ* / √(1 + π/8·σ²*))
               from Rasmussen & Williams (2006) §3.4.3, eq. 3.82.
    """
    def __init__(self, train_x, train_y, likelihood, kernel):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SparseGPModel(gpytorch.models.ApproximateGP):
    """Sparse GP using Variational Inference with inducing points.
    Reduces complexity from O(n³) to O(nm²) where m = number of inducing points."""
    def __init__(self, inducing_points, kernel):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = gpytorch.variational.VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True
        )
        super(SparseGPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


MIN_CLUSTER_SIZE = 20
MIN_POSITIVE_SAMPLES = 30  # absolute minimum positives — below this no meaningful boundary exists


class Client:
    def __init__(self, client_id, csv_path, num_inducing_points=100, expected_columns=None,
                 kernel_type="matern", standardize=False):
        self.id = client_id
        self.csv_path = csv_path
        self.has_data = False
        self.num_inducing_points = num_inducing_points
        self.kernel_type = kernel_type
        self.standardize = standardize
        self.expected_columns = expected_columns
        self.optimal_threshold = 0.5  # Validation'da öğrenilecek

        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Client {self.id}: Dosya okunamadı! Hata: {e}")
            return

        target_map = {
            "Outcome": "diabetes", "outcome": "diabetes",
            "Diabetes_012": "diabetes", "diabetes012": "diabetes",
            "Diabetes_binary": "diabetes"
        }
        df = df.rename(columns=target_map)

        if "diabetes" not in df.columns:
            print(f"Client {self.id}: 'diabetes' sütunu yok.")
            return

        y = df["diabetes"]
        X = df.drop(columns=["diabetes"])

        pos_rate = y.mean()
        print(f"  [Client {self.id}] class dağılımı: "
              f"pozitif={pos_rate:.1%}  negatif={(1-pos_rate):.1%}  (n={len(y)})")

        if len(df) < MIN_CLUSTER_SIZE:
            print(f"Client {self.id}: Veri sayısı çok az ({len(df)} < {MIN_CLUSTER_SIZE}), atlanıyor.")
            return

        n_pos = int(y.sum())
        if n_pos < MIN_POSITIVE_SAMPLES:
            print(f"Client {self.id}: Pozitif örnek çok az ({n_pos} < {MIN_POSITIVE_SAMPLES}), "
                  f"anlamlı sınır öğrenilemez — atlanıyor.")
            return

        self.scaler = StandardScaler()
        X = pd.get_dummies(X, drop_first=True)

        if self.expected_columns is not None:
            for col in self.expected_columns:
                if col not in X.columns:
                    X[col] = 0
            X = X[self.expected_columns]

        # Determine split sizes BEFORE scaling so we know model_type early
        n = len(X)
        if n < 1000:
            train_end = int(0.8 * n)
            val_end   = int(0.9 * n)
        else:
            train_end = int(0.6 * n)
            val_end   = int(0.8 * n)

        # Auto-select model based on training set size:
        #   small  (< 2000): ExactGP + GaussianLikelihood + ExactMLL + Laplace probit at predict
        #   sparse (≥ 2000): SVGP  + BernoulliLikelihood  + VariationalELBO + MC samples at predict
        self.model_type = "small" if train_end < 2000 else "sparse"

        if self.standardize:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = X.values.astype(np.float64)
        print(f"  [Client {self.id}] standardization={'ON' if self.standardize else 'OFF'}  n_train={train_end}")

        self.X_tensor = torch.tensor(X_scaled).float()
        self.y_tensor = torch.tensor(y.values).float()

        self.train_x = self.X_tensor[:train_end]
        self.train_y = self.y_tensor[:train_end]
        self.val_x   = self.X_tensor[train_end:val_end]
        self.val_y   = self.y_tensor[train_end:val_end]
        self.test_x  = self.X_tensor[val_end:]
        self.test_y  = self.y_tensor[val_end:]

        kernel = self._build_kernel()

        if self.model_type == "small":
            # Targets mapped to {-1, +1} so latent f spans (-∞,+∞).
            # This makes the Laplace probit valid:
            #   negative class → posterior mean ≈ -1 → σ(κ·(-1)) < 0.5
            #   positive class → posterior mean ≈ +1 → σ(κ·(+1)) > 0.5
            train_y_signed = 2.0 * self.train_y - 1.0  # {0,1} → {-1,+1}
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self.model = SmallDatasetGPModel(self.train_x, train_y_signed, self.likelihood, kernel)
            # ExactGP stores likelihood as a submodule → model.parameters() already covers it
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
            self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
            self.model.covar_module.outputscale = torch.tensor(5.0)
            self.model.covar_module.base_kernel.lengthscale = torch.tensor([[1.0]])
            print(f"🔧 Client {self.id}: ExactGP+Laplace+{self.kernel_type} | n_train={len(self.train_x)}")
        else:
            self.likelihood = gpytorch.likelihoods.BernoulliLikelihood()
            actual_num_inducing = min(self.num_inducing_points, len(self.train_x))
            inducing_indices = torch.randperm(len(self.train_x))[:actual_num_inducing]
            inducing_points = self.train_x[inducing_indices].clone()
            self.model = SparseGPModel(inducing_points, kernel)
            self.optimizer = torch.optim.Adam([
                {'params': self.model.parameters()},
                {'params': self.likelihood.parameters()},
            ], lr=0.1)
            self.mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=len(self.train_y))
            # outputscale=4 → GP mean ±2 → sigmoid [0.12, 0.88] — prevents collapse to 0.5
            self.model.covar_module.outputscale = torch.tensor(4.0)
            self.model.covar_module.base_kernel.lengthscale = torch.tensor([[0.5]])
            print(f"Client {self.id}: SparseGP+Bernoulli+{self.kernel_type} | {actual_num_inducing} inducing")

        self.has_data = True

    def _build_kernel(self):
        k = gpytorch.kernels.MaternKernel(nu=2.5)
        k.register_constraint("raw_lengthscale", gpytorch.constraints.Interval(0.1, 3.0))
        return k

    def _balance_training_data(self, max_ratio=3.0):
        """
        Pozitif class'ı oversample ederek negatif/pozitif oranını max_ratio:1'e çeker.
        max_ratio=None → ham veriyi döndür (oversampling yok).
        max_ratio=3   → negatif sayısı pozitifin en fazla 3 katı olur.
        """
        if max_ratio is None:
            return self.train_x, self.train_y

        pos_idx = (self.train_y == 1).nonzero(as_tuple=True)[0]
        neg_idx = (self.train_y == 0).nonzero(as_tuple=True)[0]

        if len(pos_idx) == 0:
            return self.train_x, self.train_y

        max_neg = int(len(pos_idx) * max_ratio)
        if len(neg_idx) > max_neg:
            perm = torch.randperm(len(neg_idx))[:max_neg]
            neg_idx = neg_idx[perm]

        repeat = max(1, max_neg // len(pos_idx))
        pos_idx_repeated = pos_idx.repeat(repeat)

        all_idx = torch.cat([pos_idx_repeated, neg_idx])
        all_idx = all_idx[torch.randperm(len(all_idx))]

        balanced_x = self.train_x[all_idx]
        balanced_y = self.train_y[all_idx]

        print(f"  Balanced: {len(pos_idx_repeated)} pos + {len(neg_idx)} neg = {len(all_idx)} total")
        return balanced_x, balanced_y

    def train_local(self, training_iter=50):
        if not self.has_data:
            return

        # ExactGP handles imbalance via MLL directly; oversampling biases the posterior.
        # For extreme imbalance (pos < 10%) use 4:1 ratio to guarantee enough positive signal.
        orig_pos = self.train_y.mean().item()
        if self.model_type == "sparse":
            if orig_pos < 0.05:
                oversample_ratio = 5.0   # extreme: ≥17% positive after balance
            elif orig_pos < 0.10:
                oversample_ratio = 4.0   # severe: ≥20% positive after balance
            else:
                oversample_ratio = 3.0   # normal: ≥25% positive after balance
        else:
            oversample_ratio = None
        train_x, train_y = self._balance_training_data(max_ratio=oversample_ratio)

        self.model.train()
        self.likelihood.train()

        # ExactGP caches training data — update before each training run (signed targets)
        if self.model_type == "small":
            self.model.set_train_data(train_x, 2.0 * train_y - 1.0, strict=False)

        pos_ratio = train_y.mean().item()
        pre_os = self.model.covar_module.outputscale.item()
        pre_ls = self.model.covar_module.base_kernel.lengthscale.item()
        print(f"Client {self.id} train start | n={len(train_x)} pos={pos_ratio:.3f} "
              f"| params_before: os={pre_os:.4f} ls={pre_ls:.4f}")

        for i in range(training_iter):
            self.optimizer.zero_grad()
            if self.model_type == "small":
                # ExactMLL with {-1,+1} targets — latent f spans (-∞,+∞) for valid probit
                output = self.model(train_x)
                loss = -self.mll(output, 2.0 * train_y - 1.0)
            else:
                # VariationalELBO with BernoulliLikelihood — requires MC integration
                with gpytorch.settings.num_likelihood_samples(16):
                    output = self.model(train_x)
                    loss = -self.mll(output, train_y)
            loss.backward()
            self.optimizer.step()

            if torch.isnan(loss):
                print(f"⚠️ Client {self.id}: Loss NaN, durduruluyor.")
                break

        # DEBUG — GP latent mean range (sıfıra yakınsa kernel converge edememiş demektir)
        self.model.eval()
        with torch.no_grad():
            sample = self.val_x[:min(100, len(self.val_x))]
            gp_out = self.model(sample)
            gp_mean = gp_out.mean
            os = self.model.covar_module.outputscale.item()
            ls = self.model.covar_module.base_kernel.lengthscale.item()
            print(f"  Client {self.id} params_after: os={os:.4f} ls={ls:.4f} "
                  f"| GP mean=[{gp_mean.min():.3f}, {gp_mean.max():.3f}]")

        self._find_optimal_threshold()

    def _get_probs(self, X_tensor):
        """
        Model tipine göre class probability üretir:

        small  — Laplace probit approximation (R&W 2006, §3.4.3, eq. 3.82):
                   P(y=1|x*) ≈ σ(μ* / √(1 + π/8·σ²*))
                 ExactGP posterior mean μ* ve variance σ²* analytically tractable.

        sparse — BernoulliLikelihood + 64 MC samples:
                   P(y=1|x*) ≈ (1/S) Σ σ(f^(s)(x*))
        """
        self.model.eval()
        self.likelihood.eval()

        with torch.no_grad():
            if self.model_type == "small":
                with gpytorch.settings.fast_pred_var():
                    pred = self.likelihood(self.model(X_tensor))
                    mu  = pred.mean.detach().cpu().numpy()
                    var = pred.variance.detach().cpu().numpy()
                # Probit correction — shrinks latent mean by posterior uncertainty
                kappa = 1.0 / np.sqrt(1.0 + np.pi / 8.0 * np.clip(var, 0, None))
                return (1.0 / (1.0 + np.exp(-kappa * mu))).flatten()
            else:
                with gpytorch.settings.num_likelihood_samples(64):
                    pred = self.likelihood(self.model(X_tensor))
                    probs = pred.probs.detach().cpu().numpy()
                    if probs.ndim == 2:
                        probs = probs.mean(axis=0)  # (S, N) → (N,)
                    return probs.flatten()

    def _find_optimal_threshold(self):
        """
        Validation set üzerinde F1'i maximize eden threshold'u bul.
        Imbalanced dataset'lerde 0.5 genellikle suboptimal.
        """
        if len(self.val_x) == 0:
            return

        probs = self._get_probs(self.val_x)
        labels = self.val_y.numpy()

        # Sadece iki class varsa threshold search yap
        if len(np.unique(labels)) < 2:
            return

        # Minimum precision floor: threshold search'te "hepsini pozitif say" çözümünü engelle.
        # En az 2x base rate precision gerektir (örn. %15 pozitif → min precision %30).
        base_rate = labels.mean()
        min_precision = min(2.0 * base_rate, 0.5)  # üst sınır 0.5

        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.05, 0.95, 0.005):
            preds = (probs > t).astype(float)
            if preds.sum() == 0:
                continue
            prec = precision_score(labels, preds, zero_division=0)
            if prec < min_precision:
                continue
            f1 = f1_score(labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = float(t)

        self.optimal_threshold = best_t
        print(f"  → Client {self.id}: optimal threshold = {best_t:.3f} (val F1={best_f1:.4f})")

    def predict(self, X_test_tensor=None):
        """
        Returns (probs, hard_labels).
        probs: sigmoid(f(x)) — class-1 probability
        hard_labels: {0, 1} thresholded at optimal_threshold
        """
        if not self.has_data:
            return None, None

        if X_test_tensor is None:
            X_test_tensor = self.test_x

        probs = self._get_probs(X_test_tensor)
        y_hard = (probs > self.optimal_threshold).astype(float)

        return probs, y_hard

    def evaluate(self, X_test_tensor=None, y_true=None, verbose=True):
        """
        Tam classification metrikleri: Accuracy, F1, Precision, Recall, ROC-AUC
        """
        if not self.has_data:
            return None

        if X_test_tensor is None:
            X_test_tensor = self.test_x
        if y_true is None:
            y_true = self.test_y.numpy()
        elif torch.is_tensor(y_true):
            y_true = y_true.numpy()

        probs, y_hard = self.predict(X_test_tensor)

        if len(np.unique(y_true)) < 2:
            if verbose:
                print(f"Client {self.id}: Test set'te tek class var, metrik hesaplanamıyor.")
            return None

        acc = accuracy_score(y_true, y_hard)
        f1 = f1_score(y_true, y_hard, zero_division=0)
        prec = precision_score(y_true, y_hard, zero_division=0)
        rec = recall_score(y_true, y_hard, zero_division=0)
        auc = roc_auc_score(y_true, probs)
        cm = confusion_matrix(y_true, y_hard)

        if verbose:
            print(f"\n Client {self.id} Evaluation (threshold={self.optimal_threshold:.2f}):")
            print(f"  Accuracy  : {acc:.4f}")
            print(f"  F1 Score  : {f1:.4f}")
            print(f"  Precision : {prec:.4f}")
            print(f"  Recall    : {rec:.4f}")
            print(f"  ROC-AUC   : {auc:.4f}")
            print(f"  Confusion Matrix:\n{cm}")

        return {
            "accuracy": acc, "f1": f1, "precision": prec,
            "recall": rec, "roc_auc": auc, "confusion_matrix": cm,
            "threshold": self.optimal_threshold
        }

    def send_params(self):
        if not self.has_data:
            return None
        output_scale = self.model.covar_module.outputscale.item()
        length_scale = self.model.covar_module.base_kernel.lengthscale.item()
        mean_constant = self.model.mean_module.constant.item()
        return np.array([np.log(output_scale), np.log(length_scale), mean_constant])

    def set_params(self, params):
        if not self.has_data:
            return
        new_output_scale = float(np.exp(params[0]))
        new_length_scale = float(np.exp(params[1]))
        new_mean_constant = float(params[2])

        # Use initialize() so GPyTorch's constraint transform is respected
        self.model.covar_module.outputscale = torch.tensor(new_output_scale)
        self.model.covar_module.base_kernel.lengthscale = torch.tensor([[new_length_scale]])
        self.model.mean_module.constant.data = torch.tensor(new_mean_constant)

        # Verify what was actually set (constraint may clip the value)
        actual_ls = self.model.covar_module.base_kernel.lengthscale.item()
        actual_os = self.model.covar_module.outputscale.item()

        if self.model_type == "small":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        else:
            self.optimizer = torch.optim.Adam([
                {'params': self.model.parameters()},
                {'params': self.likelihood.parameters()},
            ], lr=0.1)

        self.optimal_threshold = 0.5

    def get_learner(self):
        return self

    def get_params(self):
        if not self.has_data:
            return None
        output_scale = self.model.covar_module.outputscale.item()
        length_scale = self.model.covar_module.base_kernel.lengthscale.item()
        mean_constant = self.model.mean_module.constant.item()
        return output_scale, length_scale, mean_constant

    def test_global_model(self):
        return self.evaluate()