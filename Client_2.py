import torch
import gpytorch
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Cihazı en başta belirliyoruz
device = torch.device("cpu") 
print(f"⚠️ GPU Sorunu nedeniyle CPU kullanılıyor. Cihaz: {device}")
# --- GP Model Sınıfı ---
class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # RBF Kernel: Standart ve güçlü bir çekirdek
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# --- Client Sınıfı ---
class Client:
    def __init__(self, client_id, csv_path):
        self.id = client_id
        self.csv_path = csv_path
        self.has_data = False # Varsayılan olarak False başlasın

        # 1. Veriyi Oku
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"❌ Client {self.id}: Dosya okunamadı! Hata: {e}")
            return

        # 2. Sütun İsimlerini Standartlaştır
        target_map = {
            "Outcome": "diabetes", "outcome": "diabetes",
            "Diabetes_012": "diabetes", "diabetes012": "diabetes",
            "Diabetes_binary": "diabetes"
        }
        df = df.rename(columns=target_map)
        
        if "diabetes" not in df.columns:
            print(f"⚠️ Client {self.id}: 'diabetes' sütunu yok.")
            return

        # 3. Veriyi Hazırla
        y = df["diabetes"]
        X = df.drop(columns=["diabetes"])
        
        # Scaling
        scaler = StandardScaler()
        X = pd.get_dummies(X, drop_first=True) 
        X_scaled = scaler.fit_transform(X)
        
        # --- KRİTİK DÜZELTME BURADA ---
        # Önce CPU üzerinde Tensör yapıyoruz (GPU'ya henüz atmıyoruz!)
        X_tensor_cpu = torch.tensor(X_scaled).float()
        y_tensor_cpu = torch.tensor(y.values).float()
                    
        # ŞİMDİ GPU'ya gönderiyoruz (Sadece 2000 tane veri gider, hata vermez)
        self.X_tensor = X_tensor_cpu.to(device)
        self.y_tensor = y_tensor_cpu.to(device)
        # -----------------------------
        
        # Train / Test Ayrımı
        train_size = int(0.8 * len(self.X_tensor))
        self.train_x = self.X_tensor[:train_size]
        self.train_y = self.y_tensor[:train_size]
        self.test_x = self.X_tensor[train_size:]
        self.test_y = self.y_tensor[train_size:]
        
        # Model Kurulumu
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
        self.model = GPModel(self.train_x, self.train_y, self.likelihood).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.1)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        
        self.has_data = True

    def train_local(self, training_iter=50):
        if not self.has_data: return # Veri yoksa işlem yapma

        self.model.train()
        self.likelihood.train()
        
        print(f"Client {self.id} GPyTorch eğitimi başladı... (Veri Sayısı: {len(self.train_x)})")
        
        for i in range(training_iter):
            self.optimizer.zero_grad()
            output = self.model(self.train_x)
            loss = -self.mll(output, self.train_y)
            loss.backward()
            self.optimizer.step()
            
            # Loss çok artarsa (NaN olursa) eğitimi durdur
            if torch.isnan(loss):
                print(f"⚠️ Client {self.id}: Loss NaN oldu, eğitim durduruluyor.")
                break

    def send_params(self):
        if not self.has_data: return None
        # Parametreleri alıp logaritmasını gönderiyoruz (Server ortalaması için daha sağlıklı)
        output_scale = self.model.covar_module.outputscale.item()
        length_scale = self.model.covar_module.base_kernel.lengthscale.item()
        params = np.array([np.log(output_scale), np.log(length_scale)])
        return params

    def set_params(self, params):
        if not self.has_data: return
        # Server'dan gelen ortalamayı geri açıp (exp) modele yüklüyoruz
        new_output_scale = np.exp(params[0])
        new_length_scale = np.exp(params[1])
        
        # Cihaz uyumluluğuna dikkat ederek yükle
        self.model.covar_module.outputscale = torch.tensor(new_output_scale, device=device)
        self.model.covar_module.base_kernel.lengthscale = torch.tensor([[new_length_scale]], device=device)

    def predict(self, X_test_tensor=None):
        if not self.has_data: return None, None
        
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
    
    # Client_2.py dosyasının içine, set_params fonksiyonunun altına ekle:

    def get_params(self):
        """
        Modelin güncel parametrelerini (OutputScale ve LengthScale) döndürür.
        Sadece ekrana yazdırmak (print) için kullanılır.
        """
        if not self.has_data: return None
        
        # Parametreleri tensor'dan alıp normal sayıya çeviriyoruz
        output_scale = self.model.covar_module.outputscale.item()
        length_scale = self.model.covar_module.base_kernel.lengthscale.item()
        
        return output_scale, length_scale