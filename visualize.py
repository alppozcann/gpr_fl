import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.metrics import confusion_matrix
import os

warnings.filterwarnings("ignore", message="The input matches the stored training data")

def get_partition_threshold(client):
    return client.optimal_threshold

def plot_gp_predictions(client, feature_name, output_dir):
    probs, _ = client.predict()  # probs = sigmoid(f(x)) on client.test_x
    y_true = client.test_y.cpu().numpy().flatten()
    partition_thresh = get_partition_threshold(client)

    # Bernoulli aleatoric uncertainty: sqrt(p*(1-p))
    p = np.array(probs).flatten()
    std = np.sqrt(p * (1 - p))
    x_axis = np.arange(len(y_true))

    plt.figure(figsize=(10, 5))

    plt.fill_between(x_axis, p - 2*std, p + 2*std,
                     alpha=0.3, color='gray', label='95% Confidence')

    plt.scatter(x_axis, y_true, c='gray', alpha=0.6, s=30, label='Actual')

    plt.plot(x_axis, p, 'b-', linewidth=2, label='GP Prediction')

    plt.axhline(y=partition_thresh, color='r', linestyle='--', alpha=0.7, label=f'Partition Threshold ({partition_thresh:.2f})')
    
    plt.xlabel('Sample Index')
    plt.ylabel('Prediction')
    plt.title(f'{feature_name} - Client {client.id} GP Predictions')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = os.path.join(output_dir, f"{feature_name}_client{client.id}.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")

def plot_confusion_matrix(client, feature_name, output_dir):
    y_pred_probs, _ = client.predict()  # probs = sigmoid(f(x)) on client.test_x
    y_true = client.test_y.cpu().numpy().flatten().astype(int)
    partition_thresh = get_partition_threshold(client)

    y_pred = (np.array(y_pred_probs).flatten() > partition_thresh).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Diabetes (0)', 'Diabetes (1)'],
                yticklabels=['No Diabetes (0)', 'Diabetes (1)'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{feature_name} - Client {client.id}\nConfusion Matrix (Threshold: {partition_thresh:.2f})')
    
    plt.tight_layout()
    filename = os.path.join(output_dir, f"{feature_name}_client{client.id}_cm.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved: {filename}")

def visualize_clients(clients, feature_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for client in clients:
        plot_gp_predictions(client, feature_name, output_dir)
        plot_confusion_matrix(client, feature_name, output_dir)
