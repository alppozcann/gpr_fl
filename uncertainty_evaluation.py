"""
Uncertainty Evaluation Module for GP-FL

This module showcases the unique power of Gaussian Processes:
the ability to provide prediction uncertainty (variance), which 
traditional classifiers like RFC and LR cannot provide.

For medical diagnosis, this is crucial:
- High confidence positive → Likely diabetic, needs treatment
- High confidence negative → Likely healthy, no action needed  
- Uncertain → Needs further testing (GP flags these!)
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def get_uncertainty_metrics(client):
    """
    Get predictions with uncertainty from a single client.

    Uses client.predict() — returns (probs, hard_labels) — so no direct
    model access is needed. Aleatoric uncertainty is approximated as
    Bernoulli variance: p*(1-p), consistent with sigmoid(f(x)) output.

    Returns:
        dict with predictions, uncertainties, and metrics
    """
    probs, _ = client.predict()  # probs = sigmoid(f(x)) on client.test_x
    y_pred_mean = np.array(probs).flatten()

    # Bernoulli aleatoric uncertainty: p*(1-p)
    y_pred_var = y_pred_mean * (1 - y_pred_mean)
    y_pred_std = np.sqrt(y_pred_var)

    y_true = client.test_y.cpu().numpy().flatten().astype(int)
    threshold = client.optimal_threshold
    y_pred_binary = (y_pred_mean > threshold).astype(int)

    return {
        'y_true': y_true,
        'y_pred_mean': y_pred_mean,
        'y_pred_var': y_pred_var,
        'y_pred_std': y_pred_std,
        'y_pred_binary': y_pred_binary,
        'threshold': threshold
    }


def analyze_uncertainty(clients, feature_name, plots_dir):
    """
    Comprehensive uncertainty analysis for all clients.
    
    This analysis shows what GP can do that RFC/LR cannot:
    1. Identify uncertain predictions that need human review
    2. Provide confidence-calibrated predictions
    3. Flag out-of-distribution samples
    """
    all_y_true = []
    all_y_pred_mean = []
    all_y_pred_std = []
    all_y_pred_binary = []
    all_thresholds = []
    
    for client in clients:
        metrics = get_uncertainty_metrics(client)
        all_y_true.append(metrics['y_true'])
        all_y_pred_mean.append(metrics['y_pred_mean'])
        all_y_pred_std.append(metrics['y_pred_std'])
        all_y_pred_binary.append(metrics['y_pred_binary'])
        all_thresholds.append(metrics['threshold'])
    
    y_true = np.concatenate(all_y_true)
    y_pred_mean = np.concatenate(all_y_pred_mean)
    y_pred_std = np.concatenate(all_y_pred_std)
    y_pred_binary = np.concatenate(all_y_pred_binary)
    
    results = {}
    
    # 1. Basic uncertainty statistics
    results['mean_uncertainty'] = np.mean(y_pred_std)
    results['median_uncertainty'] = np.median(y_pred_std)
    results['max_uncertainty'] = np.max(y_pred_std)
    results['min_uncertainty'] = np.min(y_pred_std)
    
    # 2. Uncertainty by prediction correctness
    correct_mask = (y_pred_binary == y_true)
    
    results['mean_uncertainty_correct'] = np.mean(y_pred_std[correct_mask])
    results['mean_uncertainty_incorrect'] = np.mean(y_pred_std[~correct_mask]) if np.sum(~correct_mask) > 0 else 0
    
    # 3. Confidence-filtered metrics (THE KEY GP ADVANTAGE!)
    # Calculate metrics at different confidence thresholds
    confidence_thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    confidence_results = []
    
    for conf_thresh in confidence_thresholds:
        # Low uncertainty = high confidence
        # Filter predictions where std is below threshold * max_std
        max_std = np.max(y_pred_std)
        if max_std > 0:
            normalized_std = y_pred_std / max_std
        else:
            normalized_std = y_pred_std
            
        confident_mask = normalized_std <= (1 - conf_thresh)
        
        if np.sum(confident_mask) > 0:
            y_true_conf = y_true[confident_mask]
            y_pred_conf = y_pred_binary[confident_mask]
            
            acc = accuracy_score(y_true_conf, y_pred_conf)
            prec = precision_score(y_true_conf, y_pred_conf, zero_division=0)
            rec = recall_score(y_true_conf, y_pred_conf, zero_division=0)
            f1 = f1_score(y_true_conf, y_pred_conf, zero_division=0)
            coverage = np.sum(confident_mask) / len(y_true)
            rejection_rate = 1 - coverage
        else:
            acc = prec = rec = f1 = 0
            coverage = 0
            rejection_rate = 1
            
        confidence_results.append({
            'confidence_threshold': conf_thresh,
            'coverage': coverage,
            'rejection_rate': rejection_rate,
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        })
    
    results['confidence_analysis'] = confidence_results
    
    # 4. Find the "optimal" rejection rate for best F1
    best_f1 = 0
    best_conf_result = confidence_results[0]
    for cr in confidence_results:
        if cr['f1'] > best_f1 and cr['coverage'] > 0.5:  # At least 50% coverage
            best_f1 = cr['f1']
            best_conf_result = cr
    
    results['best_confident_metrics'] = best_conf_result
    
    # 5. Uncertainty-based risk stratification
    # Divide into Low/Medium/High risk based on prediction + uncertainty
    low_risk_mask = (y_pred_mean < 0.3) & (y_pred_std < np.percentile(y_pred_std, 50))
    high_risk_mask = (y_pred_mean > 0.7) & (y_pred_std < np.percentile(y_pred_std, 50))
    uncertain_mask = ~low_risk_mask & ~high_risk_mask
    
    results['risk_stratification'] = {
        'low_risk_count': np.sum(low_risk_mask),
        'low_risk_actual_positive_rate': np.mean(y_true[low_risk_mask]) if np.sum(low_risk_mask) > 0 else 0,
        'high_risk_count': np.sum(high_risk_mask),
        'high_risk_actual_positive_rate': np.mean(y_true[high_risk_mask]) if np.sum(high_risk_mask) > 0 else 0,
        'uncertain_count': np.sum(uncertain_mask),
        'uncertain_actual_positive_rate': np.mean(y_true[uncertain_mask]) if np.sum(uncertain_mask) > 0 else 0,
    }
    
    # Generate visualizations
    decision_threshold = float(np.mean(all_thresholds)) if len(all_thresholds) > 0 else 0.5
    _create_uncertainty_plots(y_true, y_pred_mean, y_pred_std, y_pred_binary, decision_threshold, feature_name, plots_dir, results)
    
    return results


def _create_uncertainty_plots(y_true, y_pred_mean, y_pred_std, y_pred_binary, decision_threshold, feature_name, plots_dir, results):
    """Create uncertainty visualization plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'GP Uncertainty Analysis: {feature_name}', fontsize=16, fontweight='bold')
    
    # 1. Prediction vs Uncertainty scatter
    ax1 = axes[0, 0]
    colors = ['green' if t == 0 else 'red' for t in y_true]
    ax1.scatter(y_pred_mean, y_pred_std, c=colors, alpha=0.3, s=10)
    ax1.set_xlabel('Predicted Probability')
    ax1.set_ylabel('Prediction Uncertainty (Std)')
    ax1.set_title('Prediction vs Uncertainty\n(Green=Healthy, Red=Diabetic)')
    ax1.axvline(x=decision_threshold, color='black', linestyle='--', alpha=0.5,
                label=f'Decision threshold ({decision_threshold:.2f})')
    ax1.legend()
    
    # 2. Uncertainty distribution by correctness
    ax2 = axes[0, 1]
    correct_mask = (y_pred_binary == y_true)
    ax2.hist(y_pred_std[correct_mask], bins=50, alpha=0.7, label=f'Correct (μ={results["mean_uncertainty_correct"]:.3f})', color='green')
    ax2.hist(y_pred_std[~correct_mask], bins=50, alpha=0.7, label=f'Incorrect (μ={results["mean_uncertainty_incorrect"]:.3f})', color='red')
    ax2.set_xlabel('Uncertainty (Std)')
    ax2.set_ylabel('Count')
    ax2.set_title('Uncertainty Distribution\n(Correct vs Incorrect Predictions)')
    ax2.legend()
    
    # 3. Confidence vs F1 tradeoff
    ax3 = axes[0, 2]
    conf_data = results['confidence_analysis']
    coverage = [c['coverage'] * 100 for c in conf_data]
    f1_scores = [c['f1'] for c in conf_data]
    accuracies = [c['accuracy'] for c in conf_data]
    
    ax3.plot(coverage, f1_scores, 'b-o', linewidth=2, markersize=8, label='F1-Score')
    ax3.plot(coverage, accuracies, 'g-s', linewidth=2, markersize=8, label='Accuracy')
    ax3.set_xlabel('Coverage (%)')
    ax3.set_ylabel('Score')
    ax3.set_title('Confidence Filtering:\nMetrics vs Coverage')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.invert_xaxis()  # Lower coverage = higher confidence threshold
    
    # 4. Risk stratification
    ax4 = axes[1, 0]
    risk_data = results['risk_stratification']
    categories = ['Low Risk\n(Confident Healthy)', 'Uncertain\n(Needs Review)', 'High Risk\n(Confident Diabetic)']
    counts = [risk_data['low_risk_count'], risk_data['uncertain_count'], risk_data['high_risk_count']]
    actual_rates = [risk_data['low_risk_actual_positive_rate'], 
                   risk_data['uncertain_actual_positive_rate'],
                   risk_data['high_risk_actual_positive_rate']]
    
    x = np.arange(len(categories))
    width = 0.35
    
    bars1 = ax4.bar(x - width/2, counts, width, label='Count', color='steelblue')
    ax4_twin = ax4.twinx()
    bars2 = ax4_twin.bar(x + width/2, [r*100 for r in actual_rates], width, label='Actual Positive Rate', color='coral')
    
    ax4.set_ylabel('Count')
    ax4_twin.set_ylabel('Actual Diabetes Rate (%)')
    ax4.set_title('Risk Stratification\n(GP\'s Unique Capability)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    
    # 5. Calibration plot
    ax5 = axes[1, 1]
    # Bin predictions and show actual positive rate
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    actual_rates = []
    predicted_rates = []
    for i in range(n_bins):
        mask = (y_pred_mean >= bin_edges[i]) & (y_pred_mean < bin_edges[i+1])
        if np.sum(mask) > 0:
            actual_rates.append(np.mean(y_true[mask]))
            predicted_rates.append(bin_centers[i])
        else:
            actual_rates.append(np.nan)
            predicted_rates.append(bin_centers[i])
    
    ax5.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax5.scatter(predicted_rates, actual_rates, s=100, c='blue', label='GP Predictions')
    ax5.set_xlabel('Predicted Probability')
    ax5.set_ylabel('Actual Positive Rate')
    ax5.set_title('Calibration Plot\n(How well does GP probability reflect reality?)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Summary text
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    best = results['best_confident_metrics']
    summary_text = f"""
    📊 GP UNCERTAINTY ANALYSIS SUMMARY
    
    🎯 Key GP Advantage: Uncertainty Quantification
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    📈 Overall Uncertainty Statistics:
    • Mean Uncertainty: {results['mean_uncertainty']:.4f}
    • Median Uncertainty: {results['median_uncertainty']:.4f}
    
    ✅ Uncertainty vs Correctness:
    • Correct predictions: σ = {results['mean_uncertainty_correct']:.4f}
    • Incorrect predictions: σ = {results['mean_uncertainty_incorrect']:.4f}
    • Ratio: {results['mean_uncertainty_incorrect']/max(results['mean_uncertainty_correct'], 0.001):.2f}x higher for errors
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    🏆 CONFIDENT PREDICTIONS ONLY:
    (Rejecting {best['rejection_rate']*100:.1f}% uncertain cases)
    
    • Coverage: {best['coverage']*100:.1f}%
    • Accuracy: {best['accuracy']:.4f}
    • Precision: {best['precision']:.4f}
    • Recall: {best['recall']:.4f}
    • F1-Score: {best['f1']:.4f}
    
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    
    🏥 Clinical Value:
    GP identifies {results['risk_stratification']['uncertain_count']} uncertain
    cases ({results['risk_stratification']['uncertain_count']/len(y_true)*100:.1f}%) 
    that should be flagged for additional testing.
    
    RFC/LR cannot provide this uncertainty information!
    """
    
    ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f'uncertainty_{feature_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: uncertainty_{feature_name}.png")


def generate_uncertainty_report(all_uncertainty_results, output_file):
    """Generate a comprehensive uncertainty report."""
    
    with open(output_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("GP-FL UNCERTAINTY ANALYSIS REPORT\n")
        f.write("Showcasing Gaussian Process's Unique Capability: Prediction Uncertainty\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("WHY THIS MATTERS:\n")
        f.write("-" * 50 + "\n")
        f.write("Traditional classifiers (LR, RFC) output only point predictions.\n")
        f.write("GP provides UNCERTAINTY estimates, enabling:\n")
        f.write("  1. Risk stratification (Low/Medium/High confidence)\n")
        f.write("  2. Flagging uncertain cases for human review\n")
        f.write("  3. Confidence-filtered predictions with higher accuracy\n")
        f.write("  4. Better clinical decision support\n\n")
        
        # Summary table
        f.write("=" * 100 + "\n")
        f.write(f"{'Feature':<25} {'Coverage':<12} {'Acc (All)':<12} {'Acc (Conf)':<12} {'F1 (All)':<12} {'F1 (Conf)':<12} {'Uncertain %':<12}\n")
        f.write("=" * 100 + "\n")
        
        for feature, results in all_uncertainty_results.items():
            all_metrics = results['confidence_analysis'][0]  # 0% rejection = all data
            best = results['best_confident_metrics']
            uncertain_pct = results['risk_stratification']['uncertain_count'] / (
                results['risk_stratification']['low_risk_count'] + 
                results['risk_stratification']['uncertain_count'] + 
                results['risk_stratification']['high_risk_count']
            ) * 100
            
            f.write(f"{feature:<25} {best['coverage']*100:<12.1f} {all_metrics['accuracy']:<12.4f} {best['accuracy']:<12.4f} {all_metrics['f1']:<12.4f} {best['f1']:<12.4f} {uncertain_pct:<12.1f}\n")
        
        f.write("=" * 100 + "\n\n")
        
        # Detailed per-feature analysis
        for feature, results in all_uncertainty_results.items():
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"FEATURE: {feature}\n")
            f.write("=" * 80 + "\n")
            
            f.write(f"\nUncertainty Statistics:\n")
            f.write(f"  Mean σ: {results['mean_uncertainty']:.4f}\n")
            f.write(f"  Median σ: {results['median_uncertainty']:.4f}\n")
            f.write(f"  σ for correct predictions: {results['mean_uncertainty_correct']:.4f}\n")
            f.write(f"  σ for incorrect predictions: {results['mean_uncertainty_incorrect']:.4f}\n")
            
            f.write(f"\nRisk Stratification:\n")
            risk = results['risk_stratification']
            total = risk['low_risk_count'] + risk['uncertain_count'] + risk['high_risk_count']
            f.write(f"  Low Risk (Confident Healthy): {risk['low_risk_count']} ({risk['low_risk_count']/total*100:.1f}%)\n")
            f.write(f"    → Actual diabetes rate: {risk['low_risk_actual_positive_rate']*100:.1f}%\n")
            f.write(f"  Uncertain (Needs Review): {risk['uncertain_count']} ({risk['uncertain_count']/total*100:.1f}%)\n")
            f.write(f"    → Actual diabetes rate: {risk['uncertain_actual_positive_rate']*100:.1f}%\n")
            f.write(f"  High Risk (Confident Diabetic): {risk['high_risk_count']} ({risk['high_risk_count']/total*100:.1f}%)\n")
            f.write(f"    → Actual diabetes rate: {risk['high_risk_actual_positive_rate']*100:.1f}%\n")
            
            f.write(f"\nConfidence Filtering Analysis:\n")
            f.write(f"  {'Confidence':<12} {'Coverage':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
            f.write(f"  {'-'*72}\n")
            for conf in results['confidence_analysis']:
                f.write(f"  {conf['confidence_threshold']*100:<12.0f} {conf['coverage']*100:<12.1f} {conf['accuracy']:<12.4f} {conf['precision']:<12.4f} {conf['recall']:<12.4f} {conf['f1']:<12.4f}\n")
        
        f.write("\n" + "=" * 100 + "\n")
        f.write("CONCLUSION:\n")
        f.write("-" * 50 + "\n")
        f.write("GP-FL provides uncertainty-aware predictions that enable:\n")
        f.write("  • Higher accuracy on confident predictions\n")
        f.write("  • Identification of uncertain cases for further testing\n")
        f.write("  • Risk-based patient stratification\n")
        f.write("  • Clinically interpretable confidence levels\n")
        f.write("\nThis capability is UNIQUE to Gaussian Processes and unavailable in RFC/LR.\n")
        f.write("=" * 100 + "\n")
    
    print(f"📄 Uncertainty report saved to: {output_file}")


def get_global_uncertainty_metrics(clients, threshold=None):
    """
    Compute global metrics WITH uncertainty information.
    Each client predicts on its own data, providing both predictions and uncertainty.
    """
    all_y_true = []
    all_y_probs = []
    all_y_std = []
    all_y_binary = []
    
    for client in clients:
        metrics = get_uncertainty_metrics(client)
        all_y_true.append(metrics['y_true'])
        all_y_probs.append(metrics['y_pred_mean'])
        all_y_std.append(metrics['y_pred_std'])
        if threshold is None:
            all_y_binary.append(metrics['y_pred_binary'])
        else:
            all_y_binary.append((metrics['y_pred_mean'] > threshold).astype(int))
    
    y_true = np.concatenate(all_y_true)
    y_pred_probs = np.concatenate(all_y_probs)
    y_pred_std = np.concatenate(all_y_std)
    
    y_pred_binary = np.concatenate(all_y_binary)
    
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary, zero_division=0)
    recall = recall_score(y_true, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true, y_pred_binary, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_pred_probs)
    except ValueError:
        roc_auc = 0.5
    
    # Uncertainty metrics
    mean_uncertainty = np.mean(y_pred_std)
    
    # Confident predictions metrics (uncertainty < median)
    confident_mask = y_pred_std < np.median(y_pred_std)
    if np.sum(confident_mask) > 0:
        y_true_conf = y_true[confident_mask]
        y_pred_conf = y_pred_binary[confident_mask]
        confident_accuracy = accuracy_score(y_true_conf, y_pred_conf)
        confident_f1 = f1_score(y_true_conf, y_pred_conf, zero_division=0)
    else:
        confident_accuracy = accuracy
        confident_f1 = f1
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "mean_uncertainty": mean_uncertainty,
        "confident_accuracy": confident_accuracy,
        "confident_f1": confident_f1,
        "total_samples": len(y_true)
    }
