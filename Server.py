import numpy as np

def weighted_average_aggregation(params_list, sizes_list):
    """
    Makaledeki gibi veri sayısına göre ağırlıklı ortalama alır.
    : "weighting the cluster-specific coefficients according to their sample sizes"
    """
    total_samples = sum(sizes_list)
    weighted_params = np.zeros_like(params_list[0])
    
    for params, size in zip(params_list, sizes_list):
        weight = size / total_samples
        weighted_params += (params * weight)
        
    return weighted_params