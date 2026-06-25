import numpy as np

# Indici delle features (0-23):
# 0-5: actual_q_0 to 5
# 6-11: actual_current_0 to 5
# 12-17: actual_qd_0 to 5
# 18-23: actual_TCP_force_0 to 5

# Modelli sequenziali (CNN, LSTM, Transformer) ora usano tutte e 24 le feature fisiche per catturare ogni anomalia.
SEQ_FEATURES_TO_DROP = []

# Modelli tabulari (MLP) non capiscono le velocità istantanee (qd_2..5) e la corrente 3 (ultime classificate AutoGluon).
TAB_FEATURES_TO_DROP = [9, 14, 15, 16, 17]

def apply_feature_mask(data, model_type):
    """
    data: numpy array of shape (..., 24)
    model_type: 'sequential' or 'tabular'
    Returns data with 19 features.
    """
    if model_type == 'sequential':
        indices_to_drop = SEQ_FEATURES_TO_DROP
    elif model_type == 'tabular':
        indices_to_drop = TAB_FEATURES_TO_DROP
    else:
        raise ValueError("model_type must be 'sequential' or 'tabular'")
        
    all_indices = np.arange(data.shape[-1])
    keep_indices = np.setdiff1d(all_indices, indices_to_drop)
    
    return data[..., keep_indices]

def extract_tabular_features(windows):
    """
    windows: numpy array of shape (N, 50, F)
    Returns: numpy array of shape (N, F*5) containing Mean, Std, Max, Min, RMS
    """
    # 1. Mean (N, F)
    mean_v = np.mean(windows, axis=1)
    
    # 2. Std (N, F)
    std_v = np.std(windows, axis=1)
    
    # 3. Max (N, F)
    max_v = np.max(windows, axis=1)
    
    # 4. Min (N, F)
    min_v = np.min(windows, axis=1)
    
    # 5. RMS (N, F)
    rms_v = np.sqrt(np.mean(windows**2, axis=1))
    
    # Concat along feature axis
    engineered = np.concatenate([mean_v, std_v, max_v, min_v, rms_v], axis=1)
    return engineered
