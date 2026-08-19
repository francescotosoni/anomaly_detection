import os
import torch
import torch.nn as nn
import onnx
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.train_CNN import CNNAutoencoder
from models.train_LSTM import LSTMAutoencoder
from models.train_MLP import MLPAutoencoder
from models.train_Transformer import TransformerAutoencoder


class AnomalyScoreWrapper(nn.Module):
    """
    Wrapper generico per LSTM e Transformer.
    Input:  (batch, 50, 24) — segnale grezzo
    Output: (batch, 1)      — anomaly score (MSE tra input e ricostruzione)
    """
    def __init__(self, autoencoder):
        super().__init__()
        self.autoencoder = autoencoder
    
    def forward(self, x):
        reconstructed = self.autoencoder(x)
        # Per-sample MSE: mean over every dimension except batch
        mse = ((x - reconstructed) ** 2).mean(dim=[1, 2], keepdim=False)
        return mse.unsqueeze(1)  # (batch, 1)


class CNNAnomalyScoreWrapper(nn.Module):
    """
    Wrapper for the CNN (which works channels-first).
    Input:  (batch, 50, 24) - raw signal (same as every other model)
    Output: (batch, 1)      — anomaly score (MSE)
    
    Internally transposes (batch, 50, 24) -> (batch, 24, 50) for the CNN,
    then computes the MSE and returns a scalar.
    """
    def __init__(self, cnn_model):
        super().__init__()
        self.cnn = cnn_model
    
    def forward(self, x):
        # x: (batch, 50, 24) — formato standard time-first
        x_transposed = x.transpose(1, 2)           # (batch, 24, 50) — channels-first per CNN
        reconstructed = self.cnn(x_transposed)      # (batch, 24, 50)
        mse = ((x_transposed - reconstructed) ** 2).mean(dim=[1, 2], keepdim=False)
        return mse.unsqueeze(1)  # (batch, 1)


class MLPAnomalyScoreWrapper(nn.Module):
    """
    Wrapper for the MLP with built-in feature engineering + MSE.
    Input:  (batch, 50, 24) — segnale grezzo
    Output: (batch, 1)      — anomaly score (MSE sulle 95 feature tabulari)
    """
    def __init__(self, mlp_model, features_to_drop):
        super().__init__()
        self.mlp = mlp_model
        all_indices = list(range(24))
        self.keep_indices = [i for i in all_indices if i not in features_to_drop]
    
    def forward(self, x):
        # x: (batch, 50, 24)
        
        # 1. Feature masking -> (batch, 50, 19)
        x = x[:, :, self.keep_indices]
        
        # 2. Estrazione statistica
        mean_v = x.mean(dim=1)
        std_v = x.std(dim=1, correction=0)
        max_v = x.max(dim=1).values
        min_v = x.min(dim=1).values
        rms_v = torch.sqrt((x ** 2).mean(dim=1))
        
        # 3. Concatenazione -> (batch, 95)
        features = torch.cat([mean_v, std_v, max_v, min_v, rms_v], dim=1)
        
        # 4. Forward + MSE
        reconstructed = self.mlp(features)
        mse = ((features - reconstructed) ** 2).mean(dim=1, keepdim=True)
        return mse  # (batch, 1)

def export_model_to_onnx(model, model_name, dummy_input, output_dir):
    model.eval()
    onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
    
    print(f"\nEsportazione di {model_name} in ONNX...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamo=False,           # Forza l'exporter legacy (TorchScript) per compatibilita' Edge Impulse
        external_data=False
    )
    print(f"[OK] {model_name} exported successfully to: {onnx_path}")
    
    # Verify the generated ONNX model
    try:
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print(f"[OK] The ONNX checker confirms that {model_name}.onnx is valid.")
    except Exception as e:
        print(f"[X] Error while validating {model_name}.onnx: {e}")

def main():
    device = torch.device('cpu')
    output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'onnx'))
    os.makedirs(output_dir, exist_ok=True)
    
    # ---------------------------------------------------------
    # ARCHITECTURE PARAMETERS (in sync with evaluate_metrics)
    # ---------------------------------------------------------
    WINDOW_SIZE = 50
    N_FEATURES_SEQ = 24
    N_FEATURES_RAW = 24  # Numero totale di feature grezze dal robot (usato anche dal wrapper MLP)
    MLP_INPUT_DIM = 95
    
    # LSTM
    HIDDEN_DIM_LSTM = 32
    N_LAYERS_LSTM = 2
    DROPOUT_LSTM = 0.2
    
    # MLP
    BOTTLENECK_DIM_MLP = 32
    
    # Transformer
    D_MODEL_TR = 64
    NHEAD_TR = 2
    NUM_ENCODER_LAYERS_TR = 2
    NUM_DECODER_LAYERS_TR = 2
    DIM_FEEDFORWARD_TR = 128
    BOTTLENECK_DIM_TR = 32
    DROPOUT_TR = 0.1
    
    models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models'))
    
    # ---------------------------------------------------------
    # 1. CNN (wrapped: input time-first + MSE interno)
    # ---------------------------------------------------------
    cnn_path = os.path.join(models_dir, 'cnn_autoencoder.pth')
    if os.path.exists(cnn_path):
        cnn_model = CNNAutoencoder(N_FEATURES_SEQ).to(device)
        cnn_model.load_state_dict(torch.load(cnn_path, map_location=device, weights_only=True))
        wrapped_cnn = CNNAnomalyScoreWrapper(cnn_model).to(device)
        # Input unificato: (batch, time, features) = (1, 50, 24)
        dummy_cnn = torch.randn(1, WINDOW_SIZE, N_FEATURES_RAW).to(device)
        export_model_to_onnx(wrapped_cnn, "cnn_autoencoder", dummy_cnn, output_dir)
        
    # ---------------------------------------------------------
    # 2. LSTM (wrapped: MSE interno)
    # ---------------------------------------------------------
    lstm_path = os.path.join(models_dir, 'lstm_autoencoder.pth')
    if os.path.exists(lstm_path):
        lstm_model = LSTMAutoencoder(
            N_FEATURES_SEQ, HIDDEN_DIM_LSTM, 
            seq_len=WINDOW_SIZE, n_layers=N_LAYERS_LSTM, dropout=DROPOUT_LSTM
        ).to(device)
        lstm_model.load_state_dict(torch.load(lstm_path, map_location=device, weights_only=True))
        wrapped_lstm = AnomalyScoreWrapper(lstm_model).to(device)
        dummy_lstm = torch.randn(1, WINDOW_SIZE, N_FEATURES_SEQ).to(device)
        export_model_to_onnx(wrapped_lstm, "lstm_autoencoder", dummy_lstm, output_dir)
        
    # ---------------------------------------------------------
    # 3. MLP (wrapped: feature engineering + MSE interno)
    # ---------------------------------------------------------
    mlp_path = os.path.join(models_dir, 'mlp_autoencoder.pth')
    if os.path.exists(mlp_path):
        mlp_model = MLPAutoencoder(
            input_dim=MLP_INPUT_DIM, bottleneck_dim=BOTTLENECK_DIM_MLP
        ).to(device)
        mlp_model.load_state_dict(torch.load(mlp_path, map_location=device, weights_only=True))
        TAB_FEATURES_TO_DROP = [9, 14, 15, 16, 17]
        wrapped_mlp = MLPAnomalyScoreWrapper(mlp_model, TAB_FEATURES_TO_DROP).to(device)
        dummy_mlp = torch.randn(1, WINDOW_SIZE, N_FEATURES_RAW).to(device)
        export_model_to_onnx(wrapped_mlp, "mlp_autoencoder", dummy_mlp, output_dir)
        
    # ---------------------------------------------------------
    # 4. Transformer (wrapped: MSE interno)
    # ---------------------------------------------------------
    tr_path = os.path.join(models_dir, 'transformer_autoencoder.pth')
    if os.path.exists(tr_path):
        tr_model = TransformerAutoencoder(
            N_FEATURES_SEQ, d_model=D_MODEL_TR, nhead=NHEAD_TR,
            num_encoder_layers=NUM_ENCODER_LAYERS_TR, num_decoder_layers=NUM_DECODER_LAYERS_TR,
            dim_feedforward=DIM_FEEDFORWARD_TR, bottleneck_dim=BOTTLENECK_DIM_TR,
            seq_len=WINDOW_SIZE, dropout=DROPOUT_TR
        ).to(device)
        tr_model.load_state_dict(torch.load(tr_path, map_location=device, weights_only=True))
        wrapped_tr = AnomalyScoreWrapper(tr_model).to(device)
        dummy_tr = torch.randn(1, WINDOW_SIZE, N_FEATURES_SEQ).to(device)
        export_model_to_onnx(wrapped_tr, "transformer_autoencoder", dummy_tr, output_dir)
        
    print("\nDone! Every available model has been exported and verified.")

if __name__ == "__main__":
    main()
