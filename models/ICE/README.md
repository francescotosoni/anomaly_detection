# models/ICE — Artefatti sviluppati in laboratorio

Questa cartella contiene tutto ciò che è **nuovo e ricalibrato in lab**, tenuto
separato dai modelli benchmark accademici in `models/onnx/` (che restano
immutati e riproducibili rispetto al dataset di Graabæk et al.).

## Convenzione

Ogni modello ricalibrato viaggia con il suo scaler nella STESSA cartella, così
`bridge.py` carica automaticamente lo `scaler.json` accanto all'ONNX passato con
`--onnx-model`.

```
models/ICE/
├── mlp_autoencoder.onnx    # modello ricalibrato sui dati reali del lab
└── scaler.json             # parametri di normalizzazione (mean/std per categoria)
```

## Come generarli

```bash
# Ricalibra sull'ultimo CSV normale raccolto in lab
python lab_retrain.py

# Avvia il bridge sul modello ricalibrato (trova scaler.json da solo)
python models/deploy/bridge.py --robot-ip <IP> --mode pc \
    --onnx-model models/ICE/mlp_autoencoder.onnx
```

I modelli originali in `models/onnx/` continuano a funzionare con il loro
`scaler.json` (statistiche del dataset di training) e restano il riferimento
per il confronto col paper.
