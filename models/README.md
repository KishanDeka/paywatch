Versioned runs are written here by `python -m paywatch.train`. Output directories must be empty.
Artifacts include both ONNX models, scaler, policy/schema/checksum manifest, training checkpoints,
evaluation results, predictions, and drift reference. The service needs only manifest.json,
scaler.npz, autoencoder.onnx, and xgboost.onnx. Checksums detect accidental inconsistency;
they are not an authenticity signature.
