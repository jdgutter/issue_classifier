import torch
import numpy as np
import onnxruntime as ort
import os
from src.config import settings
from src.ranking import IssueRankingModel

def export_to_onnx(pytorch_path: str, onnx_path: str) -> None:
    """
    Loads a trained PyTorch IssueRankingModel and exports it to ONNX format
    with dynamic batch axes. Performs numerical validation between the 
    PyTorch and ONNX models.
    """
    if not os.path.exists(pytorch_path):
        raise FileNotFoundError(f"Trained PyTorch model not found at {pytorch_path}")

    print(f"Loading PyTorch checkpoint from {pytorch_path}...")
    checkpoint = torch.load(pytorch_path, map_location="cpu")
    
    # Instantiate the model using the serialized structural configurations
    model = IssueRankingModel(
        num_tags=checkpoint.get("num_tags", 10),
        tag_embed_dim=checkpoint.get("tag_embed_dim", 8),
        embedding_dim=checkpoint.get("embedding_dim", 384)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Define dummy inputs representing a single candidate batch item
    dummy_cont = torch.randn(1, 3, dtype=torch.float32)
    dummy_cat = torch.zeros(1, 2, dtype=torch.long)
    dummy_emb = torch.randn(1, 384, dtype=torch.float32)

    # Ensure the parent directory of target ONNX path exists
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    print(f"Exporting model to ONNX format at {onnx_path}...")
    torch.onnx.export(
        model,
        (dummy_cont, dummy_cat, dummy_emb),
        onnx_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=["continuous_features", "categorical_features", "text_embeddings"],
        output_names=["logits"],
        dynamic_axes={
            "continuous_features": {0: "batch_size"},
            "categorical_features": {0: "batch_size"},
            "text_embeddings": {0: "batch_size"},
            "logits": {0: "batch_size"}
        }
    )
    print("ONNX model successfully serialized.")

    # --- Numerical Validation Step ---
    print("Verifying model numerical equivalence using ONNX Runtime...")
    ort_session = ort.InferenceSession(onnx_path)

    # Execute PyTorch inference
    with torch.no_grad():
        pytorch_logits = model(dummy_cont, dummy_cat, dummy_emb).numpy()

    # Execute ONNX Runtime inference
    ort_inputs = {
        "continuous_features": dummy_cont.numpy(),
        "categorical_features": dummy_cat.numpy(),
        "text_embeddings": dummy_emb.numpy()
    }
    ort_outs = ort_session.run(None, ort_inputs)
    onnx_logits = ort_outs[0]

    # Validate output tensors match within strict tolerance
    np.testing.assert_allclose(pytorch_logits, onnx_logits, rtol=1e-5, atol=1e-5)
    print("Validation passed: PyTorch and ONNX models output numerically equivalent logits.")

if __name__ == "__main__":
    pytorch_model_path = str(settings.RANKER_MODEL_PATH)
    # Define ONNX target file by swapping extension in settings or adding to models/
    onnx_model_path = str(settings.MODELS_DIR / "ranking_model.onnx")
    
    export_to_onnx(pytorch_model_path, onnx_model_path)
