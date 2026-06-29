import os
import torch
import onnxruntime as ort
from src.ranking import IssueRankingModel
from src.training.export_onnx import export_to_onnx

def test_onnx_export_flow(tmp_path):
    # 1. Create a dummy trained PyTorch checkpoint in a temporary directory
    pytorch_path = str(tmp_path / "mock_ranking_model.pt")
    onnx_path = str(tmp_path / "mock_ranking_model.onnx")
    
    model = IssueRankingModel(num_tags=10, tag_embed_dim=8, embedding_dim=384)
    state = {
        "model_state_dict": model.state_dict(),
        "num_tags": 10,
        "tag_embed_dim": 8,
        "embedding_dim": 384,
        "scaling_stats": {
            "clicks_mean": 0.0, "clicks_std": 1.0,
            "pop_mean": 0.0, "pop_std": 1.0,
            "age_mean": 0.0, "age_std": 1.0
        }
    }
    torch.save(state, pytorch_path)
    
    # 2. Run the export to ONNX function
    assert not os.path.exists(onnx_path)
    export_to_onnx(pytorch_path, onnx_path)
    assert os.path.exists(onnx_path)
    
    # 3. Verify ONNX model properties using ONNX Runtime
    session = ort.InferenceSession(onnx_path)
    
    # Verify input specifications
    input_details = session.get_inputs()
    assert len(input_details) == 3
    input_names = [inp.name for inp in input_details]
    assert "continuous_features" in input_names
    assert "categorical_features" in input_names
    assert "text_embeddings" in input_names
    
    # Verify dynamic batch shapes (first dim is dynamic/string or None)
    # The batch size axis is set dynamically, typically returned as None or 'batch_size'
    for inp in input_details:
        assert len(inp.shape) in [2, 3] # batch_size + feature dimensions
        assert inp.shape[0] is None or isinstance(inp.shape[0], str)
        
    # Verify output specifications
    output_details = session.get_outputs()
    assert len(output_details) == 1
    assert output_details[0].name == "logits"
    assert output_details[0].shape[0] is None or isinstance(output_details[0].shape[0], str)
