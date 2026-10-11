import torch
import pytest
from rop.models.hybrid_model import HybridFuzzyROPModel

@pytest.fixture
def dummy_input():
    return torch.randn(2, 3, 224, 224)

def test_model_construction():
    model = HybridFuzzyROPModel(pretrained_backbone=False)
    assert model is not None
    
def test_forward_pass_shape(dummy_input):
    model = HybridFuzzyROPModel(pretrained_backbone=False, n_classes=5)
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape == (2, 5)
    assert torch.isfinite(output).all()
    
def test_gradient_flow(dummy_input):
    model = HybridFuzzyROPModel(pretrained_backbone=False)
    output = model(dummy_input)
    loss = output.sum()
    loss.backward()
    
    # Check if gradients exist for non-frozen layers
    has_grad = False
    for param in model.parameters():
        if param.requires_grad and param.grad is not None:
            has_grad = True
            break
    assert has_grad
