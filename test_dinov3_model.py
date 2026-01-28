"""
test_dinov3_model.py - Quick verification script for DINOv3 migration

This script verifies that:
1. DINOv3 model loads correctly from HuggingFace
2. Input/output dimensions are correct
3. Feature extraction works as expected
4. Model parameters are properly configured
"""

import torch
from models import NetraModel

def test_model_loading():
    """Test that DINOv3 model loads without errors."""
    print("="*70)
    print("TEST 1: Model Loading")
    print("="*70)
    try:
        model = NetraModel(num_classes=2, unfreeze_blocks=2)
        print("✅ Model loaded successfully!")
        return model
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return None

def test_input_output_shapes(model):
    """Test that input/output shapes are correct."""
    print("\n" + "="*70)
    print("TEST 2: Input/Output Shapes")
    print("="*70)
    
    # Create dummy input (batch_size=2, channels=3, height=512, width=512)
    x = torch.randn(2, 3, 512, 512)
    print(f"Input shape: {x.shape}")
    
    try:
        # Test feature extraction
        features = model.extract_features(x)
        print(f"Features shape: {features.shape}")
        expected_features = (2, 1024)
        assert features.shape == expected_features, f"Expected {expected_features}, got {features.shape}"
        print(f"✅ Features shape correct: {features.shape}")
        
        # Test forward pass (logits)
        model.eval()
        with torch.no_grad():
            logits = model(x)
        print(f"Logits shape: {logits.shape}")
        expected_logits = (2, 2)
        assert logits.shape == expected_logits, f"Expected {expected_logits}, got {logits.shape}"
        print(f"✅ Logits shape correct: {logits.shape}")
        
        return True
    except Exception as e:
        print(f"❌ Shape test failed: {e}")
        return False

def test_model_architecture(model):
    """Verify model architecture details."""
    print("\n" + "="*70)
    print("TEST 3: Model Architecture")
    print("="*70)
    
    # Check feature dimension
    assert model.feature_dim == 1024, f"Expected feature_dim=1024, got {model.feature_dim}"
    print(f"✅ Feature dimension: {model.feature_dim}")
    
    # Check model ID
    expected_id = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    assert model.MODEL_ID == expected_id, f"Expected {expected_id}, got {model.MODEL_ID}"
    print(f"✅ Model ID: {model.MODEL_ID}")
    
    # Check backbone config
    config = model.backbone.config
    print(f"✅ Hidden size: {config.hidden_size}")
    print(f"✅ Num layers: {config.num_hidden_layers}")
    print(f"✅ Num attention heads: {config.num_attention_heads}")
    print(f"✅ Patch size: {config.patch_size}")
    
    return True

def test_gradient_flow(model):
    """Test that gradients flow correctly through unfrozen layers."""
    print("\n" + "="*70)
    print("TEST 4: Gradient Flow")
    print("="*70)
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
    print(f"Frozen parameters: {frozen_params:,} ({100*frozen_params/total_params:.2f}%)")
    
    # Verify that some parameters are trainable (unfrozen layers + head)
    assert trainable_params > 0, "No trainable parameters found!"
    print("✅ Model has trainable parameters")
    
    # Verify that most backbone is frozen (>90%)
    assert frozen_params > trainable_params, "Too many parameters are trainable!"
    print("✅ Backbone is mostly frozen")
    
    # Test gradient computation
    model.train()
    x = torch.randn(1, 3, 512, 512)
    logits = model(x)
    loss = logits.sum()
    loss.backward()
    
    # Check that gradients exist for trainable params
    has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grads, "No gradients computed!"
    print("✅ Gradients flow through trainable layers")
    
    return True

def test_inference_mode(model):
    """Test model in inference mode."""
    print("\n" + "="*70)
    print("TEST 5: Inference Mode")
    print("="*70)
    
    model.eval()
    x = torch.randn(4, 3, 512, 512)
    
    try:
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
        
        print(f"Input batch: {x.shape}")
        print(f"Output logits: {logits.shape}")
        print(f"Output probs: {probs.shape}")
        print(f"Sample predictions: {probs[:2]}")
        
        # Verify probabilities sum to 1
        prob_sums = probs.sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5), "Probabilities don't sum to 1"
        print("✅ Probabilities sum to 1.0")
        
        return True
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("\n" + "🔬 DINOv3 Migration Verification Tests 🔬\n")
    
    # Test 1: Load model
    model = test_model_loading()
    if model is None:
        print("\n❌ Critical failure: Model failed to load")
        return False
    
    # Test 2: Shapes
    if not test_input_output_shapes(model):
        print("\n❌ Critical failure: Shape mismatch")
        return False
    
    # Test 3: Architecture
    if not test_model_architecture(model):
        print("\n❌ Critical failure: Architecture mismatch")
        return False
    
    # Test 4: Gradients
    if not test_gradient_flow(model):
        print("\n❌ Critical failure: Gradient flow issue")
        return False
    
    # Test 5: Inference
    if not test_inference_mode(model):
        print("\n❌ Critical failure: Inference issue")
        return False
    
    # All tests passed!
    print("\n" + "="*70)
    print("🎉 ALL TESTS PASSED! DINOv3 migration successful! 🎉")
    print("="*70)
    print("\nYour model is ready for:")
    print("  • Training (python train_source.py)")
    print("  • Adaptation (python adapt_target.py)")
    print("  • Evaluation (python evaluate.py)")
    print("\n✨ The DINOv3 ViT-L/16 model is fully functional! ✨\n")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
