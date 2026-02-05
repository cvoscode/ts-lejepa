"""
Test script to verify all encoders follow the BaseEncoder pattern consistently.
"""

import torch
from .rnn import LSTMEncoder, GRUEncoder
from .cnn import CNNEncoder
from .tcn import TCNEncoder
from .transformer import TransformerEncoder
from .mamba import MambaEncoder
from .graphlstm import GraphLSTMEncoder
from .covariate import CovariateEncoder, FusedEncoder


def test_encoder(encoder_class, name, **kwargs):
    """Test that an encoder follows the BaseEncoder pattern."""
    print(f"\n{'='*60}")
    print(f"Testing {name}")
    print(f"{'='*60}")
    
    # Create encoder
    encoder = encoder_class(input_channels=10, output_dim=64, **kwargs)
    
    # Verify attributes
    assert hasattr(encoder, 'input_channels'), f"{name} missing input_channels"
    assert hasattr(encoder, 'output_dim'), f"{name} missing output_dim"
    assert hasattr(encoder, 'pool_mode'), f"{name} missing pool_mode"
    assert hasattr(encoder, 'forward_backbone'), f"{name} missing forward_backbone"
    assert hasattr(encoder, '_init_weights'), f"{name} missing _init_weights"
    
    print(f"✓ Attributes check passed")
    
    # Test forward_backbone (should return [B, T, D])
    B, C, T = 4, 10, 50
    x = torch.randn(B, C, T)
    
    # Get backbone output
    with torch.no_grad():
        output = encoder.forward_backbone(x)
    
    assert output.dim() == 3, f"{name} forward_backbone should return 3D tensor, got {output.dim()}D"
    assert output.shape[0] == B, f"{name} batch dimension mismatch"
    assert output.shape[2] == 64, f"{name} output_dim mismatch"
    print(f"✓ forward_backbone returns correct shape: {tuple(output.shape)}")
    
    # Test different pool modes
    for pool_mode in ['mean', 'max', 'last', 'none']:
        encoder.pool_mode = pool_mode
        with torch.no_grad():
            pooled = encoder(x)
        
        if pool_mode == 'none':
            assert pooled.dim() == 3, f"{name} with pool_mode='none' should return 3D"
            assert pooled.shape == (B, output.shape[1], 64)
        else:
            assert pooled.dim() == 2, f"{name} with pool_mode='{pool_mode}' should return 2D"
            assert pooled.shape == (B, 64)
    
    print(f"✓ All pool modes work correctly")
    print(f"✅ {name} passed all tests!")


def main():
    print("Testing Encoder Consistency with BaseEncoder Pattern")
    print("=" * 60)
    
    # Test RNN encoders
    test_encoder(LSTMEncoder, "LSTMEncoder")
    test_encoder(GRUEncoder, "GRUEncoder")
    
    # Test CNN encoder
    test_encoder(CNNEncoder, "CNNEncoder")
    
    # Test TCN encoder
    test_encoder(TCNEncoder, "TCNEncoder")
    
    # Test Transformer encoder
    test_encoder(TransformerEncoder, "TransformerEncoder")
    
    # Test Mamba encoder
    test_encoder(MambaEncoder, "MambaEncoder")
    
    # Test Graph LSTM encoder (requires edge_index)
    print(f"\n{'='*60}")
    print("Testing GraphLSTMEncoder")
    print(f"{'='*60}")
    
    # Create a simple fully connected graph
    num_nodes = 10
    edge_index = torch.tensor([
        [i for i in range(num_nodes) for j in range(num_nodes) if i != j],
        [j for i in range(num_nodes) for j in range(num_nodes) if i != j]
    ], dtype=torch.long)
    
    encoder = GraphLSTMEncoder(
        input_channels=10, 
        output_dim=64,
        edge_index=edge_index
    )
    
    B, C, T = 4, 10, 50
    x = torch.randn(B, C, T)
    
    with torch.no_grad():
        output = encoder.forward_backbone(x)
    
    assert output.dim() == 3
    assert output.shape[0] == B
    assert output.shape[2] == 64
    print(f"✓ forward_backbone returns correct shape: {tuple(output.shape)}")
    print(f"✅ GraphLSTMEncoder passed tests!")
    
    # Test Covariate encoder
    print(f"\n{'='*60}")
    print("Testing CovariateEncoder")
    print(f"{'='*60}")
    
    cov_encoder = CovariateEncoder(input_channels=6, output_dim=32)
    
    B, L, F = 4, 50, 6
    time_features = torch.randn(B, L, F)
    dummy_x = torch.randn(B, 6, L)  # Dummy input
    
    with torch.no_grad():
        output = cov_encoder.forward_backbone(dummy_x, time_features=time_features)
    
    assert output.shape == (B, L, 32)
    print(f"✓ CovariateEncoder returns correct shape: {tuple(output.shape)}")
    print(f"✅ CovariateEncoder passed tests!")
    
    # Test Fused encoder
    print(f"\n{'='*60}")
    print("Testing FusedEncoder")
    print(f"{'='*60}")
    
    sensor_encoder = LSTMEncoder(input_channels=10, output_dim=64)
    cov_encoder = CovariateEncoder(input_channels=6, output_dim=32)
    fused_encoder = FusedEncoder(sensor_encoder, cov_encoder, output_dim=128)
    
    B, C, L = 4, 10, 50
    sensor_data = torch.randn(B, C, L)
    time_features = torch.randn(B, L, 6)
    
    with torch.no_grad():
        output = fused_encoder.forward_backbone(sensor_data, time_features=time_features)
    
    assert output.shape == (B, L, 128)
    print(f"✓ FusedEncoder returns correct shape: {tuple(output.shape)}")
    print(f"✅ FusedEncoder passed tests!")
    
    print("\n" + "=" * 60)
    print("🎉 ALL ENCODERS PASSED CONSISTENCY TESTS!")
    print("=" * 60)


if __name__ == "__main__":
    main()
