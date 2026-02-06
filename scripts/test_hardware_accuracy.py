#!/usr/bin/env python3
"""
Test Hardware Simulator Accuracy

Compares hardware simulator output with original PyTorch modules
to quantify precision loss for each component.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from einops import rearrange

from integration import create_model_loader
from feature_extractor import (
    TransplatFeatureExtractor,
    MVSplatFeatureExtractor, 
    DepthSplatFeatureExtractor,
    CNNEncoderSimulator,
    TransformerSimulator,
    ViTSimulator,
    ViTConfig,
)
from feature_extractor.types import CNNConfig, TransformerConfig


def compute_psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute PSNR between two tensors."""
    mse = F.mse_loss(a.float(), b.float())
    if mse < 1e-10:
        return float('inf')
    return 10 * torch.log10(1.0 / mse).item()


def test_cnn_simulator(model_name: str) -> dict:
    """Test CNN simulator accuracy."""
    print(f"\n{'='*60}")
    print(f"Testing CNN Simulator for {model_name.upper()}")
    print('='*60)
    
    loader = create_model_loader(model_name)
    if model_name == 'transplat':
        ckpt = '/home/mazirui/transplat/SCARF/transplat/checkpoints/re10k.ckpt'
    elif model_name == 'mvsplat':
        ckpt = '/home/mazirui/transplat/SCARF/mvsplat/checkpoints/re10k.ckpt'
    else:
        ckpt = '/home/mazirui/transplat/SCARF/depthsplat/checkpoints/re10k.ckpt'
    
    bundle = loader.load_model(ckpt)
    model = bundle.model
    device = next(model.parameters()).device
    
    # Get backbone
    if model_name == 'depthsplat':
        backbone = model.encoder.depth_predictor.backbone
    else:
        backbone = model.encoder.backbone.cnet
    
    # Create simulator
    cnn_config = CNNConfig(input_channels=3, output_dim=128)
    cnn_sim = CNNEncoderSimulator(cnn_config, device)
    cnn_sim.load_from_pytorch(backbone)
    
    # Test input
    B, V = 1, 2
    H, W = 256, 256
    images = torch.rand(B, V, 3, H, W, device=device)
    
    # Normalize
    mean = torch.tensor([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).reshape(1, 1, 3, 1, 1).to(device)
    images_norm = (images - mean) / std
    concat = rearrange(images_norm, 'b v c h w -> (b v) c h w')
    
    with torch.no_grad():
        # Original PyTorch
        orig_output = backbone(concat)
        if isinstance(orig_output, list):
            orig_features = orig_output[::-1][0]
        else:
            orig_features = orig_output
        
        # Hardware simulator
        sim_features, cycles = cnn_sim.forward(concat)
    
    print(f"Original shape: {orig_features.shape}")
    print(f"Simulator shape: {sim_features.shape}")
    
    result = {
        'model': model_name,
        'component': 'CNN',
        'orig_shape': tuple(orig_features.shape),
        'sim_shape': tuple(sim_features.shape),
        'cycles': cycles,
    }
    
    if orig_features.shape == sim_features.shape:
        psnr = compute_psnr(orig_features, sim_features)
        result['psnr'] = psnr
        result['match'] = psnr > 100
        print(f"PSNR: {psnr:.2f} dB")
        if psnr > 100:
            print("✓ Bit-accurate match!")
        else:
            print(f"⚠ Precision loss detected")
    else:
        result['psnr'] = None
        result['match'] = False
        print("✗ Shape mismatch - architecture incompatible")
    
    return result


def test_vit_simulator() -> dict:
    """Test ViT simulator accuracy for DINOv2."""
    print(f"\n{'='*60}")
    print("Testing ViT Simulator for DepthSplat DINOv2")
    print('='*60)
    
    loader = create_model_loader('depthsplat')
    ckpt = '/home/mazirui/transplat/SCARF/depthsplat/checkpoints/re10k.ckpt'
    bundle = loader.load_model(ckpt)
    model = bundle.model
    device = next(model.parameters()).device
    
    dp = model.encoder.depth_predictor
    
    # Create ViT simulator
    vit_config = ViTConfig(vit_type='vits', patch_size=14)
    vit_sim = ViTSimulator(vit_config, device)
    vit_sim.load_from_dinov2(dp.pretrained)
    
    # Test input
    H, W = 252, 252  # Patch-aligned
    images = torch.rand(2, 3, H, W, device=device)
    
    layer_indices = [2, 5, 8, 11]
    
    with torch.no_grad():
        # Original DINOv2
        orig_features = list(dp.pretrained.get_intermediate_layers(
            images, layer_indices, return_class_token=False
        ))
        
        # Hardware simulator
        sim_features, cycles = vit_sim.get_intermediate_layers(
            images, layer_indices, return_class_token=False
        )
    
    print(f"Number of layers: {len(orig_features)}")
    
    result = {
        'model': 'depthsplat',
        'component': 'DINOv2_ViT',
        'cycles': cycles,
        'layer_psnrs': [],
    }
    
    all_match = True
    for i, (orig, sim) in enumerate(zip(orig_features, sim_features)):
        print(f"\nLayer {layer_indices[i]}:")
        print(f"  Original: {orig.shape}")
        print(f"  Simulator: {sim.shape}")
        
        if orig.shape == sim.shape:
            psnr = compute_psnr(orig, sim)
            result['layer_psnrs'].append(psnr)
            print(f"  PSNR: {psnr:.2f} dB")
            if psnr < 100:
                all_match = False
        else:
            result['layer_psnrs'].append(None)
            all_match = False
            print("  ✗ Shape mismatch")
    
    result['match'] = all_match
    if all_match:
        print("\n✓ All layers bit-accurate!")
    else:
        print("\n⚠ Precision loss detected in some layers")
    
    return result


def test_full_pipeline():
    """Test full feature extraction pipeline for all models."""
    print("\n" + "="*70)
    print("FULL PIPELINE TEST - Hardware vs Original")
    print("="*70)
    
    results = []
    
    for model_name in ['transplat', 'mvsplat', 'depthsplat']:
        print(f"\n{'='*60}")
        print(f"Testing {model_name.upper()}")
        print('='*60)
        
        loader = create_model_loader(model_name)
        if model_name == 'transplat':
            ckpt = '/home/mazirui/transplat/SCARF/transplat/checkpoints/re10k.ckpt'
        elif model_name == 'mvsplat':
            ckpt = '/home/mazirui/transplat/SCARF/mvsplat/checkpoints/re10k.ckpt'
        else:
            ckpt = '/home/mazirui/transplat/SCARF/depthsplat/checkpoints/re10k.ckpt'
        
        bundle = loader.load_model(ckpt)
        model = bundle.model
        device = next(model.parameters()).device
        
        # Create extractor
        if model_name == 'transplat':
            extractor = TransplatFeatureExtractor.from_encoder(model.encoder)
        elif model_name == 'mvsplat':
            extractor = MVSplatFeatureExtractor.from_encoder(model.encoder)
        else:
            extractor = DepthSplatFeatureExtractor.from_encoder(model.encoder)
        
        # Test input
        B, V, C, H, W = 1, 2, 3, 256, 256
        images = torch.rand(B, V, C, H, W, device=device)
        ext = torch.eye(4, device=device).unsqueeze(0).unsqueeze(0).expand(B, V, 4, 4).contiguous()
        
        with torch.no_grad():
            output = extractor.forward(images, ext)
        
        result = {
            'model': model_name,
            'trans_features': output.trans_features is not None,
            'cnn_cycles': output.cnn_cycles,
            'transformer_cycles': output.transformer_cycles,
            'total_cycles': output.total_cycles,
        }
        
        if hasattr(output, 'dinov2_cycles'):
            result['dinov2_cycles'] = output.dinov2_cycles
        
        if output.trans_features is not None:
            result['feature_shape'] = tuple(output.trans_features.shape)
        
        print(f"Features produced: {result['trans_features']}")
        if output.trans_features is not None:
            print(f"Feature shape: {result['feature_shape']}")
        print(f"CNN cycles: {result['cnn_cycles']:,}")
        print(f"Transformer cycles: {result['transformer_cycles']:,}")
        print(f"Total cycles: {result['total_cycles']:,}")
        
        results.append(result)
    
    return results


def main():
    print("="*70)
    print("SCARF Hardware Simulator Accuracy Test")
    print("="*70)
    
    # Test CNN for each model
    cnn_results = []
    for model_name in ['transplat', 'mvsplat']:
        try:
            result = test_cnn_simulator(model_name)
            cnn_results.append(result)
        except Exception as e:
            print(f"Error testing {model_name}: {e}")
    
    # Test ViT for DepthSplat
    vit_result = None
    try:
        vit_result = test_vit_simulator()
    except Exception as e:
        print(f"Error testing ViT: {e}")
        import traceback
        traceback.print_exc()
    
    # Test full pipeline
    pipeline_results = test_full_pipeline()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    print("\n### CNN Simulator")
    for r in cnn_results:
        status = "✓" if r.get('match') else "✗"
        psnr = r.get('psnr', 'N/A')
        if psnr == float('inf'):
            psnr = "∞"
        elif psnr is not None:
            psnr = f"{psnr:.1f}"
        print(f"  {status} {r['model']}: PSNR={psnr} dB, cycles={r['cycles']:,}")
    
    print("\n### ViT Simulator (DINOv2)")
    if vit_result:
        status = "✓" if vit_result.get('match') else "✗"
        psnrs = vit_result.get('layer_psnrs', [])
        avg_psnr = sum(p for p in psnrs if p is not None and p != float('inf')) / max(1, len([p for p in psnrs if p is not None and p != float('inf')]))
        print(f"  {status} depthsplat: avg PSNR={avg_psnr:.1f} dB, cycles={vit_result['cycles']:,}")
    
    print("\n### Full Pipeline Cycles")
    for r in pipeline_results:
        print(f"  {r['model']}: {r['total_cycles']:,} total cycles")
    
    print("\n" + "="*70)
    print("Note: PSNR > 100 dB = bit-accurate, PSNR < 30 dB = significant loss")
    print("="*70)


if __name__ == '__main__':
    main()
