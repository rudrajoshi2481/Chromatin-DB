#!/usr/bin/env python3
"""
Generate detailed model structure for MQ-VAE
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from model import MQVAE

def main():
    # Create model with default configuration
    model = MQVAE()
    
    # Print model structure
    print("=" * 80)
    print("MQ-VAE Model Structure")
    print("=" * 80)
    print(model)
    print("=" * 80)
    
    # Print parameter counts
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # Print component-wise parameter counts
    print("\n" + "=" * 80)
    print("Component Parameter Counts")
    print("=" * 80)
    
    components = {
        'encoder': model.encoder,
        'masker': model.masker,
        'vq': model.vq,
        'demasker': model.demasker,
        'decoder': model.decoder,
        'contact_head': model.contact_head,
        'classifier_head': model.classifier_head
    }
    
    for name, component in components.items():
        if component is not None:
            params = sum(p.numel() for p in component.parameters())
            print(f"{name:20s}: {params:,} parameters")
    
    print("=" * 80)

if __name__ == "__main__":
    main()
