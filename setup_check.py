"""
Setup verification script to check if all dependencies are installed correctly.
"""

import sys

def check_import(module_name, package_name=None):
    """Check if a module can be imported"""
    if package_name is None:
        package_name = module_name
    
    try:
        __import__(module_name)
        print(f"✓ {package_name} is installed")
        return True
    except ImportError:
        print(f"✗ {package_name} is NOT installed")
        return False

def check_flash_attn():
    """Check FlashAttention-2 specifically"""
    try:
        import flash_attn
        print(f"✓ flash-attn is installed (version: {flash_attn.__version__})")
        
        # Try to import the function we need
        from flash_attn import flash_attn_func
        print("✓ flash_attn_func can be imported")
        return True
    except ImportError as e:
        print(f"✗ flash-attn is NOT installed or not working: {e}")
        print("  Install with: pip install flash-attn --no-build-isolation")
        return False

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA is available")
            print(f"  Device: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            return True
        else:
            print("✗ CUDA is NOT available (CPU only)")
            return False
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        return False

def main():
    print("=" * 60)
    print("Checking training setup...")
    print("=" * 60)
    
    all_ok = True
    
    print("\n1. Core dependencies:")
    all_ok &= check_import("torch", "PyTorch")
    all_ok &= check_import("transformers", "transformers")
    all_ok &= check_import("datatrove", "datatrove")
    all_ok &= check_import("yaml", "PyYAML")
    all_ok &= check_import("tqdm", "tqdm")
    
    print("\n2. FlashAttention-2:")
    all_ok &= check_flash_attn()
    
    print("\n3. CUDA:")
    all_ok &= check_cuda()
    
    print("\n4. Model files:")
    try:
        from model import create_1b_model
        print("✓ model.py can be imported")
    except Exception as e:
        print(f"✗ Error importing model.py: {e}")
        all_ok = False
    
    try:
        from data_loader import create_dataloader
        print("✓ data_loader.py can be imported")
    except Exception as e:
        print(f"✗ Error importing data_loader.py: {e}")
        all_ok = False
    
    try:
        from train import train
        print("✓ train.py can be imported")
    except Exception as e:
        print(f"✗ Error importing train.py: {e}")
        all_ok = False
    
    print("\n" + "=" * 60)
    if all_ok:
        print("✓ All checks passed! Ready for training.")
    else:
        print("✗ Some checks failed. Please install missing dependencies.")
        print("\nInstall with:")
        print("  pip install -r requirements.txt")
        print("  bash install_flash_attn.sh")
        print("  # OR: pip install flash-attn --no-build-isolation")
    print("=" * 60)
    
    return 0 if all_ok else 1

if __name__ == '__main__':
    sys.exit(main())

