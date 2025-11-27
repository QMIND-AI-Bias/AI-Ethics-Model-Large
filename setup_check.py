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
            capability = torch.cuda.get_device_capability(0)
            arch_tag = f"sm_{capability[0]}{capability[1]}"
            print(f"  Compute capability: {arch_tag}")
            
            get_arch_list = getattr(torch.cuda, "get_arch_list", None)
            compiled_arches = get_arch_list() if callable(get_arch_list) else []
            if compiled_arches:
                print(f"  PyTorch built for: {', '.join(compiled_arches)}")
                if arch_tag not in compiled_arches:
                    print(f"✗ Current PyTorch build does not include kernels for {arch_tag}.")
                    print("  Install a CUDA 12.4+ nightly build as described in docs/BLACKWELL.md.")
                    return False
            
            cuda_version = torch.version.cuda
            if capability[0] >= 12:
                try:
                    cuda_float = float(cuda_version)
                except (TypeError, ValueError):
                    cuda_float = None
                if cuda_float is not None and cuda_float < 12.4:
                    print("✗ CUDA runtime too old for Blackwell GPUs. Need CUDA 12.4+ build of PyTorch.")
                    print("  Follow docs/BLACKWELL.md to reinstall torch/flash-attn.")
                    return False
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

