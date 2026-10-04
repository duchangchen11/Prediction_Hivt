# HiVT environment

env=ped_intent
torch=2.5.1+cu124
cuda_runtime=12.4
cuda_available=True
GPU=NVIDIA GeForce RTX 3080
torch_geometric=2.6.1
torch_scatter=2.1.2+pt25cu124
torch_sparse=0.6.18+pt25cu124
training_wrapper=native PyTorch
lightning=not installed; not used
existing_package_changes={}

New PyG packages installed using constraints pinning all previously installed packages. scatter/sparse use PyTorch2.5 CUDA12.4 wheels. CUDA scatter kernel verified. No new environment, no torch/CUDA downgrade, no core package upgrade.
