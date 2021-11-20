# config.py
USE_FL = False
cfg = {
    'name': 'FaceCPU',
    'feature_maps': [[64, 64], [32, 32], [16, 16], [8, 8]],
    'min_dim': 1024,
    'steps': [16, 32, 64, 128],
    'min_sizes': [[8, 16], [32, 64, 96, 128], [192, 256], [384, 512]],
    'aspect_ratios': [[1], [1], [1], [1]],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 1.0,
    'conf_weight': 5.0,
    'gpu_train': True
    
}
