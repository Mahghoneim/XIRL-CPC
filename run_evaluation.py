
import yaml
import os

# Create metadata.yaml
metadata = {
    'algo': 'xirl',
    'embodiment': 'gripper',
    'mode': 'same'  # from experiment name: 'same_algo'
}

exp_dir = 'C:/tmp/xirl/pretrain_runs/dataset=xmagical_mode=same_algo=xirl_embodiment=gripper_uid=99a3c9d6-35e7-46ea-ad5b-eea016f01612'
metadata_path = os.path.join(exp_dir, 'metadata.yaml')

with open(metadata_path, 'w') as f:
    yaml.dump(metadata, f)

print(f'Created metadata.yaml at {metadata_path}')
print('Contents:')
print(yaml.dump(metadata, default_flow_style=False))
