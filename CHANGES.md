# Changes Summary

## Overview
Updated the federated learning project to:
1. Remove all LFW (Labeled Faces in the Wild) dataset support
2. Remove pretrained model options - all models now train from scratch
3. Update CelebA to use official dataset structure with identity labels and evaluation partitions
4. Update VGGFace to use VGGFace2 with official train/val splits

## Detailed Changes

### 1. Configuration (`config.py`)
- **Removed**: LFW data path
- **Updated**: Dataset paths to use `./data/CelebA` and `./data/VGGFace2`

### 2. Model Architectures
All models now train from scratch (no pretrained weights):

#### `models/resnet.py`
- Removed `pretrained` parameter from `ResNet18Classifier.__init__()`
- Changed to `models.resnet18(pretrained=False)`
- Updated `create_resnet18()` factory function

#### `models/mobilenet.py`
- Removed `pretrained` parameter from `MobileNetV2Classifier.__init__()`
- Changed to `models.mobilenet_v2(pretrained=False)`
- Updated `create_mobilenetv2()` factory function

#### `models/__init__.py`
- Removed `pretrained` parameter from `create_model()` function

### 3. Client Dataset Handlers

#### `clients/celebA_client.py`
- **Updated** `CelebADataset` to use official CelebA structure:
  - Uses `list_eval_partition.txt` for train/val/test splits (0=train, 1=val, 2=test)
  - Loads identity labels from `Anno/identity_CelebA.txt`
  - Falls back to pseudo-identities if identity file not available
  - No longer uses manual split_dataset function
- **Updated** `load_celeba_data()` to create separate train/val/test datasets
- Data structure expected:
  ```
  CelebA/
  ├── img_align_celeba/
  ├── Anno/
  │   └── identity_CelebA.txt
  └── list_eval_partition.txt
  ```

#### `clients/vggface_client.py`
- **Renamed** from VGGFace to VGGFace2
- **Updated** `VGGFaceDataset` → `VGGFace2Dataset`
- Uses official VGGFace2 structure with train/val folders
- **Updated** `load_vggface_data()` → `load_vggface2_data()`
- **Updated** `explore_vggface()` → `explore_vggface2()`
- Data structure expected:
  ```
  VGGFace2/
  ├── train/
  │   ├── n000001/
  │   ├── n000002/
  │   └── ...
  └── val/
      ├── n000001/
      └── ...
  ```

#### `clients/lfw_client.py`
- **Deleted** entirely

#### `clients/__init__.py`
- Removed all LFW imports and references
- Updated `get_client_data()` to accept only 'celeba' or 'vggface2'
- Updated `explore_client_data()` accordingly

### 4. Main Scripts

#### `main.py`
- Updated `explore_datasets()` to only explore CelebA and VGGFace2
- Updated `print_project_info()` to show 2 clients instead of 3
- Added note about training from scratch

#### `train_local.py`
- Updated argument choices to `['celeba', 'vggface2']`

#### `run_federated.py`
- Updated `create_federated_clients()` to only create CelebA and VGGFace2 clients
- Removed LFW-specific logic

#### `evaluate.py`
- Updated client choices to `['celeba', 'vggface2']`
- Updated `generate_full_report()` to iterate over 2 clients

### 5. Centralized Training

#### `centralized/train_global.py`
- Removed LFW import
- Updated `combine_datasets()` to:
  - Only combine CelebA and VGGFace2
  - Use official train/val/test splits from each dataset
  - Create separate combined datasets for each split

### 6. Documentation

#### `README.md`
- Updated to reflect 2 clients (CelebA, VGGFace2)
- Added dataset structure requirements
- Noted all models train from scratch
- Updated all command examples

## Dataset Requirements

### CelebA
- Must have `img_align_celeba/` folder with images
- Must have `list_eval_partition.txt` for train/val/test splits
- Should have `Anno/identity_CelebA.txt` for identity labels (creates pseudo-identities if missing)

### VGGFace2
- Must have `train/` folder with person subdirectories
- Must have `val/` folder with person subdirectories
- Each person folder contains their face images

## Migration Notes

If you have existing code using the old structure:
1. Replace `'lfw'` with either `'celeba'` or `'vggface2'`
2. Remove `pretrained=True/False` arguments from model creation
3. Update dataset paths in `config.py`
4. Ensure datasets follow the required structure above
