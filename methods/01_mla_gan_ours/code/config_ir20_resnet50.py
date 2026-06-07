"""IR=20 + ResNet50 引導分類器變體 config。

差異 vs 預設 config.py:
  - dataset_root  -> 999_mlagan_Export_2026/data
  - ct_dir = ct_256_ir20, mask_dir = masks_ir20  (IR=20 子集)
  - pretrained_classifier -> 999/data/classifiers/resnet50_v3_ir20/resnet50_v3_best.pth
  - classifier_arch = 'resnet50'
  - output_dir -> ./output_ir20_resnet50/
"""
from dataclasses import dataclass
from pathlib import Path

from config import MLAGANConfig

_HERE = Path(__file__).resolve().parent
_999_DATA = _HERE.parents[2] / 'data'


@dataclass
class MLAGANConfigIR20ResNet50(MLAGANConfig):
    dataset_root: str = str(_999_DATA)
    ct_dir: str = "ct_256_ir20"
    mask_dir: str = "masks_ir20"
    pretrained_classifier: str = str(_999_DATA / "classifiers" / "resnet50_v3_ir20" / "resnet50_v3_best.pth")
    classifier_arch: str = 'resnet50'
    output_dir: str = str(_HERE / "output_ir20_resnet50")
    kfold_bd_json: str = str(_999_DATA / "classifiers" / "kfold_bd_ir20.json")
