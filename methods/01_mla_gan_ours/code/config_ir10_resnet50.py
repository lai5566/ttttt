"""IR=10 + ResNet50 引導分類器變體 config。

差異 vs 預設 config.py:
  - dataset_root  -> 999_mlagan_Export_2026/data  (乾淨 baseline，無 _synth 污染)
  - ct_dir = ct_256, mask_dir = masks
  - pretrained_classifier -> 999/data/classifiers/resnet50_v3_ir10/resnet50_v3_best.pth
  - classifier_arch = 'resnet50'
  - output_dir -> ./output_ir10_resnet50/
"""
from dataclasses import dataclass
from pathlib import Path

from config import MLAGANConfig

_HERE = Path(__file__).resolve().parent
_999_ROOT = _HERE.parents[2]
_999_DATA = _999_ROOT / 'data'


@dataclass
class MLAGANConfigIR10ResNet50(MLAGANConfig):
    dataset_root: str = str(_999_DATA)
    ct_dir: str = "ct_256"
    mask_dir: str = "masks"
    pretrained_classifier: str = str(_999_DATA / "classifiers" / "resnet50_v3_ir10" / "resnet50_v3_best.pth")
    classifier_arch: str = 'resnet50'
    output_dir: str = str(_HERE / "output_ir10_resnet50")
    kfold_bd_json: str = str(_999_DATA / "classifiers" / "kfold_bd.json")
