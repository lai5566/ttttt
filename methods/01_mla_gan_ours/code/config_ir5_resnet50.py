"""IR=5 + ResNet50 引導分類器變體 config。

差異 vs IR=10:
  - ct_dir = ct_256_ir5
  - mask_dir = masks_ir5
  - output_dir -> ./output_ir5_resnet50/
  - 引導分類器仍用 IR=5 訓的 ResNet50 v3（同一個 class 邊界）
"""
from dataclasses import dataclass
from pathlib import Path

from config import MLAGANConfig

_HERE = Path(__file__).resolve().parent
_999_DATA = _HERE.parents[2] / 'data'


@dataclass
class MLAGANConfigIR5ResNet50(MLAGANConfig):
    dataset_root: str = str(_999_DATA)
    ct_dir: str = "ct_256_ir5"
    mask_dir: str = "masks_ir5"
    pretrained_classifier: str = str(_999_DATA / "classifiers" / "resnet50_v3_ir5" / "resnet50_v3_best.pth")
    classifier_arch: str = 'resnet50'
    output_dir: str = str(_HERE / "output_ir5_resnet50")
    kfold_bd_json: str = str(_999_DATA / "classifiers" / "kfold_bd_ir5.json")
