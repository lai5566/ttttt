# v3(原版 / 非 K-fold)4-cell 清單

建立:2026-06-06。method 01「原版(非 kfold)」分支,跑 4 個 cell,與 kfold 版並排比較。

- **訓練**:`train.py` 的 `train()`,BD = on-the-fly `'rank'`(base config 預設,**非 K-fold**)。
- **引導分類器**:ir5→`resnet50_v3_ir5`(本批新訓)、ir10→`resnet50_v3_ir10`、ir20→`resnet50_v3_ir20`。
- **生成 profile**:`pure_boundary`(bd≈0.05),與 `generate_*_kfold.sh` 一致 → 只差訓練。
- **eval**:`eval/evaluate_multi_v3.py`,archs = resnet50 / efficientnet_b0 / efficientnet_b3 / vit_small_patch16_224 / densenet121,runs=5(5 seed),epochs=20,含 Real-only。

| cell | train.sh | checkpoint | generate.sh | 生成圖 | eval results |
|---|---|---|---|---|---|
| ir5 bs8  | `train_ir5_v3.sh`       | `output_ir5_v3/run_seed7/best_model.pth`       | `generate_ir5_v3.sh`       | `generated_ir5_v3/`       | `eval/results/eval_mlagan_v3_ir5_5arch_5seed/` |
| ir5 bs16 | `train_ir5_v3_bs16.sh`  | `output_ir5_v3_bs16/run_seed7/best_model.pth`  | `generate_ir5_v3_bs16.sh`  | `generated_ir5_v3_bs16/`  | `eval/results/eval_mlagan_v3_ir5_bs16_5arch_5seed/` |
| ir10 bs16| `train_ir10_v3_bs16.sh` | `output_ir10_v3_bs16/run_seed7/best_model.pth` | `generate_ir10_v3_bs16.sh` | `generated_ir10_v3_bs16/` | `eval/results/eval_mlagan_v3_ir10_bs16_5arch_5seed/` |
| ir20 bs16| `train_ir20_v3_bs16.sh` | `output_ir20_v3_bs16/run_seed7/best_model.pth` | `generate_ir20_v3_bs16.sh` | `generated_ir20_v3_bs16/` | `eval/results/eval_mlagan_v3_ir20_bs16_5arch_5seed/` |

> 資料根:ir5→`data/ct_256_ir5`、ir10→`data/ct_256`、ir20→`data/ct_256_ir20`。
> 一鍵跑:`bash run_pipeline_v3_4cells.sh`(train→generate→eval,有 skip 守衛可續跑)。
> kfold 對照在 `eval/results/eval_mlagan_kfold_ir{5,20}_6arch_5seed/`。
