# YOLO11 AdaptiveTFE：指标持平时如何诊断

两组均未加载预训练权重时，可以排除“只有基线用了预训练”的差异。
先检查已有结果和已训练模型；本工具不假定权重已经退化，也不承诺改模块就能提高精度。

## 先分析已有训练，不需要重训

在 PyCharm 项目终端执行，下列路径换成真实文件。PowerShell 路径有空格时保留双引号。

```powershell
python diagnose_tfe.py results "runs/detect/基线/results.csv" "runs/detect/改进版/results.csv"
python diagnose_tfe.py weights --weights "runs/detect/改进版/weights/best.pt" --source "C:/date/Visdrone2019/images/val2019" --max-images 32 --device 0
```

第一个命令分别报告最佳 mAP50-95 所在轮和最后一轮的 P、R、mAP50、mAP50-95。
最佳轮的验证指标不能当成独立测试集成绩。

第二个命令只推理，统计每个 AdaptiveTFE 的三路平均权重、每张图内的空间标准差、归一化熵等。
统计来自真正参与预测的 logits；跳过模型预热，不重复计算模块前向。

- 熵接近 1、空间变化很小且权重接近 1/3：门控可能与均值融合接近，需要用 mean 对照验证。
- 某一路长期占优：说明模型偏好该尺度，不能仅凭这个判断权重崩溃或设计错误。
- 有明显空间差异但指标持平：学出了尺度选择，但尚未证明选择对检测有收益。
- 统计包含背景和 letterbox 填充，不能据此判断目标区域权重，也不代表遮挡程度。

## 三组主要对照

| 组别 | 配置 | 目的 |
|---|---|---|
| baseline | `yolo11n.yaml`，解析到已有 `yolo11.yaml` | 原有三尺度 YOLO11 |
| mean | `yolo11n-tfe-mean.yaml` | 固定均分的三尺度 TFE |
| adaptive | `yolo11n-adaptive-tfe.yaml` | 逐位置自适应权重 |

新配置位于 `ultralytics/cfg/models/11/`，与聊天中给出的 YOLO11 拓扑一致。
保持 P3、P4、P5 三个检测输出，不使用此前 YOLOv8 的四检测头配置。
原 `AdaptiveTFE` 实现不变。

若已有 baseline/adaptive 实验与下面设置不同，不要直接拿它们和新实验比较。
三组必须使用同一数据划分、训练轮数、优化器、图像尺寸、batch 和增强设置。
下面显式固定 SGD，避免 auto 优化器随模型配置变化；从随机初始化开始训练。
deterministic=True 尽量减少随机差异，不保证不同硬件/版本完全一致。

```powershell
yolo detect train model=ultralytics/cfg/models/11/yolo11n.yaml data="C:/你的路径/VisDrone.yaml" epochs=200 imgsz=640 batch=4 device=0 workers=0 pretrained=False seed=0 deterministic=True optimizer=SGD lr0=0.01 patience=0 project=runs/tfe_ablation name=baseline_seed0
yolo detect train model=ultralytics/cfg/models/11/yolo11n-tfe-mean.yaml data="C:/你的路径/VisDrone.yaml" epochs=200 imgsz=640 batch=4 device=0 workers=0 pretrained=False seed=0 deterministic=True optimizer=SGD lr0=0.01 patience=0 project=runs/tfe_ablation name=mean_seed0
yolo detect train model=ultralytics/cfg/models/11/yolo11n-adaptive-tfe.yaml data="C:/你的路径/VisDrone.yaml" epochs=200 imgsz=640 batch=4 device=0 workers=0 pretrained=False seed=0 deterministic=True optimizer=SGD lr0=0.01 patience=0 project=runs/tfe_ablation name=adaptive_seed0
```

先分析已有实验再决定是否重训。需要检验小幅差异时，对 seed=1、2 重复匹配实验，比较均值和波动。
同 seed 不代表不同结构能得到逐参数相同的初始化。

## 可选的拼接对照

还提供 `yolo11n-tfe-concat.yaml`，用于检查加权求和的信息压缩是否值得进一步研究。
它在三路拼接后用 1×1 CBS 投影回 C，保持外部接口一致；并非论文原始 TFE，且参数量与加权版本不同。

结果判断：

1. mean 与 adaptive 都弱于 baseline：优先研究连接、对齐和通道压缩，不应只修改权重预测器。
2. adaptive 稳定优于 mean：说明自适应权重相对均分有收益；仍要单独与 baseline 比较。
3. 三者持平：当前实验未显示收益，可考虑保留原结构以节省开销。
4. concat 优于两种求和：支持继续调查求和的信息损失，但还需考虑额外投影层和参数量差异。

## 本地检查

```powershell
python tests/test_tfe_diagnostics.py
python tests/smoke_yolo11_tfe.py
```

整网检查使用随机输入和合成框，只验证四种配置的尺寸、梯度及检测损失可运行，不是精度实验。
