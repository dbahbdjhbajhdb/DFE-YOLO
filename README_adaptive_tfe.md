# AdaptiveTFE：TFE 与 ASFF 式空间加权融合

这是一个新的实验模块，不是原论文公开实现，也不是完整 SAIR-YOLO 复现。
实验配置以 YOLOv8-P2 为基础，加入两个渐进三尺度融合节点；不包含 SSFF、DyHead、DCNv2 或论文混合损失。

## 结构

输入按空间分辨率从高到低排列，三个输入通道数可以不同：

```text
高分辨率 → 1×1 CBS → MaxPool + AvgPool → 3×3 CBS → A_h ─┐
中分辨率 → 1×1 CBS ────────────────────────────→ A_m ─┼→ 空间加权求和 → 3×3 CBS
低分辨率 → 1×1 CBS → 最近邻上采样 → 3×3 CBS ──→ A_l ─┘

A_h、A_m、A_l → 分别压缩到 8 通道 → 拼接 → 1×1 Conv → 3 通道 logits
logits → 沿尺度维度 softmax → w_h、w_m、w_l
```

`Y = CBS(w_h * A_h + w_m * A_m + w_l * A_l)`。

- 三路对齐到中间输入的 H×W，通道统一为 C。
- 权重是 `[B,3,H,W]`，在每个像素上三路权重之和为 1；不是三个全局标量。
- 权重对通道共享，随图像内容和空间位置变化，通过检测损失端到端学习。
- 主特征融合为加权求和。只有预测权重所用的低维描述子使用拼接。
- 输出始终为 `[B,C,H,W]`；如果直接替换输出为 3C 的旧 TFE，必须修改下游输入通道。
- logits 卷积用小随机权重和零偏置初始化，让初始权重接近均分，同时让前级权重预测层第一步就能得到梯度。
- 为适配奇数尺寸，下采样使用自适应池化。这是实现选择，不是论文明确指定的池化超参数。
- 权重可以同时抑制两个不合适的尺度，但 softmax 必须将总权重分配给某个尺度，不能直接表示“所有尺度都无用”。

## 独立使用

`adaptive_tfe.py` 只依赖 PyTorch，可以单独复制到其他项目。

```python
import torch
from ultralytics.nn.modules import AdaptiveTFE

module = AdaptiveTFE([64, 128, 256], 128)
features = [
    torch.randn(2, 64, 80, 80),
    torch.randn(2, 128, 40, 40),
    torch.randn(2, 256, 20, 20),
]
output, weights = module.forward_with_weights(features)
# output: [2,128,40,40]; weights: [2,3,40,40]
# weights[:,0], weights[:,1], weights[:,2] 对应高、中、低分辨率来源。
```

展示权重时使用 `module.eval()` 和 `torch.no_grad()`，并将权重 detach 后保存。
应结合目标标注、误检位置一起分析权重图；高权重并不能单独证明该尺度有因果贡献。

## YOLO 实验配置

配置：`ultralytics/cfg/models/v8/yolov8-adaptive-tfe-p2.yaml`。
模型解析器已注册模块并处理宽度缩放。

两个节点依次融合：

1. `[P3, P4, P5] → F4`。
2. `[P2, P3, F4] → F3`。

后续保留 P2 融合和 PAN 的 Concat，用于只考察三输入 TFE 的融合方式。

从仓库根目录运行，换成真实数据集路径：

```bash
yolo detect train model=ultralytics/cfg/models/v8/yolov8-adaptive-tfe-p2.yaml data=/path/to/data.yaml imgsz=640 epochs=100 seed=0
```

默认选用 n 缩放。类别数由训练数据覆盖。不要假定旧检测头权重可以直接加载；需检查实际加载报告。

## 消融

将 YAML 中两个 AdaptiveTFE 的最后一个参数同步改成以下值：

| fusion | 方式 | 含义 |
|---|---|---|
| adaptive | 空间 softmax 加权求和 | 主实验 |
| mean | 三路固定 1/3 求和 | 检查收益是否来自自适应选择 |
| concat | 三路拼接后 1×1 CBS 投影回 C | 外部接口一致的拼接对照 |

concat 对照多一个投影层，参数量不同；它不是原论文 TFE 的精确复现。
先用相同数据划分、训练设置和随机种子，再对多个种子复验。
比较 mAP、AP_small、小目标召回率、背景误检，以及相同硬件上的延迟和峰值显存。
权重预测也有计算开销，不能只根据输出通道数或参数量推断加速。

## 验证

```bash
python tests/test_adaptive_tfe.py
PYTHONPATH=. python tests/smoke_adaptive_tfe_model.py
```

模块测试涵盖奇数及非方形尺寸、逐位置权重归一化、三路与权重预测层梯度、强制选择某个尺度时的实际融合结果、对照模式及 CPU bfloat16。
整网检查采用 nano、2 类、128×128 随机图像和合成框，验证四尺度输出、真实检测损失反向传播和推理。
这些检查不代表数据集上的检测效果，也未验证 CUDA、导出或设备部署。
