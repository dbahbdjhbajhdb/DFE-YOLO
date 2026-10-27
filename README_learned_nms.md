# 几何 + ROI 分类器辅助 NMS：YOLO11 独立实验

这个实验直接实现“几何 + 视觉”版本，冻结检测器，仅训练一个框对分类器。
它预测的是“两个框是否对应两个真实目标”，不是遮挡程度；不需要遮挡标签。
当前是 PyTorch 实验原型，没有验证 VisDrone 上的精度收益、CUDA 延迟或部署导出。

## 结构

```text
冻结 YOLO11 一次前向
  ├─ NMS 前候选框：xyxy、分数、类别
  └─ backbone 第 2 层 P2 特征图（标准 YOLO11，stride=4）

同类重叠小框 A、B
  ├─ 16 维对称几何/分数特征
  ├─ A 的 3×3 ROI
  ├─ B 的 3×3 ROI
  └─ A、B 并集外扩 10% 的上下文 ROI

三个 ROI 共用 Linear(C×3×3,16) + LayerNorm + SiLU 编码器
拼接 [几何, (eA+eB)/2, |eA-eB|, eContext]，共 64 维
Linear(64,32) → SiLU → Linear(32,1) → sigmoid
```

当 P2 有 64 通道时，分类器为 11,377 个参数，约 1.14 万。它不是在原图上额外运行一个检测器。
A、B 交换顺序时，几何与融合特征保持不变。
ROI 可能包含邻近目标和背景，浅层特征也不保证一定提供足够区分信息。

## NMS 的替换位置

1. 与普通 class-aware NMS 一样，按置信度排序。
2. 只对 IoU 超过阈值、同类别、双方均为小目标的框对调用分类器。
3. 高置信度判断为不同目标时，取消这一次抑制；其他框对仍按原规则抑制。
4. 被保留的候选继续参加后续 NMS，仍可能被其他重复框抑制。

小目标定义为原始图像坐标下 `sqrt(box_area) <= small_size`，默认 32 像素。
输入缩放和 letterbox 通过现有验证器处理；ROI 在对应的网络输入坐标中提取。
分类器不改变检测分数，不跨类别抑制，也不恢复检测器没有产生的框。

## 第一步：从训练集导出框对

先 `git pull origin main`，然后在项目根目录的 PyCharm 终端运行。
下面是 PowerShell 单行命令，替换权重和数据配置路径：

```powershell
python learned_nms_experiment.py export --weights "runs/detect/你的训练目录/weights/best.pt" --data "C:/你的路径/VisDrone.yaml" --out "work/nms_pairs" --device 0
```

- 使用已有训练好的检测器。不要在这一步重新训练 YOLO11。
- 命令固定读取数据 YAML 的 train split，通过无数据增强的验证加载器获取候选和 GT。
- 每张图最多保存 64 个可靠框对，float16 保存 ROI；数千张图仍可能占数百 MB 至数 GB。
- 默认候选上限 3000、冲突框对上限 1024。超过框对预算的部分沿用原 NMS。
- 导出目录必须为空。中断后 meta.json 标为未完成，不能拿残缺导出直接训练。
- 导出过程会显示训练集上的检测统计，它不是独立评估成绩。

标签：

| 匹配关系 | 标签 |
|---|---|
| 两个框可靠匹配到不同 GT | 1：保留这对候选 |
| 两个框可靠匹配到同一个 GT | 0：不取消 NMS 抑制 |
| 一个可靠匹配、另一个明显为背景 | 0 |
| 定位或 GT 身份不确定 | 排除 |

可靠匹配默认要求同类 IoU≥0.5，且最佳与次佳匹配 IoU 相差≥0.1。
明显背景要求同类 GT 的最大 IoU≤0.1。不能把所有未匹配框都标成不同目标。
若有效正样本太少，应检查 proposal、GT 和采样预算；脚本不会捏造正样本来强行训练。
由于拥挤区本身更容易匹配歧义，这种保守标签会丢掉一部分最难的样本，存在分布偏差。

## 第二步：训练框对分类器

```powershell
python learned_nms_experiment.py train --pairs "work/nms_pairs" --out "runs/pair_classifier/exp1" --device 0 --epochs 30
```

默认按文件名第一个下划线前的场景前缀分组，80% 训练、20% 校准/模型选择。
VisDrone 等连续帧数据应核对 split.json，确保同场景没有跨组；错误分组可能造成泄漏。
若数据确实是独立图像且没有场景前缀，导出时可显式使用 `--grouping image`。
分类器分组隔离不意味着检测器没见过这些训练图像；真正的检测效果必须看独立 val/test。
训练与校准两组都必须含正负样本，否则报错并要求重新检查数据。

模型使用加权 BCE 和 AdamW。输出分数不视为校准后的真实概率。
在留出的框对上，选择保留精度达到 98%、至少正确保留 5 对的阈值，优先提高保留召回。
如果没有任何阈值达标，阈值设为 1.01，自动退回普通 NMS；这是无合格保留策略，不是训练成功的精度收益。
阈值是框对级启发式，不能保证最终检测误检率；最终仍需检测级验证。
best.pt 根据留出集的框对 AP 保存，这里的 pair_AP 不是目标检测 AP。

## 第三步：同一检测器对比两种后处理

```powershell
python learned_nms_experiment.py eval --mode nms --weights "runs/detect/你的训练目录/weights/best.pt" --data "C:/你的路径/VisDrone.yaml" --device 0
python learned_nms_experiment.py eval --mode learned --weights "runs/detect/你的训练目录/weights/best.pt" --pair-model "runs/pair_classifier/exp1/best.pt" --data "C:/你的路径/VisDrone.yaml" --device 0
```

两组使用相同多标签候选入口、候选预算、conf、IoU、max_det、检测器和验证数据。
不能直接与历史实验不同 NMS 设置下的结果对比。默认 max_det=1000，适用于密集场景实验。
配对模型绑定检测器文件的 SHA256，禁止换另一份检测器权重却继续沿用分类器。
imgsz、conf、iou、small-size、候选数、框对预算、特征层也必须与导出设置一致，否则报错。
如果设置了非默认参数，应在 export 和两次 eval 中一致地传入。

标准验证器报告总体 P、R、mAP50、mAP50-95 和各阶段时间，**不会自动报告 COCO AP_small**。
每次 eval 的保存目录中还会生成 `postprocess_summary.json`，记录指标、设置和各阶段耗时。
小目标收益需要额外用一致的尺寸分组评估确认，不能用总体指标代替。
多次调参应另设最终测试集，避免把反复查看的 val 当作独立结果。

## 速度和适用范围

- 参数量小不代表整体后处理快：冲突矩阵最坏 O(N²)，ROIAlign 和 Python 贪心循环也有开销。
- 分类器按最多 128 对分块计算，框对总量受 pair-cap 限制。框对截断可能漏掉可挽救目标。
- 该版本提供检测级验证入口，尚未集成普通 `yolo predict`、ONNX 或 TensorRT。
- 目前只支持标准非 end-to-end 的 YOLO 检测 .pt、class-aware NMS、batch=1、FP32。
- 特征层默认针对标准 YOLO11。自定义骨干必须确认第 2 层是合适的 BCHW 特征。
- 在相同设备、分辨率和候选设置下，测量充分预热后的延迟、峰值显存和密集场景的尾延迟。

## 代码与检查

核心：`ultralytics/utils/learned_nms.py`；实验入口：`learned_nms_experiment.py`。

```powershell
python tests/test_learned_nms.py
```

测试验证框对交换对称性、ROI 与梯度、GT 标签、场景隔离、未获合格阈值时回退，
以及“取消一次抑制后，候选仍能在后续去重”的行为。测试和合成管线检查不证明真实精度提升。
