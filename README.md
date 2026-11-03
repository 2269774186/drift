# YODA + Drifting 真实视频复现

本目录在不修改官方源码的前提下，将 YODA 的单步 SANA DiT + LoRA 去噪器替换为 Drifting 的条件单次前向生成器。当前已在服务器 RTX 4090 上使用真实 YODA 权重、真实 TA-AE/CLC 编解码、真实熵编码码流和真实视频帧完成小样本实验。

完整设计、消融、复现命令、指标和限制见 [docs/PLAN.md](docs/PLAN.md)。

## 当前结果

实验使用 `Akiyo` 的 300 帧，前 200 帧训练、后 100 帧 held-out。Drifting 分支的 v4 配置为 800 steps、batch size 8、hidden channels 64、depth 2、学习率 `1e-4`、重建权重 100。

| held-out 模式 | PSNR | MS-SSIM | LPIPS | DISTS | bpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| YODA original | 26.8037 | 0.920381 | 0.039073 | 0.061155 | 0.010803 |
| Drift identity | 21.1865 | 0.853360 | 0.092875 | 0.125448 | 0.012847 |
| Drift trained (v4) | 26.8291 | 0.920928 | 0.039201 | 0.061463 | 0.010855 |

v4 相对 YODA original：PSNR `+0.0254 dB`、MS-SSIM `+0.000547`、LPIPS `+0.000128`、DISTS `+0.000307`、bpp `+0.0000529`。因此 v4 已基本追平 YODA，但不能据此宣称感知质量或率失真性能明显超过 YODA。

## 完整视频

以下文件均为完整 300 帧视频，不是潜变量预览图：

- [2x2 视频对比](outputs/real_akiyo_yoda_drift_v4/comparison_2x2.mp4)：704x576
- [并排视频对比](outputs/real_akiyo_yoda_drift_v4/comparison_side_by_side.mp4)：1408x288
- [YODA original](outputs/real_akiyo_yoda_drift_v4/yoda_original.mp4)
- [Drift trained](outputs/real_akiyo_yoda_drift_v4/drift_trained.mp4)
- [Drift identity](outputs/real_akiyo_yoda_drift_v4/drift_identity.mp4)
- [原始视频](outputs/real_akiyo_yoda_drift_v4/original.mp4)
- [YODA 误差图视频](outputs/real_akiyo_yoda_drift_v4/error_yoda.mp4)
- [Drift 误差图视频](outputs/real_akiyo_yoda_drift_v4/error_drift.mp4)

所有 MP4 已用 `ffprobe` 验证为 H.264、`yuv420p`、300 帧、`10.01 s`；服务器与本地文件的 SHA256 已逐一核对一致。

## 实现

- `yoda_drift/drift_loss.py`：Drifting 多温度核漂移损失的 PyTorch 实现。
- `yoda_drift/memory_bank.py`：按条件上下文分桶的正样本与历史负样本 memory bank。
- `yoda_drift/drift_model.py`：接收量化潜变量和 TA-AE 多尺度上下文的单次前向 `ConditionalDriftGenerator`。
- `yoda_drift/trainer.py`：漂移损失与条件重建辅助损失。
- `yoda_drift/yoda_hybrid.py`：保留 YODA 编解码接口，支持 `original` / `drift` 两种 denoiser。
- `scripts/run_real_yoda_drift.py`：真实视频、真实 YODA 权重和真实码流的训练/评测入口。
- `scripts/run_real_yoda_drift_v4.sh`：v4 `stable_rec100` 固定配置启动脚本。

训练正样本是 YODA 原 SANA 分支输出的 `encoded["x_denoised"][0, :32]`，标记为 `yoda_original_sana_x_denoised`。它不是输入的 `lq_latent`。Drifting 分支保持 1 NFE，不加载 prompt embedding。

## 复现命令

服务器项目目录为 `/home/amax/Public/YODA`。完整实验可使用：

```bash
cd /home/amax/Public/YODA
bash scripts/run_real_yoda_drift_v4.sh --detach
```

训练与评测结束后生成 `summary.json`、`validation.json`、逐帧指标和完整 MP4。快速接口与算法验证：

```bash
cd /home/amax/Public/YODA
.venv/bin/python -m unittest discover -s tests -v
bash -n scripts/run_real_yoda_drift_v4.sh
```

当前共有 29 项 `unittest`。

## 实验边界

- 当前结果是 Akiyo 单视频、300 帧、200/100 帧训练/held-out 划分，不是 UVG、HEVC Class B 或 MCL-JCV 的完整 benchmark。
- YODA 的公开训练代码和原始训练损失不可得；本实现是 YODA 条件路径上的 Drifting 适配研究，不是论文原始训练配方的逐行复现。
- Drifting 官方实现为 JAX/Flax，本仓库使用数学等价的 PyTorch 移植。
- v4 只说明该小样本配置可稳定追平已有 YODA 输出；泛化能力仍需多视频基准验证。
- GPU0 测量期间存在其他用户进程竞争，延迟数字只代表本次实测，不能视为稳定加速比。
