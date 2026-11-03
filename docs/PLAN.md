# YODA 复现与 Drifting 去噪器替换计划及结果

## 1. 目标

本任务包含两部分：

1. 复现 YODA 的真实视频压缩推理与码流路径，并先在小规模真实视频上验证。
2. 将 YODA 原流程中的单步 SANA DiT + LoRA 替换为 Drifting 论文的一步条件漂移生成器。

验收重点是完整视频上的真实编解码结果，不以潜变量预览图或合成数据 smoke 作为最终结果。

## 2. 论文路径与替换关系

YODA 使用 TA-AE 提取参考帧多尺度上下文并编码当前帧潜变量，CLC 对潜变量进行条件熵编码，再由 SANA-Sprint DiT + LoRA 完成单步去噪，最后通过 TA-AE 解码。公开源码提供推理、码流和评估路径，但没有端到端训练代码和原始训练损失。

Drifting 从正样本和历史生成负样本构造多温度核漂移场，训练生成器用一次前向直接生成目标。官方实现为 JAX/Flax；本仓库将其核心损失与训练目标数学等价地移植到 PyTorch。

| YODA 原实现 | 本实现 |
| --- | --- |
| SANA DiT + LoRA | `ConditionalDriftGenerator` |
| `DiT(lq_latent_hat, prompt, ctxs)` | `G_phi(lq_latent_hat, ctxs, noise)` |
| 单步 SCM 去噪 | 条件漂移生成器单次前向，1 NFE |
| 原始 DiT 训练损失 | 漂移损失 + 条件重建辅助损失 |

YODA 的条件输入是量化压缩潜变量和 TA-AE 多尺度上下文，而 Drifting 原论文以类别条件生成为主。因此，本任务是“压缩条件化 Drifting”的适配扩展，不等同于严格复现任一论文的完整训练配方。

## 3. 实现结构

```text
yoda_drift/
├── drift_loss.py          # 多温度核漂移损失
├── memory_bank.py         # 按条件桶保存正/负样本
├── drift_model.py         # 条件一步漂移生成器
├── trainer.py             # 漂移训练步与重建辅助损失
└── yoda_hybrid.py         # TA-AE + CLC + 可切换 denoiser
scripts/
├── run_real_yoda_drift.py # 真实视频训练、码流和评测入口
├── run_real_yoda_drift_v4.sh
├── diagnose_drift_latents.py
└── ablate_real_drift.py
tests/                     # 29 项接口、损失、训练和真实路径测试
```

接口约定为：

```python
denoised = denoiser(lq_latent_hat, ctxs, noise=noise)
```

其中 `lq_latent_hat` 为 `[B, 32, H/32, W/32]`，`ctxs` 为 TA-AE 多尺度特征。`denoiser="original"` 保留原 SANA 路径，`denoiser="drift"` 使用条件漂移生成器且不加载 prompt embedding。

## 4. 真实数据和训练设计

### 数据与划分

- 视频：Akiyo。
- 总帧数：300。
- 训练：前 200 帧。
- held-out：后 100 帧。
- 输入分辨率：352x288。
- 真实模块：YODA TA-AE、CLC、SANA original 分支、熵编码和 `.bin` 码流。

这是单视频小样本可行性实验，不是完整 benchmark。后续仍需在 UVG、HEVC Class B、MCL-JCV 等数据集上按统一协议评测。

### 正样本定义

第一版适配错误地把输入 `lq_latent` 当成 YODA 原 SANA 分支的训练目标。修正后，正样本为：

```python
encoded["x_denoised"][0, :32]
```

其标记为 `yoda_original_sana_x_denoised`。该张量是 YODA 原始单步 SANA 去噪结果，而不是量化输入潜变量 `lq_latent_hat`，也不是未去噪的 `lq_latent`。

YODA 编码器、量化、熵编码和码流逻辑保持原样。Drifting 训练只替换单步去噪器对应的目标生成路径，使 trained 分支可以直接和 YODA original 在同一编解码协议下比较。

### 漂移训练目标

对生成样本 `gen`、正样本和 memory bank 中的历史负样本：

1. 使用 `R_list=(0.2, 0.05, 0.02)` 计算多温度核注意力。
2. 正样本形成吸引项，负样本形成排斥项。
3. 对漂移目标停止梯度。
4. 最小化生成样本到目标漂移的 MSE。
5. 增加条件重建辅助损失以稳定 YODA 潜变量适配。

## 5. 消融和版本迭代

| 版本 | 主要变化 | held-out PSNR | 结论 |
| --- | --- | ---: | --- |
| v1/v2 | 早期条件漂移适配 | - | 暴露正样本定义和稳定性问题 |
| v3 | 修正正样本前的对照训练 | 23.8815 | 比 YODA 低 2.9221 dB，不可接受 |
| v4 `stable_rec100` | SANA `x_denoised` 正样本、重建权重 100、800 steps、batch 8 | 26.8291 | 基本追平 YODA |

v4 配置：

- steps：800
- batch size：8
- hidden channels：64
- depth：2
- learning rate：`1e-4`
- reconstruction weight：100
- gradient clip：1
- NFE：1

v3 的结果说明，仅修正漂移目标和训练步数不足以保证接近 YODA；正样本必须对齐 YODA 原始 SANA 去噪输出。v4 的稳定配置将 PSNR 从 23.8815 恢复到 26.8291 dB。

## 6. v4 held-out 结果

| 模式 | PSNR | MS-SSIM | LPIPS | DISTS | bpp |
| --- | ---: | ---: | ---: | ---: | ---: |
| YODA original | 26.8037 | 0.920381 | 0.039073 | 0.061155 | 0.010803 |
| Drift identity | 21.1865 | 0.853360 | 0.092875 | 0.125448 | 0.012847 |
| Drift trained | 26.8291 | 0.920928 | 0.039201 | 0.061463 | 0.010855 |

v4 trained 相对 YODA original：

| 指标 | 差值 | 方向 |
| --- | ---: | --- |
| PSNR | `+0.0254 dB` | 越高越好，略优 |
| MS-SSIM | `+0.000547` | 越高越好，略优 |
| LPIPS | `+0.000128` | 越低越好，略差 |
| DISTS | `+0.000307` | 越低越好，略差 |
| bpp | `+0.0000529` | 越低越好，略差 |

结论：v4 在 PSNR 和 MS-SSIM 上略高，在 LPIPS、DISTS 和 bpp 上略差，整体是基本追平，不能宣称明显超过 YODA。Drift identity 明显劣于 trained 和 YODA，说明训练本身贡献了主要恢复。

训练统计：

- 训练 loss：`20.147791` 降至 `7.109575`。
- 最终重建项：`0.000257893`。
- 训练耗时：约 `105.07 s`。
- CUDA 峰值：约 `5323 MB`。

## 7. 潜变量诊断

诊断使用 10 个训练帧，`batch-size=1`，结果位于 `outputs/real_akiyo_yoda_drift_v4/diagnostics/training_latent_decode.json`。

| 诊断项 | 数值 |
| --- | ---: |
| condition decode PSNR | 19.0997 |
| target decode PSNR | 27.0087 |
| generated decode PSNR | 26.9936 |
| condition -> target latent MSE | 0.113100 |
| generated -> target latent MSE | 0.0002368 |
| generated -> condition latent MSE | 0.111998 |

这说明 Drifting 已几乎精确拟合 YODA 原单步 SANA 输出的目标潜变量。最终效果仍需由完整视频和像素域指标确认。

## 8. 完整视频与验证

输出目录为 `outputs/real_akiyo_yoda_drift_v4/`：

- `comparison_2x2.mp4`：704x576，2x2 完整视频对比。
- `comparison_side_by_side.mp4`：1408x288，完整并排视频对比。
- `yoda_original.mp4`：YODA 原始分支输出。
- `drift_trained.mp4`：Drifting v4 输出。
- `drift_identity.mp4`：未充分训练的 identity 对照。
- `original.mp4`：原始视频。
- `error_yoda.mp4`：YODA 重建误差视频。
- `error_drift.mp4`：Drifting 重建误差视频。
- `summary.json`：配置、指标、码流和逐帧数据入口。
- `validation.json`：MP4 属性及交叉检查结果。
- `per_frame_metrics.json`：逐帧指标。
- `train_history.json`：训练历史。

验证结果：

- 8 个 MP4 全部为 H.264、`yuv420p`、300 帧、`10.01 s`、`30000/1001 fps`。
- 服务器和本地文件已逐一通过 SHA256 校验一致。
- `validation.json` 中记录了各视频尺寸、采样亮度范围和交叉视频平均绝对差。

最终展示必须使用完整 MP4；`latent_preview.png` 只用于早期算法调试，不能作为复现效果。

## 9. 延迟

在同一次复杂模型重构 run 的 held-out 测量中：

| 模式 | 编码 | 解码 |
| --- | ---: | ---: |
| YODA | 306.47 ms | 254.37 ms |
| Drift trained | 130.46 ms | 93.70 ms |

该 run 使用 reconstruction weight 100 的复杂模型配置，且 GPU0 有其他进程竞争。数值只能作为本次实测，不能夸大为稳定加速比；延迟优势仍需隔离显存和 GPU 负载后复测。

## 10. 复现命令

服务器环境：

```bash
cd /home/amax/Public/YODA
bash scripts/run_real_yoda_drift_v4.sh --detach
```

脚本固定使用：

```text
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
outputs/real_akiyo_yoda_drift_v4
```

测试和脚本语法检查：

```bash
cd /home/amax/Public/YODA
.venv/bin/python -m unittest discover -s tests -v
bash -n scripts/run_real_yoda_drift_v4.sh
```

潜变量诊断：

```bash
cd /home/amax/Public/YODA
.venv/bin/python scripts/diagnose_drift_latents.py \
  --output-dir outputs/real_akiyo_yoda_drift_v4 \
  --batch-size 1 \
  --device cuda:0
```

## 11. 验收结论

已完成：

1. 真实 YODA 权重和小规模真实视频上的压缩/解压缩路径。
2. 真实 `.bin` 码流生成和读取。
3. SANA 单步 DiT + LoRA 到条件 Drifting 单次前向的替换。
4. 正样本修正、v3 失败对照、v4 稳定配置及逐帧/视频级指标。
5. 完整 300 帧 MP4 对比、ffprobe 属性检查和服务器/本地 SHA256 一致性检查。
6. 29 项单元测试覆盖核心损失、生成器、memory bank、trainer、hybrid 接口和 YODA 路径。

未完成：

1. 完整 UVG/HEVC Class B/MCL-JCV benchmark 和论文同协议对比。
2. 多视频泛化、多个随机种子和独立 GPU 负载下的稳定延迟测试。
3. Drifting 官方 JAX/Flax 训练配方的端到端同条件复现。

## 12. 已知限制

- YODA 公开源码不含原始训练代码和训练损失，无法逐行复现其训练目标。
- Drifting 原论文是类别条件生成；本任务将其扩展到 YODA 压缩潜变量条件，属于适配研究。
- 当前 held-out 只有 Akiyo 后 100 帧，结果不能代表完整数据集性能。
- v4 基本追平而非明显超过 YODA；若目标是获得稳定感知质量增益，需要扩大数据、调整漂移/感知目标和做多视频验证。
- 熵编码依赖上游扩展和服务器环境；复现实验必须在已安装完整依赖的服务器 `.venv` 中运行。
