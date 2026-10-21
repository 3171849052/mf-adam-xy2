# exp1：Qwen3.5-9B-Base 数学 LoRA / IID DP-AdamW

实现范围：原始 Base、非私有 LoRA、IID 高斯噪声 DP-AdamW。没有实现 MF、Momentum-Bias、Scale。
使用现有 **Conda `pndp`**（按用户后续指示替代 `curve`）、单张 NVIDIA A800 80GB。
不要把微型随机模型的测试输出作为 9B 模型的实验结果。

## 目录与环境

- `exp1/*.py`：实现；`exp1/configs/`：配置与不可变上游版本/SHA256 锁。
- `exp1/tests/`：标准库 unittest；无需额外安装 pytest。
- `exp1/artifacts/`：审计、参与日程、测试报告；`exp1/logs/`：本次执行日志。
- `exp1/runs/<名称>/`：手动启动后的配置、逐步日志、adapter、验证/最终评价、冻结清单。
- 根目录 `data/{raw,processed}/`：所有实验共享的数据；根目录 `model/Qwen3.5-9B-Base/`：权重与 tokenizer。
- 模型/HF/Torch 缓存在根目录 `model/.cache/`；数据缓存在 `data/.cache/`；编译缓存在 `exp1/.cache/`。

实际环境见 `artifacts/environment.json`，依赖固定在 `requirements.txt`。已安装 Python 3.11.15、Torch 2.8.0 / CUDA 12.8、Transformers 5.12.1、PEFT 0.19.1、Opacus 1.6.0。
现有环境缺少 FLA / causal-conv1d：本实现**显式选择 Transformers 上游 torch 参考实现**处理 gated delta、卷积及 recurrent decode，并使用 SDPA 完整注意力。
此选择不会动态切换为其他后端。其吞吐量可能低于优化内核；使用新内核前必须实现并重新验证明确的后端选择。

本机 Conda 不在初始 PATH 中，先执行：

```bash
export PATH=/XYFS01/HDD_POOL/sysu_qling/sysu_qling_3/anaconda3/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
conda run --no-capture-output -n pndp python -m exp1 env
```

从仓库根目录运行后文命令。没有自动切换模型、改 batch、删已有运行目录或吞掉错误；输出目录已存在会报错，请显式选新名称。

## 数据、去污染与答案规则

- [GSM8K 官方仓库](https://github.com/openai/grade-school-math)，commit `3101c7d5072418e28b9008a6636bde82a006892c`，官方 `train.jsonl`/`test.jsonl`。
- [LLM-Adapters 官方仓库](https://github.com/AGI-Edgerunners/LLM-Adapters)，commit `816657208af4db747803f87ba40a4c71383fed7a`，`ft-training_set/math_10k.json`，实际 9,919 条。
- [Qwen3.5-9B-Base](https://huggingface.co/Qwen/Qwen3.5-9B-Base)，revision `68c46c4b3498877f3ef123c856ecfde50c39f404`。
- 数据、tokenizer、配置、模型分片均锁定 SHA256，见 `configs/sources.lock.json`。下载后不匹配即报错。

GSM8K 官方训练集按 NumPy `default_rng(20261020).permutation` 抽出 512 条作内部验证，剩余 6,961 条；官方测试集 1,319 条。
对 Math-10K 与内部验证/官方测试**题目文本**执行 NFKC、casefold、去标点/空白归一化的精确匹配，以及全部词集合 Jaccard ≥ 0.80 的近似匹配；数字保留。
使用稀疏矩阵计算全部词集合交集，无随机候选漏检。此规则会保守地排除部分同模板不同数字题。
排除 419 条，Math-10K 剩余 9,500 条。报告列出每个匹配对、规则、得分、数据源启发式和内部重复。
GSM8K 官方划分自身的 5 个近似匹配对也保留在审计中，没有改动官方训练划分。

LLM-Adapters 的固定 README 明确来源为 GSM8K、MAWPS、AQuA（1000 examples）。原始 JSON 没有可靠的逐条来源字段；
来源统计中的 `likely` 只是启发式，不能当作已验证来源。某些生成解答与给定答案存在矛盾；保留上游解答，末尾添加给定答案标记。
词面审计**不能保证**排除语义改写、翻译、错误标注来源和 Base 预训练污染。
官方测试题目在冻结前只用于去污染，不生成预测、计算准确率或据此选择配置。

统一提示如下，不使用聊天模板：

```text
Solve the following math problem. Show your reasoning. End with #### followed by the final answer.

Question: {question}

Answer:
```

只监督回答和 EOS，prompt/padding 标签为 `-100`。先逐样本对有效**回答 token**的交叉熵取平均，再对样本梯度裁剪；不按整个 batch 的 token 数加权。
GSM8K 保留推理并统一末尾 `#### answer`；Math-10K 保留生成解答并附上同一标记，AQuA 的选择字母也保留。
推理只解析生成续写的最后一个 `####` 标记之后的完整首行：数字、有限小数、分数或 A–E，使用 Fraction 精确归一化。
不使用“取最后一个数字”的 fallback；缺少标记或格式不合约定计解析失败，同时计回答错误。GSM8K 的 gold 是数值。

训练长度起点 256 的审计保留在 `artifacts/data/lengths-full-256.json`：GSM8K 1,523/6,961（21.88%）、Math-10K 3,079/9,500（32.41%）会丢失回答 token。
观察到最大完整长度 1,595；因此 **pilot/full 配置已显式调整为 1,664**，重新审计为零截断。
动态 padding 只补到当前 physical batch 的最长样本。pilot/full 有任何超长样本即拒绝开始，不裁剪、不丢弃它们。
smoke 明确选不超过 256 的小子集并记录 ID。验证/最终评价 loss 使用完整解答，不沿用训练截断上限。

## 优化器、Ghost Clipping 与隐私定义

BF16 非量化骨干；PEFT LoRA rank=16、alpha=16、dropout=0、无 bias，覆盖全部语言 decoder 的 Linear（包括 full attention、gated delta 投影及 MLP），不训练视觉模块、卷积、norm、embedding 或 lm_head。
LoRA 主参数和 Adam 状态为 FP32。AdamW betas=(0.9,0.999)、eps=1e-8、weight_decay=0，无学习率调度；候选 `1e-4, 3e-4, 5e-4`。
完整训练 logical batch=64，每 64 个参与槽位更新一次；physical batch 可配置为 1/2/4/8，默认 4，恰好完成 300 次更新。

每个 dataset/seed 生成数据内容无关的固定随机排列流，遍历完再重排。非私有/DP 同 seed 使用相同流。
保存完整槽位索引、样本 ID、计划/实际参与次数及日程 SHA256。没有 Poisson 采样器。
跨轮次时同一条记录可能在同一 logical batch 出现两次，**不会忽略这个敏感度增加**。

Ghost 第一遍对逐样本平均 loss 的和反传，用 [Opacus 的序列 Linear 精确 Ghost 范数公式](https://github.com/meta-pytorch/opacus/blob/v1.6.0/opacus/grad_sample/linear.py)计算 A/B 各参数范数，汇总为全部可训练参数的每样本整体 L2 范数。
不构造逐样本参数梯度，不使用序列长度的 Cauchy 上界。裁剪系数 `min(1,C/norm)`（默认 C=1）detach 后，第二遍重新前向，对加权样本 loss 的和反传。
使用 tensor 梯度 hook 配合 non-reentrant 梯度检查点；不支持重复调用/共享训练权重或额外训练模块，发现即报错。
每个 physical batch 得到裁剪梯度和，在 FP32 中累积。每个 logical batch 对总和加入**一次**独立 `N(0,(σC)^2 I)`，再除以 64、调用一次 AdamW。
冻结的词表头采用 PyTorch checkpoint 分块交叉熵，避免分配 `[B,T,248320]` 的完整 FP32 logits；有 loss/梯度等价性测试。

隐私单元为**固定公开槽位中的一条训练记录**。add/remove-zero-out 邻接保持公开 N、槽位、日程不变，移除记录的贡献视为零；不是删除后紧凑重排，也不是 replace-one 邻接。
对记录 i 在第 t 步出现的次数记 `m_it`，单步总和敏感度为 `m_it C`。令 `K2=max_i Σ_t m_it²`，则

```text
μ = sqrt(K2) / σ
δ(ε; μ) = Φ(μ/2 - ε/μ) - exp(ε) Φ(-μ/2 - ε/μ)
```

按 [Gaussian Differential Privacy 的 Gaussian 机制与组合](https://arxiv.org/abs/1905.02383)求解 ε=6、δ=1e-5 对应 μ，校准 σ。
同 batch 无重复时 K2 就是最大参与次数；有重复时用平方重数求和。使用 log-CDF / expm1 稳定求根，无采样放大、无 Poisson 会计或 CLT 近似。
保存最大参与次数、K2、总和/平均梯度上的噪声标准差、μ 和实际 ε。

ε=6 是**单次训练**预算；三次 DP 学习率搜索及三种 seed 的发布要组合。`privacy_composition.json` 保守地将两套数据所有 DP fits 按 μ² 相加，不假设跨数据集不相交。
非私有模型、未加噪数据审计和预处理本身不满足该 DP 定义；不能称整个混合实验包为 ε=6-DP 发布。
验证数据视为公开数据；不保存 DP 的原始训练 loss、逐样本梯度范数或裁剪系数。
数据/模型/日程种子固定并公开；DP 噪声生成器用独立 OS 熵初始化，**不公开、不保存噪声 seed**。因此 DP fits 不承诺逐比特复现。
这里是 PyTorch 浮点高斯实现的研究会计，不声称密码学安全随机数或完整有限精度 DP 证明。

## 已实现的运行入口

`python -m exp1 --help` 列出实际入口。`run` 执行以下流程，**停止在配置冻结后**：

1. 全量长度审计；在真实 9B 权重上做 Ghost/autograd 校验，分别测试 checkpoint off/on、B=0/非零、physical 1/2/4/8；恢复原始 adapter。
2. 原始 Base 的内部验证集评价。
3. 两套数据 × 非私有/DP × 三个 LR 的搜索，仅按内部验证准确率选 LR；平局取更低验证 loss、再取较小 LR。
4. 每组用选定 LR 运行三个 seed 的正式 fits，验证参与日程一致。
5. 保存含源码、配置、数据/模型清单、adapter 哈希的 `frozen.json`。此时仍未评价官方测试集。

`final` 只接受冻结的 **full** 运行；验证冻结哈希后，对原始 Base 和 12 个正式 adapter 评价官方 GSM8K。
`results.json` 包含准确率、逐样本平均验证/测试 loss、解析失败率、生成上限命中率、训练样本/token 吞吐量、峰值显存、三 seed 均值/样本标准差。
同 seed/同题配对比较 Math-10K 对 GSM8K、DP 对非私有；保存不一致正确计数、精确 McNemar p 值、配对 seed 差及描述性数据选型结论。
生成统一 greedy；full 最大生成 512 tokens，命中上限不自动重跑。

## 复现与手动完整实验命令

先下载/校验数据与真实模型。默认直接从固定 HF revision 下载：

```bash
conda run --no-capture-output -n pndp python -m exp1 prepare
conda run --no-capture-output -n pndp python -m exp1 download --transport huggingface
conda run --no-capture-output -n pndp python -m exp1 audit --config exp1/configs/full.json
conda run --no-capture-output -n pndp python -m unittest discover -s exp1/tests -v
```

HF 大文件连接在本机有超时问题。可**显式选择** `download --transport modelscope`：仅改用 Qwen 官方 ModelScope 的字节传输，逐文件核对已锁定的 HF 内容 SHA256（包括配置/tokenizer），不改变模型。
该服务器指向当前文件，但任何内容与固定版本不同都会报错；没有自动 fallback。该入口可以按 HTTP Range 续传已有 `.partial`，服务器未正确响应 Range 时直接报错，最终仍须核对完整 SHA256。

真实 9B 小规模 smoke（8 槽/步、1 步、每数据/方法一个 LR、验证 2 题、生成上限 8；不允许访问官方测试）：

```bash
conda run --no-capture-output -n pndp python -m exp1 run --config exp1/configs/smoke.json --output exp1/runs/smoke-9b
```

可独立复核真实模型的梯度正确性：

```bash
conda run --no-capture-output -n pndp python -m exp1 check-ghost --config exp1/configs/full.json --output exp1/artifacts/ghost-9b-manual.json
```

准备好后，由你手动启动完整实验（下列两个入口均已实现）：

```bash
conda run --no-capture-output -n pndp python -m exp1 run --config exp1/configs/full.json --output exp1/runs/full
conda run --no-capture-output -n pndp python -m exp1 final --run exp1/runs/full
```

第一条完成搜索、正式训练和冻结；第二条才运行官方测试并生成最终结论。不要在两条命令之间改冻结源码、配置或 adapter。
需要 pilot 时，将第一条的配置改成 `exp1/configs/pilot.json`、输出目录改成 `exp1/runs/pilot`；其 20 steps / 验证 32 题的结果不属于正式结果，也不能运行 `final`。
不用模型即可审阅全部 300-step 日程、参与次数和预算（`artifacts/plans/full/` 已生成，重跑需新目录）：

```bash
conda run --no-capture-output -n pndp python -m exp1 plan --config exp1/configs/full.json --output exp1/artifacts/plans/full-manual
```

测试夹具 smoke 使用微型随机初始化的**真实 Qwen3.5 混合模块**及合成 tokenizer/题目，覆盖训练、两种优化方案、保存/重载 adapter、验证及配对统计：

```bash
conda run --no-capture-output -n pndp python -m exp1.tests.smoke --output exp1/artifacts/fixture-smoke-manual
```

本次执行状态、错误和未完成部分见 `artifacts/implementation_status.json`。完整实验没有启动；不存在正式准确率或数据集胜负结论。
本次通过的是 16 项单元测试、A800 BF16 微型混合架构 Ghost 对照及夹具 smoke。
真实 9B 只完成 meta 形状/权重键审计（496 个 Ghost 支持的 Linear、43,278,336 个 LoRA 参数）；未加载真实权重。
大文件传输缓慢且 HF 曾超时，下载已停止，保留 partial 供上述显式命令续传；真实 9B smoke 因缺少 SHA256 验证后的权重被入口拒绝。
因此真实规模下的数值正确性、80GB 峰值显存和吞吐量**尚未验证**；正式 `run` 会先在真实权重上通过 Ghost 检查，失败即停止。

## 已授权的后台完整实验

2026-10-20（北京时间），用户后续明确要求启动完整实验。已启动独立后台监督进程，按 `download → unit_tests → smoke_9b → full → final` 顺序调用现有入口；任一步非零退出即停止后续阶段。不会修改 batch、模型或测试通过阈值。

当前阶段、心跳、权重字节进度、子进程 PID 和各阶段退出码以 `exp1/launches/full-20261020/status.json` 为准。主日志为 `exp1/logs/full_supervisor_20261020.log`，各阶段日志在该 launch 目录。原始配置及源码哈希已记录在 `inputs.json`；后续阶段发现输入被修改会拒绝继续。

```bash
cat exp1/launches/full-20261020/status.json
tail -f exp1/logs/full_supervisor_20261020.log
```

下载和预检期间 `full_experiments_started` 为 false；只有进入 full 阶段才置 true。正式结果目录为 `exp1/runs/full/`，smoke 目录为 `exp1/runs/smoke-9b/`。

只读监视进程每 30 秒检查监督进程/子进程存活、阶段心跳、权重字节进度、GPU 利用率/显存及日志错误。监视不会修改训练或配置。最新观测为 `exp1/launches/full-20261020/monitor/latest.json`，历史为 `observations.jsonl`，阶段变化和异常为 `events.jsonl`。下载连续 15 分钟无新增字节会记录异常；任务完成或失败后记录终态并退出。

```bash
cat exp1/launches/full-20261020/monitor/latest.json
tail -f exp1/launches/full-20261020/monitor/events.jsonl
```

下载恢复更新：第一次后台启动在北京时间 20:12 因网络读取超时停止，保留 5,486,149,632 字节。已增加 64 MiB 有界 Range 请求和明确记录的暂时网络错误重试（每文件最多 12 次）；不重试错误 Range、无效文件或 SHA256 不匹配。恢复逻辑的 3 项测试通过，保留了第一次失败日志。新任务位于 `exp1/launches/full-20261020-retry1/`，`exp1/launches/current/` 指向当前任务；当前状态请读该目录，而非旧任务的终止快照。训练参数未修改。

## 2026-10-21 实际进度：真实 9B 正确性检查未通过

固定权重于北京时间 2026-10-20 22:00 完成下载和 SHA256 校验，随后 19 项单元测试通过。22:06 的真实 9B 检查在 physical batch=2 时出现裁剪梯度相对误差 0.0320353，超过现有 0.03 门槛，因此尚未开始任何正式 optimizer update，也未评价官方测试集。

独立诊断发现未裁剪 BF16 梯度跨 batch 已有约 3.59% 差异；几种数值后端设置仍未解决该问题。FP32 精度对照的 physical batch=2 裁剪梯度误差约 5.66e-6，说明应继续定位混合精度数值问题。FP32 仅用于诊断，未切换正式训练精度；未改生产源码、配置或检查阈值。诊断报告为 `exp1/artifacts/ghost_failure_analysis.json`。

[PyTorch 2.8 数值精度说明](https://docs.pytorch.org/docs/2.8/notes/numerical_accuracy.html)介绍了批量计算与逐条计算的浮点差异；这不足以自动认可本次 BF16 检查结果，因此完整流程继续保持停止。
