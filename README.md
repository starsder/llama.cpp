# llama.cpp：Qwen MoE host 推理实验分支

本仓库是 [starsder/llama.cpp](https://github.com/starsder/llama.cpp)，直接派生自 [unslothai/llama.cpp](https://github.com/unslothai/llama.cpp)，基础推理引擎来自 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) 与 [ggml](https://github.com/ggml-org/ggml)。本 fork 主要探索大于显存容量的 MoE 模型在单机 CPU/GPU 混合推理中的专家缓存、预取与执行重叠。

**状态：不稳定、研究用途，不是生产稳定版。** 当前发布的保守 host 代码基线为 [`7e01451b2`](https://github.com/starsder/llama.cpp/commit/7e01451b2d7aab6a4ff58ecfdd2f3b13fcaeb0bf)，包含 host 分区路由错位修复及混合量化权重寻址保护；不包含后续 repack/KV 实验及尚未提交的缓存策略候选。“保守基线”不代表无崩溃、无计算错误或已经完成全面回归。

> **严重资源风险：推荐参数以外的开关、组合和并发运行，可能耗尽整机主存、提交额度或显存，造成进程崩溃、系统无响应，甚至需要重启。128 GB 内存也不代表安全。推荐参数同样不是安全保证。请先保存其他工作，不要在承担重要任务的机器上无人值守运行。**

## 实验档案：路线、失败与更正

本项目的研发从 **SSD→主存的 PLE 行缓存**开始，随后是静态表＋XT、Fate、共享专家 SMoE、在线缓存与双门控、dev 路径时序、TBQ4 与 NXQ。最值得保留的是每次为什么尝试、为什么转向，以及哪些漂亮结果后来被推翻。

- **[完整档案索引](docs/experiments/README.md)**：路径覆盖表、版本边界、保留／放弃／未完成状态。
- **[按研发顺序阅读](docs/experiments/00-research-chronology.md)**：包括维护者补充的 PLE 约1G／90%+、SMoE teacher 99%，与在线指标分开记账。
- [PLE、host、devpart、prefill、MTP](docs/experiments/01-host-and-devpart.md) · [预测、缓存与双门控](docs/experiments/02-prediction-and-cache.md)。
- [权重量化选型与 IQ/Q 内核差异](docs/experiments/03-weight-quantization-and-kernels.md) · [TBQ3/TBQ4、NXQ 与 KV 质量](docs/experiments/04-kv-tbq-and-nxq.md)。
- [正确性、测量失效与工程事故](docs/experiments/05-correctness-and-methodology.md) · **[原始证据与哈希清单](docs/experiments/evidence/README.md)**。

档案也收录未发布和未验收的研究，**上传文档不等于合入这些实现**。原生代码基线仍为 `7e01451b2`；没有因本次归档重跑模型或启用新优化。

## 1. 测试硬件与适用范围

| 项目 | 本机配置／测试工作点 |
|---|---|
| CPU | AMD Ryzen 9 5950X，16 核／32 线程；推荐推理线程数 16 |
| 主存 | DDR4-2666，128 GB；DDR4-2666 表示 2666 MT/s |
| GPU | NVIDIA RTX A5000 Laptop GPU，16 GB 显存 |
| GPU 链路 | PCIe 4.0 ×8 |
| 系统 | Windows 11 x64；CUDA backend |
| 测试模型 | 本地使用的 `Qwen3.8-Flash-Next-UD-IQ3_XXS`，Unsloth UD 混合量化 GGUF，三个分片 |
| 推荐上下文／KV | `-c 8192`，K/V 均为 `q8_0` |
| 推荐加载方式 | `--cpu-moe --no-mmap --lazy-mode off`，权重完整装入主存 |
| 推荐专家缓存预算 | `6144 MiB`；预算不等于整个进程的显存占用 |
| 长跑测量 | 单请求、贪心解码、请求 400 token，并开启 `--ignore-eos` |

内存频率、容量与 PCIe 链路由设备使用者提供；它们不是本次文档更新重新采集的硬件遥测。Laptop GPU 的功耗限制、散热、驱动及后台负载都会影响结果，不可据此推算桌面 A5000、PCIe ×16、其他内存频率或其他模型的性能。测试用大模型权重不随本文提供，必须自行取得并核对其来源与许可证。

## 2. 性能基准：提高了多少

### 历史修复后的记录（不是严格同条件 A/B）

仓库 [handoff.md §6.32](handoff.md) 在修复 host split 的路由错位之后，记录了下列 **8k 上下文、q8_0 KV、auto 缓存预算（97 槽／层）** 的结果：

| 历史配置 | Generation | 与历史 9.2 数值的算术比较 |
|---|---:|---:|
| 同一 fork 的缓存关闭路径 | 9.2 token/s | 1.00× |
| host split，auto 缓存，`AHEAD=3` | 20.8 token/s | 2.26×，约 **+126.1%** |
| host split，auto 缓存，`AHEAD=2` | 18.4 token/s | 不与下方 6 GiB 推荐工作点混同 |

历史条目中的提升按 `(20.8 / 9.2 - 1) × 100% = 126.1%` 计算，约为 2.26 倍吞吐，不是延迟降低 126%。参照来自本 fork 的缓存关闭路径，**不是最新上游 llama.cpp、Unsloth、Fate 或 HybriMoE 的横向排名**。

历史记录包含 Eiffel 文本回归通过的说明，但未建立两侧提示词、生成 token 数和全部运行条件一致的完整证据链。因此，**+126.1% 仅是历史记录数字的算术比较，本 README 不将其认定为严格控制变量后的实测净提升**，也不能据此声称质量无损或稳定加速。`auto/AHEAD=3` 不是本文推荐的保守配置；当前推荐配置提高了多少，仍需要同条件的新一轮对照才能可靠回答。

### 当前推荐的 6 GiB／400-token 工作点

后续本地开发构建采用下面的保守参数，正常退出的 CLI 记录为 **19.2、19.3 token/s**；后来一次相同参数的旧基线复测为 **18.4 token/s**。其中 19.3 的观测运行记录：

| 指标 | 单次观测值 |
|---|---:|
| 请求生成／逐图记录 | 400 token（忽略 EOS）／403 个 graph，后者包含非解码图 |
| 专家缓存 | 请求 6144 MiB，64 槽／层，物理缓存约 6116.3 MiB |
| 整卡采样显存峰值 | 12458 MiB，包含当时其他显存使用，不是纯模型分配量 |
| 页锁定权重缓冲 | 72.6 GiB，日志报告 0 次锁定失败 |
| 缓存访问命中率 | 69.7%，不是预测器准确率 |

这些是维护者本地记录，**不是对 `7e01451b2` 发布构建的新一轮验收**；完整原始日志尚未随本 README 发布，无法仅凭此表独立复核。没有对这一工作点重新进行同条件缓存关闭对照，故**不把 19.3 除以历史 9.2 来宣称当前配置的净提升**。19 token/s 只是本机当前的内部筛选下限，既不保证每次达到，也不是达到后就代表正确。

以下数字明确不计入有效成果：

- 旧文档中路由错位修复之前的 host 20.3–22 token/s；后续记录已撤回它们作为正确性能证据的资格。
- devpart 在计算错误状态下出现的约 28–32 token/s。
- 后续缓存策略候选的单次峰值、失败运行、只降低传输字节却没有稳定整步收益的结果。
- 不同提示词、缓存预算、PLE 设置、上下文、短跑预热状态之间直接相除的“提升”。

**推荐参数不等于历史最高分参数。** 后续 400-token／6 GiB 的 `AHEAD=3` 运行出现过非零退出，本机优先使用 `AHEAD=2`。无法据已有数据确认后台软件就是速度下降的原因。

## 3. 推荐参数：非 devpart 的 host split

`host` 指 **`LLAMA_MOE_DEVPART=0` 且 `LLAMA_MOE_SPLIT=1`**，不是纯 CPU，也不是关闭 split。模型真实选中的专家仍需要计算：驻留专家由 GPU 执行，未驻留部分由 CPU 执行。

| 参数组 | 推荐设置 |
|---|---|
| 执行与缓存布局 | `DEVPART=0`、`SPLIT=1`、`DIRECT_READ=1`、`GLOBAL_POOL=0`、`WINDOW_LAYERS=0` |
| 容量 | `CACHE_MIB=6144`、`VRAM_LIMIT_MIB=15360`、`VRAM_GUARD_MIB=1024` |
| 淘汰策略 | `MRS=1`、`FIFO=0`、`EVICT_SCORE=0` |
| 预测 | `PREDICT_SMOE=1`、`SMOE_NONBLOCK=1`、`SMOE_AHEAD=2`、`PREDICT_TOPK=26` |
| 传输与回填 | `PREFETCH=1`、`PREFETCH_JOIN=0`、`HOT_BACKFILL=8` |
| CPU 部分 | `CPU_ASYNC=1`，`LLAMA_ARG_THREADS=16` |
| 不启用的其他路径 | `PREDICT_FATE=0`、`PREDICT_XT=0`、`INSERT_ON_MISS=0`、`FALLBACK_PREFETCH=0` |
| 主存／PLE／KV | `PIN_WEIGHTS=1`、`LLAMA_PLE_CACHE_MIB=0`、`LLAMA_PLE_GPU_CACHE_MIB=0`，使用普通 q8_0 KV |

上表未写完整的变量名均以 `LLAMA_MOE_` 为前缀。`PREDICT_TOPK=26` 是预取候选参数，**不是更改模型实际路由的 top-k**。MRS 是此分支沿用的名称；`EVICT_SCORE=0` 使用实际路由频率，不等于开启论文中的完整 gate-score 策略。

### Windows 构建与运行示例

使用装有 Visual Studio C++ 工具链、CMake、Ninja、CUDA Toolkit 和 Python 的 **x64 Developer PowerShell**。详细平台要求见 [构建文档](docs/build.md)。从本 fork 构建，不要把上游发布页的通用二进制当成包含本分支功能的版本：

```powershell
git clone --branch master https://github.com/starsder/llama.cpp.git llama.cpp-host
Set-Location llama.cpp-host
cmake -S . -B build-ple-trace-mrs -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build-ple-trace-mrs --target llama-cli -j 16
```

将三个 GGUF 分片放在同一目录，修改 [tools-run.py](tools-run.py) 中的 `MODEL`，指向真实的 `...-00001-of-00003.gguf`。脚本当前仍含维护者机器的绝对路径；不要直接照抄该路径，也不要只下载第一个分片。

然后在一个**新的** Developer PowerShell、仓库根目录内执行以下命令。环境清理只影响当前终端及其后续子进程，不修改系统永久环境：

```powershell
Get-ChildItem Env: |
    Where-Object { $_.Name -match '^(LLAMA_|GGML_)' } |
    ForEach-Object { Remove-Item -LiteralPath ("Env:" + $_.Name) }

$moeArgs = @(
    'LLAMA_MOE_DEVPART=0',
    'LLAMA_MOE_SPLIT=1',
    'LLAMA_MOE_GLOBAL_POOL=0',
    'LLAMA_MOE_WINDOW_LAYERS=0',
    'LLAMA_MOE_DIRECT_READ=1',
    'LLAMA_MOE_CACHE_MIB=6144',
    'LLAMA_MOE_VRAM_LIMIT_MIB=15360',
    'LLAMA_MOE_VRAM_GUARD_MIB=1024',
    'LLAMA_MOE_MRS=1',
    'LLAMA_MOE_FIFO=0',
    'LLAMA_MOE_EVICT_SCORE=0',
    'LLAMA_MOE_PREFETCH=1',
    'LLAMA_MOE_PREFETCH_JOIN=0',
    'LLAMA_MOE_PREDICT_SMOE=1',
    'LLAMA_MOE_PREDICT_FATE=0',
    'LLAMA_MOE_PREDICT_XT=0',
    'LLAMA_MOE_PREDICT_TOPK=26',
    'LLAMA_MOE_SMOE_NONBLOCK=1',
    'LLAMA_MOE_SMOE_AHEAD=2',
    'LLAMA_MOE_HOT_BACKFILL=8',
    'LLAMA_MOE_CPU_ASYNC=1',
    'LLAMA_MOE_INSERT_ON_MISS=0',
    'LLAMA_MOE_FALLBACK_PREFETCH=0',
    'LLAMA_MOE_PIN_WEIGHTS=1',
    'LLAMA_PLE_CACHE_MIB=0',
    'LLAMA_PLE_GPU_CACHE_MIB=0',
    'LLAMA_ARG_THREADS=16'
)
python .\tools-run.py --tag host400 --tokens 400 --ignore-eos `
    --min-free-mib 90000 --wait-mem 120 --timeout 900 @moeArgs
```

该发布版本的脚本**没有 `--profile` 或 `--json` 参数**。以上命令显式覆盖旧脚本的 auto 缓存等默认值；不要用 `--keep` 混入其他实验环境。脚本自带 `--cpu-moe --no-mmap --lazy-mode off -ngl 49 -c 8192`、q8_0 KV、贪心采样及单轮运行设置；这些参数只针对上述模型工作点，不应推广到所有 GGUF。

检查 `host400-out.txt`、`host400-err.txt`、`stats-host400.csv` 以及脚本控制台报告：

- 确认 host 模式、有效缓存容量、实际线程数与页锁定结果符合预期。
- 确认报告中**模型的真实 `exit=0`**、有最终 Generation 行且没有异常；旧脚本自身返回 0 不足以证明子进程成功。
- 400 token 完整性在 CLI 中主要依赖请求数量、`--ignore-eos` 与正常退出；graph 数不是生成 token 数。基准应保存完整命令、环境、模型分片校验值、输出和原始退出码。
- 失败、超时、输出退化或缺失统计的运行均不能计入成功性能。`--ignore-eos` 是长跑测试设置，可能让模型在正常回答结束后继续重复输出，不是日常问答推荐。

## 4. 已知不稳定性与内存风险

1. **只能单实例运行。** 本机约 72.6 GiB 的权重缓冲被页锁定；两个实例仅这一项就可能超过 145 GiB。仓库确有并发模型导致整机内存耗尽、卡死的历史。脚本 `.tools-run.lock` 只保护同一工作目录、遵守该锁的启动方式；另一个 checkout、裸 CLI 或其他服务可以绕过它。
2. **保留至少 90000 MiB 空闲主存的启动闸门。** 它不是总内存需求的数学上界，也无法阻止其他程序在启动后抢占内存。不要降低闸门、手动删掉活进程的锁，或在未确认旧进程退出时重新启动。
3. **保持 `--no-mmap --lazy-mode off` 与当前页锁定方式。** 此分支历史出现过 mmap／权重页锁定组合带来的严重内存压力；不要自行混用。6 GiB 专家缓存只是一部分显存开销，不代表模型只需要 6 GiB RAM 或 VRAM。
4. **退出期仍可能出现 `0xC0000005`。** profiler 有对象生命周期风险；当前推荐沿用脚本的计时／`LLAMA_TOKEN_PROF=1` 观测配置，该风险尚未修复验证。即使正文已经输出，也应把非零退出视为失败，而不是“只是日志丢了，可以忽略”。
5. **推荐范围外全部按未验证组合处理。** 尤其不要直接叠加 devpart、共享池、自动扩大缓存、其他前瞻距离／在线调参器、PLE 缓存、CPU-KV/QSA、TBQ KV、超长上下文、视觉模型、MTP／draft／投机解码、多请求或多模型并发。它们可能改变内存规模、图形状和异步时序，导致 OOM、非法访问或静默计算错误。
6. **更小缓存也不自动更安全。** 旧版本存在极小预算导致零槽位时不正确退化的问题；改配置不能取代正确性验证。

CPU/GPU 对同一专家的数值实现可能产生浮点差异，缓存放置改变后自由生成也可能不同。因此不承诺逐位等价、所有 token 一致或“无损”。反过来，差异也不能未经定位就被认定为某个新 bug。强制 token replay 的文本相同不等于自由生成相同。

如果是服务部署、无人值守或其他重要用途，建议先选择适合自己工作负载且经过验证的上游版本；不要把本实验分支的推荐配置当成安全认证。本次更新只整理文档，没有继续运行模型或宣称消除了这些风险。

## 5. 使用的技术与实现边界

- **ggml／GGUF／量化 CPU 与 CUDA 算子：** 沿用上游推理与量化基础；CPU 使用实际可用的 SIMD 路径，GPU 使用 CUDA。Unsloth UD 是混合量化，不能假定所有权重都只有文件名中的一种类型。
- **每层专家 bundle 缓存：** 将专家的 gate/up/down 分量按共同槽位管理，驻留检查要求必要分量齐全；direct-read 让 GPU 从缓存槽位读取，避免不必要的中间搬运。
- **CPU/GPU 拆分与重叠：** 根据真实 router 结果分配执行位置，CPU 处理未驻留专家，异步 CPU worker 与 GPU 部分重叠，再在消费依赖处会合。没有以预测专家替代真实专家，也不把漏算当优化。
- **源码中命名为 SMoE 的侧图预测：** 使用近似中间状态生成未来层的预取候选，配合 `AHEAD=2` 和非阻塞结果消费。这是本工程的实现名称，不在此冒认某篇同名论文的完整复现。
- **实际使用频率与热回填：** 推荐 `EVICT_SCORE=0`，按实际路由使用信息管理缓存；`HOT_BACKFILL=8` 与预测预取是不同来源。
- **传输管理：** 使用 CUDA 页锁定主存、侧流、事件与驻留／在途去重，保留既有预取预算及可行性约束。命中驻留专家不等于再次传输；降低某个等待计数或传输字节也不必然缩短整步延迟。
- **针对性的正确性修复：** host 分区更新路由 leaf 后重新提交必要拷贝，避免读取上一层路由；对 UD 混合量化权重的偏移／stride 做相应处理与保护。不是对所有 backend、模型形状或并发方式的完整正确性证明。

实现与实验说明见 [ggml-backend.cpp](ggml/src/ggml-backend.cpp)、[llama-graph.cpp](src/llama-graph.cpp)、[缓存设计说明](docs/moe-cache-score-aware-prd.md) 与 [实验账本](handoff.md)。账本是按时间追加的研究记录，含后来被否定的结论和旧参数；发生冲突时应结合后续更正阅读，不能摘取其中最高速度。

## 6. 来源、参考与开源声明

### 代码来源与技术参考

| 来源 | 本仓库与其关系 |
|---|---|
| [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp)、[ggml](https://github.com/ggml-org/ggml) | 基础推理框架、GGUF、量化算子与各 backend；保留原作者版权、许可证和贡献历史 |
| [unslothai/llama.cpp](https://github.com/unslothai/llama.cpp) | 本 fork 的直接上游；模型适配及相关继承代码以 Git 历史和文件声明为准，不将已有 Qwen 模型支持冒认为本 fork 原创 |
| [starsder/llama.cpp](https://github.com/starsder/llama.cpp) | 本 fork 的缓存、调度、修复、测试工具及实验记录；不是上游官方稳定发行版 |
| [Fate: Fast Edge Inference of Mixture-of-Experts Models via Cross-Layer Gate](https://arxiv.org/abs/2502.12224) | 跨层 gate 预测／专家预取的技术参考；仓库另有 Fate 路径，但推荐配置明确 `PREDICT_FATE=0` |
| [HybriMoE: Hybrid CPU-GPU Scheduling and Cache Management for Efficient MoE Inference](https://arxiv.org/abs/2504.05897)、[作者代码仓库](https://github.com/PKU-SEC-Lab/HybriMoE) | 混合调度、预取和 score-based 缓存的技术参考；不是整套 HybriMoE／kTransformers 的移植或其性能复现 |
| [Unsloth](https://github.com/unslothai/unsloth)、[其 Hugging Face 组织](https://huggingface.co/unsloth) | 本地测试量化文件标称的来源；具体权重必须以下载时的模型卡、分片校验值及许可证为准 |

论文引用是技术归属说明，**不意味着论文、配套代码、模型权重或商标被本仓库重新许可**。这里的性能数字只属于本机实验，不能引用为上述论文或作者的实验结果。项目、模型和硬件名称仅用于识别与归属，不表示任何上游组织、模型作者或 NVIDIA 对本 fork 的认可、合作或担保。

### 许可证与分发责任

- 主项目沿用根目录 [MIT LICENSE](LICENSE)，其中保留 `Copyright (c) 2023-2026 The ggml authors`。本 fork 的新增修改按仓库 MIT 许可提供，已有文件级或第三方独立声明不因本 README 而改变。
- MIT 允许使用、修改及商业分发等行为，但分发软件副本或实质部分时，必须保留其要求的版权和完整许可声明。本文的不稳定警告及部署建议是风险说明，**不是额外的“禁止商用”等许可证限制**。
- 第三方库、内嵌代码、构建时获取的组件及随包运行库保留各自的许可与 NOTICE 要求；不能仅用顶层 MIT 声明覆盖全部依赖。请保留 [AUTHORS](AUTHORS)，并按实际分发版本核查 [licenses](licenses)、[vendor](vendor)、[gguf-py/LICENSE](gguf-py/LICENSE)、对应源码文件及构建产物中的声明。例如 [cpp-httplib](vendor/cpp-httplib/LICENSE) 为 MIT，[xxHash](vendor/hash/xxhash/LICENSE) 为 BSD-2-Clause；下方上游 README 的第三方致谢也予以保留。
- 测试模型、分词器等资产的许可与软件代码分开处理。本文未核实该具体 GGUF 发布物的完整许可证链，**不宣称其自动适用 MIT、可任意商用或可再分发**；使用、下载、转换及再分发应遵守原模型和量化发布者适用的条款。
- CUDA Toolkit、驱动及其他专有运行时按各自条款使用和分发，不因本项目开源而自动获得再分发授权。请自行从合法来源安装，不要无条件打包第三方组件。
- 软件按许可证的 **“AS IS”** 条款提供，不作适销性、特定用途适用性或其他保证；责任限制以原许可证及适用法律为准。本 README 不是对全部依赖完成法律审计的证明，也不替代正式法律意见。

如发现遗漏的版权声明、来源引用或许可冲突，请通过 [Issues](https://github.com/starsder/llama.cpp/issues) 提供具体文件、版本与原始来源，以便核查和修正。提交代码时请注明借用来源及许可，并提供可复现的正确性／性能证据；请勿上传无权分发的模型、私密日志或凭据。

---

## 上游原始 README（保留）

以下保留原有上游介绍、链接与致谢。其中的发行徽章、安装入口和默认下载示例主要指向上游，**不代表包含本 fork 的实验功能**；本 fork 请使用上面的源码构建和参数说明。

# llama.cpp

![llama](https://raw.githubusercontent.com/ggml-org/llama.brand/refs/heads/master/cover/llama-cpp/cover-llama-cpp-dark.svg)

<div align="center">

<b>LLM inference in C/C++</b>

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Release](https://img.shields.io/github/v/release/ggml-org/llama.cpp?filter=v*&color=brightgreen)](https://github.com/ggml-org/llama.cpp/releases?q=tag:v0)
[![Nightly](https://img.shields.io/github/v/release/ggml-org/llama.cpp?label=nightly&filter=b*&color=orange)](https://github.com/ggml-org/llama.cpp/releases?q=b)
[![Server](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/server.yml?label=Server)](https://github.com/ggml-org/llama.cpp/actions/workflows/server.yml)
[![Docker](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/docker.yml?label=Docker)](https://github.com/ggml-org/llama.cpp/actions/workflows/docker.yml)
[![Winget](https://img.shields.io/github/actions/workflow/status/ggml-org/llama.cpp/winget.yml?label=Winget)](https://github.com/ggml-org/llama.cpp/actions/workflows/winget.yml)

[ggml](https://github.com/ggml-org/ggml) / [ops](https://github.com/ggml-org/llama.cpp/blob/master/docs/ops.md) / [maintainer PRs](https://github.com/ggml-org/llama.cpp/issues?q=is%3Apr%20is%3Aopen%20draft%3AFalse%20(author%3Argerganov%20OR%20author%3AKitaitiMakoto%20OR%20author%3Adanbev%20OR%20author%3Aaldehir%20OR%20author%3Amax-krasnyansky%20OR%20author%3ACISC%20OR%20author%3Aggerganov%20OR%20author%3Aam17an%20OR%20author%3Abartowski1182%20OR%20author%3Anikwen%20OR%20author%3Ahipudding%20OR%20author%3AServeurpersoCom%20OR%20author%3Apwilkin%20OR%20author%3Areeselevine%20OR%20author%3Angxson%20OR%20author%3Ajeffbolznv%20OR%20author%3Amarty1885%20OR%20author%3A0cc4m%20OR%20author%3ATitaniumtown%20OR%20author%3Aangt%20OR%20author%3AIMbackK%20OR%20author%3Aarthw%20OR%20author%3AJohannesGaessler%20OR%20author%3AORippler%20OR%20author%3Aruixiang63%20OR%20author%3Axctan%20OR%20author%3Aallozaur%20OR%20author%3Ayomaytk%20OR%20author%3Aaendk%20OR%20author%3Agaugarg-nv%20OR%20author%3Ataronaeo%20OR%20author%3Aforforever73%20OR%20author%3Alhez%20OR%20author%3Anetrunnereve%20OR%20author%3Afairydreaming)%20sort%3Aupdated-desc) / [dev stats](https://github.com/ggml-org/llama.cpp-dev) / [lib llama API](https://github.com/ggml-org/llama.cpp/issues/9289) / [llama-server REST API](https://github.com/ggml-org/llama.cpp/issues/9291)

</div>

## Quick start

A few options to get `llama.cpp` installed on your machine:

- Visit https://llama.app and follow the instructions
- Run with Docker - see our [Docker documentation](docs/docker.md)
- Download pre-built binaries from the [releases page](https://github.com/ggml-org/llama.cpp/releases)
- Build from source by cloning this repository - check out [our build guide](docs/build.md)

Once installed:

```sh
# Download and run a model directly from Hugging Face
llama cli -hf ggml-org/Qwen3.5-0.8B-GGUF

# Launch OpenAI-compatible API server
llama serve -hf ggml-org/Qwen3.5-0.8B-GGUF
```

<table align="center">
    <tr>
        <td align="center" width=50%>
            <img width="1310" height="888" alt="VLM session with `llama cli`" src="https://github.com/user-attachments/assets/88726b48-1713-48aa-a525-95a02e78afc4" />
            <i>VLM session with <b>llama cli</b></i>
        </td>
        <td align="center">
            <img width="1392" height="958" alt="Built-in web UI against `llama serve` running Qwen 3.6" src="https://github.com/user-attachments/assets/b402f972-2e32-4def-8771-8d849f08cf2e" />
            <i>Built-in web UI against <b>llama serve</b></i>
        </td>
    </tr>
<table>

## Description

The main goal of `llama.cpp` is to enable LLM (and VLM) inference with minimal setup and state-of-the-art performance on
a wide range of hardware - locally and in the cloud.

- Plain C/C++ implementation without any dependencies
- Apple silicon is a first-class citizen - optimized via ARM NEON, Accelerate and Metal frameworks
- AVX, AVX2, AVX512 and AMX support for x86 architectures
- RVV, ZVFH, ZFH, ZICBOP and ZIHINTPAUSE support for RISC-V architectures
- 1.5-bit, 2-bit, 3-bit, 4-bit, 5-bit, 6-bit, and 8-bit integer quantization for faster inference and reduced memory use
- Custom CUDA kernels for running LLMs on NVIDIA GPUs (support for AMD GPUs via HIP and Moore Threads GPUs via MUSA)
- Vulkan and SYCL backend support
- CPU+GPU hybrid inference to partially accelerate models larger than the total VRAM capacity

The `llama.cpp` project is build on top of the [ggml](https://github.com/ggml-org/ggml) library.

## Supported backends

| Backend | Target devices |
| --- | --- |
| [BLAS](docs/build.md#blas-build) | All |
| [BLIS](docs/backend/BLIS.md) | All |
| [CANN](docs/build.md#cann) | Ascend NPU |
| [CUDA](docs/build.md#cuda) | Nvidia GPU |
| [HIP](docs/build.md#hip) | AMD GPU |
| [Hexagon [In Progress]](docs/backend/snapdragon/README.md) | Snapdragon |
| [IBM zDNN](docs/backend/zDNN.md) | IBM Z & LinuxONE |
| [MUSA](docs/build.md#musa) | Moore Threads GPU |
| [Metal](docs/build.md#metal-build) | Apple Silicon |
| [OpenCL](docs/backend/OPENCL.md) | Adreno GPU |
| [OpenVINO [In Progress]](docs/backend/OPENVINO.md) | Intel CPUs, GPUs, and NPUs |
| [RPC](https://github.com/ggml-org/llama.cpp/tree/master/tools/rpc) | All |
| [SYCL](docs/backend/SYCL.md) | Intel GPU |
| [VirtGPU](docs/backend/VirtGPU.md) | VirtGPU APIR |
| [Vulkan](docs/build.md#vulkan) | GPU |
| [WebGPU](docs/build.md#webgpu) | All |
| [ZenDNN](docs/build.md#zendnn) | AMD CPU |

## Documentation

#### Tools

- [cli](tools/cli/README.md)
- [completion](tools/completion/README.md)
- [server](tools/server/README.md)
- [GBNF grammars](grammars/README.md)

#### Development

- [How to build](docs/build.md)
- [Running on Docker](docs/docker.md)
- [Build on Android](docs/android.md)
- [Multi-GPU usage](docs/multi-gpu.md)
- [Performance troubleshooting](docs/development/token_generation_performance_tips.md)
- [GGML tips & tricks](https://github.com/ggml-org/llama.cpp/wiki/GGML-Tips-&-Tricks)
- [XCFramework](docs/xcframework.md)
- [Completions](docs/completions.md)
- [Models](docs/models.md)
- [Release process](docs/release.md)

## Contributing

- Contributors can open PRs
- Collaborators will be invited based on contributions
- Maintainers can push to branches in the `llama.cpp` repo and merge PRs into the `master` branch
- Any help with managing issues, PRs and projects is very appreciated!
- Read the [CONTRIBUTING.md](CONTRIBUTING.md) for more information

## Acknowledgements

- [yhirose/cpp-httplib](https://github.com/yhirose/cpp-httplib) - Single-header HTTP server, used by `llama-server` - MIT license
- [nothings/stb](https://github.com/nothings/stb) - Single-header image format decoder, used by multimodal subsystem - Public domain
- [nlohmann/json](https://github.com/nlohmann/json) - Single-header JSON library, used by various tools/examples - MIT License
- [mackron/miniaudio](https://github.com/mackron/miniaudio) - Single-header audio format decoder, used by multimodal subsystem - Public domain
- [sheredom/subprocess.h](https://github.com/sheredom/subprocess.h) - Single-header process launching solution for C and C++ - Public domain
