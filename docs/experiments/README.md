[中文](README.md) · [English](README.en.md)

# Qwen MoE 混合推理：完整实验档案

这不是一张“最终提升多少”的成绩单，而是一份保留研发顺序、失败路径、工程事故和撤回结论的研究记录。每个专题说明：**为什么尝试 → 怎么实现／测量 → 实际发生什么 → 为什么保留或放弃 → 还有什么没解决**。

**先读[时间主线](00-research-chronology.md)。** 顺序以维护者补充为准：SSD→主存 PLE → 静态热表＋XT → Fate → SMoE teacher → 在线缓存与双门控 → devpart 时序／正确性 → 权重内核与 TBQ → NXQ。`handoff.md` 的45个小节只是其中一部分，不是整个研发过程的起点。

## 1. 阅读地图

| 文档 | 回答的问题 |
|---|---|
| [00：研发时间主线](00-research-chronology.md) | 为什么从 PLE 转向专家预测，又为什么研究 KV；维护者补充的1G／90%+与 teacher 99%如何入账 |
| [01：PLE、host 与 devpart](01-host-and-devpart.md) | SSD行缓存、GPU PLE、逐层会合、异步、页锁定、split合并、设备分区、prefill、256k、MTP，以及未实现的算子／分层存储设计 |
| [02：预测与缓存](02-prediction-and-cache.md) | 静态／XT／CrossLayer／Fate／SMoE、离线与在线口径、真实使用频率、淘汰、准入、双门控、全局池、LFU_POS、回填、固定历史与最后400-token候选 |
| [03：权重量化与内核](03-weight-quantization-and-kernels.md) | 为什么用UD-IQ3_XXS；名字不等于实际dtype；IQ2_S／IQ4_NL与Q系列的CPU成本；寻址、压缩、repack和失败的AVX2改写 |
| [04：TBQ与NXQ](04-kv-tbq-and-nxq.md) | TBQ3／TBQ4的容量、速度、PPL与KLD；FlashAttention路径；NXQ端点、物化、旋转、参考链差距、fake-quant与边界保护 |
| [05：正确性与方法](05-correctness-and-methodology.md) | 为什么流畅输出不够；静默错算、共享池stride、无效对照、测量装置错误、内存事故、日志误删、未定位到堆栈的退出崩溃 |
| [证据目录](evidence/README.md) | 125份小日志、聚合JSON、历史快照、只读解析输出、哈希、未找到和未上传的材料 |

查具体问题时，可在专题中搜索路线编号：`H01–H17`、`D01–D05`、`PC-01–PC-36`、`WQ-01–WQ-11`、`R-KV-01–R-KV-04`、`R-NXQ-01–R-NXQ-07`、`C-01–C-11`。这些编号包括研究路线与诊断事件，不代表91项独立且有效的优化。

## 2. 版本边界：记录了，不等于发布了

| 对象 | 身份 |
|---|---|
| [`7e01451b2`](https://github.com/starsder/llama.cpp/commit/7e01451b2d7aab6a4ff58ecfdd2f3b13fcaeb0bf) | 本次归档保留的原生代码基线：host路由修复及混合量化寻址保护；不是全面验收证明 |
| [`0862af564`](https://github.com/starsder/llama.cpp/commit/0862af564b48b898f7d84a57a895420b86c23824) | 本次文档提交的父版本，在上述代码基线上更新过README |
| 源工作区 `17ca0de85` ＋ 未提交改动 | 后续repack／TBQ／NXQ与缓存策略研究的来源；不因为写进文档就进入master |
| 本档案 | 只增加文档和选定证据，不上传实验源码、模型、二进制或大型logits数组 |

“已发布”仅表示相关代码在上述发布基线内；“默认关闭”“已回退”“仅设计”“本地候选”“未验收”另行标记。历史文档中的“已完成”“封板”“正确”“天花板”等原话都要结合后续更正阅读。

本次归档**没有重跑推理、性能基准或模型测试，没有编译模型程序**。仅对已有GGUF头、已有调度日志做只读解析，并检查文档和证据文件的一致性。发布基线不因这次整理获得新的速度或正确性背书。

## 3. 最容易读错的结论

1. **PLE有两个不同层级。** 最早是SSD→主存的行缓存，维护者报告约1G、90%+命中，确实节省PLE常驻内存；后来GPU PLE是另一层。不能用GPU层93.1%的日志替前一个命中率证明分母，也不能把4倍容量＋不同gather实现的对比归因于预取本身。全RAM工作点使原来的SSD收益不再突出，不等于原方案无效。
2. **teacher 99%是在线化动机，不是在线缓存命中。** 该数字按维护者回述保留。现存N+k离线报告另有68.53%等结果；缺少99%那轮完整协议，既不能合并，也不能断言两者必然采用不同分母。Fate没有得到有用信号后，SMoE才成为主线。
3. **双门控是价值门＋时限门。** 一个判断每字节预取是否值得，一个判断能否赶上消费；字节速率预算是另外的控制量。默认开关、冷启动和worker旁路需看源码条件。“全来源频率准入”是后来的另一个失败候选，不要混为一谈。
4. **命中、字节、等待和吞吐不是同一个目标。** 更高命中可以伴随更多传输和更低速度；砍掉53%逻辑预取字节也没有建立稳定加速。提交字节不是PCIe总线计数，计时斜率倒数也不是带宽实测。
5. **漂亮的host／devpart成绩曾被正确性问题推翻。** CPU半边零数据、路由拷贝时序错误都曾产生“流畅且快”。devpart的旧host对照也已失效，不能用那组数字判定修复后两条路径的净优劣。
6. **UD标签不等于所有权重同一种3-bit格式。** 实际主体是IQ2_S与IQ4_NL，另有IQ3_S例外。低bpw不保证CPU快；IQ2_S的索引／查表成本很重要，但两次失败的AVX2尝试不能证明所有实现都“到顶”。内核工具把GiB/s打印为GB/s，专题保留原数并纠正换算。
7. **TQ3/TQ4在这里指TBQ3/TBQ4，不是常规权重Q3/Q4。** TBQ4在已测512-context的1／8-chunk质量协议下不可用；同batch单chunk值是63.7301，65.0026来自另一batch探针。TBQ3缺CUDA FA白名单，发布源码路由为CPU `FLASH_ATTN_EXT`；历史二进制实际placement缺日志，不能编成“GPU先反量化再非FA”。
8. **NXQ不是原项目的忠实复现，也不是原生packed FA。** 本地GPU先存packed KV，再物化完整F16交给现有FA。参考项目运行时fake-quant、离线压缩核算与本地定长格式不同；编码／旋转对齐和局部控制没有补齐全部差距，保护边界仍有质量尾部问题。
9. **最新候选没有取得完整验收。** 18.6 t/s的CLI低于19筛选线；固定历史400步有398/400 top-1一致，另一次CLI以`0xC0000005`退出。逻辑夹具通过和24个stride GPU探针无误，都不能外推成模型数值等价或稳定加速。

更正没有悄悄覆盖原账本：专题写当前可支持的结论，[sources](sources/handoff.md)保留历史说法与状态提示，便于看见结论如何被撤回。

## 4. handoff全部45节的逐项去向

下表保留历史标题中的判断性措辞，**标题不是本档案认可的结论**；以目标专题的“观察与边界／保留与放弃”栏为准。完整原文见[handoff历史快照](sources/handoff.md)。

| 原节号 | 历史标题 | 专题去向 |
|---|---|---|
| §6.1 | 已完成的测量（2026-09-12，同一二进制、`run-cur-ref.ps1` 同级配置） | [01／H01–H05](01-host-and-devpart.md) |
| §6.2 | 已实现但中性的改动 | [01／H06](01-host-and-devpart.md) |
| §6.3 | 下一步（已与用户确认方向） | [01／H09](01-host-and-devpart.md) |
| §6.4 | 2026-09-12：延迟结构的最终结论（**拷贝机制无关**） | [01／H02／H03／H07／H09](01-host-and-devpart.md) |
| §6.5 | 2026-09-12：devpart 的 host-leaf 实验（速度已达标，正确性未解决） | [01／H09：早期虚高](01-host-and-devpart.md) |
| §6.6 | 2026-09-12：预取量与准入截止线（本次会话结论，**量是成本不是收益**） | [02／PC-12–PC-16／PC-24](02-prediction-and-cache.md) |
| §6.7 | 2026-09-12 晚：热区根治 —— **淘汰分数用错了**（本会话最大收益） | [02／PC-17–PC-21](02-prediction-and-cache.md) |
| §6.8 | 2026-09-12 深夜：自适应门槛（**极值搜索控制器**，已收敛） | [02／PC-22](02-prediction-and-cache.md) |
| §6.9 | 2026-09-12 深夜二：模型驱动自整定（B）—— 门槛与预算都自己算出来 | [02／PC-23／PC-24](02-prediction-and-cache.md) |
| §6.10 | 2026-09-12 深夜三：退出期崩溃定位 + 整机事故与防复发闸 | [05／C-07](05-correctness-and-methodology.md) |
| §6.11 | 2026-09-12 深夜四：CPU_ASYNC 转正 + 逐项开销盘点（当前 48.7ms/token） | [01／H06／H08](01-host-and-devpart.md) |
| §6.12 | 2026-09-12 深夜五：pinned 读回（无效）+ split 结构 + devpart 与自适应系统的兼容性 | [01／H02／H08／H10](01-host-and-devpart.md) |
| §6.13 | 2026-09-12 深夜六：split 合并调查 —— **那 13.1ms 是用 VRAM 换来的，不做** | [01／H08](01-host-and-devpart.md) |
| §6.14 | 2026-09-13：prefill/decode 倾向分离（实测） | [01／H13](01-host-and-devpart.md) |
| §6.15 | 2026-09-13：SMoE/缓存对 prefill 的影响（实测：**中性**） | [01／H13](01-host-and-devpart.md) |
| §6.16 | 2026-09-13：prefill 读缓存（D2D 暂存）—— **实测零收益，已回退** | [01／H12](01-host-and-devpart.md) |
| §6.17 | 2026-09-13：devpart 复查 —— **收益确认 +54%，正确性缺陷已定位到设备分区输出的可见性** | [01／H09：旧收益失效](01-host-and-devpart.md) |
| §6.18 | 2026-09-13：devpart 正确性修复 —— 定位到根因，修到一半（图构建崩） | [01／H09：图构建异常](01-host-and-devpart.md) |
| §6.19 | 2026-09-13：devpart 正确性 —— **修好了（输出连贯）**，剩余是浮点次序差异 + 同步开销 | [05／C-01：连贯不等于正确](05-correctness-and-methodology.md) |
| §6.20 | 2026-09-13：devpart —— 速度路线打通（+70%），CPU 半边数据仍待确诊 | [01／H09：CPU零数据虚高](01-host-and-devpart.md) |
| §6.21 | 2026-09-13：devpart 剩余缺陷收敛为**单点** —— 提前读回没有提交 | [01／H09：提前回读未提交](01-host-and-devpart.md) |
| §6.22 | 2026-09-13：devpart 收尾状态 —— 管道已通，缺陷收敛到"驻留表在设备侧的内容" | [01／H09：设备驻留表](01-host-and-devpart.md) |
| §6.23 | 2026-09-13：devpart 已修复（正确性对齐宿主，逐字一致） | [01／H09：当时探针修正](01-host-and-devpart.md) |
| §6.24 | 2026-09-13：devpart 400-token 稳态与"缓存不填充"根因 | [01／H09／H10：驻留决策仍在主机](01-host-and-devpart.md) |
| §6.25 | 2026-09-13：封板记录（本轮结论与状态） | [01／H09／H11：封板后撤回](01-host-and-devpart.md) |
| §6.26 | 2026-09-13：MTP 兼容性调查（MTP 能跑；缓存 × 投机前端仍崩，已二分到位） | [01／H15：MTP调查](01-host-and-devpart.md) |
| §6.27 | 2026-09-13：MTP×缓存 —— 决策与二分结论（当前状态） | [01／H15：二分](01-host-and-devpart.md) |
| §6.28 | 2026-09-13：**决定：放弃 MTP×缓存方案**（WIP 已全部回退） | [01／H15：回退](01-host-and-devpart.md) |
| §6.29 | 2026-09-13：视觉编码器 + 256k + TBQ4 KV 下的缓存参数（结论：用 auto，不要手工调） | [04／R-KV-04：auto口径纠正](04-kv-tbq-and-nxq.md) |
| §6.30 | 2026-09-13：重大发现 —— `LLAMA_MOE_SPLIT=1` 会**静默算错**（一致性/竞态 bug） | [05／C-02：路由静默错误](05-correctness-and-methodology.md) |
| §6.31 | 2026-09-13 夜：可部署性矩阵 + 安全锁（用户已睡，本轮自主收尾） | [01／H11：部署结论撤回](01-host-and-devpart.md) |
| §6.32 | 2026-09-13 深夜（自主）：**修复 split 静默算错 + 恢复速度 + 前瞻自适应**（全部已验证） | [02／PC-25／PC-26：修复与ahead](02-prediction-and-cache.md) |
| §6.33 | 检查项 1 结案：`SMOE_AHEAD=1` 的 r1 异常 = 非阻塞读回的"晚一拍" | [02／PC-27：读回晚一拍](02-prediction-and-cache.md) |
| §6.34 | 评估：IQ4_NL 的 AVX2 dot/repack 算子对我们**没有帮助**（实测） | [03／WQ-05：初判](03-weight-quantization-and-kernels.md) |
| §6.35 | 权重类型实测（unsloth UD = 动态量化，混合多格式）+ 6.34 的更正 | [03／WQ-01／WQ-02／WQ-06：dtype更正](03-weight-quantization-and-kernels.md) |
| §6.36 | 混合类型（UD）对读取策略的影响：审计 + 运行期保险 | [03／WQ-03：寻址](03-weight-quantization-and-kernels.md) |
| §6.37 | 缓存压缩空间评估：**基本没有**（实测） | [03／WQ-04：压缩与padding更正](03-weight-quantization-and-kernels.md) |
| §6.38 | 内核覆盖矩阵：我们的格式 × 计算路径 × CPU/GPU 是否都有加速算子 | [03／WQ-07：覆盖矩阵](03-weight-quantization-and-kernels.md) |
| §6.39 | CPU 算子效率实测 + "4n/repass 拉满"的可行性评估（结论：不是当前的瓶颈） | [03／WQ-08：CPU效率与单位](03-weight-quantization-and-kernels.md) |
| §6.40 | AVX2 内核可行性边界 + "做完值多少"（决定是否投工） | [03／WQ-09：可行性边界](03-weight-quantization-and-kernels.md) |
| §6.41 | AVX2 i-quant 内核优化：两个实验都做了，结论是**此路在 AVX2 上到顶** | [03／WQ-10／WQ-11：两次失败，不是普遍上界](03-weight-quantization-and-kernels.md) |
| §6.42 | KV 量化类型评测：速度 / MoE 缓存槽 / PPL / KL 散度（含一个 TBQ4 的坏点） | [04／R-KV-01–R-KV-03](04-kv-tbq-and-nxq.md) |
| §6.43 | NXQ GPU KV 压缩接入 | [04／R-NXQ-01／R-NXQ-02／R-NXQ-07](04-kv-tbq-and-nxq.md) |
| §6.44 | NexusQuant 对齐：参考真实链路、本次两处修改、两处结构性缺口 | [04／R-NXQ-02／R-NXQ-03／R-NXQ-06](04-kv-tbq-and-nxq.md) |
| §6.45 | NexusQuant 保真复现（fp16 fake-quant 路径）与结论 | [04／R-NXQ-04–R-NXQ-06](04-kv-tbq-and-nxq.md) |

## 5. 不在这45节里、但同样收录的路线

| 补充来源／路线 | 去向 |
|---|---|
| 维护者补充的最早PLE行缓存、约1G／90%+、后来GPU PLE收益小、全RAM转向 | [00](00-research-chronology.md)、[01 H14](01-host-and-devpart.md)、[原话记录](sources/author-recollection.md) |
| 静态热表、XT、CrossLayer、Fate；SMoE teacher 99%的动机 | [02 PC-01–PC-06](02-prediction-and-cache.md) |
| SMoE N+k、FIFO模拟、trace缺口、ffn_input单项与完整三项 | [02 PC-07–PC-11](02-prediction-and-cache.md)、[退化实验计划快照](sources/smoe-nk-degradation-plan.md) |
| 共享全局池、LFU_POS、跨层牺牲者、位置价值、byte-stride与UID风险 | [02 PC-28–PC-31](02-prediction-and-cache.md)、[05 C-03](05-correctness-and-methodology.md) |
| 生命周期诊断、失败的全来源频率门、空缓存／轮转回填修复、400-token固定历史与退出失败 | [02 PC-32–PC-36](02-prediction-and-cache.md)、[05 C-04／C-05／C-08／C-09](05-correctness-and-methodology.md) |
| IQ4_XS AArch64 repack修复及与本机AVX2收益的边界 | [03 WQ-11](03-weight-quantization-and-kernels.md)、[原始commit notes](evidence/commit-notes.txt) |
| 独立MoE算子、RAM／VRAM／SSD分层、专家聚类、共享专家前置、deadline重写 | [01 D01–D05](01-host-and-devpart.md)、[性能计划](sources/moe-decode-perf-plan.md)、[重建规范](sources/rebuild-spec.md) |
| 日志误删事故与未恢复事实、无效噪声容差、正确性先于速度 | [05 C-09–C-11](05-correctness-and-methodology.md)、[事件记录](evidence/deleted-logs-incident.json) |

## 6. 如何使用这份档案

- 想复用代码：先核对master里的符号和默认值，不从WIP说明推断功能已发布。
- 想比较速度：先核对二进制哈希、提示词、上下文、KV、缓存预算、步数和运行相位。`auto`、6144 MiB、2048 MiB以及不同batch不能拼成一组A/B。
- 想引用质量：区分roundtrip、teacher-forced、自由生成、PPL、KLD、top-1和logit差；KLD后的`±`在对应工具中是均值标准误，不是8个chunk的标准差，也不是等价性检验。
- 想重开失败路线：按专题中写明的前置条件重开，不沿用已撤回速度、错误单位或缺少协议的高命中数字。
- 想审计证据：从[证据索引](evidence/README.md)进入。原始哈希与公开副本哈希分列；缺失材料明确写缺失，不用重跑伪装成恢复。

## 7. 来源与致谢

本fork直接派生自 [unslothai/llama.cpp](https://github.com/unslothai/llama.cpp)，基础引擎来自 [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp) 与 [ggml](https://github.com/ggml-org/ggml)。模型文件名中的Unsloth UD指混合量化发布口径，不是本档案自创的单一格式。

NXQ参考项目是 [NexusQuant](https://github.com/jagmarques/nexusquant)，对应 [nexusquant-kv 0.6.3](https://pypi.org/project/nexusquant-kv/0.6.3/)；来源、许可证与实际实现差距详见04章。这里只分析公开材料和本地实验，没有把原项目的压缩率／质量承诺算成本fork成绩，也没有运行原项目来冒充复现。

返回[仓库README](../../README.md)。
