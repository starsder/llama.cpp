# 证据索引与来源清单

返回[完整档案](../README.md)。本目录保存的是**已有实验的可携带证据**，不是一次新的性能／质量实验。

## 1. 有什么，以及没有什么

- **125份原始小日志**，源文件合计1,599,928字节。保留所选文件的完整文字，仅作编码、路径脱敏和CRLF→LF处理；不是截取成功段落。
- [measurements.json](measurements.json)：原有CLI、pressure128、固定历史400步和旧baseline400的聚合结果。保留失败、退出码、缺失吞吐及top-1分歧；剔除巨大的逐行数组和全词表logits。来源JSON本身并非全部逐字上传。
- [provenance.json](provenance.json)：原始来源SHA-256、公开副本SHA-256、变换规则、14份数值来源、125份日志及缺失材料。原始哈希证明所取材料的身份，不代表未上传的原件可由本目录恢复。
- [publication-files.json](publication-files.json)：档案文件的最终字节哈希，清单自身除外；不包含仓库根README。
- [commit-notes.txt](commit-notes.txt)：14个相关提交的原始主题／正文。提交信息是历史陈述，不是正确性或性能证明。
- [deleted-logs-incident.json](deleted-logs-incident.json)：30份旧日志被错误通配删除的事件记录。维护者之后明确无需恢复；本目录没有恢复或伪造它们。
- 5份历史文档快照、维护者补充记录，以及6份只读解析输出。第1个GGUF分片的24字节头检查另存于provenance：tensor_count=0；权重类型输出来自含张量的第2／3分片。

**没有上传**模型权重、tensor dumps、完整logits数组、可执行文件／DLL／PDB／dump、实验源码、大型构建日志和完整调度debug日志。调度器原日志只公开解析结果与原件哈希。原始大数组缺席，意味着读者不能仅凭此目录重算全部统计；已有汇总与所保留的分歧记录仍可审查。

## 2. 身份、变换与计量规则

- `SOURCE_TREE`：维护者的实验源码工作区；`LOCAL_EVIDENCE`：本机临时证据根目录（其下可有`moe-cache/`）；`LOCAL_MODELS`：本机模型目录；`OTHER_SOURCE_TREES`：其他本地检出。它们不是仓库里的实际目录。
- 文件路径里的工作点、运行编号、文件名保留；机器根路径替换。历史快照增加状态头，旧结论不在快照中悄悄修正。
- `original_sha256`对应变换前原始字节；`archived_sha256`对应公开UTF-8副本。解码或换行变化会使两者不同，这是预期行为。
- 档案自己的`.gitattributes`固定`eol=lf`，避免Windows检出改变证据哈希。单独的终端回车字符保留。
- GGUF只读头／目录，不读张量数据，**没有计算整个GGUF的SHA-256**；不能把文件名、大小和metadata检查当作完整模型身份认证。
- MB／MiB、GB／GiB按专题的具体口径解释。`test-quantize-perf`原显示`GB/s`实际除以1024³；原输出保留，换算更正在03章。
- `acceptance.status=pass`等继承字段，只表示当时runner的判据，不能覆盖后来的正确性、可比性或失败结论。废弃的放大噪声容差不作为本档案验收依据。

在仓库根目录可用PowerShell核对一个公开副本，然后与provenance中的`archived_sha256`比较：

```powershell
Get-FileHash -Algorithm SHA256 .\docs\experiments\evidence\logs\repo\ppl-single-tbq4_0.log
```

## 3. 历史文档与维护者补充

| 快照 | 原始位置 | 原始行数 |
|---|---|---:|
| [handoff.md](../sources/handoff.md) | `handoff.md` | 1703 |
| [smoe-nk-degradation-plan.md](../sources/smoe-nk-degradation-plan.md) | `docs/smoe-nk-degradation-plan.md` | 227 |
| [rebuild-spec.md](../sources/rebuild-spec.md) | `docs/rebuild-spec.md` | 546 |
| [moe-decode-perf-plan.md](../sources/moe-decode-perf-plan.md) | `docs/moe-decode-perf-plan.md` | 293 |
| [moe-cache-score-aware-prd.md](../sources/moe-cache-score-aware-prd.md) | `docs/moe-cache-score-aware-prd.md` | 286 |

[author-recollection.md](../sources/author-recollection.md)是维护者在本次归档中补充的事实与顺序，不是上述历史文档的旧版本。PLE约1G／90%+、SMoE teacher 99%按回述保存；缺少完整运行协议的地方明确留空，不拿别的层级／指标替代。

## 4. 数值JSON来源

下列原件的哈希和大小见provenance；公开数值入口统一为[measurements.json](measurements.json)。

| 来源ID | 类别 |
|---|---|
| `optimization400/lifetime-before.json` | `measurement-json` |
| `optimization400/lifetime-after.json` | `measurement-json` |
| `optimization400/final-r1.json` | `measurement-json` |
| `optimization400/final-r2.json` | `measurement-json` |
| `optimization400/baseline-paired-r2.json` | `measurement-json` |
| `optimization400/baseline-paired.json` | `measurement-json` |
| `optimization400/timeout-smoke.json` | `measurement-json` |
| `optimization400/lock-smoke.json` | `measurement-json` |
| `global-auto128.summary.json` | `measurement-json` |
| `layer-auto128.summary.json` | `measurement-json` |
| `global-pressure128-v2.summary.json` | `measurement-json` |
| `optimization400/fixed400/experiment.json` | `measurement-json` |
| `optimization400/fixed400-final/experiment.json` | `measurement-json` |
| `baseline400-6g.summary.json` | `measurement-json` |

主要读法：CLI记录同时看请求步数、退出码、Generation是否存在、运行时二进制哈希；固定历史看`decode_all`的全部400步，steady384步只是性能子集；pressure128不是400步验收的替代。CPU逻辑夹具15 PASS和GPU stride探针24组无误也不证明模型整体数值一致。

## 5. 原始小日志逐文件索引

### 早期PLE、静态热表与XT（13份）

| 公开文件 | 原始来源 | 原始字节数 |
|---|---|---:|
| [ablation-lazy-moe2-xt.log](logs/early/ablation-lazy-moe2-xt.log) | `LOCAL_MODELS\qwen38\traces\ablation-lazy-moe2-xt.log` | 16632 |
| [ablation-lazy-moe4.log](logs/early/ablation-lazy-moe4.log) | `LOCAL_MODELS\qwen38\traces\ablation-lazy-moe4.log` | 32493 |
| [ablation-lazy-moe6.log](logs/early/ablation-lazy-moe6.log) | `LOCAL_MODELS\qwen38\traces\ablation-lazy-moe6.log` | 32493 |
| [ablation-lazy-moe8-gpuple.log](logs/early/ablation-lazy-moe8-gpuple.log) | `LOCAL_MODELS\qwen38\traces\ablation-lazy-moe8-gpuple.log` | 32896 |
| [ablation-lazy-moe8.log](logs/early/ablation-lazy-moe8.log) | `LOCAL_MODELS\qwen38\traces\ablation-lazy-moe8.log` | 32494 |
| [ablation-ple-only.csv](logs/early/ablation-ple-only.csv) | `LOCAL_MODELS\qwen38\traces\ablation-ple-only.csv` | 3155 |
| [moe-static90-8192.csv](logs/early/moe-static90-8192.csv) | `LOCAL_MODELS\qwen38\traces\moe-static90-8192.csv` | 33748 |
| [moe-xt-12288.csv](logs/early/moe-xt-12288.csv) | `LOCAL_MODELS\qwen38\traces\moe-xt-12288.csv` | 33730 |
| [ple-cpu-l2-with-gpu-l1.csv](logs/early/ple-cpu-l2-with-gpu-l1.csv) | `LOCAL_MODELS\qwen38\traces\ple-cpu-l2-with-gpu-l1.csv` | 13446 |
| [ple-gpu-l1-rawpages-1g-async.csv](logs/early/ple-gpu-l1-rawpages-1g-async.csv) | `LOCAL_MODELS\qwen38\traces\ple-gpu-l1-rawpages-1g-async.csv` | 13214 |
| [ple-gpu-l1-rawpages-1g.csv](logs/early/ple-gpu-l1-rawpages-1g.csv) | `LOCAL_MODELS\qwen38\traces\ple-gpu-l1-rawpages-1g.csv` | 10465 |
| [ple-gpu-overlap-stats.csv](logs/early/ple-gpu-overlap-stats.csv) | `LOCAL_MODELS\qwen38\traces\ple-gpu-overlap-stats.csv` | 5424 |
| [ple-lru-1g-no-prefetch.csv](logs/early/ple-lru-1g-no-prefetch.csv) | `LOCAL_MODELS\qwen38\traces\ple-lru-1g-no-prefetch.csv` | 340 |

### host／devpart、预算、计时、KV与质量原始输出（78份）

| 公开文件 | 原始来源 | 原始字节数 |
|---|---|---:|
| [L2048-err.txt](logs/repo/L2048-err.txt) | `SOURCE_TREE/L2048-err.txt` | 34698 |
| [L6144-err.txt](logs/repo/L6144-err.txt) | `SOURCE_TREE/L6144-err.txt` | 34694 |
| [a400a-err.txt](logs/repo/a400a-err.txt) | `SOURCE_TREE/a400a-err.txt` | 37407 |
| [a400b-err.txt](logs/repo/a400b-err.txt) | `SOURCE_TREE/a400b-err.txt` | 37415 |
| [auto-256k-novis-err.txt](logs/repo/auto-256k-novis-err.txt) | `SOURCE_TREE\auto-256k-novis-err.txt` | 35810 |
| [auto-256k-vis-err.txt](logs/repo/auto-256k-vis-err.txt) | `SOURCE_TREE\auto-256k-vis-err.txt` | 35837 |
| [auto-256k-vis-out.txt](logs/repo/auto-256k-vis-out.txt) | `SOURCE_TREE/auto-256k-vis-out.txt` | 2061 |
| [c1-3-err.txt](logs/repo/c1-3-err.txt) | `SOURCE_TREE\c1-3-err.txt` | 35396 |
| [cap27-a-err.txt](logs/repo/cap27-a-err.txt) | `SOURCE_TREE\cap27-a-err.txt` | 35726 |
| [cap27-a-out.txt](logs/repo/cap27-a-out.txt) | `SOURCE_TREE/cap27-a-out.txt` | 1870 |
| [cap27-b-out.txt](logs/repo/cap27-b-out.txt) | `SOURCE_TREE/cap27-b-out.txt` | 1970 |
| [cpunode-run.log](logs/repo/cpunode-run.log) | `SOURCE_TREE/cpunode-run.log` | 75057 |
| [cur-ref-err.txt](logs/repo/cur-ref-err.txt) | `SOURCE_TREE/cur-ref-err.txt` | 72910 |
| [cur-ref-out.txt](logs/repo/cur-ref-out.txt) | `SOURCE_TREE/cur-ref-out.txt` | 2166 |
| [dsh-G2-err.txt](logs/repo/dsh-G2-err.txt) | `SOURCE_TREE/dsh-G2-err.txt` | 4100 |
| [dsh-G2-out.txt](logs/repo/dsh-G2-out.txt) | `SOURCE_TREE/dsh-G2-out.txt` | 1932 |
| [dsh-r3-timing.txt](logs/repo/dsh-r3-timing.txt) | `SOURCE_TREE/dsh-r3-timing.txt` | 1103 |
| [dsh-r4-timing.txt](logs/repo/dsh-r4-timing.txt) | `SOURCE_TREE/dsh-r4-timing.txt` | 1378 |
| [dsh-r5-timing.txt](logs/repo/dsh-r5-timing.txt) | `SOURCE_TREE/dsh-r5-timing.txt` | 3037 |
| [dsh-r8.txt](logs/repo/dsh-r8.txt) | `SOURCE_TREE\dsh-r8.txt` | 772 |
| [dsh-r9.txt](logs/repo/dsh-r9.txt) | `SOURCE_TREE\dsh-r9.txt` | 656 |
| [fused-final.log](logs/repo/fused-final.log) | `SOURCE_TREE/fused-final.log` | 17375 |
| [guard-1-out.txt](logs/repo/guard-1-out.txt) | `SOURCE_TREE\guard-1-out.txt` | 1571 |
| [kld-matrix.log](logs/repo/kld-matrix.log) | `SOURCE_TREE\kld-matrix.log` | 2035 |
| [kld-tbq4_0.log](logs/repo/kld-tbq4_0.log) | `SOURCE_TREE\kld-tbq4_0.log` | 552 |
| [kv-f16-err.txt](logs/repo/kv-f16-err.txt) | `SOURCE_TREE\kv-f16-err.txt` | 3483 |
| [kv-f16-out.txt](logs/repo/kv-f16-out.txt) | `SOURCE_TREE\kv-f16-out.txt` | 1442 |
| [kv-q4_0-err.txt](logs/repo/kv-q4_0-err.txt) | `SOURCE_TREE\kv-q4_0-err.txt` | 35406 |
| [kv-q4_0-out.txt](logs/repo/kv-q4_0-out.txt) | `SOURCE_TREE\kv-q4_0-out.txt` | 1737 |
| [kv-q8_0-err.txt](logs/repo/kv-q8_0-err.txt) | `SOURCE_TREE\kv-q8_0-err.txt` | 35407 |
| [kv-q8_0-out.txt](logs/repo/kv-q8_0-out.txt) | `SOURCE_TREE\kv-q8_0-out.txt` | 1496 |
| [kv-tbq3_0-err.txt](logs/repo/kv-tbq3_0-err.txt) | `SOURCE_TREE\kv-tbq3_0-err.txt` | 35407 |
| [kv-tbq3_0-out.txt](logs/repo/kv-tbq3_0-out.txt) | `SOURCE_TREE\kv-tbq3_0-out.txt` | 1484 |
| [kv-tbq4_0-err.txt](logs/repo/kv-tbq4_0-err.txt) | `SOURCE_TREE\kv-tbq4_0-err.txt` | 35407 |
| [kv-tbq4_0-out.txt](logs/repo/kv-tbq4_0-out.txt) | `SOURCE_TREE\kv-tbq4_0-out.txt` | 1824 |
| [long-run.log](logs/repo/long-run.log) | `SOURCE_TREE/long-run.log` | 27109 |
| [phase-run.log](logs/repo/phase-run.log) | `SOURCE_TREE/phase-run.log` | 17615 |
| [ppl-8ch-tbq4-retry.log](logs/repo/ppl-8ch-tbq4-retry.log) | `SOURCE_TREE\ppl-8ch-tbq4-retry.log` | 1178 |
| [ppl-matrix.log](logs/repo/ppl-matrix.log) | `SOURCE_TREE\ppl-matrix.log` | 280 |
| [ppl-nb-tbq4_0.log](logs/repo/ppl-nb-tbq4_0.log) | `SOURCE_TREE\ppl-nb-tbq4_0.log` | 18047 |
| [ppl-nofa-tbq4.log](logs/repo/ppl-nofa-tbq4.log) | `SOURCE_TREE\ppl-nofa-tbq4.log` | 867 |
| [ppl-rot-tbq3_0-rot_off.log](logs/repo/ppl-rot-tbq3_0-rot_off.log) | `SOURCE_TREE\ppl-rot-tbq3_0-rot_off.log` | 18106 |
| [ppl-rot-tbq3_0-rot_on.log](logs/repo/ppl-rot-tbq3_0-rot_on.log) | `SOURCE_TREE\ppl-rot-tbq3_0-rot_on.log` | 17924 |
| [ppl-rot-tbq4_0-rot_off.log](logs/repo/ppl-rot-tbq4_0-rot_off.log) | `SOURCE_TREE\ppl-rot-tbq4_0-rot_off.log` | 18108 |
| [ppl-single-f16.log](logs/repo/ppl-single-f16.log) | `SOURCE_TREE\ppl-single-f16.log` | 17924 |
| [ppl-single-q4_0.log](logs/repo/ppl-single-q4_0.log) | `SOURCE_TREE\ppl-single-q4_0.log` | 17924 |
| [ppl-single-q8_0.log](logs/repo/ppl-single-q8_0.log) | `SOURCE_TREE\ppl-single-q8_0.log` | 17924 |
| [ppl-single-tbq3_0.log](logs/repo/ppl-single-tbq3_0.log) | `SOURCE_TREE\ppl-single-tbq3_0.log` | 17924 |
| [ppl-single-tbq4_0.log](logs/repo/ppl-single-tbq4_0.log) | `SOURCE_TREE\ppl-single-tbq4_0.log` | 17926 |
| [ppl-tbq4_0.log](logs/repo/ppl-tbq4_0.log) | `SOURCE_TREE\ppl-tbq4_0.log` | 671 |
| [sweep-2x2.log](logs/repo/sweep-2x2.log) | `SOURCE_TREE/sweep-2x2.log` | 180 |
| [sweep-ab.csv](logs/repo/sweep-ab.csv) | `SOURCE_TREE/sweep-ab.csv` | 281 |
| [sweep-ahead.csv](logs/repo/sweep-ahead.csv) | `SOURCE_TREE\sweep-ahead.csv` | 210 |
| [sweep-ahead2.csv](logs/repo/sweep-ahead2.csv) | `SOURCE_TREE/sweep-ahead2.csv` | 298 |
| [sweep-auto.csv](logs/repo/sweep-auto.csv) | `SOURCE_TREE/sweep-auto.csv` | 157 |
| [sweep-auto8k.csv](logs/repo/sweep-auto8k.csv) | `SOURCE_TREE/sweep-auto8k.csv` | 128 |
| [sweep-budget.csv](logs/repo/sweep-budget.csv) | `SOURCE_TREE/sweep-budget.csv` | 247 |
| [sweep-c1.csv](logs/repo/sweep-c1.csv) | `SOURCE_TREE/sweep-c1.csv` | 301 |
| [sweep-c1nb.csv](logs/repo/sweep-c1nb.csv) | `SOURCE_TREE/sweep-c1nb.csv` | 197 |
| [sweep-corrupt.csv](logs/repo/sweep-corrupt.csv) | `SOURCE_TREE\sweep-corrupt.csv` | 332 |
| [sweep-fix.csv](logs/repo/sweep-fix.csv) | `SOURCE_TREE\sweep-fix.csv` | 195 |
| [sweep-hd.csv](logs/repo/sweep-hd.csv) | `SOURCE_TREE/sweep-hd.csv` | 433 |
| [sweep-kv.csv](logs/repo/sweep-kv.csv) | `SOURCE_TREE\sweep-kv.csv` | 144 |
| [sweep-limit.csv](logs/repo/sweep-limit.csv) | `SOURCE_TREE/sweep-limit.csv` | 427 |
| [sweep-neutral.csv](logs/repo/sweep-neutral.csv) | `SOURCE_TREE\sweep-neutral.csv` | 203 |
| [sweep-press.csv](logs/repo/sweep-press.csv) | `SOURCE_TREE/sweep-press.csv` | 306 |
| [sweep-reg.csv](logs/repo/sweep-reg.csv) | `SOURCE_TREE\sweep-reg.csv` | 297 |
| [sweep-smoke.csv](logs/repo/sweep-smoke.csv) | `SOURCE_TREE/sweep-smoke.csv` | 124 |
| [sweep-t2.csv](logs/repo/sweep-t2.csv) | `SOURCE_TREE/sweep-t2.csv` | 196 |
| [sweep-tune.csv](logs/repo/sweep-tune.csv) | `SOURCE_TREE/sweep-tune.csv` | 195 |
| [sweep-u.csv](logs/repo/sweep-u.csv) | `SOURCE_TREE/sweep-u.csv` | 141 |
| [sweep-v.csv](logs/repo/sweep-v.csv) | `SOURCE_TREE/sweep-v.csv` | 196 |
| [sweep-x.csv](logs/repo/sweep-x.csv) | `SOURCE_TREE/sweep-x.csv` | 174 |
| [t-4096-err.txt](logs/repo/t-4096-err.txt) | `SOURCE_TREE\t-4096-err.txt` | 35283 |
| [t-4096-out.txt](logs/repo/t-4096-out.txt) | `SOURCE_TREE\t-4096-out.txt` | 1517 |
| [t-auto-err.txt](logs/repo/t-auto-err.txt) | `SOURCE_TREE\t-auto-err.txt` | 35365 |
| [t-auto-out.txt](logs/repo/t-auto-out.txt) | `SOURCE_TREE\t-auto-out.txt` | 1714 |
| [x-tiny-err.txt](logs/repo/x-tiny-err.txt) | `SOURCE_TREE\x-tiny-err.txt` | 3369 |

### 最后400-token CLI与失败运行（14份）

| 公开文件 | 原始来源 | 原始字节数 |
|---|---|---:|
| [opt400-baseline-paired-err.txt](logs/cli/opt400-baseline-paired-err.txt) | `SOURCE_TREE\opt400-baseline-paired-err.txt` | 0 |
| [opt400-baseline-paired-out.txt](logs/cli/opt400-baseline-paired-out.txt) | `SOURCE_TREE\opt400-baseline-paired-out.txt` | 0 |
| [opt400-baseline-paired-r2-err.txt](logs/cli/opt400-baseline-paired-r2-err.txt) | `SOURCE_TREE\opt400-baseline-paired-r2-err.txt` | 35428 |
| [opt400-baseline-paired-r2-out.txt](logs/cli/opt400-baseline-paired-r2-out.txt) | `SOURCE_TREE\opt400-baseline-paired-r2-out.txt` | 1784 |
| [opt400-final-r1-err.txt](logs/cli/opt400-final-r1-err.txt) | `SOURCE_TREE\opt400-final-r1-err.txt` | 37179 |
| [opt400-final-r1-out.txt](logs/cli/opt400-final-r1-out.txt) | `SOURCE_TREE\opt400-final-r1-out.txt` | 2226 |
| [opt400-final-r2-err.txt](logs/cli/opt400-final-r2-err.txt) | `SOURCE_TREE\opt400-final-r2-err.txt` | 16699 |
| [opt400-final-r2-out.txt](logs/cli/opt400-final-r2-out.txt) | `SOURCE_TREE\opt400-final-r2-out.txt` | 2163 |
| [opt400-lifetime-after-err.txt](logs/cli/opt400-lifetime-after-err.txt) | `SOURCE_TREE\opt400-lifetime-after-err.txt` | 36524 |
| [opt400-lifetime-after-out.txt](logs/cli/opt400-lifetime-after-out.txt) | `SOURCE_TREE\opt400-lifetime-after-out.txt` | 1751 |
| [opt400-lifetime-before-err.txt](logs/cli/opt400-lifetime-before-err.txt) | `SOURCE_TREE\opt400-lifetime-before-err.txt` | 36518 |
| [opt400-lifetime-before-out.txt](logs/cli/opt400-lifetime-before-out.txt) | `SOURCE_TREE\opt400-lifetime-before-out.txt` | 1784 |
| [opt400-timeout-smoke-err.txt](logs/cli/opt400-timeout-smoke-err.txt) | `SOURCE_TREE\opt400-timeout-smoke-err.txt` | 0 |
| [opt400-timeout-smoke-out.txt](logs/cli/opt400-timeout-smoke-out.txt) | `SOURCE_TREE\opt400-timeout-smoke-out.txt` | 0 |

### 逻辑夹具与stride memcheck（4份）

| 公开文件 | 原始来源 | 原始字节数 |
|---|---|---:|
| [logic-empty-before.log](logs/logic/logic-empty-before.log) | `LOCAL_EVIDENCE\moe-cache\optimization400\logic-empty-before.log` | 933 |
| [logic-final.log](logs/logic/logic-final.log) | `LOCAL_EVIDENCE\moe-cache\optimization400\logic-final.log` | 967 |
| [logic-round-robin-before.log](logs/logic/logic-round-robin-before.log) | `LOCAL_EVIDENCE\moe-cache\optimization400\logic-round-robin-before.log` | 671 |
| [stride-memcheck.log](logs/logic/stride-memcheck.log) | `LOCAL_EVIDENCE\moe-cache\stride-memcheck.log` | 64 |

### NXQ／E8／KV局部质量诊断（16份）

| 公开文件 | 原始来源 | 原始字节数 |
|---|---|---:|
| [e8-kld-all.log](logs/kv/e8-kld-all.log) | `LOCAL_EVIDENCE\e8-kld-all.log` | 1551 |
| [e8-m1b.log](logs/kv/e8-m1b.log) | `LOCAL_EVIDENCE\e8-m1b.log` | 20377 |
| [e8-m2-direct.log](logs/kv/e8-m2-direct.log) | `LOCAL_EVIDENCE\e8-m2-direct.log` | 640 |
| [e8-m2b.log](logs/kv/e8-m2b.log) | `LOCAL_EVIDENCE\e8-m2b.log` | 20377 |
| [e8-protect.log](logs/kv/e8-protect.log) | `LOCAL_EVIDENCE\e8-protect.log` | 660 |
| [nxq-align-kld-v64.log](logs/kv/nxq-align-kld-v64.log) | `LOCAL_EVIDENCE\nxq-align-kld-v64.log` | 894 |
| [nxq-align-kld.log](logs/kv/nxq-align-kld.log) | `LOCAL_EVIDENCE\nxq-align-kld.log` | 1309 |
| [nxq-align-test.log](logs/kv/nxq-align-test.log) | `LOCAL_EVIDENCE\nxq-align-test.log` | 718 |
| [nxq-quality-nxq22.log](logs/kv/nxq-quality-nxq22.log) | `LOCAL_EVIDENCE\nxq-quality-nxq22.log` | 37473 |
| [nxq-quality-nxq32.log](logs/kv/nxq-quality-nxq32.log) | `LOCAL_EVIDENCE\nxq-quality-nxq32.log` | 37472 |
| [nxq-quality-q8.log](logs/kv/nxq-quality-q8.log) | `LOCAL_EVIDENCE\nxq-quality-q8.log` | 37467 |
| [nxq-tbo-cpy.log](logs/kv/nxq-tbo-cpy.log) | `LOCAL_EVIDENCE\nxq-tbo-cpy.log` | 2070 |
| [nxq-tbo-sr.log](logs/kv/nxq-tbo-sr.log) | `LOCAL_EVIDENCE\nxq-tbo-sr.log` | 79658 |
| [nxq_base_f16_nkvo.log](logs/kv/nxq_base_f16_nkvo.log) | `LOCAL_EVIDENCE\nxq_base_f16_nkvo.log` | 552 |
| [nxq_ppl.log](logs/kv/nxq_ppl.log) | `LOCAL_EVIDENCE\nxq_ppl.log` | 179 |
| [nxq_ppl_nkvo.log](logs/kv/nxq_ppl_nkvo.log) | `LOCAL_EVIDENCE\nxq_ppl_nkvo.log` | 552 |

## 6. 归档时的只读核对

这些输出解析已有材料，不运行模型、不加载权重张量、不构建程序。脚本哈希、命令、输入大小与可获得的输入哈希在provenance中。

| 输出 | 范围 |
|---|---|
| [gguf-shard2-types.txt](scans/gguf-shard2-types.txt) | GGUF头／tensor目录 |
| [gguf-shard3-types.txt](scans/gguf-shard3-types.txt) | GGUF头／tensor目录 |
| [gguf-shard2-geometry.txt](scans/gguf-shard2-geometry.txt) | GGUF头／tensor目录 |
| [gguf-shard3-geometry.txt](scans/gguf-shard3-geometry.txt) | GGUF头／tensor目录 |
| [scheduler-text.txt](scans/scheduler-text.txt) | 已有调度日志的backend汇总 |
| [scheduler-vision.txt](scans/scheduler-vision.txt) | 已有调度日志的backend汇总 |

第2、3分片分别列出468与756个tensor，共1224；各行GiB显示值已经四舍五入，不要把分项显示数的末位差误认作原始字节差。

## 7. 缺失证据与不可补推的部分

- `SOURCE_TREE/dsh-r6-timing.txt`：`not-found-at-archive-time`。
- `SOURCE_TREE/dsh-r7-timing.txt`：`not-found-at-archive-time`。

此外，teacher 99%的完整原协议、部分早期独立profiling输出、最新退出失败的原生故障堆栈和原项目NXQ实际复现结果没有进入本档案。某些历史数字只有当时笔记或提交说明，专题按证据等级标注；不能因缺少反例就判通过，也不能以新跑的相似结果冒充原件。

原日志中的warning、旧结论、失败和异常不等于本档案的推荐设置。请优先阅读对应专题的更正与结论边界。
