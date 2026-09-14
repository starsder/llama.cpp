# 路由研究轻量数据包

先提供容易下载、便于核对的原始小文件：**164份，公开正文合计735,256字节（约0.70 MiB）**，另附本说明与来源清单。大体积hidden／router二进制不进Git、不使用Git LFS，后续由维护者上传云盘后单独提供入口。

返回[实验档案](../../README.md) · [证据总览](../../evidence/README.md)。

## 文件入口

| 内容 | 数量 | 入口 |
|---|---:|---|
| 多域提示词原件 | 51 | [collect-prompts2](files/collect-prompts2/) |
| 最早三提示词原件 | 3 | [collect-prompts](files/collect-prompts/) |
| 51请求的token记录 | 51 | [collect-run2](files/collect-run2/)；各请求目录内为tokens.csv |
| 同三提示词两次独立运行的token记录 | 6 | [collect-run1](files/collect-run1/) · [collect-run1b](files/collect-run1b/) |
| 20请求采集清单及token记录 | 21 | [standard-smoe-20-20260910b](files/standard-smoe-20-20260910b/) |
| 8请求补采清单及token记录 | 9 | [standard-smoe-8-20260911](files/standard-smoe-8-20260911/) |
| 20请求阶段张量的派生汇总 | 5 | [standard summary](files/standard-smoe-20-20260910b-summary/) |
| 使用teacher-forced attention的下一层gate评估 | 4 | [direct recall](files/direct-smoe-recall-20-20260910b/) |
| N+k消融 | 2 | [N+k结果](files/smoe-nk-8-20260911/) |
| 51请求的FIFO离线评估及预算变体 | 10 | [eval](files/offline-smoe-eval-20260910/) · [budget](files/offline-smoe-budget-20260910/) |
| hidden预测探针的历史输出 | 2 | [generalize-out](files/generalize-out.txt) · [probe4-out](files/probe4-out.txt) |
| 来源、字节数与SHA-256 | — | [manifest.json](manifest.json) |

## 哪些不能混在一起

- **51 → 20 → 8是提示词子集关系，不是79个独立提示词。** 20与8来自新一轮采集，不能将不同运行的hidden和router文件交叉拼接。同提示词的不同采集也应放在同一训练／测试分组。
- `collect-*`的token表是`graph_id,ctx_id,pos,token_id`；`standard-smoe-*`增加`request_id,round,phase,batch_size`。不同采集器的`pos`不能未经核实直接当作全局上下文位置。
- 采集图包含非decode图；token表行数不是生成token数。短于生成上限的记录原样保留，不补齐或冒充等长序列。
- `direct recall`的99%级recall@16使用了记录下来的下一层attention残差，是有条件离线模拟，不是在线提前预测准确率；也没有据此认定它就是维护者回述的那一次teacher实验。
- FIFO报告不是SMoE gate预测器的实现结果；报告自身拒绝在缺attention／专家贡献时给出完整反事实hidden重建结论。
- 20请求原始张量集没有直接保存`ffn_moe_input`，8请求补采才增加它。当前轻量包只公开清单和token记录，不包含这两批张量数据。

## 身份与分发边界

这些是已有实验的小文件副本，没有重跑模型。正文仅统一UTF-8／LF并替换本机根路径；旧报告的结果与措辞保留，解读边界见本页及[预测专题](../../02-prediction-and-cache.md)。`manifest.json`分列原件与公开副本哈希。

提示词内容、输入token与生成token记录均保留，便于核对实验；本包不是完整tokenizer或模型快照。模型权重、抽取的gate／HC权重、激活数组、完整词表logits、可执行文件和本机环境凭据均不包含在内。不要把本仓库许可证自动外推为未提供的模型或第三方数据的许可。

**大数据下载：尚无云盘链接。** 大体积原始数据将只在本地打包，待维护者上传并提供链接后更新下载说明；仓库克隆不需要下载这些数组。
