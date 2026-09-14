# Qwen3.8 Flash Next hidden／router原始采集包

这是一份离线研究数据包，不是模型权重，不是可直接替代上游llama.cpp的发布版。原始数组仅通过外部压缩包分发，不进Git或Git LFS。云盘链接尚未提供。

## 主包内容

| 目录 | 请求数 | 定位 |
|---|---:|---|
| collect-run1 | 3 | 早期hidden／expert采集，无router目录 |
| collect-run1b | 3 | 同提示词独立重跑，增加router；不能与run1跨运行拼接 |
| collect-run2 | 51 | 多域hidden／router／expert采集 |
| standard-smoe-20-20260910b | 20 | attention、HC、gate及专家贡献等阶段张量，未直接保存ffn_moe_input |
| standard-smoe-8-20260911 | 8 | 增加ffn_moe_input的补采 |
| DATASET_INFO | — | 本说明及元数据清单 |

提示词关系为51 ⊃ 20 ⊃ 8，不能按79个独立提示词拆分训练／测试；相同提示词的重复运行应同组划分。最早三提示词是另一个历史集合，本表列的是采集目录数，不是全包去重后的提示词数。

提示词、token记录、历史报告及原件哈希已放在[仓库轻量数据包](https://github.com/starsder/qwen3.8-flash-next-inference-research/tree/master/docs/experiments/data/routing-small)。本地原始目录不改名、不改数据；归档排除原始日志和杂项TXT，避免携带本机路径，相关脱敏报告在轻量包提供。没有模型权重、抽取的gate／HC矩阵、GGUF、NPY或可执行文件。

## 读取格式与对齐边界

### collect系列

- hidden_meta.csv：`graph_id,sched_id,layer,ne0,ne1,ne2,nb1,nb2,type,file`。该批hidden记录均为F32、ne0=2560，逻辑形状通常为[2560,1,nt]。
- experts.csv：`graph_id,split_id,layer,tensor,token_row,rank,expert_id,sched_id`。
- tokens.csv：`graph_id,ctx_id,pos,token_id`；pos为microbatch内索引，不能直接当全局上下文位置。
- router的`rl_*.bin`／`rp_*.bin`是512个F32值，每文件2048字节；只覆盖采集器接受的单token图，不覆盖完整prefill。
- 请求身份由目录给出。sched_id与ctx_id分别属于后端／上下文命名空间，不是全局request_id。
- collect-run2有1,197,360条hidden元数据记录，逻辑hidden字节14,348,677,120；experts.csv合计14,012,380条记录。这些不是独立训练样本数量。
- 29,285条token记录包含非decode记录，不能称为29,285个生成token。短序列原样保留。第47层prefill覆盖不同，需结合图的输出token选择对齐，不能仅凭形状判为损坏。

### standard系列

- tensors.csv包含`graph_id,sched_id,kind,layer,occurrence,ne0..ne3,nb0..nb3,type,bytes,file`。
- **bin按逻辑张量行紧密打包；CSV的nb*保留原tensor步长，仅用于审计，不是文件步长。** 例如topk的逻辑形状[10,2]、F32等宽int32数据只有80字节，即使原nb1为2048；不能按原nb1跳读文件。
- topk为int32，浮点张量按记录type读取。相同kind可能有不同形状和occurrence，`_o`编号不可随意丢弃。
- tokens.csv：`request_id,graph_id,round,pos,token_id,phase,batch_size`，与collect版schema不同。
- 20批每请求9,823条张量记录，总计196,460；8批每请求11,647条，总计93,176。8批每层新增两个ffn_moe_input形状，但应按实际metadata选择。

## 质量与适用范围

完整扫描过collect-run2的hidden元数据及experts.csv字段，并检查了每请求首尾hidden大小和单token router样本。20／8批采集manifest的退出码均为0。**尚未逐一验证全部数组的数值，也未重新运行模型；压缩包校验不等于实验数值正确性认证。** collect系列缺完整启动／退出码清单，不能仅凭旧文档恢复全部采集参数或二进制身份。

已知截断的standard-smoe-20-20260910、发生同名覆盖的早期smoke及其他开发smoke不混入主包，原因列在inventory.json。旧的99%级direct-recall报告使用记录下来的下一层attention残差，不是在线提前预测准确率。原始数据来自特定模型及实验分支，不保证对其他Qwen版本或上游llama.cpp适用。

## 完整性与解压

压缩文件名为`qwen3.8-hidden-routing-research.7z`，外部同时提供SHA256SUMS.txt。下载后先核对SHA-256，再用7-Zip执行`7z t qwen3.8-hidden-routing-research.7z`。解压到独立空目录，不要覆盖源码工作树；文件数量非常多，应预留足够容量与解压时间。

Git仓库只保存本说明、元数据和轻量小文件。外部分发不改变模型或第三方材料的许可要求，不应把仓库许可证自动外推到其他资产。
