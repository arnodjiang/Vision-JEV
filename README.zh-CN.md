# Vision-JEV

**Efficient Multimodal Question Answering through Structured Extraction and Decisions**

通过结构化信息抽取与判别实现高效多模态问答。

[English](README.md) · [方法](docs/METHOD.md) · [数据协议](docs/DATA_FORMAT.md) · [实验协议](docs/EXPERIMENTS.md)

基于 **Qwen3.5-0.8B** 的多模态信息抽取与类型化判别研究库。

输入图片、问题以及推理时可获取的候选字段；保留 Qwen 原生视觉编码器和混合语言主干，使用新训练的 pointer head 直接输出候选概率，无需逐 token 生成答案。定位为 QA 系统的抽取与判别组件。

**状态：0.1 研究原型。未发布训练完成的 0.8B 权重，未验证精度或加速倍数。** 新增预测头随机初始化，必须训练。已完成官方 0.8B 权重的一步合成图像微调、保存恢复、批量与 LoRA 合并验证；软件原型测试不等于真实任务效果评估。见 [验证记录](docs/VALIDATION.md)。

## 已实现

- `extract`：选择 OCR 片段／表格单元格候选，返回值与候选携带的证据。
- `choice`：动态候选类别判别；`boolean`：是非判别。
- Qwen3.5 多模态主干 + 可训练候选 pointer head，绕过 vocabulary LM head。
- 冻结主干训练头，或语言 FFN LoRA 微调；保存／恢复 backbone 或 adapter 与预测头。
- JSONL 训练、推理、候选准确率／NLL／覆盖率评测；训练和验证按 source group 防泄漏。
- 显式 null 候选及低概率拒答。概率尚未校准，不等于可靠的正确率。

## 安装

Python 3.11+；建议 CPU 先验证、CUDA 做正式训练。MPS 可选择，但混合层算子支持和速度取决于环境。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest -q
vision-jev --help
```

## 数据与训练

每行一个问题。`group_id` 应标识原始文档／图表，全部翻译与重绘变体共享该 ID。图片路径相对于 JSONL 文件；无图片记录用于文本调试。

```json
{"id":"q1","group_id":"doc1","image":"table.png","task":"extract","question":"收入是多少？","candidates":[{"id":"c1","text":"12 USD","value":12,"bbox":[0.1,0.2,0.4,0.3]},{"id":"none","text":"未找到","is_null":true}],"target":"c1"}
```

`bbox` 是归一化 xyxy，由 OCR／候选生成器提供，**不是本库预测的定位框**。`target` 仅训练与评分使用，绝不拼接到模型输入。`context` 是可选的推理时外部正文。候选不能通过查阅标准答案生成。跨语言值需由候选生成器或后续呈现模块提供；当前不会自动翻译任意字段。

```bash
vision-jev train --data examples/train.jsonl --eval-data examples/eval.jsonl \
  --base Qwen/Qwen3.5-0.8B --mode lora --shuffle-candidates \
  --output checkpoints/toy --device cpu
vision-jev predict --checkpoint checkpoints/toy --data examples/eval.jsonl \
  --output outputs/predictions.jsonl --device cpu
vision-jev evaluate --data examples/eval.jsonl --predictions outputs/predictions.jsonl
```

示例仅为两个文本记录的流程演示，不足以训练有效模型。真实训练需独立的图像与候选标注集。`--revision` 可固定基础模型 commit。默认 float32、单样本训练以优先保证实现简单；0.8B 下载和真实训练需要额外时间与内存。输出目录存在时拒绝覆盖。

将训练后的 LoRA 合并到多模态主干权重，继续使用 Vision-JEV 的预测头和接口：

```bash
vision-jev merge --checkpoint checkpoints/toy --output checkpoints/toy-merged
```

## Python API

```python
from vision_jev.pipeline import VisionJEVPipeline
from vision_jev.schema import read_records

pipeline = VisionJEVPipeline("checkpoints/toy", device="cpu")
result = pipeline(read_records("examples/eval.jsonl")[0])
print(result["value"], result["probabilities"])
```

生成带图片的独立流程测试数据：`python examples/make_visual_demo.py`，随后将训练与验证路径改为 `outputs/visual-demo/train.jsonl` 和 `outputs/visual-demo/eval.jsonl`。这也是流程测试，不能代表真实 benchmark 效果。

## 架构与范围

每个问题作为独立 batch 行经过多模态主干；问题与所有候选之后的 readout token 对候选结束位置打分。交叉熵训练指针头和可选 LoRA。候选之间仍是因果顺序，**候选顺序不变性不受保证**，应使用候选重排训练与评测。

Python API 的 `pipeline.batch(records)` 支持独立问题按 batch 并行，但重复图片仍会重复编码。当前没有共享前缀缓存、在线多字段服务、OCR 引擎、自由字符串生成、自动计算规划或校准训练。Qwen3.5 的 DeltaNet 循环状态意味着不能仅修改 attention mask 实现问题隔离；后续缓存实现必须同时处理 KV、循环状态、卷积状态与多模态位置。

与生成式 QA 的集成：先获得候选 → Vision-JEV 选择证据和值 → 确定性计算或 QA 模型消费结构化结果。找不到候选时应 null／回退，而非强制选择。比较模型效果时必须计入候选生成成本与召回率。

## MStructBench

MStructQA 的 128 个基础案例、24 种语言、8,960 个 QA 变体作为冻结评测，不用来微调本模型。候选预测指标不等于原始 benchmark 的语义 ACC。现有数据没有完整的候选指针监督，接入前需独立建立候选和证据标注。见 [设计文档](docs/DESIGN.md) 与 [本地数据参考](docs/MSTRUCT_REFERENCE.md)。

## 来源与许可

- [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B)
- [Transformers Qwen3.5 实现](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3_5)
- [TypeSafe Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev)：类型化决策理念。
- [Kev](https://github.com/jaredpalmer/kev)：相关开放实现；本项目没有声称复现 Jev 私有架构。

本库原创代码 MIT；基础模型与上游数据分别遵守其许可。本库与 TypeSafe 无隶属关系。
