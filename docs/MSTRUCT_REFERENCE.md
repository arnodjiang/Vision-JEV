# MStructQA / MStructBench 本地参考

2026-09-21 只读核查；未修改 MVisQA 项目。

- 本地项目：`$MSTRUCT_ROOT`（用户自己的 MStructQA 项目目录）
- 活跃版本：`data/visual_benchmark/final_128_24lang_v5_visual_types`
- 公共仓库：https://huggingface.co/datasets/arnodjiang/MStructBench
- 本地发布清单：`data/hf_publish/MStructBench_visual_types_v1/release.json`
- canonical references SHA256（发布清单声明）：`ab1cd0fa0e2dbf212edf43d883e56d996aa66f0cde84539244f576bb06f557f1`

实际计数：128 base_id、128 case_id、24 图片语言、3072 PNG、8960 QA；87 chart、41 table；答案类型 71 numeric、55 short_text、2 list。

`benchmark.jsonl` 完整 cohort：1631 accepted、7329 needs_review。`validation_release/val.jsonl` 是 1631 条筛选子集，不等于完整 8960 条公开 evaluation cohort。自动 accepted 不等于人工核验。

真实问题包括单元格读取、极值差、条件筛选与比值排序、曲线交点计数、语义说明。可优先在字段读取和受限判别子任务检验本库，但必须保留原始全量评测并报告回退情况。

只允许 image、query 和协议允许的 source_context 进入推理。answer、audit、provenance、render.py 中的隐藏结构只能用于评分／审计；不得直接作为测试候选或模型输入。

一个案例当前仅有一个基础 QA。70 个语言配置不是 70 个不同字段；多字段吞吐实验需独立构建。外部正文拼接应遵循原项目 `scripts.evaluation.context_input.input_text`，避免遗漏依赖上下文的问题。

当前库尚未自动把 MStructQA 变成候选监督集：原 QA 缺少候选指针标签，不应把合成标签伪装成原始人工标签。不要用这 128 个案例及其语言变体训练后，再将原集合成绩报告为泛化性能。
