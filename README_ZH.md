# MigraineGuidelineQA (偏头痛指南智能问答系统)

## 概述

本项目面向偏头痛指南的知识问答场景，目标是把 PDF 医学指南转成可检索、可审核、可跨文档推理的本地医疗知识系统。它不仅支持普通问答，还能够将表格中的药物、剂量、适应症和证据等关键信息抽取出来，写入 Elasticsearch 进行精确检索；同时保留 Markdown 文本用于语义检索；再在多份指南之间进行跨文档图谱推理。

与直接使用微软开源 GraphRAG 的默认配置不同，本项目对图谱构建流程做了医疗领域改造：

- 设计医疗专属实体和关系 schema
- 手动调优覆盖索引与查询的 Prompt
- 明确限制 LLM 自由创造无关实体
- 集成 MinerU 处理 PDF，保留图片和表格结构
- 使用自定义 chunking strategy 适配医疗指南文本
- 支持多篇文档的增量图谱更新，而不是简单覆盖历史图谱

本项目有意拆分为两个环境：
- 主应用：`${PROJECT_ROOT}`
- GraphRAG 运行环境：`${GRAPH_RAG_ROOT}`

这种隔离设计既避免了 GraphRAG 与主应用的依赖冲突，又通过受控的子进程调用实现了两者的紧密集成。

## 为什么要做这个项目

医学指南通常以 PDF 形式分发，其中不仅包含说明性文本，还包含大量表格、药物名称、剂量信息以及适应症说明。传统的检索方式难以覆盖如下典型需求：

- 某药物的推荐剂量是多少？
- 哪份指南推荐这种治疗方案？
- 不同指南之间对同类治疗的差异是什么？
- 对于特定治疗，是否有最新公开证据或更新？

本系统围绕这些真实需求设计了完整知识链路：

- PDF 上传与解析
- 表格抽取与标准化
- Elasticsearch 精准索引
- 人工审核与审批
- 向量检索支持常规语义问答
- GraphRAG 支持跨文档关系推理
- Tavily 提供最新外部知识补充

## 系统架构

```text
User uploads PDF
        |
        v
Streamlit frontend
        |
        v
Pipeline orchestration
        |
        +--> MinerU parsing -> JSON + Markdown
        |       |
        |       +--> JSON tables -> Elasticsearch extraction
        |       |                     -> review center -> final indexing
        |       |
        |       +--> Markdown -> Chroma semantic index
        |
        +--> GraphRAG (separate environment)
                |
                +--> input/ stores historical documents
                +--> output/ stores graph state
                +--> standard / standard-update incremental strategy

Query routing:
  - dosage / exact facts -> ES
  - semantic / general guidelines -> Chroma + hybrid retrieval
  - cross-guideline reasoning -> GraphRAG
  - latest external evidence -> Tavily
```

## 核心能力

### 1. PDF 导入与解析

- 支持上传一篇或多篇指南 PDF
- 通过 MinerU 在线 API 执行解析
- 自动生成 JSON 和 Markdown 两种输出
- 能保留真实的指南名称，并对历史上传文档进行累积管理

### 2. 结构化抽取

- 从解析后的 PDF 页面中提取表格内容
- 识别药物名称、剂量、适应症、禁忌与说明等字段
- 对异常或不一致结构做规范化处理
- 将有效记录写入 Elasticsearch，支持精确查询

### 3. 人工审核与确认流程

- 抽取结果不直接全部写入最终库
- 先进入审核中心进行人工检查与修正
- 审核通过后的记录才进入正式检索体系
- 该机制适用于医疗场景中的高可信要求和审计追踪

### 4. 本地语义检索

- 将 Markdown 内容纳入本地向量数据库
- 支持常规的语义问答和文档级检索
- 使用 Ollama 托管的本地模型和 embedding，支持本地化部署

### 5. GraphRAG 跨文档推理

- 构建多篇指南之间的知识图谱
- 支持跨文档比较、关联分析和复杂知识问答
- 对历史图谱状态做增量更新，避免多篇上传时覆盖旧知识
- 当图谱状态不完整时，自动回退到完整重建策略

### 6. 问题意图路由

系统能够根据用户提问类型选择不同的检索路径：

- 剂量或精确事实问题 → Elasticsearch
- 一般知识问答 → 本地向量检索
- 多指南关系或跨文档比较 → GraphRAG
- 最新公开知识 → Tavily 网络搜索

## 技术栈

### 前端与流程编排
- Python
- Streamlit
- 流程脚本负责上传、解析、抽取和索引联动

### 检索与 AI 能力
- Ollama 本地模型
- Qwen 2.5 系列模型
- 本地 embeddings
- Elasticsearch 精确/结构化检索
- Chroma 向量数据库
- GraphRAG 图谱推理
- Tavily 最新信息搜索

### 存储与数据流
- 使用 MinerU 进行 PDF 解析
- Elasticsearch 保存结构化药物和剂量记录
- Chroma 保存语义索引
- GraphRAG 保存图谱输入/输出状态
- 本地存储上传的 PDF、处理后的输出和审核记录


## 运行环境要求

### 主项目环境
- Python 3.11+
- Conda 环境: `rag_env`
- 依赖项: `requirements.txt`
- 本地服务:
  - Elasticsearch: ${ES_HOST} (默认 http://localhost:9200)
  - Ollama: ${OLLAMA_BASE_URL} (默认 http://localhost:11434)


### GraphRAG 环境

- Python 3.11
- Conda 环境: `graphrag_env`
- GraphRAG 根目录：${GRAPH_RAG_ROOT}
GraphRAG 代码仓库：Medical_migraine_graphrag
- 通过 Ollama 使用的本地模型：
  - qwen2.5:7b 或 qwen2.5:14b-instruct-q4_K_M
  - Embedding 模型：如 bge-m3
- 注意：部署 GraphRAG 仓库后，必须在其 settings.yaml 中将输入源配置为 Markdown 格式，即设置 file_type: text 以及 file_pattern: ".*\\.md$$"，以便直接使用主应用传入的 MD 文件。

## 快速开始

### 1. 安装依赖

```bash
cd ${PROJECT_ROOT}
pip install -r requirements.txt
```

### 2. 拉取本地模型

```bash
ollama pull qwen2.5:14b-instruct-q4_K_M
ollama pull bge-m3
```

### 3. 启动 Elasticsearch

确保 Elasticsearch 在 ${ES_HOST} 本地运行。

### 4. 启动应用

```bash
cd ${PROJECT_ROOT}
streamlit run app.py
```

### 5. 上传 PDFs

工作流将自动执行以下操作:

- 通过 MinerU 解析 PDF
- 生成 JSON + Markdown 输出
- 抽取剂量和治疗表格
- 展示审核记录以供批准
- 索引到 Elasticsearch
- 构建语义检索数据
- 触发独立环境中的 GraphRAG 图谱构建


## 项目结构

- [app.py](app.py) – 前端与用户工作流
- [pipeline.py](pipeline.py) – 编排引擎 (Orchestration pipeline)
- [src/ingestion/](src/ingestion/) – MinerU 解析、信息抽取、审核与 GraphRAG 准备工作
- [src/retrieval/](src/retrieval/) – 路由与检索逻辑
- [src/utils/](src/utils/) – 元数据、环境变量与路径工具

## 核心价值

本项目绝不仅仅是在 PDF 之上套了一层聊天机器人。它是一个临床知识系统，强调：

- 准确抽取治疗与剂量事实
- 入库前的可信审核
- 多文档增量累积而非盲目覆盖
- 隐私优先的本地化部署
- 用于跨指南比较的图谱推理

这使得它比单纯基于 RAG 的实现更适合医疗指南问答场景。

## 许可证

本项目仅用于研究、内部实验和演示目的。若用于临床、商业或生产环境，请务必核实底层工具（包括 MinerU、GraphRAG、Ollama、Elasticsearch 和 Tavily）的许可协议与合规要求。

## 附录：医疗 GraphRAG 定制说明与公开部署安全指南

本项目并未将 GraphRAG 用作通用的开箱即用图谱构建器。相反，它被视为一个医疗领域的知识图谱引擎，并对其进行了修改以符合指南问答的需求。

关键的领域特定改动：

- 医疗实体 Schema：Disease, Symptom, Drug, Diagnostic Criteria, Treatment, Evidence Level, Contraindication
- 基于诊断、治疗、禁忌症和证据逻辑的受控关系 Schema
- 覆盖索引与查询阶段的纯手工 Prompt 工程
- 明确限制以防止 LLM 捏造无支持的临床实体
- 采用 MinerU 进行 PDF 优先处理，以及针对指南文本的领域特定切片
- 多文档图谱累积而非盲目覆盖

这一点至关重要，因为微软默认的 GraphRAG 是通用设计，并使用了如 PERSON 和 ORGANIZATION 这样的通用标签，这与临床证据任务并不匹配。对于基于指南的问答，医疗图谱必须在语义上精确并扎根于临床。

## 公开部署注意事项

请勿在面向公众的文档中硬编码个人根目录路径。项目应使用环境变量或本地配置文件，推荐的配置模式：

- 主应用根目录：${PROJECT_ROOT} 或在 .env 中配置的本地路径
- GraphRAG 根目录：${GRAPH_RAG_ROOT}
- 模型端点：${OLLAMA_BASE_URL}
- MinerU 端点与 Token：使用环境变量，而非硬编码的绝对路径

环境变量配置示例：

```bash
export PROJECT_ROOT="${HOME}/projects/migraine-guideline-qa"
export GRAPH_RAG_ROOT="${HOME}/projects/graphrag-main"
export OLLAMA_BASE_URL="http://localhost:11434"
export ES_HOST="http://localhost:9200"
```

这确保了项目文档在不同机器间的可移植性，并避免了泄露用户本地文件系统的结构。