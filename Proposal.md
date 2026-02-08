# 项目提案：S&P 500 收益电话会议中的 AI 叙事——基于迁移学习与对话动态的分析

# (Proposal: Deciphering AI Narratives in S&P 500 Earnings Calls via Transfer Learning)

### 1. 动机与贡献 (Motivation and Contribution)

人工智能（AI）已成为市场的主导叙事，但企业对 AI 的披露动机各不相同。目前的难题在于区分**实质性的技术布局**与**管理层的“蹭热度”（Greenwashing/AI-washing）**。

* **现状痛点：** 简单的关键词匹配无法捕捉复杂的语境，且忽略了收益电话会议（Earnings Calls）中“有稿演讲（Prepared Remarks）”与“问答环节（Q&A）”的结构性差异。
* **核心贡献：** 本项目旨在通过**迁移学习（Transfer Learning）**构建一个高精度的 AI 话题检测器，并利用该工具深入剖析 AI 话题在电话会议中的**发起机制（Initiation Mechanism）**——即 AI 话题是由管理层主动推销的，还是由分析师追问出来的？

### 2. 研究问题 (Research Questions)

根据教授建议，我们将重心从“预测股价”转向“检测趋势与互动模式”：

* **RQ1 (方法论验证):** 基于外部 AI 新闻数据训练的编码器（Encoder），能否通过**迁移学习**有效识别收益电话会议中特定且微妙的 AI 讨论（如生成式 AI、自动化、算力基础设施），其表现是否优于传统的关键词词典？
* **RQ2 (互动动态 - Who Drives the Narrative?):** AI 叙事主要是由管理层在演讲中**自上而下（Top-down）**发起的，还是由分析师在 Q&A 中**自下而上（Bottom-up）**引发的？这两者在时间轴上（如 ChatGPT 发布前后）有何变化？
* **RQ3 (特征关联):** “管理层主动提及 AI 率”较高的公司，具备什么样的财务特征（基于 WRDS 数据，如研发投入占比、市盈率、股价波动率）？是否存在管理层在业绩不佳时利用 AI 叙事进行转移视线的现象？

### 3. 数据描述 (Data Description)

我们将整合三个不同来源的数据集：

1. **主要语料库：S&P 500 Earnings Transcripts (2013-2025)**
* 来源：Hugging Face (`glopardo/sp500-earnings-transcripts`) 。


* **关键处理：** 我们将编写脚本，将每份会议记录**严格切分**为“管理层演讲（Speech）”和“问答环节（Q&A）”两部分。对于 Q&A，我们将进一步细化为“提问（Question）”与“回答（Answer）”的对话轮次（Turn-level）数据。


2. **迁移学习训练集：AI News & Media Dataset**
* 来源：自备的富含 AI 主题的新闻 CSV 数据集。
* 用途：作为源域（Source Domain），用于训练深度学习模型以捕捉 AI 相关的语义特征。


3. **财务元数据：WRDS S&P 500 Compustat Quarterly**
   * **数据文件**：`Sp500_meta_data.csv`（已从 WRDS 下载，约 10,400+ 条记录）
   * **时间范围**：2020Q1 - 2025Q1（季度频率）
   * **核心字段**：
     | 字段名 | 描述 | 用途 |
     |--------|------|------|
     | `tic` / `gvkey` / `conm` | 股票代码 / GVKEY / 公司全称 | 公司标识与匹配 |
     | `gsector` / `gsubind` / `sic` | GICS 行业大类 / 子行业 / SIC 代码 | 行业分类分析 |
     | `datacqtr` / `datadate` / `rdq` | 日历季度 / 数据日期 / 报告发布日 | 时间序列对齐与事件研究 |
     | `epspxq` | 每股收益（排除异常项目） | 业绩表现 (Beat/Miss) 判定 |
     | `xrdq` | 研发费用（季度） | R&D Intensity 计算 |
     | `mkvaltq` / `prccq` | 市值 / 季末股价 | 公司规模与市场反应 |
     | `cshoq` | 普通股发行数量 | 估值与加权计算 |
   * **数据来源**：WRDS Compustat North America Quarterly Fundamentals
   * **用途**：与 Earnings Transcripts 基于 `tic` (股票代码) 和 `datacqtr` (季度) 进行合并，解决 Hugging Face 数据集元数据不足的问题，并支持 RQ3 中的截面回归分析。





### 4. 方法论 (Methodology)

#### 4.1 迁移学习与 AI 话题检测 (Transfer Learning Pipeline)

为了响应“Transfer Learning”的要求，我们将摒弃简单的关键词计数：

* **第一步（预训练/微调）：** 使用**AI 新闻数据集**对预训练语言模型（如 FinBERT 或 RoBERTa）进行微调（Fine-tuning）或训练一个分类头（Classification Head）。该模型将被训练去识别“AI 创新”、“AI 风险”、“AI 落地”等具体主题。
* **第二步（领域迁移）：** 将训练好的模型应用到**收益电话会议（目标域）**的句子级别数据上。
* 
**优势：** 这种方法能识别出没有直接提及 "AI" 单词但语义相关的句子（例如描述“由于自动化算法提高了效率”），解决了传统方法的漏判问题 。



#### 4.2 结构化文本分析 (Structure-Aware Text Mining)

我们将分别计算以下指标：

* **Speech AI Intensity:** 管理层在演讲稿中提及 AI 的频率（反映战略意图/公关策略）。
* **Q&A AI Intensity:** 问答环节中 AI 的讨论热度。
* **AI Initiation Score (创新指标):** 在 Q&A 环节中，计算 AI 话题由**分析师提问**首先引入的比例，与**管理层在回答无关问题时强行引入** AI 的比例。这能有效区分“市场关注”与“管理层炒作”。

#### 4.3 描述性统计与回归分析 (Descriptive & Explanatory Analysis)

* **趋势图：** 绘制演讲与问答中 AI 热度随时间的分离趋势（例如：是否存在管理层热度先于分析师热度的情况？）。
* **截面回归 (Cross-sectional Regression):**
*  = AI Initiation Score (管理层主动性)
*  = 公司特征 (Lagged Returns, R&D Intensity, Miss/Beat Earnings)
* *目的：* 检验是否是业绩较差的公司更倾向于主动谈论 AI。



### 5. 预期成果与图表 (Planned Outputs)

1. 
**模型性能报告：** 展示迁移学习模型在电话会议样本上的分类准确率（Precision/Recall），并与基于字典（Dictionary-based）的基准模型进行对比 。


2. **时间序列图：** "Speech vs. Q&A" 的 AI 提及率对比图，标注关键时间点（如 ChatGPT 发布）。
3. **象限图：** 将公司分为四类——“言行一致型”（演讲和Q&A都多）、“被动响应型”（只在Q&A被问）、“自嗨型”（只在演讲中说，没人问）、“沉默型”。
4. **回归表：** 展示公司财务基本面如何驱动不同类型的 AI 叙事模式。

### 6. 局限性 (Limitations)

* 
**因果关系：** 我们的分析主要是相关性分析，难以确立因果关系（例如是 AI 叙事导致股价上涨，还是股价上涨的公司更爱聊 AI） 。


* **领域适应挑战：** 新闻语料与电话会议语料在文风上存在差异，迁移学习模型可能需要通过领域适应（Domain Adaptation）技术进行调整。