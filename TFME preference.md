# Text Mining for Economics & Finance: 教授研究偏好深度分析报告

这份文档通过分析TMEF课程的完整代码作业（从Assignment 1到Assignment 6），总结了你们教授在文本挖掘领域的核心研究哲学、方法论偏好以及对大语言模型（LLM）的独特视角。

## 1. 核心哲学：解释性 > 预测力 (Interpretability > Accuracy)

这是贯穿整个课程最显著的特征。你的教授不仅仅关注模型能否做出准确预测，更执着于**打开黑盒**，理解模型背后的逻辑。

*   **证据**：
    *   在Assignment 1、3、5、6中，几乎每个作业的最后一步都是提取Lasso回归的**系数（Coefficients）**。
    *   **标志性可视化**：他非常推崇绘制“火山图”（Volcano Plot），横轴是特征对预测结果的影响力（Coefficient），纵轴是特征出现的频率（Frequency）。
*   **潜台词**：在社会科学与经济学研究中，知道“为什么这个词会导致股价下跌”比单纯“预测准股价下跌”更有价值。这种因果推断风格（Causal-Style Inference）是经济学家的典型特征。

## 2. 方法论偏好：稀疏线性模型与混合特征

教授偏好使用**可解释的线性模型**（如Lasso/Elastic Net）来处理高维文本数据，而不是直接使用深度学习黑盒。

### A. 稀疏性与特征选择 (Sparsity & Feature Selection)
*   **工具**：高度依赖 `glmnet` 包。
*   **逻辑**：通过L1正则化（Lasso）将不重要的特征系数压缩为0，从而筛选出真正具有经济学意义的关键特征（Key Drivers）。

### B. 混合特征工程 (Hybrid Feature Engineering)
他不满足于简单的词袋模型（Bag-of-Words），而是强调结合**领域知识**构建高级特征：
*   **结构性特征**：例如Assignment 6中，专门计算了“财报会议中第一个提问的字数”和“第一个回答的字数”，研究其与EPS的关联。
*   **社会语言学特征**：
    *   **关键证据**：Assignment 5中明确提到 `library(politeness) # a package I wrote :)`。
    *   **含义**：这直接表明教授的研究重心是**社会语言学（Sociolinguistics）**，特别是如何量化“礼貌”、“地位”、“情感”等抽象社会概念，并研究它们对经济结果的影响。

### C. “代理模型”解释法 (Surrogate Modeling)
*   **高阶技巧**：在Assignment 6中，即便使用了先进的Embedding（嵌入向量），他依然训练了一个简单的线性模型（N-gram Lasso）去拟合Embedding模型的预测结果。
*   **目的**：利用线性模型的透明性，去解释复杂的Embedding模型到底看重了什么。这是非常经典的“打开黑盒”的研究手段。

## 3. 评估指标：经济学视角的排序 (Ranking > Classification)

*   **首选指标**：**Kendall’s Tau (Kendall Rank Correlation Coefficient)**。
*   **原因**：在金融和经济学中，我们往往更关心**相对排序**（例如：这类公司的风险是否比那类公司高？这篇财报是否比那篇更乐观？），而不是绝对的分类准确率。Kendall相关系数能更好地衡量这种排序的一致性。
*   **对比**：传统的Accuracy或F1-Score在仅仅关心“涨/跌”时有用，但在评估连续的经济指标（如收益率、评分、风险等级）时，Kendall Tau更具鲁棒性。

## 4. 对LLM的态度：强大的工具，但非终点

教授对大语言模型（LLM）持开放但审慎的态度。他将其视为增强传统研究的工具，而不是替代品。

*   **LLM的角色**：
    1.  **高效标注员** (Zero-shot Annotation)：利用LLM（如Gemini）直接生成标签（Assignment 6中的情感、价格、性别预测），解决标注数据稀缺的问题。
    2.  **特征提取器** (Feature Extractor)：利用LLM生成高质量的文本嵌入（Embeddings），捕捉深层语义。
*   **坚持底线**：即便用了LLM，最后一步依然要回归到统计模型（Regression/Lasso）来进行分析和归因。

## 5. 战略建议：如何在课程/项目中拿高分？

基于上述分析，如果你想在期末项目或考试中获得教授的青睐，建议遵循以下策略：

1.  **必须画出系数图**：无论你用了多复杂的模型，最后一定要尝试提取出Top Features（正向/负向影响最大的词或短语），并画图展示。
2.  **讲好“经济学故事”**：对着系数图解释：“模型发现 *'uncertainty'* 这个词与股价下跌显著相关，这符合...理论”。不要只扔一个Accuracy数字就结束。
3.  **尝试构建“聪明”的特征**：不要只用N-gram。试着想一想：句子的长度、被动语态的使用率、特定的行业术语密度……这些结构化特征往往比纯文本更能打动老师。
4.  **关注排序能力**：在模型评估时，记得计算一下Kendall Tau，证明你的模型能正确地给样本排序。

## 总结

你的老师是一位典型的**计算社会科学家（Computational Social Scientist）**。他的研究范式是：

> **Text Data + Domain Knowledge (Features) -> Interpretable Model (Lasso) -> Economic Insight**

掌握了这个核心公式，你就掌握了TMEF这门课的通关密码。