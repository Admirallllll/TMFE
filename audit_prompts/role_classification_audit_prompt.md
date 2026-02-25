# Q&A Role Classification (角色识别) 深度审查提示词

请你作为一名的高级自然语言处理工程师和数据科学家，帮我深度审查和诊断我项目中的 **“财报电话会议 Q&A 角色识别算法”**。

在最近的人工标注验证中，我发现我们代码抽取的 Q&A 角色（analyst / management / operator）出现了大范围的错误，如果这个问题不解决，将直接摧毁我们后续关于“谁推动了 AI 讨论 (AI initiation)”的核心经济学结论。

请你遵循以下步骤对我提供的代码和数据进行联合诊断，并给我出具一份**带有实际代码修复方案**的诊断报告。

## 步骤 1：阅读核心算法代码
请你必须先阅读负责角色分类的核心函数 `classify_role`，该函数位于项目中：
- 文件路径：`src/preprocessing/transcript_parser.py`
- 请重点检查 `ANALYST_PATTERNS`、`MANAGEMENT_KEYWORDS` 的正则表达式设计，以及它是如何利用文本上下文特征（如问号和开场白）进行二次判定的。

## 步骤 2：对撞真实抽样数据进行逆向推理 (Reverse Engineering)
请你读取我们的标注样本文件，观察算法在真实战场下的翻车情况：
- 请分析文件：`outputs/annotation_samples/role_audit_qa_turns.csv`
- 若需要查看出错 turn 的完整上下文对峙情况，请连带读取 `outputs/annotation_samples/role_audit_qa_turns_full_call_contexts.jsonl`

**在诊断中，请强制回答以下三个硬核问题：**
1. **错杀漏洞：** 我们的 `MANAGEMENT_KEYWORDS` 和 `ANALYST_PATTERNS` 词表中，是否漏掉了 S&P 500 公司高频出现的典型投行名称或高管头衔？（如 SVP、Head of IR 等）
2. **逻辑穿透漏洞：** 如果一个分析师在提问中**没有带问号**，或者用陈述句开了个长篇大论，我们的算法是否会因为正则匹配失败而将其归置为 `unknown` 甚至误认为 `management`？
3. **被截断或缩写的机构名：** 在数据集中找出那些因为机构名字只被写了一点点（如 JPM 代替了 J.P. Morgan），从而导致匹配失效的具体真实案例。

## 步骤 3：输出重构补救方案
在查明问题后，请：
- 为 `transcript_parser.py` 里的 `classify_role` 方法写一版**防御性最高、最严密的重构代码**。
- 给出你补充的新正则表达式池。
- 如果你有除了死抠字典以外，更好且资源消耗不大的轻量级判定方案（如：统计整场会议谁主导抛出问题、谁一直在接锅），请务必补充到代码中作为增强启发式规则 (Augmented Heuristics)。
