# Q&A Boundary Segment & Pairing (问答边界与配对切割) 深度审查提示词

请你以一位精通金融文本工程和结构化数据提取的架构师身份，帮我深度调查我项目中**“财报电话会议 Q&A 边界寻找与问答配对算法”**出现的灾难级错误。

最近的人工标注验证暴露出：我的 `qa_start_index` 找的切割点有很多是歪的！原本该是高管的致辞直接被算成了问答开始；原本该是分析师的一连串组合提问，在问答轮次对齐（pairing）中彻底散架，导致我们后面算 Initiation Score（话题发动得分）全是错乱的。

为了挽救本项目极其关键的经济学叙事线索，请按以下严苛链路帮我进行逆向诊断与重构。

## 步骤 1：彻查核心边界切分代码逻辑
请第一时间深入阅读我们的边界定位功能并挖掘它的缺陷：
- 文件路标：由于 `find_qa_start_index` 是切割的生死线，请你仔细审查 `src/preprocessing/transcript_parser.py` 中的该函数。
- 文件路标：由于配对逻辑是第二灾区，请连带审查负责这部分的算法 `src/metrics/initiation_score.py` （尤其是如何匹配 `question` 对应 `answer` 的逻辑）。

**致命质检问题一：主持人发誓 (Operator Pledge 滥撞)**
在 `QA_START_PATTERNS` 正则捕获网里，当某位高管长篇阔论中无意提到了 "take the first question for our new product..." 这种话语时，代码是否直接把这句当成 Q&A 开始了？

## 步骤 2：剖解前线坍塌实录
不要纸上谈兵。请深入 `outputs/annotation_samples/qa_boundary_audit_docs.csv`，将 `pairing_quality` 为 `major_issue` 甚至直接标为 `unusable` 的那批毒瘤数据单拎出来！

如果光看 CSV 无法理解切分为何偏移，请直接去 `outputs/annotation_samples/qa_boundary_audit_docs_full_call_contexts.jsonl` 这本厚账簿里查阅那场出事会议的全样本流水账。

**致命质检问题二：连环追问崩盘 (Consecutive Queries Crash)**
在真实的 Q&A 录音文本里，分析师经常提一个大问题抛出后，还会补充一句。当遇到“分析师紧接着说了一段 -> 高管开始答”或者“高管拉下属一起答（CEO说完 CFO补充）”时，我们的 1对1 极简配对（Simple Pairing）算法是不是把“CEO对CFO的话”当成了新的一轮独立问答？找出这些反面教材片段展示给我。

## 步骤 3：交接加装重型防御装甲的代码修复版
诊断结束不能只是空谈。请对我那套千疮百孔的启发式流水线进行翻新。

1. **加固 `find_qa_start_index` (抗干扰)：** 请为切割代码加上一层“预热抗干扰期（Warm-up buffer）”。告诉算法，会议开始前 5 分钟内出现的正则触发词大概率不是真实的 Q&A。另外强化“主持人宣布口吻”的强锚定效应。
2. **重铸交手回合 (Exchange Pairing)：** 放弃机械的一问一答对对碰，为 `src/metrics/initiation_score.py` 提供一个能够容忍“多人组合上阵（Consecutive same-role speakers merging）”的上下文融合拼盘算法。即将多个分析师的连续追问打包成一个“大问题 (Macro-Question)”，将多个高管的接龙回应合成一个“大回应 (Macro-Answer)”。
3. 请确保修正后的代码可以直接插入替换旧管道，不会引发底层表结构地震。
