"""分类 Prompt（Few-shot）。

迁移自原 Notebook，保留原始 wording，仅追加两点改动：
1. 新增第 5 类「知识库问答」；
2. 追加严格的输出格式约束，降低解析失败率（配合结构化输出使用）。
"""

from __future__ import annotations

CATEGORIZE_PROMPT = """Categorize the following customer query into one of these categories:
1: Learn Generative AI Technology
2: Resume Making
3: Interview Preparation
4: Job Search
5: Knowledge Base Q&A (questions about documents the user uploaded)
6: JD Match Analysis (score how well the user's resume matches a specific job description)
7: Interview Review (retrospective on an interview that already happened)

Output rules (MUST follow):
- Reply with ONE lowercase word only: learning, resume, interview, job_search, knowledge, jd_match or interview_review
- Do NOT output the number, do NOT explain
- Use `knowledge` only when the question must be answered from the user's uploaded documents
- Use `jd_match` when the user wants a match score / gap list between a resume and a JD
- Use `interview_review` when the user asks to review or reflect on a finished interview

Examples:
1. Query: 'What are the basics of generative AI, and how can I start learning it?' -> learning
2. Query: 'Can you help me improve my resume for a tech position?' -> resume
3. Query: 'What are some common questions asked in AI interviews?' -> interview
4. Query: 'Are there any job openings for AI engineers?' -> job_search
5. Query: '资料里提到的 RAG 切分策略是什么？' -> knowledge
6. Query: '我上传的文档里怎么讲 rerank 的？' -> knowledge
7. Query: '帮我看下我的简历和这个 JD 的匹配度，还差哪些能力？' -> jd_match
8. Query: '刚才那场模拟面试帮我复盘一下，哪些地方答得不好？' -> interview_review

Now, categorize the following customer query:
Query: {query}
"""

LEARNING_SUB_PROMPT = """Categorize the following user query into one of these categories:

Categories:
- tutorial: For queries related to creating tutorials, blogs, or documentation on generative AI.
- question: For general queries asking about generative AI topics.
- Default to question if the query doesn't fit either of these categories.

Output rules (MUST follow):
- Reply with ONE lowercase word only: tutorial or question

Examples:
1. User query: 'How to create a blog on prompt engineering for generative AI?' -> tutorial
2. User query: 'Can you provide a step-by-step guide on fine-tuning a generative model?' -> tutorial
3. User query: 'Provide me the documentation for Langchain?' -> tutorial
4. User query: 'What are the main applications of generative AI?' -> question
5. User query: 'Is there any generative AI course available?' -> question

Now, categorize the following user query:
The user query is: {query}
"""

INTERVIEW_SUB_PROMPT = """Categorize the following user query into one of these categories:

Categories:
- mock: For requests related to mock interviews.
- question: For general queries asking about interview topics or preparation.
- Default to question if the query doesn't fit either of these categories.

Output rules (MUST follow):
- Reply with ONE lowercase word only: mock or question

Examples:
1. User query: 'Can you conduct a mock interview with me for a Gen AI role?' -> mock
2. User query: 'What topics should I prepare for an AI Engineer interview?' -> question
3. User query: 'I need to practice interview focused on Gen AI.' -> mock
4. User query: 'Can you list important coding topics for AI tech interviews?' -> question

Now, categorize the following user query:
The user query is: {query}
"""

FALLBACK_MESSAGE = (
    "我目前聚焦在这几件事上，换个问法试试吧：\n\n"
    "1. **学习生成式 AI**：概念答疑、生成教程（如「帮我写一篇 LangGraph 教程」）\n"
    "2. **简历制作**：分步收集信息生成简历，或根据 JD 改写简历\n"
    "3. **面试准备**：整理面试真题，或进行一场模拟面试并给出评价\n"
    "4. **职位搜索**：按城市和岗位检索职位清单\n"
    "5. **知识库问答**：先在左侧上传文档建库，再基于资料提问\n"
    "6. **JD 匹配诊断**：把简历和岗位 JD 放一起，给出匹配分、命中项与缺口清单\n"
    "7. **面试复盘**：把刚结束的面试问答交给我，输出薄弱点与改进动作\n"
)
