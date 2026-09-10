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

Output rules (MUST follow):
- Reply with ONE lowercase word only: learning, resume, interview, job_search or knowledge
- Do NOT output the number, do NOT explain

Examples:
1. Query: 'What are the basics of generative AI, and how can I start learning it?' -> learning
2. Query: 'Can you help me improve my resume for a tech position?' -> resume
3. Query: 'What are some common questions asked in AI interviews?' -> interview
4. Query: 'Are there any job openings for AI engineers?' -> job_search
5. Query: '根据我上传的简历，我和这个岗位匹配吗？' -> knowledge
6. Query: '资料里提到的 RAG 切分策略是什么？' -> knowledge

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
)
