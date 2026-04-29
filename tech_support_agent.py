import os
import json
import openai
import numpy as np
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict, Any

# 配置：请设置你的 OpenAI API Key
openai.api_key = os.getenv("OPENAI_API_KEY")  # 或者直接填写字符串

# ==========================================
# 工具函数：模拟拉取服务器日志
# ==========================================
def fetch_server_logs(server_id: str, error_pattern: str, hours: int = 24) -> str:
    """
    模拟从 ELK 接口拉取日志。实际项目中可替换为真实 HTTP 请求。
    这里根据 server_id 和 error_pattern 返回预制的日志样本。
    """
    # 模拟数据库：实际可连接 ElasticSearch
    mock_logs_db = {
        "web-01": {
            "500_internal_error": """[2025-01-20 14:32:10] ERROR web-01.nginx: 500 Internal Server Error - upstream response timeout
[2025-01-20 14:32:12] ERROR web-01.app: Exception in /api/orders: Database connection pool exhausted
[2025-01-20 14:32:13] CRITICAL web-01.monitor: Memory usage 98%, swap usage 100%
[2025-01-20 14:32:15] ERROR web-01.db: pg_stat_activity: idle in transaction count=150
[2025-01-20 14:32:20] FATAL web-01.kernel: OOM killer invoked for process python3 (PID 2843).
""",
            "permission_denied": """[2025-01-20 14:33:00] WARN web-01.auth: Permission denied for user deploy accessing resource /deploy/config
[2025-01-20 14:33:01] ERROR web-01.ssh: Failed authentication attempt from 192.168.1.100
[2025-01-20 14:33:05] NOTICE web-01.sudo: User deploy not in sudoers file; this incident will be reported."""
        },
        "db-02": {
            "500_internal_error": """[2025-01-20 14:34:00] ERROR db-02.postgres: duplicate key value violates unique constraint "idx_orders_id"
[2025-01-20 14:34:01] ERROR db-02.postgres: current transaction is aborted, commands ignored until end of transaction block
[2025-01-20 14:34:05] WARN db-02.disk: /data partition 95% full
""",
            "timeout": """[2025-01-20 14:35:00] ERROR db-02.postgres: statement timeout: SELECT * FROM orders WHERE created_at > ... ran for 30 seconds and was cancelled
[2025-01-20 14:35:02] WARN db-02.monitor: CPU iowait 60%"""
        }
    }

    # 模糊匹配：如果找不到完全匹配的，返回默认错误日志
    logs = mock_logs_db.get(server_id, {}).get(error_pattern)
    if logs is None:
        logs = f"[{hours}h simulated] No logs matched pattern '{error_pattern}' on {server_id}. Generic output: CPU high, possible memory leak."
    return logs.strip()


# ==========================================
# RAG 知识库：历史工单 & 解决方案（向量化存储）
# ==========================================
class TicketKnowledgeBase:
    def __init__(self):
        self.tickets = [
            {"problem": "Nginx 返回 500 错误，日志显示 upstream timeout 和数据库连接池耗尽", 
             "solution": "1. 增加数据库连接池大小至 max_connections=200；2. 优化长事务，设置 idle_in_transaction_session_timeout = 5min；3. 重启 app 服务释放泄漏连接。"},
            {"problem": "OOM Killer 杀死 python 进程，内存使用98%", 
             "solution": "1. 立即重启服务释放内存；2. 检查代码内存泄漏，使用 memory_profiler 分析；3. 临时增加 swap 空间，长期升级服务器内存或启用容器内存限制。"},
            {"problem": "数据库唯一索引冲突 duplicate key value violates unique constraint", 
             "solution": "1. 确认业务逻辑是否允许重复键，如允许则删除或修改约束；2. 检查插入前是否有幂等校验逻辑；3. 如数据异常，回滚事务并清理脏数据。"},
            {"problem": "磁盘分区 /data 达到 95 % 使用率", 
             "solution": "1. 执行日志轮转和清理临时文件；2. 扩展磁盘或迁移大表到新挂载点；3. 设置磁盘使用率监控告警（阈值 85%）。"},
            {"problem": "权限拒绝 Permission denied for user deploy", 
             "solution": "1. 检查用户 deploy 的文件系统权限，确保对所需目录有 rx 权限；2. 若需 sudo 权限，添加至 sudoers 文件；3. 核查安全组策略，禁止公共网络 ssh。"},
            {"problem": "数据库查询超时 statement timeout", 
             "solution": "1. 分析查询计划，添加缺失索引；2. 优化 SQL 语句或拆分为批量小查询；3. 增加 statement_timeout 配置，但需先治本。"}
        ]
        self.index = None
        self.texts = []
        self._build_index()

    def _get_embedding(self, text: str) -> np.ndarray:
        """调用 OpenAI Embeddings API"""
        response = openai.Embedding.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        return np.array(response['data'][0]['embedding'], dtype=np.float32)

    def _build_index(self):
        """构建 FAISS 索引"""
        dim = 1536  # ada-002 维度
        self.index = faiss.IndexFlatL2(dim)
        embeddings = []
        for ticket in self.tickets:
            emb = self._get_embedding(ticket["problem"])
            embeddings.append(emb)
        embeddings = np.array(embeddings).astype(np.float32)
        self.index.add(embeddings)
        self.texts = [t["problem"] for t in self.tickets]

    def retrieve(self, query: str, top_k: int = 2) -> List[str]:
        """检索最相似的历史工单解决方案"""
        query_emb = self._get_embedding(query).reshape(1, -1)
        distances, indices = self.index.search(query_emb, top_k)
        results = []
        for idx in indices[0]:
            if idx != -1:
                results.append(self.tickets[idx]["solution"])
        return results

# 初始化知识库（运行时会调用 Embedding API，可能有费用）
ticket_kb = TicketKnowledgeBase()


# ==========================================
# Agent 1: 意图识别 Agent
# ==========================================
def intent_recognition(user_input: str) -> dict:
    """
    从用户报错描述中提取关键实体：服务器ID、错误类型、严重级别。
    使用 LLM 结构化输出。
    """
    prompt = f"""You are an expert system for support ticket intent recognition.
Extract the following information from the user's error report in JSON format:
- server_id: the server that has the problem (e.g., web-01, db-02). If not mentioned, infer "web-01".
- error_type: concise keyword for main error pattern (choose from: 500_internal_error, permission_denied, timeout, out_of_memory, disk_full, other).
- severity: critical/high/medium/low.
User report: "{user_input}"
Output only a JSON object with keys server_id, error_type, severity. Do not add any extra text.
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )
    try:
        content = response.choices[0].message.content.strip()
        return json.loads(content)
    except:
        # 降级方案
        return {"server_id": "web-01", "error_type": "500_internal_error", "severity": "high"}


# ==========================================
# Agent 2: 诊断 Agent（工具调用 + 长链推理）
# ==========================================
def diagnose_with_tools(intent: dict, user_input: str) -> str:
    """
    诊断 Agent 先调用 fetch_server_logs 获取日志，然后进行长链推理 (CoT) 分析。
    利用 OpenAI Function Calling 实现。
    """
    # 定义工具描述
    functions = [
        {
            "name": "fetch_server_logs",
            "description": "拉取指定服务器的错误日志，参数：server_id（字符串），error_pattern（日志匹配模式，例如 '500_internal_error'），hours（往前拉取的小时数，默认24）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "server_id": {"type": "string"},
                    "error_pattern": {"type": "string"},
                    "hours": {"type": "integer", "default": 24}
                },
                "required": ["server_id", "error_pattern"]
            }
        }
    ]

    messages = [
        {"role": "system", "content": """You are an expert diagnostic agent. Follow these steps:
1. Call the fetch_server_logs function with the given server_id and error_type to retrieve recent logs.
2. Once you have the logs, perform a thorough root cause analysis using Chain-of-Thought reasoning (step-by-step).
   - Identify the primary error lines.
   - Infer the sequence of events (e.g., timeout -> connection pool exhaustion -> OOM).
   - Explain the likely root cause in plain language.
3. Summarize your findings in one paragraph that starts with 'DIAGNOSIS:'. 
Use step-by-step thinking but only output the final diagnosis."""},
        {"role": "user", "content": f"User report: {user_input}\nIntents extracted: {json.dumps(intent)}"}
    ]

    # 第一次请求，可能返回函数调用
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto",
        temperature=0.2
    )
    response_message = response.choices[0].message

    # 处理函数调用
    if response_message.get("function_call"):
        func_name = response_message["function_call"]["name"]
        arguments = json.loads(response_message["function_call"]["arguments"])
        if func_name == "fetch_server_logs":
            # 实际调用我们的模拟工具
            logs = fetch_server_logs(
                server_id=arguments.get("server_id", intent.get("server_id")),
                error_pattern=arguments.get("error_pattern", intent.get("error_type")),
                hours=arguments.get("hours", 24)
            )
            # 将工具结果附加到消息中
            messages.append(response_message)
            messages.append({
                "role": "function",
                "name": func_name,
                "content": logs
            })
            # 再次请求模型，要求最终推理
            final_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=messages,
                functions=functions,
                temperature=0.2
            )
            diagnosis = final_response.choices[0].message.content
            return diagnosis
    else:
        # 模型直接返回了诊断（未调用工具）
        return response_message.content


# ==========================================
# Agent 3: 方案生成 Agent（RAG 增强）
# ==========================================
def generate_solution(diagnosis: str) -> str:
    """
    结合检索到的历史解决方案 + 诊断结果，生成最终修复报告。
    """
    # 检索类似历史工单
    similar_solutions = ticket_kb.retrieve(diagnosis, top_k=2)
    combined_solutions = "\n\n".join(f"- {sol}" for sol in similar_solutions)

    prompt = f"""You are a senior SRE engineer. Based on the following diagnosis and similar past solutions, 
produce a detailed action plan to resolve the issue. 
Include immediate mitigation steps (within next 30 minutes) and long-term preventive measures.
Format the answer with markdown headers: ## 诊断摘要, ## 紧急修复步骤, ## 长期预防措施, ## 相关历史经验.

Diagnosis: 
{diagnosis}

Relevant historical solutions (for reference only, adapt as needed):
{combined_solutions}

Output the final report in Chinese.
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_tokens=800
    )
    return response.choices[0].message.content


# ==========================================
# 主流程控制器
# ==========================================
def handle_support_ticket(user_input: str) -> str:
    print("=" * 60)
    print(f"用户报错: {user_input}\n")
    
    # Step 1: 意图识别
    intent = intent_recognition(user_input)
    print(f">>> 意图识别结果: {json.dumps(intent, indent=2)}\n")
    
    # Step 2: 日志诊断
    print(">>> 开始诊断...")
    diagnosis = diagnose_with_tools(intent, user_input)
    print(f">>> 诊断结论:\n{diagnosis}\n")
    
    # Step 3: 方案生成
    print(">>> 生成修复方案...")
    solution = generate_solution(diagnosis)
    print("=" * 60)
    return solution


# ==========================================
# 演示入口
# ==========================================
if __name__ == "__main__":
    # 测试案例
    user_reports = [
        "web-01 突然所有接口返回 500 错误，用户无法下单，已经持续 10 分钟，需要立即排查！",
        "数据库 db-02 查询特别慢，日志显示很多超时，怀疑索引失效了",
        "我的部署脚本提示 Permission denied，用户 deploy 没有权限访问 /deploy/config"
    ]
    
    for i, report in enumerate(user_reports, 1):
        print(f"\n案例 {i}:\n")
        final_report = handle_support_ticket(report)
        print(final_report)
        print("\n" + "=" * 60 + "\n")
