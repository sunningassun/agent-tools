#!/usr/bin/env python3
"""
agent_system.py – 基于 Qwen3 的 Agent + RAG，使用丰富的 SQLite 数据库（纯命令行，无 Emoji）
"""

import os
import sys
import time
import json
import logging
import re
import sqlite3
from typing import List, Dict, Any, Tuple, Callable
from datetime import datetime, timedelta
from dotenv import load_dotenv
import requests
from openai import OpenAI

load_dotenv()

# ==================== 配置 ====================
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "**************")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "Qwen/Qwen3-8B")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
FREEDUB_API_URL = "https://api.pearapi.ai/api/freedub"
SCIENCENEWS_API_URL = "https://api.pearapi.ai/api/sciencenews/"
DB_PATH = "agent_kb.db"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MOCK_MODE = not SILICONFLOW_API_KEY
if MOCK_MODE:
    logger.warning("Mock 模式，API 调用将返回模拟数据")


# ==================== 数据库管理类 ====================
class AgentDatabase:
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._ensure_db()

    def _ensure_db(self):
        if not os.path.exists(self.db_path):
            logger.error(f"数据库 {self.db_path} 不存在，请先运行 init_kb_db.py")
            sys.exit(1)

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    # ---------- RAG 全文检索 ----------
    def search_documents(self, query: str, top_k: int = 3) -> List[Dict]:
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                SELECT title, content, bm25(documents_fts) as rank
                FROM documents_fts
                WHERE documents_fts MATCH ?
                ORDER BY rank
                LIMIT ?
            ''', (query, top_k))
            rows = cursor.fetchall()
            results = []
            for row in rows:
                results.append({"title": row[0], "content": row[1], "score": 1.0 / (row[2] + 1e-6)})
            return results
        except sqlite3.OperationalError:
            return []
        finally:
            conn.close()

    # ---------- 配音角色查询 ----------
    def get_voice_profile(self, name: str) -> Dict:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT role_code, style FROM voice_profiles WHERE name = ?", (name,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return {"role_code": row[0], "style": row[1]}
        return None

    def list_voice_profiles(self) -> List[str]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM voice_profiles")
        rows = cursor.fetchall()
        conn.close()
        return [row[0] for row in rows]

    # ---------- 新闻缓存 ----------
    def get_cached_news(self, max_age_days: int = 1) -> List[Dict]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cutoff = (datetime.now() - timedelta(days=max_age_days)).strftime("%Y-%m-%d")
        cursor.execute('''
            SELECT title, time, summary FROM news_cache
            WHERE fetch_date >= ? ORDER BY time DESC LIMIT 10
        ''', (cutoff,))
        rows = cursor.fetchall()
        conn.close()
        return [{"title": r[0], "time": r[1], "summary": r[2]} for r in rows]

    def update_news_cache(self, articles: List[Dict]):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM news_cache")
        for art in articles:
            cursor.execute('''
                INSERT INTO news_cache (title, time, summary, fetch_date)
                VALUES (?, ?, ?, ?)
            ''', (
                art.get("title", ""), art.get("time", ""), art.get("summary", ""), datetime.now().strftime("%Y-%m-%d")))
        conn.commit()
        conn.close()

    # ---------- 持久化记忆 ----------
    def save_conversation(self, session_id: str, user_input: str, assistant_response: str):
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO conversation_memory (session_id, user_input, assistant_response)
            VALUES (?, ?, ?)
        ''', (session_id, user_input, assistant_response))
        conn.commit()
        conn.close()

    def load_recent_conversations(self, session_id: str, limit: int = 5) -> List[Tuple]:
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT user_input, assistant_response FROM conversation_memory
            WHERE session_id = ? ORDER BY timestamp DESC LIMIT ?
        ''', (session_id, limit))
        rows = cursor.fetchall()
        conn.close()
        return rows[::-1]


# ==================== LLM 客户端 ====================
class SiliconFlowLLMClient:
    def __init__(self, model_name: str = DEFAULT_MODEL, temperature: float = TEMPERATURE):
        self.model_name = model_name
        self.temperature = temperature
        if not MOCK_MODE:
            self.client = OpenAI(api_key=SILICONFLOW_API_KEY, base_url="https://api.siliconflow.cn/v1")

    def generate(self, prompt: str, max_tokens: int = MAX_TOKENS) -> str:
        if MOCK_MODE:
            if "置信度" in prompt:
                return "0.85"
            return "Mock 回答"
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM 失败: {e}")
            return f"错误: {str(e)}"

    def generate_with_confidence(self, prompt: str) -> Tuple[str, float]:
        answer = self.generate(prompt)
        conf_prompt = f"评估置信度(0-1只输出数字)：\n问题：{prompt[:200]}\n回答：{answer}\n置信度："
        conf_str = self.generate(conf_prompt, max_tokens=10)
        try:
            confidence = float(conf_str.strip())
        except:
            confidence = 0.5
        return answer, min(max(confidence, 0.0), 1.0)


# ==================== 工具定义（使用数据库） ====================
class Tool:
    def __init__(self, name: str, description: str, func: Callable):
        self.name = name
        self.description = description
        self.func = func

    def run(self, input_str: str) -> str:
        try:
            return self.func(input_str)
        except Exception as e:
            return f"工具错误: {str(e)}"


def freedub_tool(input_str: str) -> str:
    db = AgentDatabase()
    if "||" in input_str:
        parts = input_str.split("||")
        text = parts[0].strip()
        role_name = parts[1].strip() if len(parts) > 1 else "少女"
        style = parts[2].strip() if len(parts) > 2 else "cheerful"
        profile = db.get_voice_profile(role_name)
        if profile:
            role_code = profile["role_code"]
            style = profile["style"] if len(parts) <= 2 else style
        else:
            role_code = "zh-CN-XiaoyiNeural"
            style = "cheerful"
    else:
        raw = input_str.strip()
        text = ""
        role_name = "少女"
        colon_match = re.search(r'[：:]\s*(.+)$', raw)
        if colon_match:
            text = colon_match.group(1).strip()
        else:
            quote_match = re.search(r'["\'](.+?)["\']', raw)
            if quote_match:
                text = quote_match.group(1).strip()
        for vname in db.list_voice_profiles():
            if vname in raw:
                role_name = vname
                break
        if not text:
            text = re.sub(r'^(用|以|请|帮我|我要).*?(配音|朗读|说)\s*', '', raw).strip()
            if not text:
                text = raw
        profile = db.get_voice_profile(role_name)
        if profile:
            role_code = profile["role_code"]
            style = profile["style"]
        else:
            role_code = "zh-CN-XiaoyiNeural"
            style = "cheerful"

    if not text:
        return "错误：无法识别配音文本"

    try:
        resp = requests.post(FREEDUB_API_URL, json={"text": text, "role": role_code, "style": style}, timeout=30)
        data = resp.json()
        if data.get("code") == 200:
            audio_url = data.get("data", {}).get("audio_url", "")
            return f"语音合成成功！\n文本：{text}\n角色：{role_name}({role_code}) 风格：{style}\n音频链接：{audio_url}"
        return f"配音失败：{data.get('msg')}"
    except Exception as e:
        return f"请求失败：{str(e)}"


def sciencenews_tool(_: str) -> str:
    db = AgentDatabase()
    cached = db.get_cached_news(max_age_days=1)
    if cached:
        result = "最新科技资讯（缓存，24小时内更新）:\n"
        for i, art in enumerate(cached[:5], 1):
            result += f"{i}. [{art['time']}] {art['title']}\n   {art['summary'][:100]}\n"
        return result
    try:
        resp = requests.get(SCIENCENEWS_API_URL, timeout=15)
        data = resp.json()
        if data.get("code") == 200:
            raw_data = data.get("data", "")
            try:
                articles = json.loads(raw_data) if isinstance(raw_data, str) else raw_data
                if isinstance(articles, list):
                    to_cache = []
                    result = "最新科技资讯（来自API）:\n"
                    for idx, art in enumerate(articles[:5], 1):
                        title = art.get("title", "无标题")
                        time_str = art.get("time", "")
                        summary = art.get("summary", title)
                        result += f"{idx}. [{time_str}] {title}\n   {summary[:100]}\n"
                        to_cache.append({"title": title, "time": time_str, "summary": summary})
                    db.update_news_cache(to_cache)
                    return result
            except Exception as e:
                return f"科技资讯解析失败：{str(e)}\n原始数据：{raw_data[:500]}"
        return f"获取失败：{data.get('msg', '未知错误')}"
    except Exception as e:
        return f"请求失败：{str(e)}"


# ==================== Agent 决策器 ====================
class AgentDecisionMaker:
    def __init__(self, llm_client: SiliconFlowLLMClient, db: AgentDatabase, session_id: str = "default"):
        self.llm = llm_client
        self.db = db
        self.session_id = session_id
        persistent = self.db.load_recent_conversations(session_id, limit=5)
        self.memory = persistent
        self.tools = {
            "rag_retrieve": Tool("rag_retrieve", "从知识库检索相关文档", self._rag_wrapper),
            "freedub": Tool("freedub", "文本配音（支持自然语言，如'用少女声说你好'）", freedub_tool),
            "sciencenews": Tool("sciencenews", "获取最新科技资讯（带缓存）", sciencenews_tool),
        }

    def get_tools_info(self):
        return [{"name": name, "description": tool.description} for name, tool in self.tools.items()]

    def _rag_wrapper(self, query: str) -> str:
        docs = self.db.search_documents(query, top_k=2)
        if not docs:
            return "未找到相关文档"
        context = "\n".join(f"[{d['title']}] {d['content']}" for d in docs)
        return f"检索到的内容：\n{context}"

    def _classify_intent(self, user_input: str) -> str:
        prompt = f"""判断意图，只输出工具名（rag_retrieve / freedub / sciencenews / none）。
用户输入：{user_input}
工具名："""
        tool = self.llm.generate(prompt, max_tokens=20).strip().lower()
        return tool if tool in self.tools else "none"

    def _retrieve_context(self, query: str, top_k: int) -> str:
        docs = self.db.search_documents(query, top_k=top_k)
        if not docs:
            return ""
        context = "\n".join(d["content"] for d in docs)
        return f"参考知识库：\n{context}\n"

    def answer_with_retry(self, user_input: str) -> Tuple[str, Dict]:
        start = time.time()
        retries = 0
        tool_calls = []
        final_answer = ""
        confidence = 0.0
        last_top_k = 3

        while retries <= MAX_RETRIES:
            tool_name = self._classify_intent(user_input)
            tool_result = None
            if tool_name != "none":
                tool_result = self.tools[tool_name].run(user_input)
                tool_calls.append({"tool": tool_name, "input": user_input[:50], "result": tool_result[:200]})
                if tool_name == "freedub":
                    final_answer = tool_result
                    confidence = 1.0
                    break

            memory_str = ""
            if self.memory:
                recent = self.memory[-3:]
                memory_str = "对话历史：\n" + "\n".join(f"用户：{u}\n助手：{a}" for u, a in recent) + "\n"
            current_top_k = 3 + retries * 2
            last_top_k = current_top_k
            rag_str = self._retrieve_context(user_input, top_k=current_top_k) if tool_name != "rag_retrieve" else ""
            tool_str = f"工具结果：{tool_result}\n" if tool_result else ""
            prompt = f"""{memory_str}{rag_str}{tool_str}
用户：{user_input}
助手："""
            answer, conf = self.llm.generate_with_confidence(prompt)
            if conf >= CONFIDENCE_THRESHOLD or retries == MAX_RETRIES:
                final_answer = answer
                confidence = conf
                break
            else:
                logger.info(f"置信度低({conf:.2f})，重试{retries + 1}")
                retries += 1

        self.memory.append((user_input, final_answer))
        if len(self.memory) > 10:
            self.memory = self.memory[-10:]
        self.db.save_conversation(self.session_id, user_input, final_answer)

        info = {
            "tool_calls": tool_calls,
            "retries": retries,
            "confidence": confidence,
            "response_time": round(time.time() - start, 2),
            "top_k_used": last_top_k
        }
        return final_answer, info


# ==================== 命令行界面 ====================
def print_tools(agent):
    print("\n" + "=" * 60)
    print("可用工具：")
    for t in agent.get_tools_info():
        print(f"  - {t['name']}: {t['description']}")
    print("=" * 60)


def run_cli():
    print("Agent 命令行模式（输入 exit 退出，clear 清空内存记忆）")
    llm = SiliconFlowLLMClient()
    db = AgentDatabase()
    agent = AgentDecisionMaker(llm, db, session_id="cli_user")
    print_tools(agent)
    while True:
        user = input("\n用户：").strip()
        if user.lower() == "exit":
            break
        if user.lower() == "clear":
            agent.memory.clear()
            print("内存记忆已清空（持久化记忆仍保留）")
            continue
        ans, info = agent.answer_with_retry(user)
        print(f"Agent：{ans}")
        print(f"[信息] 工具调用次数: {len(info['tool_calls'])} | 重试次数: {info['retries']} | 置信度: {info['confidence']:.2f} | 耗时: {info['response_time']}秒")


if __name__ == "__main__":
    run_cli()