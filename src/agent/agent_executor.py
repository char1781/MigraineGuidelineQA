"""
src/agent/agent_executor.py
职责：创建和管理LangGraph Agent

改动说明：
- 从原代码中提取agent创建逻辑
- 封装为build_agent()函数，便于app.py调用
- 逻辑完全保留原代码
- 【新增】兼容直接运行和模块导入两种方式

测试方法：
# 方式1：作为模块运行
python -m src.agent.agent_executor

# 方式2：直接运行
python src/agent/agent_executor.py
"""

import sys
from pathlib import Path

# 【关键】必须在任何src模块导入之前处理路径
if __package__ is None or __package__ == '':
    project_root = Path(__file__).parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

# from langchain_deepseek import ChatDeepSeek
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from src.agent.tools import get_tools
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


def build_agent():
    """
    构建Agent执行器（原代码逻辑）
    
    改动说明：
    - 封装为函数，便于前端调用
    - 逻辑完全保留原代码
    
    Returns:
        LangGraph Agent实例
    """
    # 初始化LLM（原代码）
    # model = ChatDeepSeek(
    #     model="deepseek-chat",
    #     temperature=0,
    #     api_key=os.getenv("DEEPSEEK_API_KEY"),
    # )
    model = ChatOpenAI(
        model="qwen2.5:14b-instruct-q4_K_M",
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        temperature=0,
    )

    
    # 获取工具列表（原代码）
    tools = get_tools()
    
    # 创建Agent（原代码）
    agent_executor = create_react_agent(
        model=model,
        tools=tools,
        checkpointer=MemorySaver(),
    )
    
    return agent_executor


# ── 单元测试 ──
if __name__ == "__main__":
    print("=== Agent执行器测试 ===\n")
    
    try:
        agent = build_agent()
        print("✅ Agent创建成功")
        
        # 测试简单调用
        config = {"configurable": {"thread_id": "test"}}
        
        print("\n测试查询: 利扎曲普坦剂量")
        
        for event in agent.stream(
            {"messages": [{"role": "user", "content": "利扎曲普坦剂量"}]},
            config=config,
            stream_mode="values",
        ):
            if "messages" in event:
                last = event["messages"][-1]
                if last.type == "ai" and last.content:
                    print(f"\nAgent回答: {last.content[:200]}...")
                    break
        
        print("\n✅ 测试完成")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
