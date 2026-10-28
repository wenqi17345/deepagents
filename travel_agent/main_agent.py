import os
from pathlib import Path
import asyncio
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from deepagents.middleware.subagents import SubAgent
from langchain_mcp_adapters.client import MultiServerMCPClient

# 加载环境变量
load_dotenv()

# 加载提示词
def load_prompt(file_name:str):
    path = Path(__file__).parent / "prompts" / f"{file_name}.prompt"
    return path.read_text(encoding="utf-8")

# 获取记忆文件
AGENTS_PATH = Path(__file__).parent / "memory" / "AGENTS.md"

# 初始化大模型
llm = init_chat_model(
    model="deepseek-flash",
    model_provider="openai",
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL"),
)

# 配置持久化存储
backend = FilesystemBackend(
    root_dir=str(Path(__file__).parent), # 指定 Agent 读写文件的基础文件夹。
    virtual_mode=True,  # 路径锚定 绝对路径会被解释为相对于 root_dir 的路径 阻止越界 拦截指向 root_dir 外部的路径。
)

# 涉及I/O操作 异步处理
async def main():

    # 用户控制台输入
    user_query = input("请输入您的旅游需求: ").strip()

    # 高德地图 MCP服务
    amap_client = MultiServerMCPClient({
        "amap-maps": {
            "transport": "streamable_http",
            "url": os.getenv("AMAP_CLIENT_URL"),
        }
    })
    amap_tools = await amap_client.get_tools()

    # 12306 MCP服务
    ticket_client = MultiServerMCPClient({
        "12306": {
            "transport": "streamable_http",
            "url": os.getenv("TICKET_CLIENT_URL"),
        }
    })
    ticket_tools = await ticket_client.get_tools()

    # 地图子智能体：绑定高德 MCP工具
    map_agent: SubAgent = {
        "name": "map_helper",
        "description": "一名提供高德地图服务助手",
        "system_prompt": load_prompt("map_agent"),
        "tools": amap_tools,
    }

    # 车票子智能体：绑定12306 MCP工具
    ticket_agent: SubAgent = {
        "name": "ticket_helper",
        "description": "一名12306车票规划助手。",
        "system_prompt": load_prompt("ticket_agent"),
        "tools": ticket_tools
    }

    # 汇总子智能体，无工具
    summary_agent: SubAgent = {
        "name": "summary_helper",
        "description": "一名旅行方案汇总助手",
        "system_prompt": load_prompt("summary_agent"),
        "tools": []
    }

    # 主智能体
    main_agent = create_deep_agent(
        model=llm,
        subagents=[map_agent, ticket_agent, summary_agent],
        memory=[str(AGENTS_PATH)],
        system_prompt=load_prompt("main_agent"),
        backend = backend
    )

    # 调用主智能体
    result = await main_agent.ainvoke({
        "messages": [{"role": "user", "content": user_query}]
    })

    # 返回的是 AIMessage 对象，用属性 .content 访问
    print(result)
    content = result["messages"][-1].content

    print("\n===== 旅行方案结果 =====")
    print(content)

if __name__ == "__main__":
    asyncio.run(main())

