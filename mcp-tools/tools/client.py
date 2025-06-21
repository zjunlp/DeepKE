import asyncio
import json
import os
from typing import Optional
from contextlib import AsyncExitStack
from openai import OpenAI
from dotenv import load_dotenv

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

class MCPClient:
    def __init__(self):
        """初始化MCP客户端"""
        self.exit_stack = AsyncExitStack()
        self.opanai_api_key = os.getenv("API_KEY") # 调用模型的api_key
        self.base_url = os.getenv("BASE_URL") # 调用模型的base_url
        self.model = os.getenv("MODEL") # 调用模型名称
        self.client = OpenAI(api_key=self.opanai_api_key, base_url=self.base_url)
        self.session: Optional[ClientSession] = None # Optional提醒用户该属性是可选的，可能为None
        self.exit_stack = AsyncExitStack() # 用来存储和清楚对话中上下文的，提高异步资源利用率

    async def connect_to_server(self, server_script_path):
        """连接到MCP服务器并列出MCP服务器的可用工具函数"""
        server_params = StdioServerParameters(
            command="python",
            args=[server_script_path],
            env=None
        ) # 设置启动服务器的参数, python server.py

        # 启动MCP服务器并建立通信
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        # MCP官网原始写法
        response = await self.session.list_tools()
        tools = response.tools
        # langchain写法
        # tools = await load_mcp_tools(self.session)
        
        print("\nConnected to server with tools:", [tool.name for tool in tools])#打印服务端可用的工具
  
    async def process_query(self, query:str)->str:
        """使用大模型处理查询并调用MCP Server可用的MCP工具"""
        messages = [{"role":"user", "content":query}]
        response = await self.session.list_tools()

        available_tools = [{
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
        } for tool in response.tools]

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=available_tools,
            extra_body={"enable_thinking": False}
        )

        # 处理返回内容
        # print("\n[Debug] Raw response:", response)
        content = response.choices[0]

        while content.finish_reason == "tool_calls" or content.message.function_call is not None:
            # 返回结果是使用工具的建议，就解析并调用工具
            tool_call = content.message.tool_calls[0]
            tool_name = tool_call.function.name
            tool_args = json.loads(tool_call.function.arguments)
            # 执行工具
            result = await self.session.call_tool(tool_name, tool_args)
            print(f"\n\n[Calling tool {tool_name} with args {tool_args}]\n\n")
            # print(f"函数返回：\n{result.content[0].text}")
            # 将模型返回的调用工具的对话记录保存在messages中
            messages.append(content.message.model_dump())
            messages.append({
                "role": "tool",
                "content": result.content[0].text,
                "tool_call_id": tool_call.id,
            })
            # 将上面的结果返回给大模型用于生产最终结果
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                extra_body={"enable_thinking": False}
            )

            # print("\n[Debug] Raw response:", response)
            content = response.choices[0]
            # print("\n[Debug] Final response content:", content.message.content)

        return content.message.content
  
    async def chat_loop(self):
        """运行交互式聊天"""
        print("\n MCP客户端已启动！输入quit退出")

        while True:
            try:
                query = input("\n用户:").strip()
                if query.lower() == 'quit':
                    break
                response = await self.process_query(query)
                print(f"\nLLM Agent: {response}")
            except Exception as e:
                print(f"发生错误: {str(e)}")

    async def clean(self):
        """清理资源"""
        await self.exit_stack.aclose()

