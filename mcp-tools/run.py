import asyncio
from tools.client import MCPClient

async def main():

    client = MCPClient()
    try:
        await client.connect_to_server("tools/server.py")
        await client.chat_loop()
    finally:
        await client.clean()

if __name__ == "__main__":
    asyncio.run(main())
