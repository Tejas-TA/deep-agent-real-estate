import asyncio, json, sys

async def test():
    msg = json.dumps({
        "jsonrpc": "2.0", "id": 1, "method": "initialize",
        "params": {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "test", "version": "1.0"}
        }
    }) + "\n"

    proc = await asyncio.create_subprocess_exec(
        sys.executable, "notebooks/mcp_server.py",
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await asyncio.wait_for(proc.communicate(msg.encode()), timeout=30)
    print("STDOUT:", stdout.decode()[:1000] or "(empty)")
    print("STDERR:", stderr.decode()[:1000] or "(empty)")

asyncio.run(test())