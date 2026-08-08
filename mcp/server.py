"""
MCP Server — supports stdio and SSE transports.
"""
from __future__ import annotations
import sys
import asyncio
import threading
from typing import Optional
from .router import MCPRouter


class MCPServer:
    """MCP Server for MedAxis. Multiple transport options."""

    def __init__(self, router: MCPRouter):
        self.router = router
        self._running = False
        self._transport: Optional[str] = None

    async def run_stdio(self):
        """Run MCP server over stdin/stdout (for Claude Desktop, etc.)."""
        self._transport = "stdio"
        self._running = True
        loop = asyncio.get_event_loop()

        def read_stdin():
            while self._running:
                try:
                    line = sys.stdin.readline()
                    if not line:
                        break
                    line = line.strip()
                    if line:
                        asyncio.run_coroutine_threadsafe(self._handle_line(line), loop)
                except Exception:
                    break

        reader_thread = threading.Thread(target=read_stdin, daemon=True)
        reader_thread.start()

        while self._running:
            await asyncio.sleep(0.1)

    async def _handle_line(self, line: str):
        response = await self.router.handle_message(line)
        sys.stdout.write(response + "\n")
        sys.stdout.flush()

    async def run_sse(self, host: str = "127.0.0.1", port: int = 9000):
        """Run MCP server over HTTP SSE (Server-Sent Events) for browser-based agents."""
        self._transport = "sse"
        self._running = True
        # Minimal SSE server using aiohttp (imported lazily)
        try:
            from aiohttp import web

            async def handle_post(request: web.Request):
                raw = await request.text()
                response = await self.router.handle_message(raw)
                return web.Response(text=response, content_type="application/json")

            async def handle_get(request: web.Request):
                return web.Response(
                    text="event: endpoint\ndata: /message\n\n",
                    content_type="text/event-stream"
                )

            app = web.Application()
            app.router.add_post("/message", handle_post)
            app.router.add_get("/sse", handle_get)
            app.router.add_get("/", handle_get)

            runner = web.AppRunner(app)
            await runner.setup()
            site = web.TCPSite(runner, host, port)
            await site.start()
            print(f"MCP SSE server listening on http://{host}:{port}")

            while self._running:
                await asyncio.sleep(1)

            await runner.cleanup()
        except ImportError:
            print("aiohttp not installed. SSE transport unavailable.")

    def stop(self):
        self._running = False
