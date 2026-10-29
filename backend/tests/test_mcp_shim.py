import asyncio
import io
import json
import unittest
from unittest.mock import patch

from backend.mcp_shim import __main__ as shim


class TestMainConfiguresUtf8Stdio(unittest.TestCase):
    def test_main_configures_utf8_stdio_before_reading_json_rpc(self):
        text = "ação, coração, você"
        request = {
            "jsonrpc": "2.0",
            "method": "voicebox_speak",
            "params": {"text": text, "language": "pt"},
        }
        stdin = io.TextIOWrapper(
            io.BytesIO((json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8")),
            encoding="cp1252",
        )
        stdout = io.TextIOWrapper(io.BytesIO(), encoding="cp1252")

        original_run = asyncio.run

        def read_stdin_without_running_proxy(coroutine):
            coroutine.close()
            return original_run(shim._read_stdin_line())

        with (
            patch.object(shim.sys, "stdin", stdin),
            patch.object(shim.sys, "stdout", stdout),
            patch.object(shim.asyncio, "run", read_stdin_without_running_proxy),
        ):
            result = shim.main()

        expected = json.dumps(request, ensure_ascii=False) + "\n"
        assert result == expected
        assert json.loads(result)["params"]["text"] == text
        assert stdout.encoding == "utf-8"
