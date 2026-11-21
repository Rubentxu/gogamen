"""GOG-153: timeouts, drenado de stderr y recogida del proceso en GogamenClient."""

from __future__ import annotations

import sys
import time
import unittest

from gogamen import GogamenClient, GogamenProtocolError, GogamenTimeoutError

SILENT = "import sys,time\nsys.stdin.readline()\ntime.sleep(60)\n"
FLOOD = (
    "import sys,time\n"
    "sys.stdin.readline()\n"
    "e=sys.stderr.buffer\n"
    "for i in range(40):\n"
    "    e.write(b'x'*65536); e.flush()\n"
    "e.write(b'FINAL_MARK'); e.flush()\n"
    "time.sleep(60)\n"
)
ECHO = (
    "import sys,json\n"
    "for l in sys.stdin:\n"
    "    r=json.loads(l)\n"
    "    print(json.dumps({'id':r['id'],'ok':True,'closed':True}),flush=True)\n"
)
STUBBORN = (  # ignora SIGTERM: obliga a kill
    "import signal,sys,time\n"
    "signal.signal(signal.SIGTERM, signal.SIG_IGN)\n"
    "sys.stdin.readline()\n"
    "time.sleep(60)\n"
)


def _client(code: str, timeout: float = 0.5) -> GogamenClient:
    return GogamenClient(command=[sys.executable, "-P", "-B", "-c", code], request_timeout=timeout)


class Gog153TimeoutTests(unittest.TestCase):
    def test_gog_153_silent_server_times_out_and_reaps(self) -> None:
        client = _client(SILENT)
        t0 = time.monotonic()
        with self.assertRaises(GogamenTimeoutError) as ctx:
            client.close_session("s")
        self.assertLess(time.monotonic() - t0, 8.0)
        self.assertIsInstance(ctx.exception, GogamenProtocolError)
        self.assertIsNotNone(client._proc.poll())
        with self.assertRaises(GogamenProtocolError) as ctx2:
            client.close_session("s")
        self.assertEqual(ctx2.exception.code, "ClientClosed")

    def test_gog_153_stderr_flood_times_out_with_bounded_tail(self) -> None:
        client = _client(FLOOD, timeout=1.0)
        t0 = time.monotonic()
        with self.assertRaises(GogamenTimeoutError) as ctx:
            client.close_session("s")
        self.assertLess(time.monotonic() - t0, 8.0)
        tail = ctx.exception.stderr_tail
        self.assertIn("FINAL_MARK", tail)
        self.assertLessEqual(len(tail), 2 * 64 * 1024 + 65536)
        self.assertIsNotNone(client._proc.poll())

    def test_gog_153_normal_server_still_works(self) -> None:
        with _client(ECHO) as client:
            self.assertTrue(client.close_session("a"))
            self.assertTrue(client.close_session("b"))
        self.assertIsNotNone(client._proc.poll())

    def test_gog_153_close_kills_stubborn_process(self) -> None:
        client = _client(STUBBORN)
        t0 = time.monotonic()
        client.close()
        self.assertLess(time.monotonic() - t0, 10.0)
        self.assertIsNotNone(client._proc.poll())
        client.close()  # idempotente


if __name__ == "__main__":
    unittest.main()
