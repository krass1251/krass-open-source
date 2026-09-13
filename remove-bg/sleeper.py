"""
The sleeping stand-in for server.py. Plain stdlib, no torch: ~15 MB instead
of the ~1 GB an idle server.py holds.

server.py exec()s into this after --sleep-after minutes without work, with
its own argv, so the port and the PID stay the same and nohup/log keep
working. Anything that needs the real server wakes it: POST /api/wake (the
web page does it by itself when a photo is dropped, serve.sh does it for the
Mac app and the Quick Action). Waking exec()s back into server.py with the
saved argv; the port is free for ~2-3 s while torch imports.

    GET  /            tiny page that wakes the server and reloads
    GET  /api/status  {"sleeping": true}
    POST /api/wake    -> back to server.py
    POST /api/quit    exit
"""

import argparse
import http.server
import json
import os
import sys
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
ARGS = sys.argv[1:]

WAKE_PAGE = b"""<!doctype html><meta charset="utf-8"><title>remove background</title>
<body style="font:15px system-ui;padding:40px;color:#444">Waking the server up&hellip;
<script>
fetch('/api/wake', {method: 'POST'}).catch(() => {});
(async function poll() {
  try {
    const s = await (await fetch('/api/status')).json();
    if (!s.sleeping) return location.reload();
  } catch (e) {}
  setTimeout(poll, 700);
})();
</script>"""


def wake():
    os.execv(sys.executable, [sys.executable, os.path.join(HERE, "server.py"), *ARGS])


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def local(self):
        """Same rule as server.py: only the page on this machine may talk to
        us, so a web page in the browser cannot quit or wake the server."""
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]")
        if host not in ("127.0.0.1", "localhost", "::1"):
            return False
        origin = self.headers.get("Origin")
        if origin:
            o = origin.split("//", 1)[-1].rsplit(":", 1)[0].strip("[]")
            if o != host:
                return False
        return True

    def reply(self, code, body, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def do_GET(self):
        if not self.local():
            self.reply(403, b'{"error": "not a local request"}')
        elif self.path.startswith("/api/status"):
            self.reply(200, json.dumps({"sleeping": True}).encode())
        elif self.path == "/":
            self.reply(200, WAKE_PAGE, "text/html; charset=utf-8")
        else:
            self.reply(503, b'{"sleeping": true}')

    def do_POST(self):
        if not self.local():
            self.reply(403, b'{"error": "not a local request"}')
        elif self.path.startswith("/api/wake"):
            self.reply(200, b'{"waking": true}')
            threading.Timer(0.2, wake).start()   # after the response is out
        elif self.path.startswith("/api/quit"):
            self.reply(200, b'{"quit": true}')
            threading.Timer(0.2, lambda: os._exit(0)).start()
        else:
            self.reply(503, b'{"sleeping": true}')

    do_DELETE = do_POST


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8777)
    args, _ = ap.parse_known_args(ARGS)   # the rest belongs to server.py
    print(f"sleeping on http://{args.host}:{args.port}, POST /api/wake to resume", flush=True)
    http.server.ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
