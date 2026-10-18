"""Local Jev server with the compact prompt (fastjev.py). Drop-in for the jev-local arm:
same POST /v1/classifier request and the response fields JevClient reads.

usage: python -m jevdoom.fast_server --model Qwen/Qwen3.5-4B [--adapter C:\\tmp\\jev-doom\\models\\<run>\\best] [--port 8012]
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .fastjev import Scorer


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter")
    ap.add_argument("--port", type=int, default=8012)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args(argv)
    scorer = Scorer(args.model, args.adapter)
    lock = threading.Lock()   # one forward at a time on the GPU

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *a):
            pass

        def _send(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(200, {"model": args.model, "adapter": args.adapter})

        def do_POST(self):
            if self.path != "/v1/classifier":
                return self._send(404, {"error": "not found"})
            try:
                req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                t = time.perf_counter()
                with lock:
                    ans = scorer.score(req)
                (qid,) = req["questions"]
                ntok = ans.pop("input_tokens")
                self._send(200, {"answers": {qid: ans}, "usage": {"input_tokens": ntok},
                                 "server_ms": round((time.perf_counter() - t) * 1000, 1)})
            except Exception as e:     # report, keep serving
                self._send(400, {"error": f"{type(e).__name__}: {e}"})

    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"fast Jev server: {args.model} adapter={args.adapter} on http://{args.host}:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
