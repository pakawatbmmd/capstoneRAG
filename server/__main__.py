"""python -m server [--port 8000] [--mock]: start the API without remembering the uvicorn command."""

import argparse
import os

import uvicorn

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
parser.add_argument("--host", default="127.0.0.1")
parser.add_argument("--mock", action="store_true", help="canned responses, no models or GPU")
args = parser.parse_args()
if args.mock:
    os.environ["RAG_MOCK"] = "1"
uvicorn.run("server.app:app", host=args.host, port=args.port)
