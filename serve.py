#!/usr/bin/env python3
"""
DrugEx Hub — Development Server
Serves static assets on DevPort 34100 with zero-cache headers.
"""

import os
import sys
import mimetypes
from http.server import HTTPServer, SimpleHTTPRequestHandler

PORT = 34100
if len(sys.argv) > 1:
    try:
        PORT = int(sys.argv[1])
    except ValueError:
        pass

# Ensure correct custom MIME types
mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/javascript", ".mjs")
mimetypes.add_type("application/json", ".json")
mimetypes.add_type("text/plain", ".sdf")
mimetypes.add_type("text/plain", ".sq")
mimetypes.add_type("text/plain", ".tsv")
mimetypes.add_type("text/csv", ".csv")
mimetypes.add_type("image/svg+xml", ".svg")

class DrugExHubHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # Development zero-cache headers
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

    def do_GET(self):
        if self.path == "/" or self.path == "":
            self.send_response(302)
            self.send_header("Location", "/app/index.html")
            self.end_headers()
            return
        super().do_GET()

def run_server() -> None:
    """Initialize and run the DrugEx development HTTP server bound to localhost."""
    server_address = ("127.0.0.1", PORT)
    httpd = HTTPServer(server_address, DrugExHubHandler)
    print(f"================================================================")
    print(f" 🧬 DrugEx Hub · De Novo Drug Design & ROCS Shape Matching ")
    print(f"================================================================")
    print(f" 🌐 Running at: http://127.0.0.1:{PORT}/app/index.html")
    print(f" 📁 Serving directory: {os.getcwd()}")
    print(f" ⚡ Press Ctrl+C to stop.")
    print(f"================================================================")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        httpd.server_close()

if __name__ == "__main__":
    run_server()
