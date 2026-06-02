from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import os

class SimpleHTTPRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Discord Bot is online and listening.")

def run_server():
    # Render assigns a port dynamically, default to 8080 locally
    port = int(os.environ.get("PORT", 8080))
    server = HTTPServer(('0.0.0.0', port), SimpleHTTPRequestHandler)
    server.serve_forever()

def keep_alive():
    # Run the web server on a separate background thread
    t = threading.Thread(target=run_server)
    t.daemon = True
    t.start()