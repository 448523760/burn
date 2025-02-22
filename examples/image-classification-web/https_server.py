import http.server
import socketserver

if __name__ == "__main__":
    print("这是主函数")
    PORT = 8000

    Handler = http.server.SimpleHTTPRequestHandler

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Running local python HTTP server on port {PORT} ...")
        print(f"Serving HTTP on http://localhost:{PORT}/ ...")
        httpd.serve_forever()
