import socket

def check_server(ip, port):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(5)  # 5 seconds timeout
        s.connect((ip, port))
        print(f"✅ Successfully connected to {ip}:{port}")
        s.close()
    except Exception as e:
        print(f"❌ Connection failed: {e}")

check_server("10.128.31.9", 8801)
