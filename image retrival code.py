import socket

def download_image(ip, port, output_filename):
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect((ip, port))

        with open(output_filename, "wb") as f:
            while True:
                data = s.recv(4096)
                if not data:
                    break
                f.write(data)

        print(f"✅ Image saved as {output_filename}")
        s.close()

    except Exception as e:
        print(f"❌ Error: {e}")

# Example usage
download_image("10.128.31.9", 8801, "downloaded_image.jpg")
