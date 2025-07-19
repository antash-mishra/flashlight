import os;
import subprocess;
import sys;
import json;
import time;
import struct;


class NativeMessaging:
    def __init__(self):
        self.native_messaging_host = None
        self.start_native_messaging_host();

    def start_native_messaging_host(self):
        """Start the native messaging host process"""
        try:
            # Start the native messaging host
            script_path = os.path.join(os.path.dirname(__file__), 'ping_pong.py')
            self.native_messaging_host = subprocess.Popen(
                [sys.executable, script_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            print("✅ Native messaging host started")
        except Exception as e:
            print(f"❌ Failed to start native messaging host: {e}")

    def send_message(self, action, data=None):
        """Send a message to the native messaging host"""
        if not self.native_messaging_host:
            print("❌ Native messaging host not running")
            return False

        if self.native_messaging_host.poll() is not None:
            print("❌ Native messaging host process has terminated")
            return False

        try:
            message = json.dumps({
                'action': action, 
                'source': 'native_app',
                'timestamp': time.time(),
                'data': data
            })

            message_bytes = message.encode('utf-8')
            message_length = struct.pack('@I', len(message_bytes))

            # write message to host
            stdin_buffer = getattr(self.native_messaging_host.stdin, 'buffer', None)
            if stdin_buffer:
                stdin_buffer.write(message_length + message_bytes)
                stdin_buffer.flush()
            elif self.native_messaging_host.stdin:
                # Fallback to text mode
                self.native_messaging_host.stdin.write(message)
                self.native_messaging_host.stdin.flush()
                
            print(f"✅ Message sent to extension: {action}")
            return True
        except Exception as e:
            print(f"❌ Failed to send message: {e}")
            return False