#!/usr/bin/env python3
"""
Native messaging host for browser extension communication
Directly uses Router class instead of HTTP requests
"""

import sys
import json
import struct
import threading
import time

class NativeMessagingHost:
    def __init__(self):
        # self.router = Router()
        self.running = True
        
    def send_message_to_extension(self, message):
        """Send a message to the browser extension"""
        try:
            # Chrome requires messages to be prefixed with message length
            message_json = json.dumps(message)
            message_bytes = message_json.encode('utf-8')
            message_length = struct.pack('@I', len(message_bytes))
            
            # Write to stdout (Chrome reads from stdin)
            sys.stdout.buffer.write(message_length + message_bytes)
            sys.stdout.buffer.flush()
            print(f"Sent to extension: {message}", file=sys.stderr)
        except Exception as e:
            print(f"Error sending to extension: {e}", file=sys.stderr)
    
    def read_message_from_extension(self):
        """Read a message from the browser extension"""
        try:
            # Read message length (first 4 bytes)
            sendMessage(encodeMessage("ping"))
            raw_length = sys.stdin.buffer.read(4)
            if not raw_length:
                return None
            
            message_length = struct.unpack('@I', raw_length)[0]
            
            # Read the message
            message_bytes = sys.stdin.buffer.read(message_length)
            message = json.loads(message_bytes.decode('utf-8'))
            
            return message
        except Exception as e:
            print(f"Error reading from extension: {e}", file=sys.stderr)
            return None
    
    def handle_extension_message(self, message):
        """Handle messages from browser extension"""
        try:
            action = message.get('action')
            print(f"Received from extension: {message}", file=sys.stderr)
            
            if action == 'openTab':
                # Extension wants to open a tab
                tab_id = message.get('tab_id')
                window_id = message.get('window_id')
                
                # Directly use router methods instead of HTTP
                try:
                    # Log the request (the actual tab switching happens in browser)
                    print(f"Tab opening request for tab {tab_id} in window {window_id}", file=sys.stderr)
                    
                    self.send_message_to_extension({
                        'action': 'openTab',
                        'success': True,
                        'message': f'Tab opening request processed for tab {tab_id}',
                        'tab_id': tab_id,
                        'window_id': window_id
                    })
                except Exception as e:
                    self.send_message_to_extension({
                        'action': 'openTab',
                        'success': False,
                        'error': str(e)
                    })
                    
            # elif action == 'getTabStats':
            #     # Extension wants tab statistics
            #     try:
            #         stats = self.router.get_memory_stats()
            #         self.send_message_to_extension({
            #             'action': 'getTabStats',
            #             'success': True,
            #             'stats': stats
            #         })
            #     except Exception as e:
            #         self.send_message_to_extension({
            #             'action': 'getTabStats',
            #             'success': False,
            #             'error': str(e)
            #         })
                    
            # elif action == 'searchTabs':
            #     # Extension wants to search tabs
            #     query = message.get('query', '')
            #     try:
            #         results = self.router.search_tabs(query)
            #         self.send_message_to_extension({
            #             'action': 'searchTabs',
            #             'success': True,
            #             'results': results
            #         })
            #     except Exception as e:
            #         self.send_message_to_extension({
            #             'action': 'searchTabs',
            #             'success': False,
            #             'error': str(e)
            #         })
                    
            elif action == 'ping':
                # Simple ping/pong for testing
                self.send_message_to_extension({
                    'action': 'pong',
                    'success': True,
                    'timestamp': time.time()
                })
                
            else:
                # Unknown action
                self.send_message_to_extension({
                    'action': action,
                    'success': False,
                    'error': f'Unknown action: {action}'
                })
                
        except Exception as e:
            # Send error response
            self.send_message_to_extension({
                'action': message.get('action', 'unknown'),
                'success': False,
                'error': str(e)
            })
    
    def run(self):
        """Main loop for native messaging"""
        print("Native messaging host started", file=sys.stderr)
        
        while self.running:
            try:
                message = self.read_message_from_extension()
                if message is None:
                    sendMessage(encodeMessage("ping"))
                    break
                
                self.handle_extension_message(message)
                
            except Exception as e:
                print(f"Error in native messaging: {e}", file=sys.stderr)
                break
        
        print("Native messaging host stopped", file=sys.stderr)



def getMessage():
    rawLength = sys.stdin.buffer.read(4)
    if len(rawLength) == 0:
        sys.exit(0)
    messageLength = struct.unpack('@I', rawLength)[0]
    message = sys.stdin.buffer.read(messageLength).decode('utf-8')
    return json.loads(message)

# Encode a message for transmission, given its content.
def encodeMessage(messageContent):
    # https://docs.python.org/3/library/json.html#basic-usage
    # To get the most compact JSON representation, you should specify
    # (',', ':') to eliminate whitespace.
    # We want the most compact representation because the browser rejects
    # messages that exceed 1 MB.
    encodedContent = json.dumps(messageContent, separators=(',', ':')).encode('utf-8')
    encodedLength = struct.pack('@I', len(encodedContent))
    return {'length': encodedLength, 'content': encodedContent}

# Send an encoded message to stdout
def sendMessage(encodedMessage):
    sys.stdout.buffer.write(encodedMessage['length'])
    sys.stdout.buffer.write(encodedMessage['content'])
    sys.stdout.buffer.flush()

# while True:
#     # receivedMessage = getMessage()
#     # if receivedMessage == "ping":
#     #     sendMessage(encodeMessage("pong"))
#     host = NativeMessagingHost()


if __name__ == "__main__":
    host = NativeMessagingHost()
    host.run()