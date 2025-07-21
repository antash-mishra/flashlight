import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any
import threading
import faiss
import hashlib
from sentence_transformers import SentenceTransformer
from vecstore import VectorStore
# You can import your Router class here if you want to reuse logic
# from .router import Router

app = FastAPI()

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store connected WebSocket clients
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
        self.lock = threading.Lock()

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        with self.lock:
            self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        with self.lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def send_message(self, message: dict):
        data = json.dumps(message)
        with self.lock:
            for connection in self.active_connections:
                try:
                    asyncio.create_task(connection.send_text(data))
                except Exception as e:
                    print(f"Error sending message: {e}")

    def get_connections(self):
        with self.lock:
            return list(self.active_connections)

manager = ConnectionManager()

# Create a single VectorStore instance
vectorstore = VectorStore()

def handle_update_tabs(data):
    open_tabs = data.get('openTabs', [])
    tab_index = data.get('tabIndex', {})
    # Hash to check if update is needed
    current_hash = vectorstore._hash_tab_index(tab_index)
    if current_hash != vectorstore.last_tab_index_hash:
        print(f"Tab changes detected. Updating FAISS index...")
        vectorstore.update_index(open_tabs, tab_index)
        vectorstore.last_tab_index_hash = current_hash
    else:
        print("No tab changes detected. Skipping re-indexing.")
    return {"status": "success", "received": len(open_tabs)}

def handle_search_tabs(data):
    query = data.get('query', '')
    results = vectorstore.search(query)
    return {"results": results}

def handle_open_tab(data):
    tab_id = data.get('tab_id')
    window_id = data.get('window_id')
    print(f"[WS] Open tab requested: tab_id={tab_id}, window_id={window_id}")
    return {"status": "success", "message": f"Tab switching request sent for tab {tab_id}", "tab_id": tab_id, "window_id": window_id}

def handle_memory_stats(data):
    stats = vectorstore.get_memory_stats()
    return stats

def handle_memory_clear(data):
    vectorstore.clear_memory()
    return {"status": "success", "message": "Memory cleared"}

# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                
                print("MEssage from extension: ", message)

                action = message.get('action')
                if action == 'updateTabs':
                    response = handle_update_tabs(message)
                    await websocket.send_text(json.dumps({"action": "updateTabsResponse", **response}))
                elif action == 'searchTabs':
                    response = handle_search_tabs(message)
                    await websocket.send_text(json.dumps({"action": "searchTabsResponse", **response}))
                elif action == 'openTab':
                    response = handle_open_tab(message)
                    await websocket.send_text(json.dumps({"action": "openTabResponse", **response}))
                elif action == 'memoryStats':
                    response = handle_memory_stats(message)
                    await websocket.send_text(json.dumps({"action": "memoryStatsResponse", **response}))
                elif action == 'memoryClear':
                    response = handle_memory_clear(message)
                    await websocket.send_text(json.dumps({"action": "memoryClearResponse", **response}))
                else:
                    response = handle_update_tabs(message)
                    await websocket.send_text(json.dumps({"action": "updateTabsResponse", **response}))

            except Exception as e:
                await websocket.send_text(json.dumps({"error": str(e)}))
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        print("[WS] Client disconnected")

# Function to send a message to all connected clients (for use in on_tab_selected)
def send_message_to_extension(message: dict):
    asyncio.run(manager.send_message(message))

# Example usage: from another thread or function
# send_message_to_extension({"action": "openTab", "tab_id": 123, "window_id": 1}) 