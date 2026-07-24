import asyncio
import json
import logging
from typing import List, Dict
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from src.services.training_service import register_ws_listener, unregister_ws_listener

router = APIRouter(tags=["WebSockets"])

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                self.disconnect(connection)

manager = ConnectionManager()

@router.websocket("/ws/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    
    # Callback wrapper for training updates
    loop = asyncio.get_event_loop()
    
    def on_training_update(data: dict):
        asyncio.run_coroutine_threadsafe(manager.broadcast(data), loop)
        
    register_ws_listener(on_training_update)
    
    try:
        while True:
            # Receive client ping or keepalive
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        unregister_ws_listener(on_training_update)
    except Exception:
        manager.disconnect(websocket)
        unregister_ws_listener(on_training_update)
