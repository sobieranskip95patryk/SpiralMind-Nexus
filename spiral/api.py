"""REST API for SpiralMind-Nexus.

Provides FastAPI-based REST endpoints and WebSocket support.
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field, validator
    import uvicorn
except ImportError:
    raise ImportError("FastAPI dependencies not installed. Install with: pip install fastapi uvicorn websockets")

from .pipeline.double_pipeline import execute, batch_execute, create_event, get_pipeline_statistics, reset_pipeline
from .config.loader import load_config, Cfg
from .memory.persistence import MemoryPersistence
from .utils.logging_config import get_logger
from . import __version__

logger = get_logger(__name__)

# Pydantic models for request/response
class ProcessRequest(BaseModel):
    """Request model for text processing."""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to process")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Processing context")
    mode: str = Field(default="quantum", pattern="^(quantum|gokai|hybrid|debug)$", description="Processing mode")
    save_to_memory: bool = Field(default=False, description="Save result to memory")
    
    @validator('text')
    def validate_text(cls, v):
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()


class BatchProcessRequest(BaseModel):
    """Request model for batch processing."""
    texts: List[str] = Field(..., min_items=1, max_items=100, description="List of texts to process")
    contexts: Optional[List[Dict[str, Any]]] = Field(default=None, description="Optional contexts for each text")
    mode: str = Field(default="quantum", pattern="^(quantum|gokai|hybrid|debug)$", description="Processing mode")
    parallel: bool = Field(default=True, description="Enable parallel processing")
    save_to_memory: bool = Field(default=False, description="Save results to memory")
    
    @validator('texts')
    def validate_texts(cls, v):
        if not v:
            raise ValueError("Texts list cannot be empty")
        for i, text in enumerate(v):
            if not text or not text.strip():
                raise ValueError(f"Text at index {i} cannot be empty")
        return [text.strip() for text in v]


class EventRequest(BaseModel):
    """Request model for event creation."""
    text: str = Field(..., min_length=1, max_length=10000, description="Event text")
    event_type: str = Field(default="processing", description="Event type")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Event metadata")
    process_immediately: bool = Field(default=True, description="Process event immediately")


class ProcessResponse(BaseModel):
    """Response model for processing results."""
    success: bool
    result: Optional[Dict[str, Any]] = None
    decision: Optional[str] = None
    confidence: Optional[float] = None
    processing_time: Optional[float] = None
    quantum_score: Optional[float] = None
    gokai_score: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class MemorySearchRequest(BaseModel):
    """Request model for memory search."""
    query: str = Field(..., min_length=1, description="Search query")
    memory_type: Optional[str] = Field(default=None, description="Memory type filter")
    limit: int = Field(default=50, ge=1, le=200, description="Maximum results")


# WebSocket connection manager
class ConnectionManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket client {client_id} connected")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"WebSocket client {client_id} disconnected")
    
    async def send_personal_message(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)
    
    async def broadcast(self, message: dict):
        disconnected = []
        for client_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected.append(client_id)
        
        for client_id in disconnected:
            self.disconnect(client_id)


# Global instances
manager = ConnectionManager()
memory = MemoryPersistence()


def create_app(config: Cfg = None) -> FastAPI:
    """Create FastAPI application.
    
    Args:
        config: Optional configuration
        
    Returns:
        Configured FastAPI app
    """
    if config is None:
        config = Cfg()
    
    app = FastAPI(
        title="SpiralMind-Nexus API",
        description="Advanced Text Processing and Analysis System",
        version=__version__,
        docs_url="/docs" if config.api.enable_docs else None,
        redoc_url="/redoc" if config.api.enable_docs else None
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Error handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request, exc):
        return JSONResponse(
            status_code=exc.status_code,
            content={"success": False, "error": exc.detail}
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request, exc):
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "Internal server error"}
        )
    
    # Health check endpoint
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "version": __version__,
            "timestamp": datetime.now().isoformat()
        }
    
    # Processing endpoints
    @app.post("/process", response_model=ProcessResponse)
    async def process_text(request: ProcessRequest):
        """Process single text."""
        try:
            result = execute(
                text=request.text,
                context=request.context,
                mode=request.mode,
                save_to_memory=request.save_to_memory
            )
            
            return ProcessResponse(
                success=result['success'],
                result=result.get('result'),
                decision=result.get('decision'),
                confidence=result.get('confidence'),
                processing_time=result.get('processing_time'),
                quantum_score=result.get('quantum_score'),
                gokai_score=result.get('gokai_score'),
                metadata=result.get('metadata'),
                error=result.get('error')
            )
            
        except Exception as e:
            logger.error(f"Error processing text: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.post("/batch", response_model=List[ProcessResponse])
    async def batch_process_texts(request: BatchProcessRequest):
        """Process multiple texts."""
        try:
            results = batch_execute(
                texts=request.texts,
                contexts=request.contexts,
                mode=request.mode,
                parallel=request.parallel,
                save_to_memory=request.save_to_memory
            )
            
            response = []
            for result in results:
                response.append(ProcessResponse(
                    success=result['success'],
                    result=result.get('result'),
                    decision=result.get('decision'),
                    confidence=result.get('confidence'),
                    processing_time=result.get('processing_time'),
                    quantum_score=result.get('quantum_score'),
                    gokai_score=result.get('gokai_score'),
                    metadata=result.get('metadata'),
                    error=result.get('error')
                ))
            
            return response
            
        except Exception as e:
            logger.error(f"Error in batch processing: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Event endpoints
    @app.post("/events")
    async def create_processing_event(request: EventRequest):
        """Create and optionally process an event."""
        try:
            result = create_event(
                text=request.text,
                event_type=request.event_type,
                metadata=request.metadata,
                process_immediately=request.process_immediately
            )
            
            # Broadcast event to WebSocket clients
            await manager.broadcast({
                "type": "event_created",
                "event": result.get('event'),
                "timestamp": datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating event: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Memory endpoints
    @app.post("/memory/search")
    async def search_memory(request: MemorySearchRequest):
        """Search memory for patterns."""
        try:
            results = memory.search_memories(
                query=request.query,
                memory_type=request.memory_type,
                limit=request.limit
            )
            
            return {
                "success": True,
                "results": results,
                "count": len(results)
            }
            
        except Exception as e:
            logger.error(f"Error searching memory: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/memory/recent")
    async def get_recent_memories(hours: int = 24, memory_type: str = None, limit: int = 50):
        """Get recent memories."""
        try:
            results = memory.get_recent_memories(
                hours=hours,
                memory_type=memory_type,
                limit=limit
            )
            
            return {
                "success": True,
                "results": results,
                "count": len(results)
            }
            
        except Exception as e:
            logger.error(f"Error getting recent memories: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/memory/{memory_id}")
    async def get_memory_by_id(memory_id: int):
        """Get specific memory by ID."""
        try:
            result = memory.get_memory(memory_id)
            
            if result:
                return {"success": True, "memory": result}
            else:
                raise HTTPException(status_code=404, detail="Memory not found")
                
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error getting memory {memory_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Statistics endpoints
    @app.get("/statistics")
    async def get_statistics():
        """Get system statistics."""
        try:
            pipeline_stats = get_pipeline_statistics()
            memory_stats = memory.get_statistics()
            
            return {
                "success": True,
                "pipeline": pipeline_stats.get('statistics', {}),
                "memory": memory_stats,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # Management endpoints
    @app.post("/reset")
    async def reset_system():
        """Reset system statistics and cache."""
        try:
            result = reset_pipeline()
            return result
            
        except Exception as e:
            logger.error(f"Error resetting system: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # WebSocket endpoint
    @app.websocket("/ws/{client_id}")
    async def websocket_endpoint(websocket: WebSocket, client_id: str):
        """WebSocket endpoint for real-time communication."""
        await manager.connect(websocket, client_id)
        
        try:
            # Send welcome message
            await manager.send_personal_message({
                "type": "welcome",
                "message": "Connected to SpiralMind-Nexus",
                "client_id": client_id,
                "version": __version__
            }, client_id)
            
            while True:
                # Receive message from client
                data = await websocket.receive_text()
                
                try:
                    message = json.loads(data)
                    message_type = message.get('type')
                    
                    if message_type == 'process':
                        # Process text via WebSocket
                        text = message.get('text', '')
                        context = message.get('context', {})
                        mode = message.get('mode', 'quantum')
                        
                        if text:
                            # Add client context
                            context['websocket_client'] = client_id
                            context['realtime_processing'] = True
                            
                            result = execute(
                                text=text,
                                context=context,
                                mode=mode,
                                save_to_memory=False
                            )
                            
                            await manager.send_personal_message({
                                "type": "process_result",
                                "request_id": message.get('request_id'),
                                "result": result
                            }, client_id)
                        else:
                            await manager.send_personal_message({
                                "type": "error",
                                "message": "No text provided",
                                "request_id": message.get('request_id')
                            }, client_id)
                    
                    elif message_type == 'ping':
                        await manager.send_personal_message({
                            "type": "pong",
                            "timestamp": datetime.now().isoformat()
                        }, client_id)
                    
                    else:
                        await manager.send_personal_message({
                            "type": "error",
                            "message": f"Unknown message type: {message_type}"
                        }, client_id)
                        
                except json.JSONDecodeError:
                    await manager.send_personal_message({
                        "type": "error",
                        "message": "Invalid JSON message"
                    }, client_id)
                    
        except WebSocketDisconnect:
            manager.disconnect(client_id)
        except Exception as e:
            logger.error(f"WebSocket error for client {client_id}: {e}")
            manager.disconnect(client_id)
    
    return app


def run_server(host: str = "127.0.0.1", 
              port: int = 8000,
              config_path: str = None,
              debug: bool = False) -> None:
    """Run the FastAPI server.
    
    Args:
        host: Server host
        port: Server port
        config_path: Optional configuration file path
        debug: Enable debug mode
    """
    # Load configuration
    if config_path:
        config = load_config(config_path)
    else:
        config = Cfg()
    
    # Override with parameters
    if host != "127.0.0.1":
        config.api.host = host
    if port != 8000:
        config.api.port = port
    if debug:
        config.api.debug = debug
    
    # Create app
    app = create_app(config)
    
    logger.info(f"Starting SpiralMind-Nexus API server on {config.api.host}:{config.api.port}")
    
    # Run server
    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port,
        debug=config.api.debug,
        log_level="info" if not config.api.debug else "debug"
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SpiralMind-Nexus API Server")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8000, help="Server port")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    
    run_server(
        host=args.host,
        port=args.port,
        config_path=args.config,
        debug=args.debug
    )
