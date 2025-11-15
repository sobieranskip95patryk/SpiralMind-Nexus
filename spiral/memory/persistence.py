"""Memory persistence for SpiralMind-Nexus.

Provides SQLite-based memory storage and retrieval.
"""

import sqlite3
import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import threading
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class MemoryPersistence:
    """SQLite-based memory persistence system."""
    
    def __init__(self, db_path: str = "spiral_memory.db"):
        """Initialize memory persistence.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_database()
        logger.info(f"MemoryPersistence initialized with database: {db_path}")
    
    def _init_database(self) -> None:
        """Initialize database tables."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Create memories table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    data TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    memory_type TEXT DEFAULT 'general',
                    tags TEXT,
                    metadata TEXT
                )
            """)
            
            # Create index for faster queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_timestamp 
                ON memories(timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_memories_type 
                ON memories(memory_type)
            """)
            
            # Create sessions table for tracking processing sessions
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT UNIQUE NOT NULL,
                    start_time DATETIME DEFAULT CURRENT_TIMESTAMP,
                    end_time DATETIME,
                    metadata TEXT
                )
            """)
            
            # Create events table for event tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id TEXT UNIQUE NOT NULL,
                    event_type TEXT NOT NULL,
                    data TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    session_id INTEGER,
                    FOREIGN KEY (session_id) REFERENCES sessions (id)
                )
            """)
            
            conn.commit()
            logger.debug("Database tables initialized")
    
    @contextmanager
    def _get_connection(self):
        """Get database connection with automatic cleanup.
        
        Yields:
            sqlite3.Connection: Database connection
        """
        conn = None
        try:
            conn = sqlite3.connect(
                self.db_path,
                timeout=30.0,
                check_same_thread=False
            )
            conn.row_factory = sqlite3.Row  # Enable column access by name
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()
    
    def save_memory(self, 
                   data: Dict[str, Any],
                   memory_type: str = "general",
                   tags: List[str] = None,
                   metadata: Dict[str, Any] = None) -> int:
        """Save memory to database.
        
        Args:
            data: Memory data to save
            memory_type: Type of memory
            tags: Optional tags for categorization
            metadata: Optional metadata
            
        Returns:
            Memory ID
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                data_json = json.dumps(data, default=str)
                tags_json = json.dumps(tags) if tags else None
                metadata_json = json.dumps(metadata, default=str) if metadata else None
                
                cursor.execute("""
                    INSERT INTO memories (data, memory_type, tags, metadata)
                    VALUES (?, ?, ?, ?)
                """, (data_json, memory_type, tags_json, metadata_json))
                
                memory_id = cursor.lastrowid
                conn.commit()
                
                logger.debug(f"Saved memory {memory_id} of type {memory_type}")
                return memory_id
    
    def get_memory(self, memory_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve memory by ID.
        
        Args:
            memory_id: Memory ID to retrieve
            
        Returns:
            Memory data or None if not found
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, data, timestamp, memory_type, tags, metadata
                FROM memories WHERE id = ?
            """, (memory_id,))
            
            row = cursor.fetchone()
            
            if row:
                return {
                    'id': row['id'],
                    'data': json.loads(row['data']),
                    'timestamp': row['timestamp'],
                    'memory_type': row['memory_type'],
                    'tags': json.loads(row['tags']) if row['tags'] else [],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {}
                }
            
            return None
    
    def get_memories_by_type(self, 
                           memory_type: str,
                           limit: int = 100,
                           offset: int = 0) -> List[Dict[str, Any]]:
        """Get memories by type.
        
        Args:
            memory_type: Type of memories to retrieve
            limit: Maximum number of memories to return
            offset: Number of memories to skip
            
        Returns:
            List of memory dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT id, data, timestamp, memory_type, tags, metadata
                FROM memories 
                WHERE memory_type = ?
                ORDER BY timestamp DESC
                LIMIT ? OFFSET ?
            """, (memory_type, limit, offset))
            
            memories = []
            for row in cursor.fetchall():
                memories.append({
                    'id': row['id'],
                    'data': json.loads(row['data']),
                    'timestamp': row['timestamp'],
                    'memory_type': row['memory_type'],
                    'tags': json.loads(row['tags']) if row['tags'] else [],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {}
                })
            
            return memories
    
    def get_recent_memories(self, 
                          hours: int = 24,
                          memory_type: str = None,
                          limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent memories within specified timeframe.
        
        Args:
            hours: Number of hours to look back
            memory_type: Optional memory type filter
            limit: Maximum number of memories to return
            
        Returns:
            List of recent memory dictionaries
        """
        since = datetime.now() - timedelta(hours=hours)
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            if memory_type:
                cursor.execute("""
                    SELECT id, data, timestamp, memory_type, tags, metadata
                    FROM memories 
                    WHERE timestamp >= ? AND memory_type = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (since.isoformat(), memory_type, limit))
            else:
                cursor.execute("""
                    SELECT id, data, timestamp, memory_type, tags, metadata
                    FROM memories 
                    WHERE timestamp >= ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """, (since.isoformat(), limit))
            
            memories = []
            for row in cursor.fetchall():
                memories.append({
                    'id': row['id'],
                    'data': json.loads(row['data']),
                    'timestamp': row['timestamp'],
                    'memory_type': row['memory_type'],
                    'tags': json.loads(row['tags']) if row['tags'] else [],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {}
                })
            
            return memories
    
    def search_memories(self, 
                       query: str,
                       memory_type: str = None,
                       tags: List[str] = None,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """Search memories by content.
        
        Args:
            query: Text to search for in memory data
            memory_type: Optional memory type filter
            tags: Optional tags filter
            limit: Maximum number of memories to return
            
        Returns:
            List of matching memory dictionaries
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            sql = """
                SELECT id, data, timestamp, memory_type, tags, metadata
                FROM memories 
                WHERE data LIKE ?
            """
            params = [f"%{query}%"]
            
            if memory_type:
                sql += " AND memory_type = ?"
                params.append(memory_type)
            
            if tags:
                for tag in tags:
                    sql += " AND tags LIKE ?"
                    params.append(f"%{tag}%")
            
            sql += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(sql, params)
            
            memories = []
            for row in cursor.fetchall():
                memories.append({
                    'id': row['id'],
                    'data': json.loads(row['data']),
                    'timestamp': row['timestamp'],
                    'memory_type': row['memory_type'],
                    'tags': json.loads(row['tags']) if row['tags'] else [],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {}
                })
            
            return memories
    
    def delete_memory(self, memory_id: int) -> bool:
        """Delete memory by ID.
        
        Args:
            memory_id: Memory ID to delete
            
        Returns:
            True if memory was deleted, False if not found
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                
                deleted = cursor.rowcount > 0
                conn.commit()
                
                if deleted:
                    logger.debug(f"Deleted memory {memory_id}")
                
                return deleted
    
    def cleanup_old_memories(self, days: int = 30) -> int:
        """Delete memories older than specified days.
        
        Args:
            days: Age threshold in days
            
        Returns:
            Number of memories deleted
        """
        cutoff = datetime.now() - timedelta(days=days)
        
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    DELETE FROM memories 
                    WHERE timestamp < ?
                """, (cutoff.isoformat(),))
                
                deleted_count = cursor.rowcount
                conn.commit()
                
                logger.info(f"Cleaned up {deleted_count} old memories (older than {days} days)")
                return deleted_count
    
    def create_session(self, session_id: str, metadata: Dict[str, Any] = None) -> int:
        """Create a new processing session.
        
        Args:
            session_id: Unique session identifier
            metadata: Optional session metadata
            
        Returns:
            Session database ID
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                metadata_json = json.dumps(metadata, default=str) if metadata else None
                
                cursor.execute("""
                    INSERT INTO sessions (session_id, metadata)
                    VALUES (?, ?)
                """, (session_id, metadata_json))
                
                session_db_id = cursor.lastrowid
                conn.commit()
                
                logger.debug(f"Created session {session_id} (DB ID: {session_db_id})")
                return session_db_id
    
    def end_session(self, session_id: str) -> bool:
        """End a processing session.
        
        Args:
            session_id: Session identifier to end
            
        Returns:
            True if session was ended, False if not found
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    UPDATE sessions 
                    SET end_time = CURRENT_TIMESTAMP
                    WHERE session_id = ? AND end_time IS NULL
                """, (session_id,))
                
                updated = cursor.rowcount > 0
                conn.commit()
                
                if updated:
                    logger.debug(f"Ended session {session_id}")
                
                return updated
    
    def log_event(self, 
                 event_id: str,
                 event_type: str,
                 data: Dict[str, Any],
                 session_id: str = None) -> int:
        """Log an event.
        
        Args:
            event_id: Unique event identifier
            event_type: Type of event
            data: Event data
            session_id: Optional session to associate with
            
        Returns:
            Event database ID
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                
                # Get session DB ID if session_id provided
                session_db_id = None
                if session_id:
                    cursor.execute(
                        "SELECT id FROM sessions WHERE session_id = ?", 
                        (session_id,)
                    )
                    row = cursor.fetchone()
                    if row:
                        session_db_id = row['id']
                
                data_json = json.dumps(data, default=str)
                
                cursor.execute("""
                    INSERT INTO events (event_id, event_type, data, session_id)
                    VALUES (?, ?, ?, ?)
                """, (event_id, event_type, data_json, session_db_id))
                
                event_db_id = cursor.lastrowid
                conn.commit()
                
                logger.debug(f"Logged event {event_id} of type {event_type}")
                return event_db_id
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory persistence statistics.
        
        Returns:
            Statistics dictionary
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Memory statistics
            cursor.execute("SELECT COUNT(*) as total FROM memories")
            total_memories = cursor.fetchone()['total']
            
            cursor.execute("""
                SELECT memory_type, COUNT(*) as count 
                FROM memories 
                GROUP BY memory_type
            """)
            memory_types = dict(cursor.fetchall())
            
            # Recent activity (last 24 hours)
            since = datetime.now() - timedelta(hours=24)
            cursor.execute("""
                SELECT COUNT(*) as recent_count 
                FROM memories 
                WHERE timestamp >= ?
            """, (since.isoformat(),))
            recent_memories = cursor.fetchone()['recent_count']
            
            # Session statistics
            cursor.execute("SELECT COUNT(*) as total FROM sessions")
            total_sessions = cursor.fetchone()['total']
            
            cursor.execute("""
                SELECT COUNT(*) as active_count 
                FROM sessions 
                WHERE end_time IS NULL
            """)
            active_sessions = cursor.fetchone()['active_count']
            
            # Event statistics
            cursor.execute("SELECT COUNT(*) as total FROM events")
            total_events = cursor.fetchone()['total']
            
            # Database file size
            db_size = 0
            try:
                db_size = Path(self.db_path).stat().st_size
            except FileNotFoundError:
                pass
            
            return {
                'database_path': self.db_path,
                'database_size_bytes': db_size,
                'total_memories': total_memories,
                'memory_types': memory_types,
                'recent_memories_24h': recent_memories,
                'total_sessions': total_sessions,
                'active_sessions': active_sessions,
                'total_events': total_events
            }
    
    def vacuum_database(self) -> None:
        """Vacuum database to reclaim space."""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("VACUUM")
                logger.info("Database vacuumed")
    
    def close(self) -> None:
        """Close database connection and cleanup."""
        logger.info("MemoryPersistence closed")
        # SQLite connections are closed automatically in context manager
    
    def export_memories(self, 
                       output_path: str,
                       memory_type: str = None,
                       since: datetime = None) -> int:
        """Export memories to JSON file.
        
        Args:
            output_path: Path to output JSON file
            memory_type: Optional memory type filter
            since: Optional datetime filter
            
        Returns:
            Number of memories exported
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            sql = """
                SELECT id, data, timestamp, memory_type, tags, metadata
                FROM memories
            """
            params = []
            
            conditions = []
            if memory_type:
                conditions.append("memory_type = ?")
                params.append(memory_type)
            
            if since:
                conditions.append("timestamp >= ?")
                params.append(since.isoformat())
            
            if conditions:
                sql += " WHERE " + " AND ".join(conditions)
            
            sql += " ORDER BY timestamp DESC"
            
            cursor.execute(sql, params)
            
            memories = []
            for row in cursor.fetchall():
                memories.append({
                    'id': row['id'],
                    'data': json.loads(row['data']),
                    'timestamp': row['timestamp'],
                    'memory_type': row['memory_type'],
                    'tags': json.loads(row['tags']) if row['tags'] else [],
                    'metadata': json.loads(row['metadata']) if row['metadata'] else {}
                })
            
            # Write to file
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(memories, f, indent=2, default=str)
            
            logger.info(f"Exported {len(memories)} memories to {output_path}")
            return len(memories)
