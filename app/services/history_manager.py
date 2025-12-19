import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any
import logging
import aiofiles

logger = logging.getLogger(__name__)


# Data directory
HISTORY_DIR = Path("data/history")
HISTORY_DIR.mkdir(parents=True, exist_ok=True)


class HistoryTab:
    """Represents a single history tab (search session)"""
    
    def __init__(
        self,
        tab_id: str,
        query: str,
        questions: List[Dict],
        created_at: str,
        metadata: Optional[Dict] = None
    ):
        self.tab_id = tab_id
        self.query = query
        self.questions = questions
        self.created_at = created_at
        self.metadata = metadata or {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "tab_id": self.tab_id,
            "query": self.query,
            "questions": self.questions,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "question_count": len(self.questions)
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "HistoryTab":
        """Create from dictionary"""
        return cls(
            tab_id=data["tab_id"],
            query=data["query"],
            questions=data["questions"],
            created_at=data["created_at"],
            metadata=data.get("metadata", {})
        )


class HistoryManager:
    """
    Manage persistent history of interview question searches
    
    Storage:
    - Each user has their own history file
    - History stored in JSONL format (one tab per line)
    - Automatic backup on write
    
    Usage:
        history = HistoryManager(user_id="user123")
        await history.initialize()
        
        # Save search
        tab_id = await history.save_search(
            query="python questions",
            questions=[...]
        )
        
        # Get all tabs
        tabs = await history.get_all_tabs()
        
        # Delete tab
        await history.delete_tab(tab_id)
    """
    
    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.history_file = HISTORY_DIR / f"{user_id}_history.jsonl"
        self.backup_file = HISTORY_DIR / f"{user_id}_history.backup.jsonl"
        self._tabs: Dict[str, HistoryTab] = {}
        self._loaded = False
        self._lock = asyncio.Lock()  # Prevent concurrent writes
    
    async def initialize(self):
        """Load history from disk"""
        if self._loaded:
            return
        
        async with self._lock:
            # Check again inside lock to avoid double loading
            if self._loaded:
                return
            
            try:
                await self._load_from_disk()
                self._loaded = True
                logger.info(f"💾 History initialized for user {self.user_id}: {len(self._tabs)} tabs")
            except Exception as e:
                logger.error(f"❌ Failed to load history: {e}")
                self._tabs = {}
                self._loaded = True # Mark as loaded even if empty to avoid retry loops
    
    async def _load_from_disk(self):
        """Load history from JSONL file"""
        if not self.history_file.exists():
            logger.info(f"No history file found for user {self.user_id}")
            return
        
        self._tabs = {}
        
        try:
            async with aiofiles.open(self.history_file, 'r', encoding='utf-8') as f:
                async for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        tab = HistoryTab.from_dict(data)
                        self._tabs[tab.tab_id] = tab
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping invalid JSON line: {e}")
                        continue
            
            logger.info(f"Loaded {len(self._tabs)} history tabs from disk")
        
        except Exception as e:
            logger.error(f"Error loading history: {e}")
            # Try to restore from backup
            await self._restore_from_backup()
    
    async def _restore_from_backup(self):
        """Restore history from backup file"""
        if not self.backup_file.exists():
            logger.warning("No backup file available")
            return
        
        try:
            logger.info("Attempting to restore from backup...")
            self._tabs = {}
            
            async with aiofiles.open(self.backup_file, 'r', encoding='utf-8') as f:
                async for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    data = json.loads(line)
                    tab = HistoryTab.from_dict(data)
                    self._tabs[tab.tab_id] = tab
            
            logger.info(f"Restored {len(self._tabs)} tabs from backup")
            
            # Save restored data to main file
            await self._save_to_disk()
        
        except Exception as e:
            logger.error(f"Backup restoration failed: {e}")
    
    async def _save_to_disk(self):
        """Save all tabs to disk with locking"""
        async with self._lock:
            try:
                # Ensure parent directory exists
                HISTORY_DIR.mkdir(parents=True, exist_ok=True)
                
                # Create backup first
                if self.history_file.exists():
                    async with aiofiles.open(self.history_file, 'r', encoding='utf-8') as src:
                        content = await src.read()
                        async with aiofiles.open(self.backup_file, 'w', encoding='utf-8') as dst:
                            await dst.write(content)
                
                # Write new history atomicaly (write to temp file then rename is better, 
                # but for now we just use a lock + backup)
                async with aiofiles.open(self.history_file, 'w', encoding='utf-8') as f:
                    for tab in self._tabs.values():
                        line = json.dumps(tab.to_dict(), ensure_ascii=False)
                        await f.write(line + '\n')
                
                logger.debug(f"✅ Saved {len(self._tabs)} tabs to disk for {self.user_id}")
            
            except Exception as e:
                logger.error(f"❌ Failed to save history for {self.user_id}: {e}", exc_info=True)
    
    async def save_search(
        self,
        query: str,
        questions: List[Dict],
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Save a search as a new history tab
        
        Args:
            query: Search query
            questions: List of questions returned
            metadata: Optional metadata (company, verified_only, etc.)
        
        Returns:
            tab_id: Unique identifier for this tab
        """
        if not self._loaded:
            await self.initialize()
        
        tab_id = str(uuid.uuid4())
        
        tab = HistoryTab(
            tab_id=tab_id,
            query=query,
            questions=questions,
            created_at=datetime.now(timezone.utc).isoformat(),
            metadata=metadata or {}
        )
        
        self._tabs[tab_id] = tab
        
        # Save to disk
        await self._save_to_disk()
        
        logger.info(f"Saved search to history: tab_id={tab_id}, query='{query}', count={len(questions)}")
        
        return tab_id
    
    async def get_tab(self, tab_id: str) -> Optional[HistoryTab]:
        """Get a specific tab by ID"""
        if not self._loaded:
            await self.initialize()
        
        return self._tabs.get(tab_id)
    
    async def get_all_tabs(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        sort_by: str = "created_at",
        ascending: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Get all history tabs
        
        Args:
            limit: Max number of tabs to return
            offset: Number of tabs to skip
            sort_by: Field to sort by (created_at, query, question_count)
            ascending: Sort order
        
        Returns:
            List of tab dictionaries
        """
        if not self._loaded:
            await self.initialize()
        
        # Convert to list
        tabs = list(self._tabs.values())
        
        # Sort
        if sort_by == "created_at":
            tabs.sort(key=lambda t: t.created_at, reverse=not ascending)
        elif sort_by == "query":
            tabs.sort(key=lambda t: t.query.lower(), reverse=not ascending)
        elif sort_by == "question_count":
            tabs.sort(key=lambda t: len(t.questions), reverse=not ascending)
        
        # Paginate
        if offset:
            tabs = tabs[offset:]
        if limit:
            tabs = tabs[:limit]
        
        return [tab.to_dict() for tab in tabs]
    
    async def update_tab(
        self,
        tab_id: str,
        query: Optional[str] = None,
        questions: Optional[List[Dict]] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Update an existing tab
        
        Returns:
            True if updated, False if tab not found
        """
        if not self._loaded:
            await self.initialize()
        
        tab = self._tabs.get(tab_id)
        if not tab:
            return False
        
        # Update fields
        if query is not None:
            tab.query = query
        if questions is not None:
            tab.questions = questions
        if metadata is not None:
            tab.metadata.update(metadata)
        
        # Save to disk
        await self._save_to_disk()
        
        logger.info(f"Updated tab: {tab_id}")
        return True
    
    async def delete_tab(self, tab_id: str) -> bool:
        """
        Delete a history tab
        
        Returns:
            True if deleted, False if tab not found
        """
        if not self._loaded:
            await self.initialize()
        
        if tab_id not in self._tabs:
            return False
        
        del self._tabs[tab_id]
        
        # Save to disk
        await self._save_to_disk()
        
        logger.info(f"Deleted tab: {tab_id}")
        return True
    
    async def delete_all_tabs(self) -> int:
        """
        Delete all history tabs
        
        Returns:
            Number of tabs deleted
        """
        if not self._loaded:
            await self.initialize()
        
        count = len(self._tabs)
        self._tabs = {}
        
        # Save empty history
        await self._save_to_disk()
        
        logger.info(f"Deleted all {count} tabs")
        return count
    
    async def search_history(
        self,
        search_query: str,
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Search within history
        
        Searches in:
        - Tab query
        - Question text
        - Question answers
        
        Returns:
            Matching tabs
        """
        if not self._loaded:
            await self.initialize()
        
        search_lower = search_query.lower()
        matches = []
        
        for tab in self._tabs.values():
            # Check query
            if search_lower in tab.query.lower():
                matches.append(tab.to_dict())
                continue
            
            # Check questions
            for question in tab.questions:
                q_text = question.get("question", "").lower()
                a_text = question.get("answer", "").lower()
                
                if search_lower in q_text or search_lower in a_text:
                    matches.append(tab.to_dict())
                    break
        
        # Sort by relevance (created_at descending)
        matches.sort(key=lambda t: t["created_at"], reverse=True)
        
        return matches[:limit]
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get history statistics"""
        if not self._loaded:
            await self.initialize()
        
        total_questions = sum(len(tab.questions) for tab in self._tabs.values())
        
        # Most common query words
        from collections import Counter
        words = []
        for tab in self._tabs.values():
            words.extend(tab.query.lower().split())
        
        common_words = Counter(words).most_common(10)
        
        return {
            "total_tabs": len(self._tabs),
            "total_questions": total_questions,
            "avg_questions_per_tab": (
                total_questions / len(self._tabs) if self._tabs else 0
            ),
            "most_common_queries": common_words,
            "oldest_tab": (
                min(self._tabs.values(), key=lambda t: t.created_at).created_at
                if self._tabs else None
            ),
            "newest_tab": (
                max(self._tabs.values(), key=lambda t: t.created_at).created_at
                if self._tabs else None
            )
        }
    
    async def export_history(self, format: str = "json") -> str:
        """
        Export entire history
        
        Args:
            format: 'json' or 'csv'
        
        Returns:
            Exported data as string
        """
        if not self._loaded:
            await self.initialize()
        
        if format == "json":
            tabs = [tab.to_dict() for tab in self._tabs.values()]
            return json.dumps(tabs, indent=2, ensure_ascii=False)
        
        elif format == "csv":
            import csv
            from io import StringIO
            
            output = StringIO()
            writer = csv.writer(output)
            
            # Header
            writer.writerow(["Tab ID", "Query", "Question Count", "Created At"])
            
            # Rows
            for tab in self._tabs.values():
                writer.writerow([
                    tab.tab_id,
                    tab.query,
                    len(tab.questions),
                    tab.created_at
                ])
            
            return output.getvalue()
        
        else:
            raise ValueError(f"Unsupported format: {format}")


# Global instance (default user)
default_history_manager = HistoryManager()


# Export
__all__ = [
    'HistoryManager',
    'HistoryTab',
    'default_history_manager'
]