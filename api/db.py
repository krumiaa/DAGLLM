# api/db.py
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Float, JSON
import datetime
from datetime import datetime, timezone

DATABASE_URL = "sqlite+aiosqlite:///./memory.db"

engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
Base = declarative_base()

class Agent(Base):
    __tablename__ = "agents"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    type = Column(String)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

class Entity(Base):
    __tablename__ = "entities"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    type = Column(String)
    agent_id = Column(Integer, ForeignKey("agents.id"))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    tags = Column(String)  # ✅ New for tags
    # In Entity model
    sentiment_value = Column(Float, nullable=True)  # -1.0 to +1.0

class Edge(Base):
    confidence = Column(Integer, nullable=True)  # 0-100
    __tablename__ = "edges"
    id = Column(Integer, primary_key=True, index=True)
    source_entity_id = Column(Integer, ForeignKey("entities.id", ondelete="CASCADE"), nullable=False)
    target_entity_id = Column(Integer, ForeignKey("entities.id", ondelete="CASCADE"), nullable=False)
    relation = Column(String)
    agent_id = Column(Integer, ForeignKey("agents.id"))
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    source_text = Column(Text)
    type = Column(String)  # ✅ New for semantic edge type
    tags = Column(Text)  # ✅ stored as JSON string
    # In Entity model
    sentiment_value = Column(Float, nullable=True)  # -1.0 to +1.0
    # NEW: timeline-specific fields
    event_time = Column(DateTime, nullable=True)        # normalized time if text says “later that day at 5pm”, etc.
    narrative_order = Column(Integer, nullable=True)    # model-assigned order: 1,2,3… (textual narrative sequence)


class AgentTextEntry(Base):
    __tablename__ = "agent_text_entries"
    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id"))
    text = Column(Text)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    
    
class EntitySentimentHistory(Base):
    __tablename__ = "entity_sentiment_history"
    id = Column(Integer, primary_key=True, index=True)
    entity_id = Column(Integer, ForeignKey("entities.id"))
    sentiment_value = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    source_text = Column(Text, nullable=True)

class EdgeSentimentHistory(Base):
    __tablename__ = "edge_sentiment_history"
    id = Column(Integer, primary_key=True, index=True)
    edge_id = Column(Integer, ForeignKey("edges.id"))
    sentiment_value = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    source_text = Column(Text, nullable=True)