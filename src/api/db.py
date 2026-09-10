"""SQLite 持久化：会话、消息、检索日志、成本日志。"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from .. import config

ENGINE = create_engine(f"sqlite:///{config.DB_PATH}", future=True, echo=False)
SessionLocal = sessionmaker(bind=ENGINE, autoflush=False, autocommit=False, expire_on_commit=False)
Base = declarative_base()


class SessionRow(Base):
    __tablename__ = "sessions"

    id = Column(String, primary_key=True)
    mode = Column(String, index=True)
    kb_name = Column(String, default="")
    created_at = Column(DateTime, default=datetime.now)
    finished = Column(Boolean, default=False)
    artifact = Column(String, default="")


class MessageRow(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, ForeignKey("sessions.id"), index=True)
    role = Column(String)
    content = Column(Text)
    created_at = Column(DateTime, default=datetime.now)


class RetrievalLog(Base):
    __tablename__ = "retrieval_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True)
    query = Column(String)
    hit_count = Column(Integer, default=0)
    latency_ms = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.now)


class CostLog(Base):
    __tablename__ = "cost_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    session_id = Column(String, index=True)
    prompt_tokens = Column(Integer, default=0)
    completion_tokens = Column(Integer, default=0)
    latency_ms = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.now)


def init_db() -> None:
    Base.metadata.create_all(ENGINE)


@contextmanager
def get_db() -> Iterator[Session]:
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def create_session(session_id: str, mode: str, kb_name: str = "") -> None:
    with get_db() as db:
        db.add(SessionRow(id=session_id, mode=mode, kb_name=kb_name))


def add_message(session_id: str, role: str, content: str) -> None:
    with get_db() as db:
        db.add(MessageRow(session_id=session_id, role=role, content=content))


def list_messages(session_id: str) -> list[MessageRow]:
    with get_db() as db:
        return (
            db.query(MessageRow)
            .filter(MessageRow.session_id == session_id)
            .order_by(MessageRow.id)
            .all()
        )


def mark_finished(session_id: str, artifact: str) -> None:
    with get_db() as db:
        row = db.query(SessionRow).filter(SessionRow.id == session_id).first()
        if row:
            row.finished = True
            row.artifact = artifact


def list_sessions(limit: int = 50) -> list[SessionRow]:
    with get_db() as db:
        return (
            db.query(SessionRow)
            .order_by(SessionRow.created_at.desc())
            .limit(limit)
            .all()
        )


def get_session(session_id: str) -> SessionRow | None:
    with get_db() as db:
        return db.query(SessionRow).filter(SessionRow.id == session_id).first()


def log_retrieval(session_id: str, query: str, hit_count: int, latency_ms: int) -> None:
    with get_db() as db:
        db.add(
            RetrievalLog(
                session_id=session_id,
                query=query[:300],
                hit_count=hit_count,
                latency_ms=latency_ms,
            )
        )


def log_cost(session_id: str, prompt_tokens: int, completion_tokens: int, latency_ms: int) -> None:
    with get_db() as db:
        db.add(
            CostLog(
                session_id=session_id,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
            )
        )
