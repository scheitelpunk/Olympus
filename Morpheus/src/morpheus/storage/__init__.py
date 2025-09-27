"""Storage layer for MORPHEUS system.

Provides persistent storage capabilities using PostgreSQL with optimized
schemas for experiences, materials, dream sessions, and learned strategies.
"""

from .postgres_storage import PostgreSQLStorage

__all__ = [
    'PostgreSQLStorage'
]