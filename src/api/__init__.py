"""
API Module
Financial Document Intelligence Platform

FastAPI backend for document upload and querying.
"""

from .app import app, create_app

__all__ = ['app', 'create_app']
