#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 22:47:32 2026

@author: dr
"""

# Configure environment
cp configs/.env.example.env
# Edit .env with your NIA API credentials, database URL, etc.

# Initialise database
python backend/models.py

# Start Flask server
python backend/app.py
# Server runs on http://localhost:5000