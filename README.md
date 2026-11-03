# CS2 Round-Start Movement Predictor

Predict Counter-Strike 2 player movement routes at round start using machine learning.

## Overview

This system ingests CS2 demo files and Steam API data to predict where players will move in the opening seconds of each round. It combines economy state, map context, round history, and historical position data to classify likely routes (A/B/Mid) and predict x,y coordinates.

## Architecture

```
Steam API ──► Demo Downloader ──► Demo Parser ──► Feature Store ──► ML Models ──► Prediction API
```

### Pipeline

1. **Data Ingestion** — Pull match history from Steam Web API, download .dem replay files
2. **Demo Parsing** — Extract tick-level player positions, economy, events using demoparser2
3. **Feature Engineering** — Build features: economy (loss bonus, equipment value), side, map, round number, score differential, buy type
4. **Model Training**
   - **Route Classifier** (XGBoost) — Predict A-site / B-site / Mid route per player
   - **Position Predictor** (MLP) — Predict x,y coordinates at T+5s, T+10s, T+15s
5. **Serving** — FastAPI endpoint returns predictions given current round state

## Project Structure

```
├── data/
│   ├── raw/demos/          # Raw .dem files
│   └── processed/          # Parquet feature tables
├── src/
│   ├── data/
│   │   ├── demo_parser.py  # Parse .dem files via demoparser2
│   │   └── steam_api.py    # Steam Web API client
│   ├── features/
│   │   └── engineer.py     # Feature extraction pipeline
│   ├── models/
│   │   ├── route_classifier.py   # XGBoost A/B/Mid classifier
│   │   └── position_predictor.py # MLP for x,y prediction
│   ├── visualization/
│   │   └── heatmap.py      # Position heatmaps on radar images
│   └── api/
│       └── server.py       # FastAPI prediction endpoint
├── configs/
│   ├── model_config.yaml   # Hyperparameters, thresholds
│   └── steam_config.yaml   # API endpoints, rate limits
├── docs/
│   ├── erd.html            # Entity Relationship Diagram
│   └── architecture.html   # System Architecture Diagram
├── requirements.txt
└── .env.example
```

## Quick Start

```bash
pip install -r requirements.txt
cp .env.example .env  # Add your Steam API key
python -m src.data.steam_api --fetch-recent 50
python -m src.data.demo_parser --input data/raw/demos/ --output data/processed/
python -m src.features.engineer
python -m src.models.route_classifier --train
python -m src.api.server
```

## Requirements

- Python 3.10+
- Steam Web API key
- CS2 demo files (.dem format)
