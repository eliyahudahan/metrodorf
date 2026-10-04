# Metrodorf v2 — Train Delay Prediction (Infrabel, Belgium)

> **Portfolio project** — not deployed in production.

## Problem

Predict whether a train will arrive **more than 5 minutes late** (>300 seconds)
at a given station, given planned schedule and historical delay patterns.

**Target audience:** One rail analyst at a European rail operator.

## Data

- **Source:** Infrabel Open Data (CC0)
- **Dataset:** Monthly raw punctuality data files
- **Period:** January 2025 (initial exploration)
- **GT:** `DELAY_ARR` (already computed by Infrabel, in seconds)

## Status

🚧 In progress — Day 1: data feasibility.

## What This Project Is

- A demonstration of methodology.
- A portfolio piece for a rail analyst.

## What This Project Is Not

- Not a production system.
- Not a real-time dispatcher tool.

## Structure
metrodorf/
├── data/
│ ├── raw/ # Infrabel CSV (gitignored)
│ └── processed/ # Parquet (gitignored)
├── models/ # Training scripts
├── features/ # Feature engineering
├── evaluation/ # Metrics
└── tests/

text
