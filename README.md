# MAPLE (Multi-Agent Protein Learning Engine)

MAPLE is a multi-agent system that autonomously explores protein sequence space and optimizes enzyme function through iterative learning.

## Core Scope
- Multi-agent loop: planner -> sequence -> structure -> property -> optimization -> evaluation
- Random mutation-based sequence exploration
- Dummy structure backend interface (future AlphaFold2/ESMFold compatible)
- Embedding + property prediction with uncertainty-aware scoring
- Diversity-aware evolutionary optimization
- Iteration artifact export (JSON/CSV)

## Install
```bash
cd /Users/hyunjoon/codex/MAPLE
python3 -m pip install -r requirements.txt
```

## CLI Run
```bash
cd /Users/hyunjoon/codex/MAPLE
python3 main.py --num-iterations 5 --selection-strategy diverse --min-hamming-distance 2
```

## Web UI Run (Streamlit)
```bash
cd /Users/hyunjoon/codex/MAPLE
streamlit run app.py
```

## UI Features
- Sidebar controls for runtime/model/scoring parameters
- One-click execution of optimization loop
- Live run summary with best sequence and score
- Iteration trend chart and ranked candidate table
- Downloadable artifacts (`history.json`, `summary.json`)

## Key Files
- `main.py`: reusable run service + CLI entrypoint
- `app.py`: Streamlit UI dashboard
- `core/pipeline.py`: orchestration loop
- `core/reporting.py`: artifact export
- `agents/*.py`: independently testable agent logic
- `tests/*.py`: unit/integration tests
