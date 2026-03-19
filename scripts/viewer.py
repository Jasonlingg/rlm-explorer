"""Local web viewer for eval trajectories.

Serves a visual UI on localhost to browse agent runs step-by-step.

Usage:
    python scripts/viewer.py              # default port 8000
    python scripts/viewer.py --port 3000
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

OUT_DIR = Path(__file__).parent.parent / "out"
TEMPLATE_DIR = Path(__file__).parent / "templates"

app = FastAPI(title="RLM Explorer — Trajectory Viewer")
cli = typer.Typer()


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = (TEMPLATE_DIR / "viewer.html").read_text()
    return HTMLResponse(content=html)


@app.get("/api/runs")
def list_runs() -> JSONResponse:
    """List all run files sorted newest first."""
    runs = sorted(OUT_DIR.glob("run_*.json"), reverse=True)
    return JSONResponse([
        {"name": r.stem, "path": r.name, "size_kb": round(r.stat().st_size / 1024, 1)}
        for r in runs
    ])


@app.get("/api/runs/{filename}")
def get_run(filename: str) -> JSONResponse:
    """Load a specific run transcript."""
    path = OUT_DIR / filename
    if not path.exists() or not path.suffix == ".json":
        return JSONResponse({"error": "not found"}, status_code=404)
    with open(path) as f:
        data = json.load(f)
    return JSONResponse(data)


@cli.command()
def main(port: int = typer.Option(8000, help="Port to serve on")) -> None:
    """Launch the trajectory viewer on localhost."""
    typer.echo(f"🔍 RLM Explorer Viewer → http://localhost:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    cli()
