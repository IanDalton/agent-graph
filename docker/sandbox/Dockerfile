# The agent's Python code-execution sandbox image (see backend/sandbox/runner.py).
# Base + fpdf2 so run_python can generate PDF artifacts (the "create a PDF" skill in
# backend/skills/sandbox_capability.py). Keep this image LEAN — every added package widens
# what model-authored code can do. Build: docker compose build sandbox
FROM python:3.12-slim

RUN pip install --no-cache-dir "fpdf2>=2.8,<3"
