# Project Aether Implementation Notes

## Current State

Project Aether is implemented as a FastAPI backend plus a Next.js frontend. The active root route, `frontend/app/page.tsx`, is the dashboard. Supporting routes live at `/visualizer`, `/shock-test`, and `/logs`.

## Data Policy

Frontend insights should be computed from the active backend data source. The backend may use deterministic CMS-like data for local development, or real CMS CSV input when configured, but UI components should not contain fixed metric values.

## Active Contracts

- `GET /api/v1/telemetry/dashboard`
- `GET /api/v1/telemetry/trajectory`
- `GET /api/v1/telemetry/logs`
- `GET /api/v1/sample-data`
- `POST /api/v1/predict`
- `POST /api/v1/shock-test`

## Verification

Run backend tests and frontend checks before considering changes complete:

```bash
cd backend
venv\Scripts\python.exe -m pytest

cd ..\frontend
npx.cmd tsc --noEmit
npm.cmd run build
```
