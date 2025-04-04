@echo off
echo [1] Building React frontend...
cd server\aisum-ui
call npm run build

echo [2] Starting FastAPI server...
cd ..
start uvicorn main:app --reload
