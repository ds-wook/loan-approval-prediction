import time
from pathlib import Path

import httpx
import pandas as pd
import yaml
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from api.routers.index import index_router

# FastAPI 객체 생성
app = FastAPI(docs_url="/docs", openapi_url="/open-api-docs")
# /api라는 경로로 index_router를 붙인다.
app.include_router(index_router, prefix="/app")
templates = Jinja2Templates(directory="app/templates")


@app.get("/", response_class=HTMLResponse)
async def get_hello(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


# 실시간 알림 대시보드
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})


# WebSocket을 통한 실시간 알림 전송
@app.websocket("/ws/notifications")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    headers = {"Content-Type": "application/json", "accept": "application/json"}
    url = "http://127.0.0.1:8000/app/loan/predict"
    # test 데이터 불러오기 (parquet 파일 기준, 필요시 CSV로 변경)
    try:
        test = pd.read_csv(Path("input/loan-approval-prediction/") / "test_features.csv")

    except Exception as e:
        await websocket.send_json({"message": f"Failed to load test data: {str(e)}"})
        await websocket.close()
        return

    # 모델에서 사용하는 피처 리스트 (Hydra의 cfg.store.features 대체)
    selected_features = yaml.safe_load(open(Path("config/store/") / "features.yaml"))["selected_features"]
    await websocket.send_json({"message": "정상 작동 중입니다."})

    async with httpx.AsyncClient() as client:
        for _, data in test.iterrows():
            data = data[selected_features].to_dict()
            # 비동기 POST 요청
            try:
                response = await client.post(url, json=data, headers=headers)
                response_data = response.json()

                await websocket.send_json({"message": f"Prediction: {response_data['prediction'][0]}"})

            except httpx.HTTPStatusError as e:
                await websocket.send_json({"message": f"Request failed: {e.response.status_code}"})

            except Exception as e:
                await websocket.send_json({"message": f"Error: {str(e)}"})

            time.sleep(5)

    await websocket.close()
