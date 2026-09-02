from pathlib import Path
import traceback
import uvicorn

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from fastapi.concurrency import run_in_threadpool

from backend import run_travel_agent, resume_travel_agent

BASE_DIR = Path(__file__).resolve().parent



app = FastAPI(
    title="TripMate AI",
    description="LangGraph Multi-Agent Travel Planner with FastAPI Frontend",
    version="1.0.0"
)



app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static"
)


templates = Jinja2Templates(
    directory=str(BASE_DIR / "templates")
)



class TravelRequest(BaseModel):
    message: str
    thread_id: str | None = None
    
    
class ApprovalRequest(BaseModel):
    thread_id: str
    approved: bool
    feedback: str = ""

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )
    
@app.post("/api/travel")
async def travel_planner(request_data: TravelRequest):
    try:
        user_message = request_data.message.strip()

        if not user_message:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Message cannot be empty."
                }
            )

        result = await run_in_threadpool(
                run_travel_agent,
                   user_message,
               request_data.thread_id
                )

        return JSONResponse(
    content={
        "success": True,
        "thread_id": result["thread_id"],
        "answer": result["answer"],

        "requires_approval": result["requires_approval"],
        "approval_request": result["approval_request"],

        "flight_results": result["flight_results"],
        "hotel_results": result["hotel_results"],
        "weather_results": result["weather_results"],
        "budget_results": result["budget_results"],
        "itinerary": result["itinerary"],

        "approved": result["approved"],
        "human_feedback": result["human_feedback"],
        "final_response": result["final_response"],

        "llm_calls": result["llm_calls"],
    }
)

    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )




@app.post("/api/travel/approve")
async def approve_travel(request_data: ApprovalRequest):
    try:
        result = await run_in_threadpool(
            resume_travel_agent,
            request_data.thread_id,
            request_data.approved,
            request_data.feedback
        )

        return JSONResponse(
            content={
                "success": True,
                "thread_id": result["thread_id"],
                "answer": result["answer"],
                "requires_approval": result["requires_approval"],
                "approval_request": result["approval_request"],
                "flight_results": result["flight_results"],
                "hotel_results": result["hotel_results"],
                "weather_results": result["weather_results"],
                "budget_results": result["budget_results"],
                "itinerary": result["itinerary"],
                "approved": result["approved"],
                "human_feedback": result["human_feedback"],
                "final_response": result["final_response"],
                "llm_calls": result["llm_calls"],
            }
        )

    except Exception as e:
        print("APPROVAL ERROR:", e)
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "message": "AI Travel Planner API is running"
    }


@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={})



if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )