✈️ TripMate AI — A Multi-Agent Travel Planner with LangGraph
An open-source AI travel planner that turns a natural-language trip request into a practical travel plan with flight suggestions, hotel ideas, and a day-by-day itinerary. The project uses a multi-agent workflow built with LangGraph, LangChain, and FastAPI.

Why this project?
Planning a trip usually means jumping between multiple websites, tools, and spreadsheets. This project brings that flow into one experience by combining:

a flight-search agent,
a hotel-research agent,
an itinerary-planning agent, and
a final response agent,
all coordinated through a LangGraph workflow.

Features
✈️ Flight research using AviationStack
🏨 Hotel suggestions using Tavily search
🧠 Multi-agent orchestration with LangGraph
📝 Structured travel itinerary generation
🌐 FastAPI backend with a simple web interface
💾 Conversation state persistence using PostgreSQL
⚡ LLM-powered responses with Groq
Tech Stack
Python 3.10+
FastAPI
Jinja2 + HTML/CSS/JavaScript frontend
LangGraph
LangChain
Groq LLMs
PostgreSQL
Tavily API
AviationStack API
Project Structure
.
├── app.py                # FastAPI app entry point
├── backend.py            # LangGraph travel workflow
├── requirements.txt      # Python dependencies
├── static/               # Static frontend assets
├── templates/            # HTML templates
└── tools/                # Flight and web search integrations
Prerequisites
Before running the project locally, make sure you have:

Python 3.10 or newer installed
PostgreSQL running and accessible
API keys for:
Groq
Tavily
AviationStack
Environment Variables
Create a .env file in the project root with the following variables:

DATABASE_URL=postgresql://user:password@localhost:5432/travel_db
GROQ_API_KEY=your_groq_api_key
AVIATIONSTACK_API_KEY=your_aviationstack_api_key
TAVILY_API_KEY=your_tavily_api_key
DEFAULT_ORIGIN_IATA=DAC
Installation
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
Running the App
Start the FastAPI server:

python app.py
Then open your browser at:

http://127.0.0.1:8000/
API Endpoints
GET /health - Health check
POST /api/travel - Submit a travel request
Example request:

curl -X POST http://127.0.0.1:8000/api/travel \
  -H "Content-Type: application/json" \
  -d '{"message":"Plan a 3-day trip to Tokyo with a budget of $1200"}'
How the Workflow Works
The user submits a travel request.
The flight agent gathers flight-related information.
The hotel agent searches for accommodation suggestions.
The itinerary agent creates a practical travel plan.
The final agent formats the result into a polished response.
Contributing
Contributions are welcome. If you want to improve the app, add new travel features, or fix issues:

Fork the repository
Create a feature branch
Make your changes
Open a pull request
Acknowledgments
This project is built with the help of modern LLM tooling and travel APIs, and it is intended as a practical example of combining LangGraph agents with real-world applications.