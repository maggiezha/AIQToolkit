from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AIQ Toolkit API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/")
async def root():
    return Response(
        content="Welcome to the AIQ Toolkit API. Use /generate endpoint for queries.",
        media_type="text/plain"
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 