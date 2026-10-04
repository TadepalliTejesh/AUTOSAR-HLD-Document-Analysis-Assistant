import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logging import logger
from app.db.database import init_db
from app.api import documents, query, inventory, audit
from app.services.llm import ollama_service
from app.services.vector_store import vector_store_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info(f"Starting {settings.APP_NAME}...")
    settings.ensure_directories()
    init_db()
    logger.info("Database and storage directories initialized.")
    yield
    # Shutdown
    logger.info(f"Shutting down {settings.APP_NAME}...")


app = FastAPI(
    title=settings.APP_NAME,
    description="Automated architectural knowledge extraction, local grounded RAG, and inventory analysis for AUTOSAR HLD documents.",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for local frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = f"{process_time:.4f}s"
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url.path}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please contact the system administrator."}
    )


# Health check endpoint
@app.get("/health", tags=["System"])
def health_check():
    ollama_info = ollama_service.check_health()
    chroma_count = 0
    try:
        chroma_count = vector_store_service.count_chunks()
        chroma_status = "operational"
    except Exception as e:
        chroma_status = f"error: {str(e)}"

    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "database": "sqlite_connected",
        "chromadb": {
            "status": chroma_status,
            "total_indexed_chunks": chroma_count
        },
        "ollama": ollama_info,
        "embedding_model": settings.EMBEDDING_MODEL
    }


# Include modular routers
app.include_router(documents.router)
app.include_router(query.router)
app.include_router(inventory.router)
app.include_router(audit.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=False
    )
