from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.query import QueryRequest, QueryResponse
from app.services.rag import rag_service
from app.core.logging import logger

router = APIRouter(tags=["Query"])


@router.post("/query", response_model=QueryResponse)
def query_hld(
    request: QueryRequest,
    db: Session = Depends(get_db)
):
    """
    Query the indexed AUTOSAR HLD using local grounded RAG.
    Strictly grounds answers in retrieved chunks, provides source citations,
    and returns explainable confidence indicators.
    """
    if not request.question.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Question must not be empty."
        )

    try:
        response = rag_service.answer_question(
            document_id=request.document_id,
            question=request.question.strip(),
            db=db
        )
        return response
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(ve)
        )
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Query processing encountered an internal error."
        )
