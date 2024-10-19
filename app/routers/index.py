from fastapi import APIRouter

from app.routers.loan import loan_router

index_router = router = APIRouter()


router.include_router(loan_router, prefix="/loan")
