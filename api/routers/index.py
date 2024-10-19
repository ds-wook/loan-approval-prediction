from fastapi import APIRouter

from api.routers.loan import loan_router

index_router = router = APIRouter()


router.include_router(loan_router, prefix="/loan")
