from fastapi import APIRouter

from api.dto.loan import LoanApprovalDto
from api.service.loan import run_model

loan_router = router = APIRouter()


@router.post("/predict")
async def get_predict_loan(body: LoanApprovalDto):
    response = await run_model(body)

    return response
