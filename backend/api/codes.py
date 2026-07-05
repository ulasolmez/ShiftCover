"""
Shift code preview endpoint.
"""

from fastapi import APIRouter

from backend.schemas import ShiftCodePreviewRequest, SolverParamsSchema
from solver import list_possible_shift_codes
from backend.api.solve import _params_to_solver

router = APIRouter()


@router.post("/shift-codes")
async def shift_codes_endpoint(req: SolverParamsSchema):
    """Return all possible shift codes given the current parameters."""
    params = _params_to_solver(req)
    codes = list_possible_shift_codes(params)
    return {"codes": codes, "count": len(codes)}