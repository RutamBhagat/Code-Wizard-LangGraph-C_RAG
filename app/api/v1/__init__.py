from fastapi import APIRouter
from app.api.v1.code_wizard import router as code_wizard_router

router = APIRouter(prefix="/v1")
router.include_router(code_wizard_router)
