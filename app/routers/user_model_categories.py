from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Dict, Any
from app.database import get_db
from app.dependencies import get_current_user
from app.models import (
    User, 
    UserModelCategoryCreate, 
    UserModelCategoryUpdate, 
    UserModelCategoryResponse,
    UserModelCategoriesListResponse
)
from app.services.user_model_category_service import UserModelCategoryService
from app.services.pricing_calculator_service import PricingCalculatorService

router = APIRouter(
    prefix="/user-model-categories",
    tags=["user-model-categories"],
    dependencies=[Depends(get_current_user)],
    responses={404: {"description": "Not found"}},
)


@router.get("/", response_model=UserModelCategoriesListResponse)
async def get_user_model_categories(
    include_inactive: bool = False,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get all model categories for the current user
    
    Returns the user's customized model categories with their assigned models.
    """
    try:
        service = UserModelCategoryService(db)
        categories = service.get_user_categories(
            user_id=current_user.id, 
            include_inactive=include_inactive
        )
        
        return UserModelCategoriesListResponse(
            categories=categories,
            total_count=len(categories)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching user model categories: {str(e)}"
        )


@router.get("/{category_name}", response_model=UserModelCategoryResponse)
async def get_user_model_category_by_name(
    category_name: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get a specific model category by name for the current user
    """
    try:
        service = UserModelCategoryService(db)
        category = service.get_category_by_name(current_user.id, category_name)
        
        if not category:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Category '{category_name}' not found"
            )
            
        return category
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching category: {str(e)}"
        )


@router.post("/", response_model=UserModelCategoryResponse)
async def create_user_model_category(
    category_data: UserModelCategoryCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create a new model category for the current user
    
    Allows users to create custom categories with their preferred models.
    """
    try:
        service = UserModelCategoryService(db)
        category = service.create_category(current_user.id, category_data)
        
        if not category:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to create category. Category name may already exist or model may not be available."
            )
            
        return category
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating category: {str(e)}"
        )


@router.put("/{category_id}", response_model=UserModelCategoryResponse)
async def update_user_model_category(
    category_id: int,
    category_data: UserModelCategoryUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Update an existing model category for the current user
    
    Allows users to modify their category settings including the assigned model.
    """
    try:
        service = UserModelCategoryService(db)
        category = service.update_category(current_user.id, category_id, category_data)
        
        if not category:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Category not found or update failed"
            )
            
        return category
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error updating category: {str(e)}"
        )


@router.delete("/{category_id}")
async def delete_user_model_category(
    category_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Delete a model category for the current user
    
    Performs a soft delete by setting is_active=False.
    """
    try:
        service = UserModelCategoryService(db)
        success = service.delete_category(current_user.id, category_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Category not found"
            )
            
        return {"message": "Category deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error deleting category: {str(e)}"
        )


@router.post("/reorder")
async def reorder_user_model_categories(
    category_orders: List[Dict[str, int]],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Reorder user model categories by updating sort_order
    
    Expects a list of objects with category_id and sort_order fields.
    Example: [{"category_id": 1, "sort_order": 1}, {"category_id": 2, "sort_order": 2}]
    """
    try:
        service = UserModelCategoryService(db)
        success = service.reorder_categories(current_user.id, category_orders)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to reorder categories"
            )
            
        return {"message": "Categories reordered successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reordering categories: {str(e)}"
        )


@router.post("/create-defaults")
async def create_default_categories(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Create default model categories for the current user
    
    Creates the standard Fast, Standard, Fast Reasoning, and Reasoning categories
    if they don't already exist.
    """
    try:
        service = UserModelCategoryService(db)
        result = service.create_default_categories_for_user(current_user.id)
        
        if not result["success"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["message"]
            )
            
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating default categories: {str(e)}"
        )


@router.get("/available-models/list", response_model=List[Dict[str, Any]])
async def get_available_models_for_categories(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get list of available models that can be used in categories
    
    Returns all active models from the model metadata with their basic information.
    """
    try:
        service = UserModelCategoryService(db)
        models = service.get_available_models_for_categories()
        
        return models
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching available models: {str(e)}"
        )


@router.post("/estimate-cost")
async def estimate_message_cost(
    request_data: Dict[str, Any],
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Estimate cost for a message before sending
    
    Expects:
    - category_name: User's category name (e.g., "FAST", "STANDARD")
    - message_length: Length of the message in characters
    - expected_response_length: Expected AI response length (optional, default 500)
    - include_images: Number of images (optional, default 0)
    """
    try:
        category_name = request_data.get("category_name")
        message_length = request_data.get("message_length", 0)
        expected_response_length = request_data.get("expected_response_length", 500)
        include_images = request_data.get("include_images", 0)
        
        if not category_name:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="category_name is required"
            )
        
        # Get user's model for this category
        service = UserModelCategoryService(db)
        model_id = service.get_model_id_for_category(current_user.id, category_name)
        
        if not model_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Category '{category_name}' not found for user"
            )
        
        # Calculate cost estimation
        cost_estimate = PricingCalculatorService.estimate_cost_for_message(
            model_id=model_id,
            message_length=message_length,
            expected_response_length=expected_response_length,
            include_images=include_images
        )
        
        return {
            "category_name": category_name,
            "model_id": model_id,
            "cost_estimate": cost_estimate,
            "input_estimation": {
                "message_length_chars": message_length,
                "estimated_input_tokens": message_length // 3,
                "expected_response_length_chars": expected_response_length,
                "estimated_output_tokens": expected_response_length // 3,
                "images_count": include_images
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error estimating cost: {str(e)}"
        )


@router.get("/{category_name}/pricing-info")
async def get_category_pricing_info(
    category_name: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """
    Get detailed pricing information for a user's category
    
    Returns detailed pricing breakdown for the model assigned to this category.
    """
    try:
        service = UserModelCategoryService(db)
        model_id = service.get_model_id_for_category(current_user.id, category_name)
        
        if not model_id:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Category '{category_name}' not found for user"
            )
        
        pricing_info = PricingCalculatorService.get_model_pricing_info(model_id)
        
        if not pricing_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Pricing information not available for model '{model_id}'"
            )
        
        return {
            "category_name": category_name,
            "pricing_info": pricing_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching pricing info: {str(e)}"
        )