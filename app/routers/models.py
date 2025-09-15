from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from app.database import get_db
from app.dependencies import require_admin
from app.models import User, ModelMetadataResponse, ModelMetadataListResponse
from app.services.model_metadata_service import ModelMetadataService
from datetime import datetime

router = APIRouter(
    prefix="/models",
    tags=["models"],
    responses={404: {"description": "Not found"}},
)


@router.post("/sync", response_model=dict)
async def sync_models_with_openrouter(
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_db)
):
    """
    Sync models with OpenRouter API (Admin only)
    
    Fetches all available models from OpenRouter API and updates the local database.
    This endpoint is restricted to admin users only.
    """
    try:
        service = ModelMetadataService(db)
        result = service.sync_models_with_openrouter()
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result["message"])
            
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error syncing models: {str(e)}")


@router.get("/", response_model=ModelMetadataListResponse)
async def get_all_models(
    active_only: bool = Query(True, description="Show only active models"),
    db: Session = Depends(get_db)
):
    """
    Get all available models
    
    Returns a list of all models available in the system.
    By default, only shows active models.
    """
    try:
        service = ModelMetadataService(db)
        models = service.get_all_models(active_only=active_only)
        
        return ModelMetadataListResponse(
            models=models,
            total_count=len(models),
            updated_at=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching models: {str(e)}")


@router.get("/search", response_model=ModelMetadataListResponse)
async def search_models(
    q: str = Query(..., description="Search query for model name, description, or ID"),
    active_only: bool = Query(True, description="Show only active models"),
    db: Session = Depends(get_db)
):
    """
    Search models by name, description, or ID
    
    Performs a case-insensitive search across model names, descriptions, and IDs.
    """
    try:
        if not q or len(q.strip()) < 2:
            raise HTTPException(status_code=400, detail="Search query must be at least 2 characters long")
            
        service = ModelMetadataService(db)
        models = service.search_models(q.strip(), active_only=active_only)
        
        return ModelMetadataListResponse(
            models=models,
            total_count=len(models),
            updated_at=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching models: {str(e)}")


@router.get("/{model_id}", response_model=ModelMetadataResponse)
async def get_model_by_id(
    model_id: str,
    db: Session = Depends(get_db)
):
    """
    Get specific model by ID
    
    Returns detailed information about a specific model.
    """
    try:
        service = ModelMetadataService(db)
        model = service.get_model_by_id(model_id)
        
        if not model:
            raise HTTPException(status_code=404, detail=f"Model with ID '{model_id}' not found")
            
        return model
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching model: {str(e)}")


@router.get("/pricing/comparison", response_model=List[dict])
async def get_pricing_comparison(
    modality: Optional[str] = Query(None, description="Filter by modality (e.g., 'text->text')"),
    min_context_length: Optional[int] = Query(None, description="Minimum context length"),
    max_context_length: Optional[int] = Query(None, description="Maximum context length"),
    active_only: bool = Query(True, description="Show only active models"),
    db: Session = Depends(get_db)
):
    """
    Get pricing comparison of models
    
    Returns a comparison of model pricing with optional filtering.
    Useful for cost analysis and model selection.
    """
    try:
        service = ModelMetadataService(db)
        models = service.get_all_models(active_only=active_only)
        
        # Apply filters
        filtered_models = []
        for model in models:
            # Filter by modality
            if modality and model.modality != modality:
                continue
                
            # Filter by context length
            if min_context_length and (not model.context_length or model.context_length < min_context_length):
                continue
            if max_context_length and (not model.context_length or model.context_length > max_context_length):
                continue
                
            # Only include models with pricing information
            if model.prompt_price is not None or model.completion_price is not None:
                filtered_models.append(model)
        
        # Create pricing comparison
        comparison = []
        for model in filtered_models:
            comparison.append({
                "id": model.id,
                "name": model.name,
                "modality": model.modality,
                "context_length": model.context_length,
                "prompt_price": model.prompt_price,
                "completion_price": model.completion_price,
                "request_price": model.request_price,
                "image_price": model.image_price,
                "input_cache_read_price": model.input_cache_read_price,
                "provider_context_length": model.provider_context_length,
                "is_moderated": model.is_moderated,
                "tokenizer": model.tokenizer
            })
        
        # Sort by prompt price (cheapest first)
        comparison.sort(key=lambda x: x["prompt_price"] or float('inf'))
        
        return comparison
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating pricing comparison: {str(e)}")


@router.get("/capabilities/summary", response_model=dict)
async def get_capabilities_summary(
    active_only: bool = Query(True, description="Show only active models"),
    db: Session = Depends(get_db)
):
    """
    Get summary of model capabilities
    
    Returns a summary of available capabilities across all models,
    including modalities, tokenizers, and feature availability.
    """
    try:
        service = ModelMetadataService(db)
        models = service.get_all_models(active_only=active_only)
        
        # Analyze capabilities
        modalities = set()
        tokenizers = set()
        input_modalities = set()
        output_modalities = set()
        features = {
            "reasoning_models": 0,
            "image_input_models": 0,
            "image_output_models": 0,
            "tool_calling_models": 0,
            "structured_output_models": 0,
            "long_context_models": 0,  # > 100k tokens
            "free_models": 0
        }
        
        for model in models:
            if model.modality:
                modalities.add(model.modality)
            if model.tokenizer:
                tokenizers.add(model.tokenizer)
                
            # Input modalities
            if model.input_modalities:
                for mod in model.input_modalities:
                    input_modalities.add(mod)
                    
            # Output modalities  
            if model.output_modalities:
                for mod in model.output_modalities:
                    output_modalities.add(mod)
                    
            # Feature analysis
            if "reasoning" in model.name.lower() or "thinking" in model.name.lower():
                features["reasoning_models"] += 1
                
            if model.input_modalities and "image" in model.input_modalities:
                features["image_input_models"] += 1
                
            if model.output_modalities and "image" in model.output_modalities:
                features["image_output_models"] += 1
                
            if model.supported_parameters and ("tool_choice" in model.supported_parameters or "tools" in model.supported_parameters):
                features["tool_calling_models"] += 1
                
            if model.supported_parameters and "structured_outputs" in model.supported_parameters:
                features["structured_output_models"] += 1
                
            if model.context_length and model.context_length > 100000:
                features["long_context_models"] += 1
                
            if model.prompt_price == 0 and model.completion_price == 0:
                features["free_models"] += 1
        
        return {
            "total_models": len(models),
            "modalities": sorted(list(modalities)),
            "tokenizers": sorted(list(tokenizers)),
            "input_modalities": sorted(list(input_modalities)),
            "output_modalities": sorted(list(output_modalities)),
            "features": features,
            "updated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating capabilities summary: {str(e)}")