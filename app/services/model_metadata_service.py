import json
import logging
import requests
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from app.models import ModelMetadata, ModelMetadataResponse, ModelMetadataListResponse
from app.database import get_db

logger = logging.getLogger(__name__)


class ModelMetadataService:
    def __init__(self, db: Session):
        self.db = db
        self.openrouter_api_url = "https://openrouter.ai/api/v1/models"

    def fetch_models_from_openrouter(self) -> Optional[List[Dict[str, Any]]]:
        """Fetch model data from OpenRouter API"""
        try:
            response = requests.get(self.openrouter_api_url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            if "data" in data:
                return data["data"]
            else:
                logger.error("No 'data' field in OpenRouter API response")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching models from OpenRouter API: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing JSON response from OpenRouter API: {e}")
            return None

    def _parse_model_data(self, model_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse raw model data from OpenRouter API into database format"""
        try:
            # Extract architecture info
            architecture = model_data.get("architecture", {})
            
            # Extract pricing info
            pricing = model_data.get("pricing", {})
            
            # Extract top provider info
            top_provider = model_data.get("top_provider", {})
            
            parsed_data = {
                "id": model_data.get("id"),
                "canonical_slug": model_data.get("canonical_slug"),
                "hugging_face_id": model_data.get("hugging_face_id"),
                "name": model_data.get("name"),
                "description": model_data.get("description"),
                "context_length": model_data.get("context_length"),
                
                # Architecture
                "modality": architecture.get("modality"),
                "input_modalities": json.dumps(architecture.get("input_modalities", [])),
                "output_modalities": json.dumps(architecture.get("output_modalities", [])),
                "tokenizer": architecture.get("tokenizer"),
                "instruct_type": architecture.get("instruct_type"),
                
                # Pricing
                "prompt_price": self._safe_float(pricing.get("prompt")),
                "completion_price": self._safe_float(pricing.get("completion")),
                "request_price": self._safe_float(pricing.get("request")),
                "image_price": self._safe_float(pricing.get("image")),
                "web_search_price": self._safe_float(pricing.get("web_search")),
                "internal_reasoning_price": self._safe_float(pricing.get("internal_reasoning")),
                "input_cache_read_price": self._safe_float(pricing.get("input_cache_read")),
                
                # Provider info
                "provider_context_length": top_provider.get("context_length"),
                "max_completion_tokens": top_provider.get("max_completion_tokens"),
                "is_moderated": top_provider.get("is_moderated", False),
                
                # Additional metadata
                "supported_parameters": json.dumps(model_data.get("supported_parameters", [])),
                "per_request_limits": json.dumps(model_data.get("per_request_limits")) if model_data.get("per_request_limits") else None,
                
                "is_active": True
            }
            
            return parsed_data
            
        except Exception as e:
            logger.error(f"Error parsing model data for {model_data.get('id', 'unknown')}: {e}")
            return None

    def _safe_float(self, value: Any) -> Optional[float]:
        """Safely convert value to float, return None if conversion fails"""
        if value is None or value == "":
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None

    def sync_models_with_openrouter(self) -> Dict[str, Any]:
        """Fetch models from OpenRouter and sync with database"""
        try:
            # Fetch models from OpenRouter API
            openrouter_models = self.fetch_models_from_openrouter()
            if not openrouter_models:
                return {
                    "success": False,
                    "message": "Failed to fetch models from OpenRouter API",
                    "synced_count": 0,
                    "updated_count": 0,
                    "errors": []
                }

            synced_count = 0
            updated_count = 0
            errors = []

            # Process each model
            for model_data in openrouter_models:
                try:
                    parsed_data = self._parse_model_data(model_data)
                    if not parsed_data or not parsed_data.get("id"):
                        errors.append(f"Failed to parse model data for {model_data.get('id', 'unknown')}")
                        continue

                    # Check if model already exists
                    existing_model = self.db.query(ModelMetadata).filter(
                        ModelMetadata.id == parsed_data["id"]
                    ).first()

                    if existing_model:
                        # Update existing model
                        for key, value in parsed_data.items():
                            if key != "id":  # Don't update the primary key
                                setattr(existing_model, key, value)
                        updated_count += 1
                    else:
                        # Create new model
                        new_model = ModelMetadata(**parsed_data)
                        self.db.add(new_model)
                        synced_count += 1

                except Exception as e:
                    error_msg = f"Error processing model {model_data.get('id', 'unknown')}: {e}"
                    logger.error(error_msg)
                    errors.append(error_msg)

            # Mark models not in OpenRouter response as inactive
            if openrouter_models:
                openrouter_model_ids = [model.get("id") for model in openrouter_models if model.get("id")]
                inactive_models = self.db.query(ModelMetadata).filter(
                    ~ModelMetadata.id.in_(openrouter_model_ids),
                    ModelMetadata.is_active == True
                ).all()
                
                for model in inactive_models:
                    model.is_active = False

            # Commit changes
            self.db.commit()

            return {
                "success": True,
                "message": f"Successfully synced {synced_count} new models and updated {updated_count} existing models",
                "synced_count": synced_count,
                "updated_count": updated_count,
                "total_models": len(openrouter_models) if openrouter_models else 0,
                "errors": errors
            }

        except Exception as e:
            self.db.rollback()
            error_msg = f"Error syncing models with OpenRouter: {e}"
            logger.error(error_msg)
            return {
                "success": False,
                "message": error_msg,
                "synced_count": 0,
                "updated_count": 0,
                "errors": [error_msg]
            }

    def get_all_models(self, active_only: bool = True) -> List[ModelMetadataResponse]:
        """Get all models from database"""
        try:
            query = self.db.query(ModelMetadata)
            if active_only:
                query = query.filter(ModelMetadata.is_active == True)
            
            models = query.order_by(ModelMetadata.name).all()
            
            result = []
            for model in models:
                # Parse JSON fields
                input_modalities = []
                output_modalities = []
                supported_parameters = []
                per_request_limits = None
                
                try:
                    if model.input_modalities:
                        input_modalities = json.loads(model.input_modalities)
                except json.JSONDecodeError:
                    pass
                    
                try:
                    if model.output_modalities:
                        output_modalities = json.loads(model.output_modalities)
                except json.JSONDecodeError:
                    pass
                    
                try:
                    if model.supported_parameters:
                        supported_parameters = json.loads(model.supported_parameters)
                except json.JSONDecodeError:
                    pass
                    
                try:
                    if model.per_request_limits:
                        per_request_limits = json.loads(model.per_request_limits)
                except json.JSONDecodeError:
                    pass

                model_response = ModelMetadataResponse(
                    id=model.id,
                    canonical_slug=model.canonical_slug,
                    hugging_face_id=model.hugging_face_id,
                    name=model.name,
                    description=model.description,
                    context_length=model.context_length,
                    modality=model.modality,
                    input_modalities=input_modalities,
                    output_modalities=output_modalities,
                    tokenizer=model.tokenizer,
                    instruct_type=model.instruct_type,
                    prompt_price=model.prompt_price,
                    completion_price=model.completion_price,
                    request_price=model.request_price,
                    image_price=model.image_price,
                    web_search_price=model.web_search_price,
                    internal_reasoning_price=model.internal_reasoning_price,
                    input_cache_read_price=model.input_cache_read_price,
                    provider_context_length=model.provider_context_length,
                    max_completion_tokens=model.max_completion_tokens,
                    is_moderated=model.is_moderated,
                    supported_parameters=supported_parameters,
                    per_request_limits=per_request_limits,
                    is_active=model.is_active,
                    last_updated=model.last_updated.isoformat() if model.last_updated else "",
                    created_at=model.created_at.isoformat() if model.created_at else ""
                )
                result.append(model_response)
                
            return result

        except Exception as e:
            logger.error(f"Error getting models from database: {e}")
            return []

    def get_model_by_id(self, model_id: str) -> Optional[ModelMetadataResponse]:
        """Get specific model by ID"""
        try:
            model = self.db.query(ModelMetadata).filter(
                ModelMetadata.id == model_id
            ).first()
            
            if not model:
                return None
                
            # Parse JSON fields
            input_modalities = []
            output_modalities = []
            supported_parameters = []
            per_request_limits = None
            
            try:
                if model.input_modalities:
                    input_modalities = json.loads(model.input_modalities)
            except json.JSONDecodeError:
                pass
                
            try:
                if model.output_modalities:
                    output_modalities = json.loads(model.output_modalities)
            except json.JSONDecodeError:
                pass
                
            try:
                if model.supported_parameters:
                    supported_parameters = json.loads(model.supported_parameters)
            except json.JSONDecodeError:
                pass
                
            try:
                if model.per_request_limits:
                    per_request_limits = json.loads(model.per_request_limits)
            except json.JSONDecodeError:
                pass

            return ModelMetadataResponse(
                id=model.id,
                canonical_slug=model.canonical_slug,
                hugging_face_id=model.hugging_face_id,
                name=model.name,
                description=model.description,
                context_length=model.context_length,
                modality=model.modality,
                input_modalities=input_modalities,
                output_modalities=output_modalities,
                tokenizer=model.tokenizer,
                instruct_type=model.instruct_type,
                prompt_price=model.prompt_price,
                completion_price=model.completion_price,
                request_price=model.request_price,
                image_price=model.image_price,
                web_search_price=model.web_search_price,
                internal_reasoning_price=model.internal_reasoning_price,
                input_cache_read_price=model.input_cache_read_price,
                provider_context_length=model.provider_context_length,
                max_completion_tokens=model.max_completion_tokens,
                is_moderated=model.is_moderated,
                supported_parameters=supported_parameters,
                per_request_limits=per_request_limits,
                is_active=model.is_active,
                last_updated=model.last_updated.isoformat() if model.last_updated else "",
                created_at=model.created_at.isoformat() if model.created_at else ""
            )

        except Exception as e:
            logger.error(f"Error getting model {model_id} from database: {e}")
            return None

    def search_models(self, query: str, active_only: bool = True) -> List[ModelMetadataResponse]:
        """Search models by name or description"""
        try:
            db_query = self.db.query(ModelMetadata)
            if active_only:
                db_query = db_query.filter(ModelMetadata.is_active == True)
            
            # Search in name and description
            search_filter = (
                ModelMetadata.name.ilike(f"%{query}%") |
                ModelMetadata.description.ilike(f"%{query}%") |
                ModelMetadata.id.ilike(f"%{query}%")
            )
            
            models = db_query.filter(search_filter).order_by(ModelMetadata.name).all()
            
            result = []
            for model in models:
                # Parse JSON fields similar to get_all_models
                input_modalities = []
                output_modalities = []
                supported_parameters = []
                per_request_limits = None
                
                try:
                    if model.input_modalities:
                        input_modalities = json.loads(model.input_modalities)
                    if model.output_modalities:
                        output_modalities = json.loads(model.output_modalities)
                    if model.supported_parameters:
                        supported_parameters = json.loads(model.supported_parameters)
                    if model.per_request_limits:
                        per_request_limits = json.loads(model.per_request_limits)
                except json.JSONDecodeError:
                    pass

                model_response = ModelMetadataResponse(
                    id=model.id,
                    canonical_slug=model.canonical_slug,
                    hugging_face_id=model.hugging_face_id,
                    name=model.name,
                    description=model.description,
                    context_length=model.context_length,
                    modality=model.modality,
                    input_modalities=input_modalities,
                    output_modalities=output_modalities,
                    tokenizer=model.tokenizer,
                    instruct_type=model.instruct_type,
                    prompt_price=model.prompt_price,
                    completion_price=model.completion_price,
                    request_price=model.request_price,
                    image_price=model.image_price,
                    web_search_price=model.web_search_price,
                    internal_reasoning_price=model.internal_reasoning_price,
                    input_cache_read_price=model.input_cache_read_price,
                    provider_context_length=model.provider_context_length,
                    max_completion_tokens=model.max_completion_tokens,
                    is_moderated=model.is_moderated,
                    supported_parameters=supported_parameters,
                    per_request_limits=per_request_limits,
                    is_active=model.is_active,
                    last_updated=model.last_updated.isoformat() if model.last_updated else "",
                    created_at=model.created_at.isoformat() if model.created_at else ""
                )
                result.append(model_response)
                
            return result

        except Exception as e:
            logger.error(f"Error searching models: {e}")
            return []