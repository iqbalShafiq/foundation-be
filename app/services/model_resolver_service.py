import logging
from typing import Optional
from sqlalchemy.orm import Session
from app.models import ModelType
from app.database import get_db
from .user_model_category_service import UserModelCategoryService

logger = logging.getLogger(__name__)


class ModelResolverService:
    """
    Service to resolve ModelType to actual model IDs based on user preferences.
    This provides backward compatibility while supporting user customization.
    """
    
    # Fallback mapping if user doesn't have custom categories
    FALLBACK_MODEL_MAPPING = {
        ModelType.FAST: "google/gemini-2.5-flash",
        ModelType.STANDARD: "anthropic/claude-sonnet-4", 
        ModelType.FAST_REASONING: "openai/o4-mini",
        ModelType.REASONING: "openai/o3",
    }

    @staticmethod
    def get_model_id_for_user(user_id: Optional[int], model_type: ModelType) -> str:
        """
        Get the actual model ID for a user's model type preference.
        Falls back to system defaults if user has no custom categories.
        """
        if not user_id:
            # No user context, use fallback
            return ModelResolverService.FALLBACK_MODEL_MAPPING.get(
                model_type, ModelResolverService.FALLBACK_MODEL_MAPPING[ModelType.STANDARD]
            )

        try:
            db = next(get_db())
            category_service = UserModelCategoryService(db)
            
            # Try to get user's custom category mapping
            category_name = model_type.value.upper().replace(" ", "_")
            model_id = category_service.get_model_id_for_category(user_id, category_name)
            
            if model_id:
                return model_id
            
            # Fallback to system default
            fallback_id = ModelResolverService.FALLBACK_MODEL_MAPPING.get(model_type)
            logger.warning(f"No user category found for {model_type.value}, using fallback: {fallback_id}")
            return fallback_id or ModelResolverService.FALLBACK_MODEL_MAPPING[ModelType.STANDARD]
            
        except Exception as e:
            logger.error(f"Error resolving model for user {user_id}, model_type {model_type}: {e}")
            # Fallback to system default on error
            return ModelResolverService.FALLBACK_MODEL_MAPPING.get(
                model_type, ModelResolverService.FALLBACK_MODEL_MAPPING[ModelType.STANDARD]
            )
        finally:
            try:
                db.close()
            except:
                pass

    @staticmethod
    def get_user_categories_as_model_types(user_id: int) -> dict:
        """
        Get user's categories mapped to ModelType enum for backward compatibility.
        Returns a mapping of ModelType -> actual_model_id
        """
        try:
            db = next(get_db())
            category_service = UserModelCategoryService(db)
            
            categories = category_service.get_user_categories(user_id)
            
            # Map category names back to ModelType enum
            mapping = {}
            for category in categories:
                try:
                    # Convert category name to ModelType
                    category_name_normalized = category.category_name.replace("_", " ")
                    model_type = ModelType(category_name_normalized)
                    mapping[model_type] = category.model_id
                except ValueError:
                    # Category doesn't match standard ModelType, skip
                    logger.debug(f"User category {category.category_name} doesn't match standard ModelType")
                    continue
            
            return mapping
            
        except Exception as e:
            logger.error(f"Error getting user categories for user {user_id}: {e}")
            return {}
        finally:
            try:
                db.close()
            except:
                pass