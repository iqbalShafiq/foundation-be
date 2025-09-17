import logging
from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from app.models import (
    UserModelCategory, 
    UserModelCategoryCreate, 
    UserModelCategoryUpdate, 
    UserModelCategoryResponse,
    ModelMetadata,
    User
)

logger = logging.getLogger(__name__)


class UserModelCategoryService:
    def __init__(self, db: Session):
        self.db = db

    # Default categories that will be created for new users
    DEFAULT_CATEGORIES = [
        {
            "category_name": "FAST",
            "display_name": "Fast",
            "model_id": "google/gemini-2.5-flash",
            "description": "Quick responses for simple tasks",
            "sort_order": 1
        },
        {
            "category_name": "STANDARD",
            "display_name": "Standard",
            "model_id": "anthropic/claude-sonnet-4",
            "description": "Balanced performance for most tasks",
            "sort_order": 2
        },
        {
            "category_name": "FAST_REASONING",
            "display_name": "Fast Reasoning",
            "model_id": "openai/o4-mini",
            "description": "Quick reasoning for analytical tasks",
            "sort_order": 3
        },
        {
            "category_name": "REASONING",
            "display_name": "Reasoning",
            "model_id": "openai/o3",
            "description": "Deep reasoning for complex problems",
            "sort_order": 4
        }
    ]

    def create_default_categories_for_user(self, user_id: int) -> Dict[str, Any]:
        """Create default model categories for a new user"""
        try:
            created_categories = []
            skipped_categories = []

            for default_cat in self.DEFAULT_CATEGORIES:
                # Check if category already exists
                existing = self.db.query(UserModelCategory).filter(
                    UserModelCategory.user_id == user_id,
                    UserModelCategory.category_name == default_cat["category_name"]
                ).first()

                if existing:
                    skipped_categories.append(default_cat["category_name"])
                    continue

                # Check if the model exists in ModelMetadata
                model_exists = self.db.query(ModelMetadata).filter(
                    ModelMetadata.id == default_cat["model_id"]
                ).first()

                if not model_exists:
                    # If model doesn't exist, try to find a fallback
                    fallback_model = self.db.query(ModelMetadata).filter(
                        ModelMetadata.is_active == True
                    ).first()
                    
                    if fallback_model:
                        model_id = fallback_model.id
                        logger.warning(f"Model {default_cat['model_id']} not found, using fallback {model_id}")
                    else:
                        logger.error(f"No models available for default category {default_cat['category_name']}")
                        continue
                else:
                    model_id = default_cat["model_id"]

                # Create the category
                category = UserModelCategory(
                    user_id=user_id,
                    category_name=default_cat["category_name"],
                    display_name=default_cat["display_name"],
                    model_id=model_id,
                    description=default_cat["description"],
                    sort_order=default_cat["sort_order"]
                )

                self.db.add(category)
                created_categories.append(default_cat["category_name"])

            self.db.commit()

            return {
                "success": True,
                "created_categories": created_categories,
                "skipped_categories": skipped_categories,
                "message": f"Created {len(created_categories)} default categories"
            }

        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating default categories for user {user_id}: {e}")
            return {
                "success": False,
                "created_categories": [],
                "skipped_categories": [],
                "message": f"Failed to create default categories: {str(e)}"
            }

    def get_user_categories(self, user_id: int, include_inactive: bool = False) -> List[UserModelCategoryResponse]:
        """Get all model categories for a user"""
        try:
            query = self.db.query(UserModelCategory, ModelMetadata).join(
                ModelMetadata, UserModelCategory.model_id == ModelMetadata.id
            ).filter(UserModelCategory.user_id == user_id)

            if not include_inactive:
                query = query.filter(UserModelCategory.is_active == True)

            query = query.order_by(UserModelCategory.sort_order, UserModelCategory.category_name)
            
            results = query.all()
            
            categories = []
            for category, model_metadata in results:
                # Build pricing info
                pricing_info = {}
                if model_metadata:
                    pricing_info = {
                        "prompt_price": model_metadata.prompt_price,
                        "completion_price": model_metadata.completion_price,
                        "request_price": model_metadata.request_price,
                        "context_length": model_metadata.context_length
                    }

                category_response = UserModelCategoryResponse(
                    id=category.id,
                    category_name=category.category_name,
                    display_name=category.display_name,
                    model_id=category.model_id,
                    model_name=model_metadata.name if model_metadata else None,
                    model_pricing=pricing_info if pricing_info else None,
                    description=category.description,
                    sort_order=category.sort_order,
                    is_active=category.is_active,
                    created_at=category.created_at.isoformat() if category.created_at else "",
                    updated_at=category.updated_at.isoformat() if category.updated_at else ""
                )
                categories.append(category_response)

            return categories

        except Exception as e:
            logger.error(f"Error getting user categories for user {user_id}: {e}")
            return []

    def get_category_by_name(self, user_id: int, category_name: str) -> Optional[UserModelCategoryResponse]:
        """Get a specific category by name for a user"""
        try:
            result = self.db.query(UserModelCategory, ModelMetadata).join(
                ModelMetadata, UserModelCategory.model_id == ModelMetadata.id
            ).filter(
                UserModelCategory.user_id == user_id,
                UserModelCategory.category_name == category_name,
                UserModelCategory.is_active == True
            ).first()

            if not result:
                return None

            category, model_metadata = result

            # Build pricing info
            pricing_info = {}
            if model_metadata:
                pricing_info = {
                    "prompt_price": model_metadata.prompt_price,
                    "completion_price": model_metadata.completion_price,
                    "request_price": model_metadata.request_price,
                    "context_length": model_metadata.context_length
                }

            return UserModelCategoryResponse(
                id=category.id,
                category_name=category.category_name,
                display_name=category.display_name,
                model_id=category.model_id,
                model_name=model_metadata.name if model_metadata else None,
                model_pricing=pricing_info if pricing_info else None,
                description=category.description,
                sort_order=category.sort_order,
                is_active=category.is_active,
                created_at=category.created_at.isoformat() if category.created_at else "",
                updated_at=category.updated_at.isoformat() if category.updated_at else ""
            )

        except Exception as e:
            logger.error(f"Error getting category {category_name} for user {user_id}: {e}")
            return None

    def create_category(self, user_id: int, category_data: UserModelCategoryCreate) -> Optional[UserModelCategoryResponse]:
        """Create a new category for a user"""
        try:
            # Validate that the model exists
            model_exists = self.db.query(ModelMetadata).filter(
                ModelMetadata.id == category_data.model_id,
                ModelMetadata.is_active == True
            ).first()

            if not model_exists:
                logger.error(f"Model {category_data.model_id} not found or inactive")
                return None

            # Create the category
            category = UserModelCategory(
                user_id=user_id,
                category_name=category_data.category_name,
                display_name=category_data.display_name,
                model_id=category_data.model_id,
                description=category_data.description,
                sort_order=category_data.sort_order
            )

            self.db.add(category)
            self.db.commit()
            self.db.refresh(category)

            # Return the created category with model info
            return self.get_category_by_name(user_id, category_data.category_name)

        except IntegrityError as e:
            self.db.rollback()
            logger.error(f"Category {category_data.category_name} already exists for user {user_id}")
            return None
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error creating category for user {user_id}: {e}")
            return None

    def update_category(self, user_id: int, category_id: int, category_data: UserModelCategoryUpdate) -> Optional[UserModelCategoryResponse]:
        """Update an existing category"""
        try:
            category = self.db.query(UserModelCategory).filter(
                UserModelCategory.id == category_id,
                UserModelCategory.user_id == user_id
            ).first()

            if not category:
                return None

            # Validate model if being updated
            if category_data.model_id:
                model_exists = self.db.query(ModelMetadata).filter(
                    ModelMetadata.id == category_data.model_id,
                    ModelMetadata.is_active == True
                ).first()

                if not model_exists:
                    logger.error(f"Model {category_data.model_id} not found or inactive")
                    return None

            # Update fields
            update_data = category_data.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(category, field, value)

            self.db.commit()
            self.db.refresh(category)

            # Return updated category with model info
            return self.get_category_by_name(user_id, category.category_name)

        except IntegrityError as e:
            self.db.rollback()
            logger.error(f"Category name conflict when updating category {category_id} for user {user_id}")
            return None
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating category {category_id} for user {user_id}: {e}")
            return None

    def delete_category(self, user_id: int, category_id: int) -> bool:
        """Delete a category (soft delete by setting is_active=False)"""
        try:
            category = self.db.query(UserModelCategory).filter(
                UserModelCategory.id == category_id,
                UserModelCategory.user_id == user_id
            ).first()

            if not category:
                return False

            category.is_active = False
            self.db.commit()
            return True

        except Exception as e:
            self.db.rollback()
            logger.error(f"Error deleting category {category_id} for user {user_id}: {e}")
            return False

    def reorder_categories(self, user_id: int, category_orders: List[Dict[str, int]]) -> bool:
        """Reorder categories by updating sort_order"""
        try:
            for order_data in category_orders:
                category_id = order_data.get("category_id")
                sort_order = order_data.get("sort_order")

                if category_id is None or sort_order is None:
                    continue

                category = self.db.query(UserModelCategory).filter(
                    UserModelCategory.id == category_id,
                    UserModelCategory.user_id == user_id
                ).first()

                if category:
                    category.sort_order = sort_order

            self.db.commit()
            return True

        except Exception as e:
            self.db.rollback()
            logger.error(f"Error reordering categories for user {user_id}: {e}")
            return False

    def get_model_id_for_category(self, user_id: int, category_name: str) -> Optional[str]:
        """Get the model ID for a specific category - used by chat services"""
        try:
            category = self.db.query(UserModelCategory).filter(
                UserModelCategory.user_id == user_id,
                UserModelCategory.category_name == category_name,
                UserModelCategory.is_active == True
            ).first()

            return category.model_id if category else None

        except Exception as e:
            logger.error(f"Error getting model ID for category {category_name}, user {user_id}: {e}")
            return None

    def get_available_models_for_categories(self) -> List[Dict[str, Any]]:
        """Get list of available models that can be used in categories"""
        try:
            models = self.db.query(ModelMetadata).filter(
                ModelMetadata.is_active == True
            ).order_by(ModelMetadata.name).all()

            return [
                {
                    "id": model.id,
                    "name": model.name,
                    "description": model.description,
                    "context_length": model.context_length,
                    "modality": model.modality,
                    "prompt_price": model.prompt_price,
                    "completion_price": model.completion_price
                }
                for model in models
            ]

        except Exception as e:
            logger.error(f"Error getting available models for categories: {e}")
            return []