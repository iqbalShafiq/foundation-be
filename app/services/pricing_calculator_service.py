import logging
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from app.models import ModelMetadata
from app.database import get_db

logger = logging.getLogger(__name__)


class PricingCalculatorService:
    """
    Service to calculate accurate pricing based on model metadata from OpenRouter.
    Replaces inaccurate OpenAI callback pricing with real OpenRouter pricing.
    """

    @staticmethod
    def calculate_cost(
        model_id: str,
        input_tokens: int,
        output_tokens: int,
        image_count: int = 0,
        web_search_count: int = 0,
        internal_reasoning_tokens: int = 0,
        cached_input_tokens: int = 0
    ) -> Dict[str, Any]:
        """
        Calculate accurate cost based on model metadata pricing.
        
        Args:
            model_id: The OpenRouter model ID
            input_tokens: Number of input/prompt tokens
            output_tokens: Number of output/completion tokens
            image_count: Number of images processed (optional)
            web_search_count: Number of web searches (optional)
            internal_reasoning_tokens: Internal reasoning tokens (optional)
            cached_input_tokens: Cached input tokens with different pricing (optional)
            
        Returns:
            Dict with cost breakdown and total cost
        """
        try:
            db = next(get_db())
            
            # Get model metadata
            model = db.query(ModelMetadata).filter(
                ModelMetadata.id == model_id,
                ModelMetadata.is_active == True
            ).first()
            
            if not model:
                logger.warning(f"Model {model_id} not found in metadata, cost calculation unavailable")
                return {
                    "total_cost": 0.0,
                    "cost_breakdown": {
                        "prompt_cost": 0.0,
                        "completion_cost": 0.0,
                        "image_cost": 0.0,
                        "web_search_cost": 0.0,
                        "internal_reasoning_cost": 0.0,
                        "cached_input_cost": 0.0
                    },
                    "model_pricing": {
                        "prompt_price": None,
                        "completion_price": None
                    },
                    "error": "Model not found in metadata"
                }
            
            # Calculate costs based on actual OpenRouter pricing
            cost_breakdown = {}
            
            # Prompt/input tokens cost
            prompt_cost = 0.0
            if model.prompt_price and input_tokens > 0:
                prompt_cost = model.prompt_price * input_tokens
            cost_breakdown["prompt_cost"] = prompt_cost
            
            # Completion/output tokens cost
            completion_cost = 0.0
            if model.completion_price and output_tokens > 0:
                completion_cost = model.completion_price * output_tokens
            cost_breakdown["completion_cost"] = completion_cost
            
            # Image processing cost
            image_cost = 0.0
            if model.image_price and image_count > 0:
                image_cost = model.image_price * image_count
            cost_breakdown["image_cost"] = image_cost
            
            # Web search cost
            web_search_cost = 0.0
            if model.web_search_price and web_search_count > 0:
                web_search_cost = model.web_search_price * web_search_count
            cost_breakdown["web_search_cost"] = web_search_cost
            
            # Internal reasoning cost (for reasoning models)
            internal_reasoning_cost = 0.0
            if model.internal_reasoning_price and internal_reasoning_tokens > 0:
                internal_reasoning_cost = model.internal_reasoning_price * internal_reasoning_tokens
            cost_breakdown["internal_reasoning_cost"] = internal_reasoning_cost
            
            # Cached input cost (different pricing for cached tokens)
            cached_input_cost = 0.0
            if model.input_cache_read_price and cached_input_tokens > 0:
                cached_input_cost = model.input_cache_read_price * cached_input_tokens
            cost_breakdown["cached_input_cost"] = cached_input_cost
            
            # Total cost
            total_cost = sum(cost_breakdown.values())
            
            return {
                "total_cost": round(total_cost, 8),  # Round to 8 decimal places
                "cost_breakdown": {k: round(v, 8) for k, v in cost_breakdown.items()},
                "model_pricing": {
                    "prompt_price": model.prompt_price,
                    "completion_price": model.completion_price,
                    "image_price": model.image_price,
                    "web_search_price": model.web_search_price,
                    "internal_reasoning_price": model.internal_reasoning_price,
                    "input_cache_read_price": model.input_cache_read_price
                },
                "tokens_used": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                    "image_count": image_count,
                    "web_search_count": web_search_count,
                    "internal_reasoning_tokens": internal_reasoning_tokens,
                    "cached_input_tokens": cached_input_tokens
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating cost for model {model_id}: {e}")
            return {
                "total_cost": 0.0,
                "cost_breakdown": {
                    "prompt_cost": 0.0,
                    "completion_cost": 0.0,
                    "image_cost": 0.0,
                    "web_search_cost": 0.0,
                    "internal_reasoning_cost": 0.0,
                    "cached_input_cost": 0.0
                },
                "model_pricing": {
                    "prompt_price": None,
                    "completion_price": None
                },
                "error": f"Calculation error: {str(e)}"
            }
        finally:
            try:
                db.close()
            except:
                pass

    @staticmethod
    def get_model_pricing_info(model_id: str) -> Optional[Dict[str, Any]]:
        """
        Get pricing information for a specific model.
        Useful for displaying pricing to users before they choose a model.
        """
        try:
            db = next(get_db())
            
            model = db.query(ModelMetadata).filter(
                ModelMetadata.id == model_id,
                ModelMetadata.is_active == True
            ).first()
            
            if not model:
                return None
                
            return {
                "model_id": model.id,
                "model_name": model.name,
                "prompt_price": model.prompt_price,
                "completion_price": model.completion_price,
                "request_price": model.request_price,
                "image_price": model.image_price,
                "web_search_price": model.web_search_price,
                "internal_reasoning_price": model.internal_reasoning_price,
                "input_cache_read_price": model.input_cache_read_price,
                "context_length": model.context_length,
                "modality": model.modality
            }
            
        except Exception as e:
            logger.error(f"Error getting pricing info for model {model_id}: {e}")
            return None
        finally:
            try:
                db.close()
            except:
                pass

    @staticmethod
    def estimate_cost_for_message(
        model_id: str,
        message_length: int,
        expected_response_length: int = 500,
        include_images: int = 0
    ) -> Dict[str, Any]:
        """
        Estimate cost for a message before sending.
        Useful for showing cost estimates to users.
        
        Args:
            model_id: The OpenRouter model ID
            message_length: Estimated length of user message in characters
            expected_response_length: Expected length of AI response in characters
            include_images: Number of images included
            
        Returns:
            Dict with estimated cost and breakdown
        """
        # Rough estimation: 1 token ≈ 3-4 characters for most languages
        estimated_input_tokens = message_length // 3
        estimated_output_tokens = expected_response_length // 3
        
        return PricingCalculatorService.calculate_cost(
            model_id=model_id,
            input_tokens=estimated_input_tokens,
            output_tokens=estimated_output_tokens,
            image_count=include_images
        )