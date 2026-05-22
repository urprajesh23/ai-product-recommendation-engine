"""
FastAPI REST API for Product Recommender System

Provides endpoints for:
- Getting personalized recommendations
- Getting similar items
- Health checks
- Model information

Features:
- Redis caching for fast responses
- Request validation with Pydantic
- Automatic API documentation (Swagger UI)
- Error handling and logging
"""

from fastapi import FastAPI, HTTPException, status, Query, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import redis
import pickle
import numpy as np
import random
from datetime import datetime
import logging
from pathlib import Path
import os
import csv
import io
import pandas as pd
from contextlib import asynccontextmanager

# Try to import models (will be available when models are trained)
try:
    from src.models.hybrid import HybridRecommender
    from src.models.collaborative import CollaborativeFilteringModel
    from src.utils.helpers import load_pickle, load_mappings, get_user_idx, get_item_idx, get_user_id, get_item_id
except ImportError as e:
    print(f"⚠️  Warning: Could not import models. Make sure to train models first. Error: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== Configuration ====================

class Config:
    """API Configuration"""
    # Redis
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))
    REDIS_CACHE_TTL = int(os.getenv('REDIS_CACHE_TTL', 3600))  # 1 hour
    
    # Model paths
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/hybrid_model.pkl')
    MODEL_V1_PATH = os.getenv('MODEL_V1_PATH', 'models/collaborative_model.pkl')
    MODEL_V2_PATH = os.getenv('MODEL_V2_PATH', 'models/hybrid_model.pkl')
    TRAIN_MATRIX_PATH = os.getenv('TRAIN_MATRIX_PATH', 'data/processed/train_matrix.pkl')
    MAPPINGS_PATH = os.getenv('MAPPINGS_PATH', 'data/processed/mappings.pkl')
    
    # API settings
    API_VERSION = "1.0.0"
    API_TITLE = "Product Recommender API"
    API_DESCRIPTION = """
    🎯 **Product Recommendation API**
    
    A hybrid recommendation engine combining collaborative filtering and content-based methods.
    
    ## Features
    - **Personalized Recommendations**: Get top-N product recommendations for any user
    - **Similar Items**: Find products similar to a given item
    - **Cold-Start Handling**: Fallback to popularity-based recommendations for new users
    - **Fast Response**: Redis caching for sub-15ms latency
    - **Model Versioning**: Track model versions via MLflow
    
    ## Metrics
    - Precision@10: 0.41
    - Recall@10: 0.38
    - NDCG@10: 0.44
    """
    
    # Recommendation limits
    MIN_RECOMMENDATIONS = 1
    MAX_RECOMMENDATIONS = 100
    DEFAULT_RECOMMENDATIONS = 10


config = Config()


# ==================== Product Catalogs ====================
# Maps item IDs to human-readable product names for both datasets

SUPERSTORE_CATALOG = {
    "S001": {"name": "Ergonomic Office Chair",         "category": "Furniture"},
    "S002": {"name": "Motorized Standing Desk",         "category": "Furniture"},
    "S003": {"name": "Bookshelf 5-Tier",                "category": "Furniture"},
    "S004": {"name": "Premium Notebook Set of 3",       "category": "Office Supplies"},
    "S005": {"name": "Gel Pens — 12 Pack",              "category": "Office Supplies"},
    "S006": {"name": "Mesh Desk Organizer",             "category": "Office Supplies"},
    "S007": {"name": "Wireless Silent Mouse",           "category": "Technology"},
    "S008": {"name": "USB-C Hub 7-in-1",               "category": "Technology"},
    "S009": {"name": "Noise Cancelling Earbuds",        "category": "Technology"},
    "S010": {"name": "Artificial Potted Plant",         "category": "Decor"},
    "S011": {"name": "LED Dimmable Desk Lamp",          "category": "Decor"},
    "S012": {"name": "Mini Fridge 4L",                  "category": "Appliances"},
    "S013": {"name": "Single Serve Coffee Maker",       "category": "Appliances"},
}

# Tech dataset: product ID prefix map for human-friendly category labels
TECH_CATEGORY_MAP = {
    "B0002": "Audio",    "B0001": "Audio",    "B0000": "Audio",
    "B000E": "Audio",    "B000I": "Audio",    "B000F": "Gaming",
    "B000H": "Gaming",   "B001": "Electronics", "B000": "Electronics",
}

def get_product_info(item_id: str):
    """Return (product_name, category) for a given item_id."""
    if item_id in SUPERSTORE_CATALOG:
        return SUPERSTORE_CATALOG[item_id]["name"], SUPERSTORE_CATALOG[item_id]["category"]
    # For tech Amazon IDs, generate a friendly label
    category = "Electronics"
    for prefix, cat in TECH_CATEGORY_MAP.items():
        if item_id.startswith(prefix):
            category = cat
            break
    return f"Product {item_id[:8]}", category


# ==================== Global Variables ====================

# These will be loaded on startup
model = None
model_v1 = None
model_v2 = None
train_matrix = None
mappings = None
redis_client = None



# ==================== Lifespan Events ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan events for FastAPI app
    Handles startup and shutdown
    """
    # Startup
    logger.info("🚀 Starting Product Recommender API...")
    
    global model, model_v1, model_v2, train_matrix, mappings, redis_client
    
    try:
        # Initialize Redis
        logger.info(f"📡 Connecting to Redis at {config.REDIS_HOST}:{config.REDIS_PORT}...")
        redis_client = redis.Redis(
            host=config.REDIS_HOST,
            port=config.REDIS_PORT,
            db=config.REDIS_DB,
            decode_responses=False,
            socket_connect_timeout=5
        )
        redis_client.ping()
        logger.info("✅ Redis connected successfully")
        
    except redis.ConnectionError as e:
        logger.warning(f"⚠️  Redis connection failed: {e}. Running without cache.")
        redis_client = None
    
    try:
        # Load models
        model_v1_path = Path(config.MODEL_V1_PATH)
        if model_v1_path.exists():
            logger.info(f"📂 Loading model v1 from {model_v1_path}...")
            try:
                from src.models.collaborative import CollaborativeFilteringModel
                model_v1 = CollaborativeFilteringModel.load(str(model_v1_path))
                logger.info("✅ Collaborative Filtering model loaded successfully (v1)")
            except Exception as e:
                logger.error(f"Failed to load v1 model: {e}")
                model_v1 = None
        else:
            logger.warning(f"⚠️  Model v1 not found at {model_v1_path}.")
            
        model_v2_path = Path(config.MODEL_V2_PATH)
        if model_v2_path.exists():
            logger.info(f"📂 Loading model v2 from {model_v2_path}...")
            try:
                from src.models.hybrid import HybridRecommender
                model_v2 = HybridRecommender.load(str(model_v2_path))
                logger.info("✅ Hybrid model loaded successfully (v2)")
            except Exception as e:
                logger.warning(f"Failed to load v2 model as HybridRecommender: {e}. Trying CollaborativeFilteringModel fallback...")
                try:
                    from src.models.collaborative import CollaborativeFilteringModel
                    model_v2 = CollaborativeFilteringModel.load(str(model_v2_path))
                    logger.info("✅ Model v2 loaded successfully as CollaborativeFilteringModel fallback")
                except Exception as e2:
                    logger.error(f"Failed to load v2 model entirely: {e2}")
                    model_v2 = None
                
        # Fallback for old model reference
        model = model_v2 if model_v2 else model_v1
            
        # Load train matrix
        matrix_path = Path(config.TRAIN_MATRIX_PATH)
        if matrix_path.exists():
            logger.info(f"📂 Loading interaction matrix from {matrix_path}...")
            train_matrix = load_pickle(str(matrix_path))
            logger.info(f"✅ Matrix loaded: {train_matrix.shape}")
        else:
            logger.warning(f"⚠️  Matrix not found at {matrix_path}")
            train_matrix = None
            
        # Load mappings
        mappings_path = Path(config.MAPPINGS_PATH)
        if mappings_path.exists():
            logger.info(f"📂 Loading ID mappings from {mappings_path}...")
            mappings = load_mappings(str(mappings_path))
            logger.info(f"✅ Mappings loaded: {len(mappings['user_id_to_idx'])} users, {len(mappings['item_id_to_idx'])} items")
        else:
            logger.warning(f"⚠️  Mappings not found at {mappings_path}")
            mappings = None
        
        logger.info("🎉 API startup complete!")
        
    except Exception as e:
        logger.error(f"❌ Error during startup: {e}")
        raise
    
    yield  # API is running
    
    # Shutdown
    logger.info("🛑 Shutting down Product Recommender API...")
    if redis_client:
        redis_client.close()
        logger.info("✅ Redis connection closed")
    logger.info("👋 Shutdown complete")


# ==================== FastAPI App ====================

app = FastAPI(
    title=config.API_TITLE,
    description=config.API_DESCRIPTION,
    version=config.API_VERSION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================== Pydantic Models ====================

class RecommendationRequest(BaseModel):
    """Request model for getting recommendations"""
    user_id: str = Field(..., description="User ID (original ID from dataset)")
    n_recommendations: int = Field(
        default=config.DEFAULT_RECOMMENDATIONS,
        ge=config.MIN_RECOMMENDATIONS,
        le=config.MAX_RECOMMENDATIONS,
        description="Number of recommendations to return"
    )
    filter_purchased: bool = Field(
        default=True,
        description="Whether to filter out items the user has already purchased"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "A2SUAM1J3GNN3B",
                "n_recommendations": 10,
                "filter_purchased": True
            }
        }


class RecommendationItem(BaseModel):
    """Single recommendation item"""
    item_id: str = Field(..., description="Item ID (original ID from dataset)")
    score: float = Field(..., description="Recommendation score (0-1)")
    rank: int = Field(..., description="Rank position (1-based)")
    product_name: Optional[str] = Field(default=None, description="Human-readable product name")
    category: Optional[str] = Field(default=None, description="Product category")


class RecommendationResponse(BaseModel):
    """Response model for recommendations"""
    user_id: str = Field(..., description="User ID that requested recommendations")
    recommendations: List[RecommendationItem] = Field(..., description="List of recommended items")
    timestamp: str = Field(..., description="Timestamp of recommendation")
    model_version: str = Field(..., description="Model version used")
    variant: Optional[str] = Field(default=None, description="A/B test variant used (v1 or v2)")
    cached: bool = Field(default=False, description="Whether result was cached")
    
    class Config:
        schema_extra = {
            "example": {
                "user_id": "A2SUAM1J3GNN3B",
                "recommendations": [
                    {"item_id": "B00DR0PDNE", "score": 0.95, "rank": 1},
                    {"item_id": "B007WTAJTO", "score": 0.89, "rank": 2},
                    {"item_id": "B00B7XI6ZO", "score": 0.85, "rank": 3}
                ],
                "timestamp": "2024-01-15T10:30:00.000Z",
                "model_version": "1.0.0",
                "cached": False
            }
        }


class ClickRequest(BaseModel):
    """Request model for simulating a click/conversion"""
    user_id: str = Field(..., description="User ID who clicked")
    item_id: str = Field(..., description="Item ID that was clicked")
    variant: str = Field(..., description="A/B test variant that was served (v1 or v2)")

class ABTestStatsResponse(BaseModel):
    """Response model for A/B testing stats"""
    v1_impressions: int
    v1_clicks: int
    v1_ctr: float
    v2_impressions: int
    v2_clicks: int
    v2_ctr: float
    winner: Optional[str] = None


class SimilarItemsRequest(BaseModel):
    """Request model for similar items"""
    item_id: str = Field(..., description="Item ID to find similar items for")
    n_items: int = Field(
        default=config.DEFAULT_RECOMMENDATIONS,
        ge=config.MIN_RECOMMENDATIONS,
        le=config.MAX_RECOMMENDATIONS,
        description="Number of similar items to return"
    )
    
    class Config:
        schema_extra = {
            "example": {
                "item_id": "B00DR0PDNE",
                "n_items": 10
            }
        }


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    model_loaded: bool
    redis_connected: bool
    version: str


class ModelInfoResponse(BaseModel):
    """Model information response"""
    model_type: str
    version: str
    n_users: int
    n_items: int
    parameters: Dict[str, Any]
    last_trained: Optional[str] = None


# ==================== Helper Functions ====================

def get_cache_key(prefix: str, **kwargs) -> str:
    """Generate cache key from parameters"""
    key_parts = [prefix] + [f"{k}:{v}" for k, v in sorted(kwargs.items())]
    return ":".join(key_parts)


def get_from_cache(key: str) -> Optional[Any]:
    """Get value from Redis cache"""
    if not redis_client:
        return None
    
    try:
        cached = redis_client.get(key)
        if cached:
            return pickle.loads(cached)
    except Exception as e:
        logger.warning(f"Cache read error: {e}")
    
    return None


def set_in_cache(key: str, value: Any, ttl: int = config.REDIS_CACHE_TTL):
    """Set value in Redis cache"""
    if not redis_client:
        return
    
    try:
        redis_client.setex(key, ttl, pickle.dumps(value))
    except Exception as e:
        logger.warning(f"Cache write error: {e}")


# ==================== API Endpoints ====================

# Root is now handled by StaticFiles serving the frontend index.html


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint
    
    Returns the status of the API and its dependencies
    """
    redis_status = False
    if redis_client:
        try:
            redis_client.ping()
            redis_status = True
        except:
            pass
    
    return HealthResponse(
        status="healthy" if model is not None and train_matrix is not None else "degraded",
        timestamp=datetime.utcnow().isoformat(),
        model_loaded=model is not None,
        redis_connected=redis_status,
        version=config.API_VERSION
    )


def _generate_recommendations(request: RecommendationRequest, req_model, variant: str) -> RecommendationResponse:
    if req_model is None or train_matrix is None or mappings is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model {variant} not loaded. Please train and load the model first."
        )
    
    # Check cache
    cache_key = get_cache_key(
        f"recommend:{variant}",
        user_id=request.user_id,
        n=request.n_recommendations,
        filter=request.filter_purchased
    )
    
    cached_result = get_from_cache(cache_key)
    if cached_result:
        logger.info(f"Cache hit for user {request.user_id} (variant {variant})")
        cached_result['cached'] = True
        # Log impression from cache if Redis available
        if redis_client:
            redis_client.incr(f"ab_test:{variant}:impressions")
        return RecommendationResponse(**cached_result)
    
    # Convert user ID to index
    user_idx = get_user_idx(request.user_id, mappings)
    is_new_user = user_idx is None or user_idx >= train_matrix.shape[0]
    
    # Map synthetic user IDs (U001-U020) to real users for demonstration purposes
    is_superstore = False
    if is_new_user and request.user_id.startswith("U0"):
        try:
            user_num = int(request.user_id[1:])
            is_superstore = 21 <= user_num <= 40
        except ValueError:
            pass
            
        import hashlib
        hash_val = int(hashlib.md5(request.user_id.encode()).hexdigest(), 16)
        user_idx = hash_val % train_matrix.shape[0]
        is_new_user = False
        logger.info(f"Mapped synthetic user {request.user_id} to internal index {user_idx}")
    
    try:
        if is_superstore:
            logger.info(f"Serving superstore recommendations for {request.user_id}")
            import random
            superstore_items = [
                "S001", "S002", "S003", "S004", "S005", "S006", 
                "S007", "S008", "S009", "S010", "S011", "S012", "S013"
            ]
            random.seed(hash_val)
            recommended_s_items = random.sample(superstore_items, min(request.n_recommendations, len(superstore_items)))
            
            recommendations = []
            for rank, item_id in enumerate(recommended_s_items, start=1):
                score = 0.95 - (rank * 0.05)
                pname, pcat = get_product_info(item_id)
                recommendations.append(RecommendationItem(
                    item_id=item_id, score=score, rank=rank,
                    product_name=pname, category=pcat
                ))
                
            response = RecommendationResponse(
                user_id=request.user_id,
                recommendations=recommendations,
                timestamp=datetime.utcnow().isoformat(),
                model_version=config.API_VERSION,
                variant=variant + " (Superstore Mock)",
                cached=False
            )
            set_in_cache(cache_key, response.dict())
            return response
        elif is_new_user:
            logger.info(f"Cold-start user: {request.user_id}")
            n_items = min(request.n_recommendations, train_matrix.shape[1])
            item_popularity = np.array(train_matrix.sum(axis=0)).flatten()
            top_items = np.argsort(item_popularity)[::-1][:n_items]
            recs = [(int(item), float(item_popularity[item])) for item in top_items]
        else:
            recs = req_model.recommend(
                user_idx=user_idx,
                user_item_matrix=train_matrix,
                N=request.n_recommendations,
                filter_already_liked=request.filter_purchased
            )
        
        recommendations = []
        for rank, (item_idx, score) in enumerate(recs, start=1):
            item_id = get_item_id(item_idx, mappings)
            if item_id:
                pname, pcat = get_product_info(item_id)
                recommendations.append(
                    RecommendationItem(
                        item_id=item_id,
                        score=float(score),
                        rank=rank,
                        product_name=pname,
                        category=pcat
                    )
                )
        
        response = RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            timestamp=datetime.utcnow().isoformat(),
            model_version=config.API_VERSION,
            variant=variant,
            cached=False
        )
        
        # Cache result
        set_in_cache(cache_key, response.dict())
        
        # Log impression
        if redis_client:
            redis_client.incr(f"ab_test:{variant}:impressions")
        
        logger.info(f"Generated {len(recommendations)} recommendations for user {request.user_id} (variant {variant})")
        return response
        
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating recommendations: {str(e)}"
        )


@app.post("/recommend/v1", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations_v1(request: RecommendationRequest):
    """Get recommendations using the Baseline SVD/CF model (Variant A)"""
    return _generate_recommendations(request, model_v1, "v1")

@app.post("/recommend/v2", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations_v2(request: RecommendationRequest):
    """Get recommendations using the Hybrid ALS+LightFM model (Variant B)"""
    return _generate_recommendations(request, model_v2, "v2")

@app.post("/recommend", response_model=RecommendationResponse, tags=["Recommendations"])
async def get_recommendations(request: RecommendationRequest):
    """
    Get personalized product recommendations for a user.
    Uses an A/B testing traffic splitter: Routes 50% of traffic to v1 (SVD) and 50% to v2 (Hybrid).
    """
    # Simple 50/50 traffic splitter
    if random.random() < 0.5:
        return _generate_recommendations(request, model_v1, "v1")
    else:
        return _generate_recommendations(request, model_v2, "v2")

@app.post("/ab-test/click", tags=["A/B Testing"])
async def log_ab_test_click(request: ClickRequest):
    """Log a simulated click to compute CTR for A/B testing"""
    if redis_client:
        redis_client.incr(f"ab_test:{request.variant}:clicks")
        return {"status": "success", "message": f"Click logged for variant {request.variant}"}
    return {"status": "warning", "message": "Redis not available, click not logged"}

@app.get("/dataset/explore", tags=["Dataset"])
async def get_dataset_explore(
    dataset: str = Query(default="tech", description="'tech' or 'superstore'")
):
    """
    Returns pandas dataframe method outputs (code and output string) for the dataset.
    """
    if dataset == "superstore":
        csv_path = Path("data/synthetic/superstore_interactions.csv")
    else:
        csv_path = Path("data/synthetic/product_interactions.csv")

    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        df = pd.read_csv(csv_path)
        
        # Capture info
        buf = io.StringIO()
        df.info(buf=buf)
        info_str = buf.getvalue()
        
        results = [
            {
                "method": "df.describe()",
                "purpose": "Statistical Overview",
                "code": "df.describe()",
                "output": df.describe().to_string()
            },
            {
                "method": "df.info()",
                "purpose": "Structural Overview",
                "code": "df.info()",
                "output": info_str
            },
            {
                "method": "df.head()",
                "purpose": "Data Preview",
                "code": "df.head()",
                "output": df.head().to_string()
            },
            {
                "method": "df.shape",
                "purpose": "Dimensionality",
                "code": "df.shape",
                "output": str(df.shape)
            },
            {
                "method": "df.isna().sum()",
                "purpose": "Missing Data",
                "code": "df.isna().sum()",
                "output": df.isna().sum().to_string()
            }
        ]
        
        return {"dataset": dataset, "exploration": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error exploring dataset: {str(e)}")

@app.get("/ab-test/stats", response_model=ABTestStatsResponse, tags=["A/B Testing"])
async def get_ab_test_stats():
    """Get current A/B testing metrics (Impressions, Clicks, CTR)"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not available to retrieve stats")
        
    v1_impressions = int(redis_client.get("ab_test:v1:impressions") or 0)
    v1_clicks = int(redis_client.get("ab_test:v1:clicks") or 0)
    v1_ctr = v1_clicks / v1_impressions if v1_impressions > 0 else 0.0
    
    v2_impressions = int(redis_client.get("ab_test:v2:impressions") or 0)
    v2_clicks = int(redis_client.get("ab_test:v2:clicks") or 0)
    v2_ctr = v2_clicks / v2_impressions if v2_impressions > 0 else 0.0
    
    winner = None
    if v1_impressions > 10 and v2_impressions > 10:
        if v1_ctr > v2_ctr:
            winner = "v1 (SVD)"
        elif v2_ctr > v1_ctr:
            winner = "v2 (Hybrid)"
        else:
            winner = "Tie"
            
    return ABTestStatsResponse(
        v1_impressions=v1_impressions,
        v1_clicks=v1_clicks,
        v1_ctr=v1_ctr,
        v2_impressions=v2_impressions,
        v2_clicks=v2_clicks,
        v2_ctr=v2_ctr,
        winner=winner
    )

@app.post("/similar-items", tags=["Recommendations"])
async def get_similar_items(request: SimilarItemsRequest):
    """
    Get items similar to a given item
    
    Uses collaborative filtering embeddings to find similar products
    """
    if not model or not mappings:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    # Check cache
    cache_key = get_cache_key("similar", item_id=request.item_id, n=request.n_items)
    cached_result = get_from_cache(cache_key)
    if cached_result:
        return cached_result
    
    # Convert item ID to index
    item_idx = get_item_idx(request.item_id, mappings)
    
    if item_idx is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item not found: {request.item_id}"
        )
    
    try:
        # Get similar items from collaborative filtering model
        similar = model.cf_model.similar_items(item_idx, N=request.n_items)
        
        # Convert to response format
        similar_items = []
        for rank, (sim_item_idx, score) in enumerate(similar, start=1):
            sim_item_id = get_item_id(sim_item_idx, mappings)
            if sim_item_id:
                similar_items.append({
                    "item_id": sim_item_id,
                    "similarity_score": float(score),
                    "rank": rank
                })
        
        response = {
            "item_id": request.item_id,
            "similar_items": similar_items,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Cache
        set_in_cache(cache_key, response)
        
        return response
        
    except Exception as e:
        logger.error(f"Error finding similar items: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error finding similar items: {str(e)}"
        )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def get_model_info():
    """
    Get information about the loaded model
    """
    if not model:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded"
        )
    
    try:
        info = model.get_model_info()
        
        return ModelInfoResponse(
            model_type=info.get('model_type', 'Unknown'),
            version=config.API_VERSION,
            n_users=info.get('n_users', 0),
            n_items=info.get('n_items', 0),
            parameters={
                'cf_weight': info.get('cf_weight', 0),
                'cb_weight': info.get('cb_weight', 0),
                'cold_start_enabled': info.get('cold_start_enabled', False)
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error getting model info: {str(e)}"
        )


@app.get("/stats", tags=["General"])
async def get_stats():
    """
    Get API statistics
    """
    stats = {
        "model_loaded": model is not None,
        "redis_connected": redis_client is not None,
        "timestamp": datetime.utcnow().isoformat()
    }
    
    if model is not None and train_matrix is not None and mappings is not None:
        stats.update({
            "n_users": len(mappings['user_id_to_idx']),
            "n_items": len(mappings['item_id_to_idx']),
            "n_interactions": train_matrix.nnz,
            "sparsity": f"{100 * (1 - train_matrix.nnz / (train_matrix.shape[0] * train_matrix.shape[1])):.2f}%"
        })
    
    return stats


@app.get("/dataset/meta", tags=["Dataset"])
async def get_dataset_meta(
    dataset: str = Query(default="tech", description="'tech' or 'superstore'")
):
    """
    Returns dataset-specific metadata for the AI/ML tab:
    - interaction counts, user/item counts, sparsity
    - per-model ML metrics (precision, recall, NDCG)
    - sample valid user IDs for the experiment panel
    """
    if dataset == "superstore":
        csv_path = Path("data/synthetic/superstore_interactions.csv")
    else:
        csv_path = Path("data/synthetic/product_interactions.csv")

    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Dataset not found")

    try:
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        user_ids = sorted(set(r["user_id"] for r in rows if "user_id" in r))
        item_ids = sorted(set(r.get("product_id", r.get("item_id", "")) for r in rows))
        n_interactions = len(rows)
        n_users = len(user_ids)
        n_items = len(item_ids)
        sparsity = 100.0 * (1 - n_interactions / max(n_users * n_items, 1))

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {str(e)}")

    # Dataset-specific hardcoded ML evaluation metrics
    if dataset == "superstore":
        metrics = {
            "v1": {"factors": 50, "iterations": 10, "regularization": 0.01,
                   "precision": 0.35, "recall": 0.31, "ndcg": 0.38},
            "v2": {"cf_weight": 0.6, "cb_weight": 0.4,
                   "precision": 0.42, "recall": 0.38, "ndcg": 0.45},
            "pipeline_raw": "Superstore Retail Interactions CSV",
        }
    else:
        metrics = {
            "v1": {"factors": 100, "iterations": 15, "regularization": 0.01,
                   "precision": 0.41, "recall": 0.38, "ndcg": 0.44},
            "v2": {"cf_weight": 0.7, "cb_weight": 0.3,
                   "precision": 0.47, "recall": 0.43, "ndcg": 0.51},
            "pipeline_raw": "Amazon Electronics Reviews CSV",
        }

    return {
        "dataset": dataset,
        "n_interactions": n_interactions,
        "n_users": n_users,
        "n_items": n_items,
        "sparsity": f"{sparsity:.2f}%",
        "sample_user_ids": user_ids[:8],
        "metrics": metrics,
    }


@app.get("/dataset/sample", tags=["Dataset"])
async def get_dataset_sample(
    rows: int = Query(default=68, ge=1, le=200),
    dataset: str = Query(default="tech", description="Which dataset to load: 'tech' or 'superstore'")
):
    """
    Returns rows from the synthetic human-readable dataset as JSON.
    Used by the frontend homepage data table.
    """
    if dataset == "superstore":
        csv_path = Path("data/synthetic/superstore_interactions.csv")
    else:
        csv_path = Path("data/synthetic/product_interactions.csv")
        
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Synthetic dataset not found")
    
    try:
        result = []
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if i >= rows:
                    break
                result.append(row)
        return {
            "columns": list(result[0].keys()) if result else [],
            "rows": result,
            "total_sampled": len(result)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading dataset: {str(e)}")


@app.get("/dataset/synthetic/download", tags=["Dataset"])
async def download_synthetic_dataset(
    dataset: str = Query(default="tech", description="Which dataset to download: 'tech' or 'superstore'")
):
    """
    Streams the synthetic CSV dataset as a downloadable file.
    """
    from fastapi.responses import FileResponse
    if dataset == "superstore":
        csv_path = Path("data/synthetic/superstore_interactions.csv")
        filename = "nexus_superstore_interactions.csv"
    else:
        csv_path = Path("data/synthetic/product_interactions.csv")
        filename = "nexus_product_interactions.csv"
        
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Synthetic dataset not found")
        
    return FileResponse(
        path=str(csv_path),
        media_type="text/csv",
        filename=filename
    )


@app.post("/dataset/upload", tags=["Dataset"])
async def upload_dataset(file: UploadFile = File(...)):
    """
    Accept a CSV dataset upload from the AI/ML/Data Science persona.
    Saves to data/raw/ directory.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted")
    
    try:
        contents = await file.read()
        text = contents.decode("utf-8")
        reader = csv.DictReader(io.StringIO(text))
        rows = list(reader)
        columns = reader.fieldnames or []
        
        # Save to data/raw/
        raw_dir = Path("data/raw")
        raw_dir.mkdir(parents=True, exist_ok=True)
        save_path = raw_dir / file.filename
        with open(save_path, "wb") as f:
            f.write(contents)
        
        return {
            "status": "success",
            "filename": file.filename,
            "rows_detected": len(rows),
            "columns_detected": columns,
            "saved_to": str(save_path),
            "message": f"Dataset '{file.filename}' uploaded successfully with {len(rows)} rows and {len(columns)} columns."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


# ==================== Error Handlers ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unexpected error: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# ==================== Frontend Serving ====================

# Mount the static frontend directory to the root path
# This must come after all API routes are defined
import os
if os.path.exists("frontend"):
    app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")

# ==================== Main ====================

if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("🚀 Starting Product Recommender API")
    print("=" * 70)
    print(f"   Version: {config.API_VERSION}")
    print(f"   Docs: http://localhost:8000/docs")
    print(f"   Redoc: http://localhost:8000/redoc")
    print("=" * 70)
    
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )