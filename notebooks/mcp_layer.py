import os
import logging
import base64
import json
import xgboost as xgb
import pickle
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any, Literal
from math import radians, sin, cos, sqrt, asin

from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# ============================================================================
# PYDANTIC SCHEMAS
# ============================================================================

class ZoningQueryInput(BaseModel):
    query: str = Field(..., description="Natural language question about Orlando zoning laws")
    top_k: int = Field(default=5, ge=1, le=20)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)

class SourceCitation(BaseModel):
    source_document: str
    page_number: Optional[int] = None
    chunk_id: str
    chunk_text: str
    similarity_score: float

class ZoningQueryOutput(BaseModel):
    query: str
    answer: str
    sources: List[SourceCitation]
    chunks_retrieved: int
    total_tokens_used: Optional[int] = None

class PropertyDamageInput(BaseModel):
    image_path: Optional[str] = None
    search_query: Optional[str] = None
    mode: Literal["analyze", "search"] = "analyze"
    top_k: int = Field(default=3, ge=1, le=10)

class PropertyDamageOutput(BaseModel):
    file_name: str
    damage_type: str
    affected_system: Literal["roofing", "plumbing", "hvac", "electrical", "structural", "cosmetic"]
    location: str
    severity: int = Field(..., ge=1, le=10)
    urgency: Literal["immediate", "short-term", "monitor"]
    visible_area_affected: Literal["small", "medium", "large"]
    estimated_repair_cost: str
    secondary_damage_risk: str
    maintenance_summary: str
    similarity_score: Optional[float] = None

class VisionSearchResult(BaseModel):
    query: str
    results: List[PropertyDamageOutput]
    results_count: int

class FMVPredictionInput(BaseModel):
    latitude: float
    longitude: float
    land_sqft: float
    living_area: float
    age: int
    structure_quality: float
    special_features_value: float = 0
    rail_dist: float
    ocean_dist: float
    water_dist: float
    center_dist: float
    subcenter_dist: float
    highway_dist: float
    month_sold: int = 6
    avno60plus: int = 0

class FMVPredictionOutput(BaseModel):
    predicted_fmv: float
    property_summary: Dict[str, Any]

class WalkabilityInput(BaseModel):
    latitude: float
    longitude: float
    radius_miles: float = 1.0

class AmenityDetail(BaseModel):
    name: Optional[str] = None
    distance_miles: float
    latitude: float
    longitude: float

class AmenityCategoryInfo(BaseModel):
    count_within_1mile: int
    nearest: Optional[AmenityDetail] = None

class WalkabilityOutput(BaseModel):
    walkability_score: int
    walkability_interpretation: str
    groceries: AmenityCategoryInfo
    restaurants: AmenityCategoryInfo
    schools: AmenityCategoryInfo
    parks: AmenityCategoryInfo
    hospitals: AmenityCategoryInfo
    transit: AmenityCategoryInfo
    cafes: AmenityCategoryInfo
    summary: str

class MarketExpertInput(BaseModel):
    query: str
    temperature: float = 0.7
    max_tokens: int = 500

class MarketExpertOutput(BaseModel):
    query: str
    response: str
    model_used: str
    tokens_used: int

class FineTunedModelConfig(BaseModel):
    model_id: str
    system_prompt: str
    project: str
    date_trained: str

# ============================================================================
# TOOL CLASSES
# ============================================================================

class ZoningLawTool:
    name: str = "zoning_law_query"
    description: str = "Query the Orlando municipal zoning code knowledge base."

    def __init__(self, index_name="orlando-zoning-index", namespace="__default__"):
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index(index_name)
        self.namespace = namespace
        self.openai_client = OpenAI()
        self.embedding_model = "text-embedding-3-small"
        self.generation_model = "gpt-4o-mini"

    def __call__(self, query: str, top_k: int = 5, similarity_threshold: float = 0.7) -> ZoningQueryOutput:
        # Implementation from your notebook
        query_embedding = self.openai_client.embeddings.create(input=query, model=self.embedding_model).data[0].embedding
        results = self.index.query(vector=query_embedding, top_k=top_k, namespace=self.namespace, include_metadata=True)
        
        chunks = []
        for match in results.matches:
            if match.score >= similarity_threshold:
                chunks.append({"id": match.id, "score": match.score, "metadata": match.metadata, "text": match.metadata.get("text", "")})
        
        # Logic for generating answer (simplified for space, use your notebook logic)
        context = "\n\n".join([f"[Source: {c['metadata'].get('source')}] {c['text']}" for c in chunks])
        response = self.openai_client.chat.completions.create(
            model=self.generation_model,
            messages=[{"role": "system", "content": "You are a zoning expert."}, {"role": "user", "content": f"Context: {context}\nQuestion: {query}"}]
        )
        
        return ZoningQueryOutput(
            query=query,
            answer=response.choices[0].message.content,
            sources=[SourceCitation(source_document=c['metadata'].get('source', 'Unknown'), chunk_id=c['id'], chunk_text=c['text'][:200], similarity_score=c['score']) for c in chunks],
            chunks_retrieved=len(chunks),
            total_tokens_used=response.usage.total_tokens
        )

class VisionTool:
    # ... metadata and schema stay the same ...

    def __init__(self, index_name="orlando-zoning-index", namespace="property-damage"):
        self.openai_client = OpenAI()
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = self.pc.Index(index_name)
        self.namespace = namespace
        self.vision_model = "gpt-4o-mini"

    def _analyze_new_image(self, image_path: Path) -> PropertyDamageOutput:
        """Helper to analyze images using vision model."""
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")
        
        response = self.openai_client.chat.completions.create(
            model=self.vision_model,
            messages=[{
                "role": "user", 
                "content": [
                    {"type": "text", "text": "Analyze property damage."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
                ]
            }],
            response_format={"type": "json_schema", "json_schema": self._damage_schema}
        )
        return PropertyDamageOutput(**json.loads(response.choices[0].message.content))

    def __call__(
        self,
        image_path: Optional[str] = None,
        search_query: Optional[str] = None,
        mode: Literal["analyze", "search"] = "analyze",
        top_k: int = 3
    ):
        if mode == "analyze":
            if not image_path:
                raise ValueError("image_path is required for analyze mode")
            return self._analyze_new_image(Path(image_path))
        
        elif mode == "search":
            if not search_query:
                raise ValueError("search_query is required for search mode")
            
            # 1. Create embedding
            query_embedding = self.openai_client.embeddings.create(
                model="text-embedding-3-small",
                input=search_query
            ).data[0].embedding
            
            # 2. Query Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                namespace=self.namespace,
                include_metadata=True
            )
            
            # 3. Format results
            damage_cases = []
            for match in results.matches:
                damage_cases.append(PropertyDamageOutput(
                    **match.metadata,
                    similarity_score=round(match.score, 3)
                ))
            
            # IMPORTANT: This must be returned!
            return VisionSearchResult(
                query=search_query,
                results=damage_cases,
                results_count=len(damage_cases)
            )
        
        else:
            raise ValueError(f"Invalid mode: {mode}")

class FMVTool:
    name: str = "predict_fair_market_value"

    def __init__(self, model_path='../output/orlando_fmv_xgboost_model_v2.json', feature_path='../output/feature_list_v2.csv', kmeans_path='../output/spatial_kmeans.pkl'):
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)
        self.feature_names = pd.read_csv(feature_path, header=None)[0].tolist()
        with open(kmeans_path, 'rb') as f:
            self.kmeans = pickle.load(f)

    def _engineer_features(self, df):
        # Insert your _engineer_features logic from cell 15 here
        return df # Placeholder

    def __call__(self, **kwargs) -> FMVPredictionOutput:
        df = pd.DataFrame([kwargs])
        # Engineering and prediction logic from notebook
        return FMVPredictionOutput(predicted_fmv=0.0, property_summary=kwargs)

class WalkabilityTool:
    name: str = "assess_walkability"

    def __init__(self, timeout=25):
        self.overpass_url = "http://overpass-api.de/api/interpreter"
        self.timeout = timeout

    def __call__(self, latitude: float, longitude: float, radius_miles: float = 1.0) -> WalkabilityOutput:
        # OSM query logic from notebook
        return WalkabilityOutput(...) # Placeholder

class MarketExpertTool:
    name: str = "orlando_market_expert"

    def __init__(self, config_path="../output/orlando_mkt_data_expert_metadata.json"):
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        self.config = FineTunedModelConfig(**config_data)
        self.openai_client = OpenAI()

    def __call__(self, query: str, temperature=0.7, max_tokens=500) -> MarketExpertOutput:
        response = self.openai_client.chat.completions.create(
            model=self.config.model_id,
            messages=[{"role": "system", "content": self.config.system_prompt}, {"role": "user", "content": query}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return MarketExpertOutput(
            query=query,
            response=response.choices[0].message.content,
            model_used=self.config.model_id,
            tokens_used=response.usage.total_tokens
        )

if __name__ == "__main__":
    print("\n" + "="*80)
    print(" 🛠️  VALIDATING MCP LAYER TOOLS")
    print("="*80)
    try:
        # Initialize one tool as a smoke test
        walk = WalkabilityTool()
        print("✅ WalkabilityTool Class loaded")
        # Add others...
        print("\n🎉 All classes defined successfully.")
    except Exception as e:
        print(f"❌ Initialization failed: {e}")