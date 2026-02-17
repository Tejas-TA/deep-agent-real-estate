# ============================================================================
# tools.py - Orlando Real Estate AI: All MCP Tool Classes
# Auto-extracted from mcp_layer.ipynb (tool cells only, no test code)
# ============================================================================

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
from typing import List, Optional, Dict, Any, Literal, Union
from math import radians, sin, cos, sqrt, asin

from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# SCHEMAS: Zoning
# ============================================================================
class ZoningQueryInput(BaseModel):
    """Input schema for zoning law queries"""
    query: str = Field(
        ...,
        description="Natural language question about Orlando zoning laws",
        examples=["What are the setback requirements for R-1A zones?"]
    )
    top_k: int = Field(default=5, description="Number of relevant document chunks to retrieve", ge=1, le=20)
    similarity_threshold: float = Field(default=0.7, description="Minimum similarity score (0-1)", ge=0.0, le=1.0)

class SourceCitation(BaseModel):
    """Individual source citation with metadata"""
    source_document: str = Field(..., description="Name of the source document")
    page_number: Optional[int] = Field(None, description="Page number in source document")
    chunk_id: str = Field(..., description="Unique identifier for this chunk")
    chunk_text: str = Field(..., description="Relevant text excerpt from the document")
    similarity_score: float = Field(..., description="Cosine similarity score (0-1)")

class ZoningQueryOutput(BaseModel):
    """Output schema for zoning law queries"""
    model_config = ConfigDict(json_schema_extra={
        "example": {
            "query": "What are setback requirements for R-1A?",
            "answer": "For R-1A residential zones, the minimum setback requirements are...",
            "sources": [{"source_document": "orlando_zoning_code.docx", "page_number": 42,
                         "chunk_id": "chunk_0123", "chunk_text": "Setback requirements...", "similarity_score": 0.89}],
            "chunks_retrieved": 5,
            "total_tokens_used": 450
        }
    })
    query: str = Field(..., description="Original query for reference")
    answer: str = Field(..., description="Generated answer based on retrieved zoning law context")
    sources: List[SourceCitation] = Field(..., description="Source citations supporting the answer")
    chunks_retrieved: int = Field(..., description="Number of chunks retrieved from vector store")
    total_tokens_used: Optional[int] = Field(None, description="Total tokens used in generation")

# ============================================================================
# TOOL: ZoningLawTool
# ============================================================================
class ZoningLawTool:
    """
    MCP-compatible tool for querying Orlando zoning law knowledge base.
    Uses RAG to answer questions about Orlando municipal zoning codes.
    """
    name: str = "zoning_law_query"
    description: str = """
    Query the Orlando municipal zoning code knowledge base using natural language.
    Use this tool when you need information about:
    - Zoning classifications (R-1A, R-2, C-1, etc.)
    - Setback requirements and lot dimensions
    - Permitted and conditional uses for zones
    - Density and height restrictions
    - Parking requirements
    - Variance and special exception rules
    - Any other Orlando zoning regulations
    """
    def __init__(
        self,
        pinecone_api_key: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        index_name: str = "orlando-zoning-laws",
        embedding_model: str = "text-embedding-3-small",
        generation_model: str = "gpt-4o-mini",
        namespace: str = "__default__"):
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.pinecone_api_key:
            raise ValueError("Pinecone API key not found. Set PINECONE_API_KEY environment variable.")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index = self.pc.Index(index_name)
        self.namespace = namespace
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.embedding_model = embedding_model
        self.generation_model = generation_model
        logger.info(f"ZoningLawTool initialized with index '{index_name}'")

    def _create_embedding(self, text: str) -> List[float]:
        response = self.openai_client.embeddings.create(model=self.embedding_model, input=text)
        return response.data[0].embedding

    def _retrieve_chunks(self, query: str, top_k: int = 5, similarity_threshold: float = 0.7) -> List[Dict[str, Any]]:
        query_embedding = self._create_embedding(query)
        results = self.index.query(vector=query_embedding, top_k=top_k, namespace=self.namespace, include_metadata=True)
        chunks = [
            {"id": m.id, "score": m.score, "metadata": m.metadata, "text": m.metadata.get("text", "")}
            for m in results.matches if m.score >= similarity_threshold
        ]
        # Adaptive fallback
        if not chunks and similarity_threshold > 0.5:
            logger.warning(f"No chunks at threshold {similarity_threshold}, retrying at 0.5")
            chunks = [
                {"id": m.id, "score": m.score, "metadata": m.metadata, "text": m.metadata.get("text", "")}
                for m in results.matches if m.score >= 0.5]

        logger.info(f"Retrieved {len(chunks)} chunks")
        return chunks

    def _generate_answer(self, query: str, chunks: List[Dict[str, Any]]) -> tuple[str, int]:
        if not chunks:
            return "No relevant information found in the Orlando zoning code database for this query.", 0
        context = "\n\n".join(
            [f"[Source {i}: {c['metadata'].get('source', 'Unknown')}, Page {c['metadata'].get('page', 'N/A')}]\n{c['text']}"
             for i, c in enumerate(chunks, 1)])
        system_prompt = """You are an expert on Orlando, Florida zoning laws and regulations.
        Answer directly based ONLY on the provided context from official Orlando municipal code documents.
        Cite specific sections when available. Do NOT make up information not present in the context."""
        response = self.openai_client.chat.completions.create(
            model=self.generation_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}\n\nAnswer:"}
            ],
            temperature=0.1,
            max_tokens=800
        )
        return response.choices[0].message.content, response.usage.total_tokens
    def __call__(self, query: str, top_k: int = 5, similarity_threshold: float = 0.7) -> ZoningQueryOutput:
        logger.info(f"Zoning query: '{query}'")
        chunks = self._retrieve_chunks(query, top_k, similarity_threshold)
        answer, tokens_used = self._generate_answer(query, chunks)
        sources = [
            SourceCitation(
                source_document=c["metadata"].get("source", "Unknown"),
                page_number=c["metadata"].get("page"),
                chunk_id=c["id"],
                chunk_text=c["text"][:300] + "..." if len(c["text"]) > 300 else c["text"],
                similarity_score=round(c["score"], 3)
            ) for c in chunks
        ]
        return ZoningQueryOutput(
            query=query, answer=answer, sources=sources,
            chunks_retrieved=len(chunks), total_tokens_used=tokens_used
        )
    def get_tool_schema(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "input_schema": ZoningQueryInput.model_json_schema(),
                "output_schema": ZoningQueryOutput.model_json_schema()}
# ============================================================================
# SCHEMAS: Property Damage
# ============================================================================
class PropertyDamageInput(BaseModel):
    """Input schema for property damage assessment"""
    image_path: Optional[str] = Field(None, description="Path to a new property damage image file")
    search_query: Optional[str] = Field(None, description="Text query to search existing damage assessments")
    mode: Literal["analyze", "search"] = Field(default="analyze")
    top_k: int = Field(default=3, ge=1, le=10)
class PropertyDamageOutput(BaseModel):
    """Output schema for property damage assessment"""
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
    """Result for search mode containing multiple similar damage cases"""
    query: str
    results: List[PropertyDamageOutput]
    results_count: int


# ============================================================================
# TOOL: VisionTool
# ============================================================================
class VisionTool:
    """
    MCP-compatible tool for property damage assessment using GPT-4o-mini vision
    and semantic search over existing damage cases.
    """
    name: str = "property_damage_assessment"
    description: str = """
    Analyze property damage images or search existing damage assessments.
    ANALYZE MODE: Provide image_path to analyze a new property damage image.
    SEARCH MODE: Provide search_query to find similar damage cases from database.
    """
    _damage_schema = {
        "name": "property_damage_assessment",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "file_name": {"type": "string"},
                "damage_type": {"type": "string"},
                "affected_system": {"type": "string", "enum": ["roofing", "plumbing", "hvac", "electrical", "structural", "cosmetic"]},
                "location": {"type": "string"},
                "severity": {"type": "integer"},
                "urgency": {"type": "string", "enum": ["immediate", "short-term", "monitor"]},
                "visible_area_affected": {"type": "string", "enum": ["small", "medium", "large"]},
                "estimated_repair_cost": {"type": "string"},
                "secondary_damage_risk": {"type": "string"},
                "maintenance_summary": {"type": "string"}},
            "required": ["file_name", "damage_type", "affected_system", "location", "severity",
            "urgency", "visible_area_affected", "estimated_repair_cost",
                         "secondary_damage_risk", "maintenance_summary"],"additionalProperties": False}}

    def __init__(
        self,
        openai_api_key: Optional[str] = None,
        pinecone_api_key: Optional[str] = None,
        index_name: str = "orlando-zoning-index",
        namespace: str = "property-damage",
        embedding_model: str = "text-embedding-3-small",
        vision_model: str = "gpt-4o-mini",
        max_tokens: int = 500):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found.")
        if not self.pinecone_api_key:
            raise ValueError("Pinecone API key not found.")
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.index = self.pc.Index(index_name)
        self.namespace = namespace
        self.embedding_model = embedding_model
        self.vision_model = vision_model
        self.max_tokens = max_tokens
        logger.info("VisionTool initialized")
    def _encode_image(self, image_path: Path) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    def _create_embedding(self, text: str) -> List[float]:
        response = self.openai_client.embeddings.create(model=self.embedding_model, input=text)
        return response.data[0].embedding
    def _analyze_new_image(self, image_path: Path, file_name: Optional[str] = None) -> PropertyDamageOutput:
        base64_image = self._encode_image(image_path)
        img_file_name = file_name or image_path.name
        response = self.openai_client.chat.completions.create(
            model=self.vision_model,
            messages=[
                {
                    "role": "system",
                    "content": """You are an expert property inspector assessing damage for real estate valuation.
                    Analyze the image and provide a detailed damage assessment.
                    Severity scores: 1-3=minor cosmetic, 4-6=moderate, 7-9=significant, 10=critical."""
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Analyze this property damage image. File name: {img_file_name}"},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "high"}}]}],
            response_format={"type": "json_schema", "json_schema": self._damage_schema},
            max_tokens=self.max_tokens)
        damage_data = json.loads(response.choices[0].message.content)
        return PropertyDamageOutput(**damage_data)
    def _search_similar_damage(self, query: str, top_k: int = 3) -> VisionSearchResult:
        query_embedding = self._create_embedding(query)
        results = self.index.query(vector=query_embedding, top_k=top_k, namespace=self.namespace, include_metadata=True)
        damage_cases = [
            PropertyDamageOutput(**m.metadata, similarity_score=round(m.score, 3))
            for m in results.matches]
        return VisionSearchResult(query=query, results=damage_cases, results_count=len(damage_cases))
    def __call__(
        self,
        image_path: Optional[str] = None,
        search_query: Optional[str] = None,
        mode: Literal["analyze", "search"] = "analyze",
        top_k: int = 3) -> Union[PropertyDamageOutput, VisionSearchResult]:
        logger.info(f"VisionTool mode='{mode}'")
        if mode == "analyze":
            if not image_path:
                raise ValueError("image_path is required for analyze mode")
            img_path = Path(image_path)
            if not img_path.exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
            return self._analyze_new_image(img_path)
        elif mode == "search":
            if not search_query:
                raise ValueError("search_query is required for search mode")
            return self._search_similar_damage(search_query, top_k)
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'analyze' or 'search'")
    def get_tool_schema(self) -> dict:
        return {"name": self.name, "description": self.description,"input_schema": PropertyDamageInput.model_json_schema()}


# ============================================================================
# SCHEMAS: FMV Prediction
# ============================================================================
class FMVPredictionInput(BaseModel):
    """Input schema for FMV prediction"""
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
    month_sold: int = Field(default=6, ge=1, le=12)
    avno60plus: int = 0
class FMVPredictionOutput(BaseModel):
    """Output schema for FMV prediction"""
    predicted_fmv: float = Field(..., description="Predicted fair market value in USD")
    property_summary: Dict[str, Any]
    confidence_interval: Optional[Dict[str, float]] = None

# ============================================================================
# TOOL: FMVTool
# ============================================================================
class FMVTool:
    """MCP-compatible tool for Fair Market Value prediction using XGBoost"""

    name: str = "predict_fair_market_value"
    description: str = """
    Predict the fair market value of Orlando properties using XGBoost model.
    Requires: location (lat/lon), size (land/living area), age, quality, distances to amenities.
    """
    def __init__(
        self,
        model_path: str = "../output/orlando_fmv_xgboost_model_v2.json",
        feature_path: str = "../output/feature_list_v2.csv",
        kmeans_path: str = "../output/spatial_kmeans.pkl"):
        self.model = xgb.XGBRegressor()
        self.model.load_model(model_path)
        self.feature_names = pd.read_csv(feature_path, header=None)[0].tolist()
        with open(kmeans_path, "rb") as f:
            self.kmeans = pickle.load(f)
        logger.info(f"FMVTool initialized with {len(self.feature_names)} features")
    def _engineer_features(self, df: pd.DataFrame, kmeans_model) -> pd.DataFrame:
        """Create derived features — must match training exactly."""
        df = df.copy()
        # Ratio features
        df["lot_utilization"] = df["TOT_LVG_AREA"] / df["LND_SQFOOT"].clip(lower=1)
        df["area_per_age"] = df["TOT_LVG_AREA"] / (df["age"] + 1)
        df["spec_feat_ratio"] = df["SPEC_FEAT_VAL"] / (df["TOT_LVG_AREA"] * 200).clip(lower=1)
        df["land_value_density"] = df["LND_SQFOOT"] / (df["CNTR_DIST"] + 1)
        # Interaction features
        df["quality_x_area"] = df["structure_quality"] * df["TOT_LVG_AREA"]
        df["quality_x_age"] = df["structure_quality"] / (df["age"] + 1)
        # Log-transform distances (drop raw)
        distance_cols = ["RAIL_DIST", "OCEAN_DIST", "WATER_DIST", "CNTR_DIST", "SUBCNTR_DI", "HWY_DIST"]
        for col in distance_cols:
            df[f"{col}_log"] = np.log1p(df[col])
        df = df.drop(columns=distance_cols)
        # Urban access score
        df["urban_access"] = 1 / (df["CNTR_DIST_log"] + df["SUBCNTR_DI_log"])
        df["near_water"] = (df["WATER_DIST_log"] < np.log1p(1000)).astype(int)
        df["hwy_convenience"] = np.where(
            (df["HWY_DIST_log"] > np.log1p(500)) & (df["HWY_DIST_log"] < np.log1p(5000)), 1, 0
        )
        # Cyclical month encoding
        df["month_sin"] = np.sin(2 * np.pi * df["month_sold"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month_sold"] / 12)
        # Location features
        df["dist_from_center"] = np.sqrt((df["LATITUDE"] - 28.5383) ** 2 + (df["LONGITUDE"] - (-81.3792)) ** 2)
        if kmeans_model is not None:
            df["spatial_cluster"] = kmeans_model.predict(df[["LATITUDE", "LONGITUDE"]])
        # Binary / nonlinear
        df["has_spec_feat"] = (df["SPEC_FEAT_VAL"] > 0).astype(int)
        df["quality_sq"] = df["structure_quality"] ** 2
        return df
    def __call__(
        self,
        latitude: float,
        longitude: float,
        land_sqft: float,
        living_area: float,
        SPEC_FEAT_VAL: float,
        structure_quality: float,
        age: int,
        rail_dist: float,
        ocean_dist: float,
        water_dist: float,
        center_dist: float,
        subcenter_dist: float,
        highway_dist: float,
        month_sold: int,
        avno60plus: int) -> FMVPredictionOutput:
        """Predict FMV for a property."""
        property_data = {
            "LATITUDE": latitude,
            "LONGITUDE": longitude,
            "LND_SQFOOT": land_sqft,
            "TOT_LVG_AREA": living_area,
            "age": age,
            "structure_quality": structure_quality,
            "SPEC_FEAT_VAL": SPEC_FEAT_VAL,
            "RAIL_DIST": rail_dist,
            "OCEAN_DIST": ocean_dist,
            "WATER_DIST": water_dist,
            "CNTR_DIST": center_dist,
            "SUBCNTR_DI": subcenter_dist,
            "HWY_DIST": highway_dist,
            "month_sold": month_sold,
            "avno60plus": avno60plus}
        df = pd.DataFrame([property_data])
        df_engineered = self._engineer_features(df, self.kmeans)
        X = df_engineered[self.feature_names]
        pred_log = self.model.predict(X)[0]
        predicted_fmv = float(np.expm1(pred_log))
        return FMVPredictionOutput(predicted_fmv=predicted_fmv, property_summary=property_data)
    def get_tool_schema(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "input_schema": FMVPredictionInput.model_json_schema(),
                "output_schema": FMVPredictionOutput.model_json_schema()}

# ============================================================================
# SCHEMAS: Walkability
# ============================================================================

class AmenityDetail(BaseModel):
    name: Optional[str] = None
    distance_miles: float
    latitude: float
    longitude: float
class AmenityCategoryInfo(BaseModel):
    count_within_1mile: int
    nearest: Optional[AmenityDetail] = None
class WalkabilityInput(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    radius_miles: float = Field(default=1.0, ge=0.1, le=5.0)
class WalkabilityOutput(BaseModel):
    walkability_score: int = Field(..., ge=0, le=100)
    walkability_interpretation: str
    groceries: AmenityCategoryInfo
    restaurants: AmenityCategoryInfo
    schools: AmenityCategoryInfo
    parks: AmenityCategoryInfo
    hospitals: AmenityCategoryInfo
    transit: AmenityCategoryInfo
    cafes: AmenityCategoryInfo
    summary: str

# ============================================================================
# TOOL: WalkabilityTool
# ============================================================================
class WalkabilityTool:
    """MCP-compatible tool for assessing property walkability using OpenStreetMap data."""
    name: str = "assess_walkability"
    description: str = """
    Assess property walkability and nearby amenities using OpenStreetMap data.
    Returns walkability score (0-100) and details about nearby groceries, restaurants,
    schools, parks, hospitals, transit, and cafes.
    """
    def __init__(self, timeout: int = 25):
        self.overpass_url = "http://overpass-api.de/api/interpreter"
        self.timeout = timeout
        logger.info("WalkabilityTool initialized")
    def _haversine_distance(self, lat1, lon1, lat2, lon2) -> float:
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
        dlat, dlon = lat2 - lat1, lon2 - lon1
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        return 3956 * 2 * asin(sqrt(a))
    def _query_osm_amenities(self, lat: float, lon: float, radius_meters: int = 1600) -> dict:
        query = f"""
        [out:json][timeout:{self.timeout}];
        (
        node["amenity"](around:{radius_meters},{lat},{lon});
        way["amenity"](around:{radius_meters},{lat},{lon});
        node["shop"="supermarket"](around:{radius_meters},{lat},{lon});
        way["shop"="supermarket"](around:{radius_meters},{lat},{lon});
        node["shop"="convenience"](around:{radius_meters},{lat},{lon});
        way["shop"="convenience"](around:{radius_meters},{lat},{lon});
        node["leisure"](around:{radius_meters},{lat},{lon});
        way["leisure"](around:{radius_meters},{lat},{lon});
        node["highway"="bus_stop"](around:{radius_meters},{lat},{lon});
        );
        out center;
        """
        response = requests.post(self.overpass_url, data={"data": query}, timeout=self.timeout)
        response.raise_for_status()
        return response.json()
    def _get_element_location(self, element: dict) -> Optional[tuple]:
        if element["type"] == "node":
            return (element.get("lat"), element.get("lon"))
        elif element["type"] == "way" and "center" in element:
            return (element["center"].get("lat"), element["center"].get("lon"))
        return None
    def _process_amenities(self, raw_data: dict, property_lat: float, property_lon: float) -> dict:
        elements = raw_data.get("elements", [])
        categories = {k: [] for k in ["groceries", "restaurants", "schools", "parks", "hospitals", "transit", "cafes"]}
        for element in elements:
            tags = element.get("tags", {})
            location = self._get_element_location(element)
            if not location:
                continue
            elem_lat, elem_lon = location
            distance = self._haversine_distance(property_lat, property_lon, elem_lat, elem_lon)
            info = {"name": tags.get("name"), "distance_miles": round(distance, 2), "latitude": elem_lat, "longitude": elem_lon}
            amenity = tags.get("amenity")
            shop = tags.get("shop")
            leisure = tags.get("leisure")
            highway = tags.get("highway")
            if shop in ["supermarket", "convenience"] or amenity == "marketplace":
                categories["groceries"].append(info)
            elif amenity in ["restaurant", "fast_food", "food_court"]:
                categories["restaurants"].append(info)
            elif amenity in ["cafe", "bar", "pub"]:
                categories["cafes"].append(info)
            elif amenity in ["school", "kindergarten", "college", "university"]:
                categories["schools"].append(info)
            elif leisure in ["park", "garden", "playground"] or amenity == "park":
                categories["parks"].append(info)
            elif amenity in ["hospital", "clinic", "doctors", "pharmacy"]:
                categories["hospitals"].append(info)
            elif highway == "bus_stop" or amenity in ["bus_station", "ferry_terminal"]:
                categories["transit"].append(info)
        result = {}
        for cat, amenities in categories.items():
            amenities.sort(key=lambda x: x["distance_miles"])
            result[cat] = {
                "count_within_1mile": len([a for a in amenities if a["distance_miles"] <= 1.0]),
                "nearest": amenities[0] if amenities else None
            }
        return result
    def _calculate_walkability_score(self, amenities_data: dict) -> int:
        score = 0
        nearest_grocery = amenities_data["groceries"]["nearest"]
        if nearest_grocery:
            d = nearest_grocery["distance_miles"]
            score += 25 if d <= 0.25 else 20 if d <= 0.5 else 15 if d <= 0.75 else 10 if d <= 1.0 else 0
        dining_total = amenities_data["restaurants"]["count_within_1mile"] + amenities_data["cafes"]["count_within_1mile"]
        score += 25 if dining_total >= 20 else 20 if dining_total >= 10 else 15 if dining_total >= 5 else 10 if dining_total >= 2 else 0
        nearest_park = amenities_data["parks"]["nearest"]
        if nearest_park:
            d = nearest_park["distance_miles"]
            score += 20 if d <= 0.5 else 15 if d <= 0.75 else 10 if d <= 1.0 else 0
        nearest_transit = amenities_data["transit"]["nearest"]
        if nearest_transit:
            d = nearest_transit["distance_miles"]
            score += 15 if d <= 0.25 else 10 if d <= 0.5 else 5 if d <= 0.75 else 0
        school_count = amenities_data["schools"]["count_within_1mile"]
        score += 15 if school_count >= 3 else 10 if school_count >= 2 else 5 if school_count >= 1 else 0
        return min(score, 100)

    def _generate_summary(self, walkability_score: int, amenities_data: dict) -> str:
        parts = []
        if walkability_score >= 90:
            parts.append("This property has excellent walkability.")
        elif walkability_score >= 70:
            parts.append("This property is very walkable.")
        elif walkability_score >= 50:
            parts.append("This property is somewhat walkable.")
        else:
            parts.append("This property is car-dependent.")
        gc = amenities_data["groceries"]["count_within_1mile"]
        if gc > 0:
            parts.append(f"{gc} grocery store(s) within 1 mile.")
        dining = amenities_data["restaurants"]["count_within_1mile"] + amenities_data["cafes"]["count_within_1mile"]
        if dining > 0:
            parts.append(f"{dining} dining option(s) nearby.")
        pc = amenities_data["parks"]["count_within_1mile"]
        if pc > 0:
            parts.append(f"{pc} park(s) for recreation.")
        return " ".join(parts)

    def __call__(self, latitude: float, longitude: float, radius_miles: float = 1.0) -> WalkabilityOutput:
        logger.info(f"Walkability check: ({latitude}, {longitude})")
        radius_meters = int(radius_miles * 1609.34)
        raw_data = self._query_osm_amenities(latitude, longitude, radius_meters)
        processed = self._process_amenities(raw_data, latitude, longitude)
        score = self._calculate_walkability_score(processed)
        interpretation = (
            "Walker's Paradise" if score >= 90 else
            "Very Walkable" if score >= 70 else
            "Somewhat Walkable" if score >= 50 else
            "Car-Dependent" if score >= 25 else
            "Very Car-Dependent")
        return WalkabilityOutput(
            walkability_score=score,
            walkability_interpretation=interpretation,
            groceries=AmenityCategoryInfo(**processed["groceries"]),
            restaurants=AmenityCategoryInfo(**processed["restaurants"]),
            schools=AmenityCategoryInfo(**processed["schools"]),
            parks=AmenityCategoryInfo(**processed["parks"]),
            hospitals=AmenityCategoryInfo(**processed["hospitals"]),
            transit=AmenityCategoryInfo(**processed["transit"]),
            cafes=AmenityCategoryInfo(**processed["cafes"]),
            summary=self._generate_summary(score, processed)
        )
    def get_tool_schema(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "input_schema": WalkabilityInput.model_json_schema(),
                "output_schema": WalkabilityOutput.model_json_schema()}
# ============================================================================
# SCHEMAS: Market Expert
# ============================================================================

class MarketExpertInput(BaseModel):
    query: str = Field(..., description="Question about Orlando real estate market")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: int = Field(default=500, ge=50, le=2000)
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
# TOOL: MarketExpertTool
# ============================================================================

class MarketExpertTool:
    """MCP-compatible tool for querying fine-tuned Orlando real estate market expert."""
    name: str = "orlando_market_expert"
    description: str = """
    Query a fine-tuned AI expert on Orlando real estate market insights.
    Trained on Orlando-specific neighborhood data, market trends, pricing, and investment opportunities.
    Use for: market analysis, neighborhood recommendations, investment advice, local market knowledge.
    """
    def __init__(
        self,
        model_config_path: str = "../output/orlando_mkt_data_expert_metadata.json",
        openai_api_key: Optional[str] = None):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key not found.")
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        with open(model_config_path, "r") as f:
            config_data = json.load(f)
        self.model_config = FineTunedModelConfig(**config_data)
        logger.info(f"MarketExpertTool initialized with model: {self.model_config.model_id}")

    def __call__(self, query: str, temperature: float = 0.7, max_tokens: int = 500) -> MarketExpertOutput:
        logger.info(f"Market expert query: '{query[:50]}...'")
        response = self.openai_client.chat.completions.create(
            model=self.model_config.model_id,
            messages=[{"role": "system", "content": self.model_config.system_prompt},{"role": "user", "content": query}],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return MarketExpertOutput(
            query=query,
            response=response.choices[0].message.content,
            model_used=self.model_config.model_id,
            tokens_used=response.usage.total_tokens
        )
    def get_tool_schema(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description,
        "input_schema": MarketExpertInput.model_json_schema(),
        "output_schema": MarketExpertOutput.model_json_schema()}


# ============================================================================
# CONVENIENCE: Initialize all tools at once
# ============================================================================

def load_all_tools(
    fmv_model_path: str = "../output/orlando_fmv_xgboost_model_v2.json",
    fmv_feature_path: str = "../output/feature_list_v2.csv",
    fmv_kmeans_path: str = "../output/spatial_kmeans.pkl",
    market_expert_config_path: str = "../output/orlando_mkt_data_expert_metadata.json",
) -> dict:
    """
    Initialize and return all tools as a dict keyed by tool name.
    Usage:
        from tools import load_all_tools
        tools = load_all_tools()
        result = tools["zoning_law_query"](query="What are R-1A setbacks?")
    """
    return {
        "zoning_law_query": ZoningLawTool(),
        "property_damage_assessment": VisionTool(),
        "predict_fair_market_value": FMVTool(fmv_model_path, fmv_feature_path, fmv_kmeans_path),
        "assess_walkability": WalkabilityTool(),
        "orlando_market_expert": MarketExpertTool(market_expert_config_path),
    }