"""
Orlando Real Estate MCP Server
Exposes all 5 tools via FastMCP over stdio transport.
Run: python mcp_server.py
"""
import logging
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================================
# LOAD TOOLS AT MODULE LEVEL — once on startup, shared across all requests
# ============================================================================

logger.info("Loading Orlando Real Estate tools...")

from tools import ZoningLawTool, VisionTool, FMVTool, WalkabilityTool, MarketExpertTool

_zoning  = ZoningLawTool(index_name="orlando-zoning-index", namespace="__default__")
_vision  = VisionTool(index_name="orlando-zoning-index", namespace="property-damage")
_fmv     = FMVTool()
_walk    = WalkabilityTool()
_expert  = MarketExpertTool()

logger.info("All tools loaded successfully.")

mcp = FastMCP("orlando_realestate_mcp")

# ============================================================================
# TOOL 1: Zoning Law Query
# ============================================================================

class ZoningInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(..., description="Natural language question about Orlando zoning laws", min_length=3)
    top_k: int = Field(default=5, ge=1, le=20, description="Number of document chunks to retrieve")
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum similarity score (0-1)")
@mcp.tool(
    name="zoning_law_query",
    annotations={
        "title": "Orlando Zoning Law Query",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,})
async def zoning_law_query(params: ZoningInput) -> str:
    """Query the Orlando municipal zoning code knowledge base using RAG.
    Use this tool when you need information about:
    - Zoning classifications (R-1A, R-2, C-1, etc.)
    - Setback requirements and lot dimensions
    - Permitted and conditional uses for zones
    - Density and height restrictions
    - Parking requirements
    - Tree and landscaping permit requirements
    - Variance and special exception rules
    Args:
        params: query (str), top_k (int), similarity_threshold (float)
    Returns:
        str: JSON with answer and source citations
    """
    result = _zoning(
        query=params.query,
        top_k=params.top_k,
        similarity_threshold=params.similarity_threshold,
    )
    return result.model_dump_json(indent=2)

# ============================================================================
# TOOL 2: Property Damage Assessment
# ============================================================================

class VisionInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    mode: str = Field(default="search", description="'analyze' for new image, 'search' for similar cases")
    image_path: Optional[str] = Field(default=None, description="Path to image file (analyze mode only)")
    search_query: Optional[str] = Field(default=None, description="Text description of damage to search for (search mode)")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of similar cases to return (search mode)")


@mcp.tool(
    name="property_damage_assessment",
    annotations={
        "title": "Property Damage Assessment",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,})
async def property_damage_assessment(params: VisionInput) -> str:
    """Analyze property damage images or search existing damage assessments.

    ANALYZE MODE: Pass image_path to analyze a new image via GPT-4o-mini vision.
    SEARCH MODE: Pass search_query to find similar damage cases from vector database.

    Returns damage type, severity (1-10), urgency, affected system,
    estimated repair cost, and secondary risk assessment.

    Args:
        params: mode, image_path or search_query, top_k
    Returns:
        str: JSON with damage assessment details
    """
    result = _vision(
        image_path=params.image_path,
        search_query=params.search_query,
        mode=params.mode,
        top_k=params.top_k,
    )
    return result.model_dump_json(indent=2)

# ============================================================================
# TOOL 3: Fair Market Value Prediction
# ============================================================================

class FMVInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    latitude: float = Field(..., description="Property latitude (e.g. 28.55)")
    longitude: float = Field(..., description="Property longitude (e.g. -81.35)")
    land_sqft: float = Field(..., gt=0, description="Land area in square feet")
    living_area: float = Field(..., gt=0, description="Living area in square feet")
    age: int = Field(..., ge=0, description="Age of property in years")
    structure_quality: float = Field(..., ge=1.0, le=5.0, description="Quality score 1-5")
    SPEC_FEAT_VAL: float = Field(default=0.0, ge=0, description="Special features value in dollars")
    rail_dist: float = Field(..., ge=0, description="Distance to nearest rail (meters)")
    ocean_dist: float = Field(..., ge=0, description="Distance to ocean (meters)")
    water_dist: float = Field(..., ge=0, description="Distance to nearest water body (meters)")
    center_dist: float = Field(..., ge=0, description="Distance to city center (meters)")
    subcenter_dist: float = Field(..., ge=0, description="Distance to subcenter (meters)")
    highway_dist: float = Field(..., ge=0, description="Distance to nearest highway (meters)")
    month_sold: int = Field(default=6, ge=1, le=12, description="Month of sale (1-12)")
    avno60plus: int = Field(default=0, ge=0, le=1, description="AVN060+ indicator (0 or 1)")

@mcp.tool(
    name="predict_fair_market_value",
    annotations={
        "title": "Predict Fair Market Value",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,})

async def predict_fair_market_value(params: FMVInput) -> str:
    """Predict the fair market value of an Orlando property using XGBoost model.
    Model trained on Orange County property data with spatial clustering,
    log-transformed distances, and cyclical month encoding.
    Args:
        params: Full property characteristics and distances
    Returns:
        str: JSON with predicted_fmv (USD) and property_summary
    """
    result = _fmv(**params.model_dump())
    return result.model_dump_json(indent=2)

# ============================================================================
# TOOL 4: Walkability Assessment
# ============================================================================

class WalkabilityInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    latitude: float = Field(..., ge=-90, le=90, description="Property latitude")
    longitude: float = Field(..., ge=-180, le=180, description="Property longitude")
    radius_miles: float = Field(default=1.0, ge=0.1, le=5.0, description="Search radius in miles")


@mcp.tool(
    name="assess_walkability",
    annotations={
        "title": "Assess Property Walkability",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,})
async def assess_walkability(params: WalkabilityInput) -> str:
    """Assess walkability score and nearby amenities for a property location.

    Queries OpenStreetMap Overpass API for nearby groceries, restaurants,
    cafes, schools, parks, hospitals, and public transit.

    Returns walkability score (0-100) with interpretation and per-category
    counts and nearest amenity details.

    Args:
        params: latitude, longitude, radius_miles

    Returns:
        str: JSON with walkability_score, interpretation, and category breakdowns
    """
    result = _walk(
        latitude=params.latitude,
        longitude=params.longitude,
        radius_miles=params.radius_miles,
    )
    return result.model_dump_json(indent=2)


# ============================================================================
# TOOL 5: Orlando Market Expert
# ============================================================================

class ExpertInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str = Field(..., min_length=5, description="Question about Orlando real estate market")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0, description="Response creativity (0=focused, 1=creative)")
    max_tokens: int = Field(default=500, ge=50, le=2000, description="Maximum response length")


@mcp.tool(
    name="orlando_market_expert",
    annotations={
        "title": "Orlando Market Expert",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    }
)
async def orlando_market_expert(params: ExpertInput) -> str:
    """Query a fine-tuned AI expert on Orlando real estate market insights.

    Trained on Orlando-specific neighborhood data, pricing trends, school
    districts, investment opportunities, and local market knowledge.

    Use for: neighborhood recommendations, market trend analysis, buyer/seller guidance, investment evaluation.

    Args:
        params: query, temperature, max_tokens

    Returns:
        str: JSON with expert response and token usage
    """
    result = _expert(
        query=params.query,
        temperature=params.temperature,
        max_tokens=params.max_tokens,
    )
    return result.model_dump_json(indent=2)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    mcp.run()  # stdio transport
