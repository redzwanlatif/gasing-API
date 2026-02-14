from __future__ import annotations

import logging
import os
import random
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from models.behaviour_classifier import BehaviourClassifier
from models.llm_generator import LLMGenerator
from models.market_analyzer import MarketAnalyzer
from services.deriv_client import DerivClient

load_dotenv()
logger = logging.getLogger(__name__)

# Deriv client (initialized in lifespan)
deriv_client = DerivClient(app_id=os.getenv("DERIV_APP_ID", "1089"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: connect to Deriv
    try:
        await deriv_client.connect()
        logger.info("Deriv WebSocket client connected")
    except Exception as e:
        logger.warning("Deriv connection failed on startup (will retry): %s", e)
    yield
    # Shutdown: disconnect
    await deriv_client.disconnect()
    logger.info("Deriv WebSocket client disconnected")


app = FastAPI(title="Spirit Trader API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

market_analyzer = MarketAnalyzer()
behaviour_classifier = BehaviourClassifier()
llm_generator = LLMGenerator(api_key=os.getenv("OPENAI_API_KEY"))


class Candle(BaseModel):
    open: float
    high: float
    low: float
    close: float
    timestamp: int
    volume: Optional[float] = None


class MarketAnalysisRequest(BaseModel):
    symbol: str
    candles: List[Candle]
    timeframe: str = "1h"


class Decision(BaseModel):
    action: str
    size: float
    timestamp: int
    price: Optional[float] = None
    result: Optional[str] = None
    pnl: Optional[float] = None


class BehaviourRequest(BaseModel):
    decisions: List[Decision]
    trader_class: str = "pemburu"
    current_pnl: float = 0.0


class CoachingRequest(BaseModel):
    behaviour_state: str
    behaviour_confidence: float
    market_analysis: Dict[str, Any]
    trader_class: str
    trader_name: str
    context: str = "general"


class QuestRequest(BaseModel):
    market_analysis: Dict[str, Any]
    trader_level: int
    trader_class: str
    quest_type: str = "pattern_recognition"


class ShareRequest(BaseModel):
    achievement_type: str
    boss_name: Optional[str] = None
    trader_name: str
    trader_class: str
    stats: Dict[str, Any]


class ProgressReportRequest(BaseModel):
    decisions: List[Decision]
    trader_class: str = "pemburu"
    session_history: Optional[List[Dict[str, Any]]] = None


class LosingTrade(BaseModel):
    action: str
    size: float
    timestamp: int
    price: float
    result: str = "loss"
    pnl: float
    symbol: str
    candles: List[Candle]
    timeframe: str = "1h"


class ExplainLossRequest(BaseModel):
    trade: LosingTrade
    previous_decisions: List[Decision]
    trader_class: str = "pemburu"
    trader_name: str = "Trader"


@app.get("/")
def health_check() -> dict[str, str]:
    return {"status": "alive", "service": "Spirit Trader API"}


@app.post("/analyse-market")
def analyse_market(request: MarketAnalysisRequest) -> dict[str, Any]:
    try:
        candles = [c.model_dump() for c in request.candles]
        result = market_analyzer.analyze(candles, request.timeframe)
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/classify-behaviour")
def classify_behaviour(request: BehaviourRequest) -> dict[str, Any]:
    try:
        decisions = [d.model_dump() for d in request.decisions]
        result = behaviour_classifier.classify(decisions, request.trader_class, request.current_pnl)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-coaching")
async def generate_coaching(request: CoachingRequest) -> dict[str, Any]:
    try:
        result = await llm_generator.generate_coaching(
            request.behaviour_state,
            request.behaviour_confidence,
            request.market_analysis,
            request.trader_class,
            request.trader_name,
            request.context,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-quest")
async def generate_quest(request: QuestRequest) -> dict[str, Any]:
    try:
        result = await llm_generator.generate_quest(
            request.market_analysis,
            request.trader_level,
            request.trader_class,
            request.quest_type,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate-share")
async def generate_share(request: ShareRequest) -> dict[str, Any]:
    try:
        result = await llm_generator.generate_share(
            request.achievement_type,
            request.boss_name,
            request.trader_name,
            request.trader_class,
            request.stats,
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/progress-report")
def progress_report(request: ProgressReportRequest) -> dict[str, Any]:
    try:
        decisions = [d.model_dump() for d in request.decisions]
        result = _calculate_progress_metrics(decisions, request.session_history or [])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/explain-trade")
async def explain_trade(request: ExplainLossRequest) -> dict[str, Any]:
    try:
        # Analyze market conditions at time of loss
        candles = [c.model_dump() for c in request.trade.candles]
        market_analysis = market_analyzer.analyze(candles, request.trade.timeframe)

        # Analyze behaviour leading to the loss
        decisions = [d.model_dump() for d in request.previous_decisions]
        behaviour = behaviour_classifier.classify(
            decisions, request.trader_class, request.trade.pnl
        )

        # Generate explanation
        result = await llm_generator.generate_trade_explanation(
            trade={
                "action": request.trade.action,
                "size": request.trade.size,
                "price": request.trade.price,
                "pnl": request.trade.pnl,
                "symbol": request.trade.symbol,
            },
            market_analysis=market_analysis,
            behaviour_analysis=behaviour,
            trader_class=request.trader_class,
            trader_name=request.trader_name,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- Live Market Data Endpoints (Deriv API) ---

NARRATIVE_TEMPLATES = {
    "bullish": [
        "The trade winds blow strong from the east — prices surge like monsoon tides.",
        "Merchants rush to the docks. The spice prices climb with the morning sun.",
        "The harbour bustles. Every ship that arrives carries fortune on its bow.",
    ],
    "bearish": [
        "Dark clouds gather over the straits. Traders whisper of falling prices.",
        "The docks grow quiet. Prices sink like stones in the harbour waters.",
        "A cold wind sweeps the market. The bold retreat, the wise wait.",
    ],
    "neutral": [
        "The waters are calm today. Prices drift like boats without wind.",
        "Neither storm nor sun — the market holds its breath, waiting.",
        "The harbour hums with quiet trade. No great fortune nor ruin today.",
    ],
    "overbought": [
        "Greed fills the air like incense smoke — prices have climbed too high, too fast.",
        "The merchants celebrate, but Seri warns: what rises without rest must fall.",
    ],
    "oversold": [
        "Fear grips the harbour. Prices have fallen far — perhaps too far.",
        "The docks are empty, but opportunity hides in the shadows of despair.",
    ],
}


def _build_narrative(analysis: dict[str, Any], tick: dict[str, Any] | None) -> str:
    """Convert technical analysis into game-world narrative."""
    trend = analysis.get("trend", "neutral")
    rsi_signal = analysis.get("indicators", {}).get("rsi_signal", "neutral")

    # Pick narrative based on RSI extremes first, then trend
    if rsi_signal == "overbought":
        pool = NARRATIVE_TEMPLATES["overbought"]
    elif rsi_signal == "oversold":
        pool = NARRATIVE_TEMPLATES["oversold"]
    else:
        pool = NARRATIVE_TEMPLATES.get(trend, NARRATIVE_TEMPLATES["neutral"])

    narrative = random.choice(pool)

    # Add price context
    if tick:
        price = tick.get("quote", 0)
        narrative += f"\n\nCurrent price: {price:.2f} gold pieces."

    return narrative


@app.get("/market/symbols")
async def get_market_symbols() -> dict[str, Any]:
    """Get available Deriv trading symbols."""
    try:
        symbols = await deriv_client.get_symbols()
        return {"symbols": symbols, "source": "deriv_live" if deriv_client.connected else "fallback"}
    except Exception as e:
        logger.warning("Failed to get symbols: %s", e)
        from services.deriv_client import DEFAULT_SYMBOLS
        return {"symbols": DEFAULT_SYMBOLS, "source": "fallback"}


@app.get("/market/candles/{symbol}")
async def get_market_candles(
    symbol: str,
    count: int = Query(default=100, ge=5, le=5000),
    granularity: int = Query(default=60, ge=60),
    analyse: bool = Query(default=False),
) -> dict[str, Any]:
    """Get live OHLC candles from Deriv, optionally with analysis."""
    if not deriv_client.connected:
        raise HTTPException(status_code=503, detail="Deriv connection unavailable")
    try:
        candles = await deriv_client.get_candles(symbol, count, granularity)
        result: dict[str, Any] = {
            "symbol": symbol,
            "candles": candles,
            "count": len(candles),
            "granularity": granularity,
            "source": "deriv_live",
        }
        if analyse and len(candles) >= 5:
            analysis = market_analyzer.analyze(candles, f"{granularity}s")
            result["analysis"] = analysis
        return result
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Deriv request timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/market/tick/{symbol}")
async def get_market_tick(symbol: str) -> dict[str, Any]:
    """Get the latest live tick for a symbol."""
    if not deriv_client.connected:
        raise HTTPException(status_code=503, detail="Deriv connection unavailable")
    try:
        tick = await deriv_client.get_tick(symbol)
        if tick is None:
            raise HTTPException(status_code=404, detail=f"No tick data for {symbol}")
        return {"tick": tick, "source": "deriv_live"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/market/snapshot/{symbol}")
async def get_market_snapshot(
    symbol: str,
    count: int = Query(default=50, ge=5, le=500),
    granularity: int = Query(default=60, ge=60),
) -> dict[str, Any]:
    """Full market snapshot: tick + candles + analysis + game narrative.

    This is the key endpoint for the live trading challenge scene.
    """
    if not deriv_client.connected:
        raise HTTPException(status_code=503, detail="Deriv connection unavailable")
    try:
        candles = await deriv_client.get_candles(symbol, count, granularity)
        tick = await deriv_client.get_tick(symbol)

        analysis = {}
        if len(candles) >= 5:
            analysis = market_analyzer.analyze(candles, f"{granularity}s")

        narrative = _build_narrative(analysis, tick)

        return {
            "symbol": symbol,
            "tick": tick,
            "candles": candles[-20:],  # Last 20 for display
            "analysis": analysis,
            "narrative": narrative,
            "source": "deriv_live",
        }
    except TimeoutError:
        raise HTTPException(status_code=504, detail="Deriv request timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _calculate_progress_metrics(
    decisions: list[dict[str, Any]], session_history: list[dict[str, Any]]
) -> dict[str, Any]:
    """Calculate learning progress and mastery scores from trading history."""
    if not decisions:
        return {
            "discipline_score": 50,
            "revenge_risk_score": 0,
            "fomo_risk_score": 0,
            "overtrading_risk_score": 0,
            "improvement_trend": "neutral",
            "strongest_bad_habit": "none",
            "strongest_good_habit": "none",
        }

    # Calculate feature metrics
    features = _extract_progress_features(decisions)

    # Calculate penalties
    revenge_penalty = min(40, features["revenge_frequency"] * 20)
    fomo_penalty = min(30, features["fomo_frequency"] * 15)
    overtrade_penalty = min(30, features["overtrade_frequency"] * 10)

    # Discipline score (0-100): starts at 100, subtract penalties
    discipline_score = 100 - revenge_penalty - fomo_penalty - overtrade_penalty
    discipline_score = max(0, min(100, discipline_score))

    # Risk scores (0-100): how often each bad habit appears
    total_trades = features["total_trades"]
    revenge_risk_score = min(100, int(features["revenge_frequency"] * 100 / max(1, total_trades)))
    fomo_risk_score = min(100, int(features["fomo_frequency"] * 100 / max(1, total_trades)))
    overtrading_risk_score = min(100, int(features["overtrade_frequency"] * 100 / max(1, total_trades)))

    # Improvement trend
    improvement_trend = _calculate_improvement_trend(decisions, session_history)

    # Strongest habits
    strongest_bad_habit = "none"
    strongest_good_habit = "none"

    risk_scores = {
        "revenge_trading": revenge_risk_score,
        "fomo": fomo_risk_score,
        "overtrading": overtrading_risk_score,
    }
    if max(risk_scores.values()) > 0:
        strongest_bad_habit = max(risk_scores, key=risk_scores.get)

    good_habits = {
        "consistent_sizing": features["consistent_sizing_score"],
        "patient_entries": features["patient_entries_score"],
        "accepting_losses": features["accepting_losses_score"],
    }
    if max(good_habits.values()) > 0:
        strongest_good_habit = max(good_habits, key=good_habits.get)

    return {
        "discipline_score": round(discipline_score),
        "revenge_risk_score": revenge_risk_score,
        "fomo_risk_score": fomo_risk_score,
        "overtrading_risk_score": overtrading_risk_score,
        "improvement_trend": improvement_trend,
        "strongest_bad_habit": strongest_bad_habit,
        "strongest_good_habit": strongest_good_habit,
        "score_breakdown": {
            "revenge_penalty": -revenge_penalty,
            "fomo_penalty": -fomo_penalty,
            "overtrade_penalty": -overtrade_penalty,
        },
        "history": _calculate_history_scores(session_history, decisions),
    }


def _extract_progress_features(decisions: list[dict[str, Any]]) -> dict[str, Any]:
    """Extract behavioral features for progress tracking."""
    if not decisions:
        return {
            "total_trades": 0,
            "revenge_frequency": 0,
            "fomo_frequency": 0,
            "overtrade_frequency": 0,
            "consistent_sizing_score": 0,
            "patient_entries_score": 0,
            "accepting_losses_score": 0,
        }

    total = len(decisions)
    sizes = [float(d.get("size", 0)) for d in decisions]
    results = [d.get("result") for d in decisions]
    timestamps = [int(d.get("timestamp", 0)) for d in decisions]

    # Count revenge trading (size increase after loss)
    revenge_count = 0
    for i in range(1, len(decisions)):
        if results[i - 1] == "loss":
            prev_size = sizes[i - 1] if sizes[i - 1] > 0 else 1.0
            if sizes[i] / prev_size > 1.5:
                revenge_count += 1

    # Count FOMO (rapid entries after losses with same direction)
    fomo_count = 0
    for i in range(1, len(decisions)):
        if results[i - 1] == "loss":
            time_diff = timestamps[i] - timestamps[i - 1]
            if time_diff < 300:  # Less than 5 minutes
                fomo_count += 1

    # Count overtrading (high frequency)
    intervals = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
    avg_interval = sum(intervals) / len(intervals) if intervals else 120
    overtrade_count = sum(1 for iv in intervals if iv < 60)

    # Consistent sizing (low variance in position sizes after wins)
    wins = [sizes[i] for i in range(len(decisions)) if results[i] == "win"]
    consistent_sizing = 50
    if len(wins) > 1:
        mean_size = sum(wins) / len(wins)
        variance = sum((s - mean_size) ** 2 for s in wins) / len(wins)
        cv = (variance ** 0.5) / mean_size if mean_size > 0 else 1
        consistent_sizing = max(0, min(100, int(100 - cv * 50)))

    # Patient entries (waiting after losses)
    patient_entries = 50
    if intervals:
        patient_count = sum(1 for iv in intervals if iv > 120)
        patient_entries = int(patient_count * 100 / len(intervals))

    # Accepting losses (no re-entry after loss within 5 min)
    loss_count = sum(1 for r in results if r == "loss")
    chase_count = 0
    for i in range(1, len(decisions)):
        if results[i - 1] == "loss" and (timestamps[i] - timestamps[i - 1]) < 300:
            chase_count += 1
    accepting_losses = 100 - int(chase_count * 100 / max(1, loss_count)) if loss_count > 0 else 50

    return {
        "total_trades": total,
        "revenge_frequency": revenge_count,
        "fomo_frequency": fomo_count,
        "overtrade_frequency": overtrade_count,
        "consistent_sizing_score": consistent_sizing,
        "patient_entries_score": patient_entries,
        "accepting_losses_score": accepting_losses,
    }


def _calculate_history_scores(
    session_history: list[dict[str, Any]], current_decisions: list[dict[str, Any]]
) -> list[int]:
    """Calculate discipline scores for time-series visualization.
    
    Returns array of scores: [previous_session_1, previous_session_2, ..., current]
    """
    history: list[int] = []
    
    # Calculate scores for each historical session
    for session in session_history[-2:]:  # Last 2 sessions max
        if isinstance(session, dict) and "decisions" in session:
            session_decisions = session["decisions"]
            if isinstance(session_decisions, list) and len(session_decisions) > 0:
                features = _extract_progress_features(session_decisions)
                score = 100
                score -= min(40, features["revenge_frequency"] * 20)
                score -= min(30, features["fomo_frequency"] * 15)
                score -= min(30, features["overtrade_frequency"] * 10)
                history.append(max(0, min(100, round(score))))
    
    # Calculate current session score
    current_features = _extract_progress_features(current_decisions)
    current_score = 100
    current_score -= min(40, current_features["revenge_frequency"] * 20)
    current_score -= min(30, current_features["fomo_frequency"] * 15)
    current_score -= min(30, current_features["overtrade_frequency"] * 10)
    history.append(max(0, min(100, round(current_score))))
    
    return history


def _calculate_improvement_trend(
    decisions: list[dict[str, Any]], session_history: list[dict[str, Any]]
) -> str:
    """Determine if trader is improving based on session-to-session comparison."""
    if not decisions or len(decisions) < 4:
        return "neutral"

    # Split into first half and second half
    mid = len(decisions) // 2
    first_half = decisions[:mid]
    second_half = decisions[mid:]

    # Calculate discipline proxy for each half
    def calc_half_score(half: list[dict]) -> float:
        results = [d.get("result") for d in half]
        timestamps = [int(d.get("timestamp", 0)) for d in half]

        win_rate = sum(1 for r in results if r == "win") / max(1, len(results))

        intervals = [timestamps[i + 1] - timestamps[i] for i in range(len(timestamps) - 1)]
        patience = sum(1 for iv in intervals if iv > 60) / max(1, len(intervals))

        return win_rate * 0.5 + patience * 0.5

    first_score = calc_half_score(first_half)
    second_score = calc_half_score(second_half)

    diff = second_score - first_score
    if diff > 0.15:
        return "improving"
    elif diff < -0.15:
        return "declining"
    return "stable"


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
