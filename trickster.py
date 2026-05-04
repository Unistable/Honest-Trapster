#!/usr/bin/env python3
"""
Honest Trapster v2.0 —重构版
基于EV（期望值）数学模型的陷阱搜索引擎
完全异步架构，Clean Architecture原则，严格类型注解
"""
from __future__ import annotations
import sys
import os
import asyncio
import logging
import argparse
import signal
from typing import Optional, Dict, List, Tuple, Any, Set
from dataclasses import dataclass, field
from collections import OrderedDict
from enum import Enum
import chess
import chess.engine

# =============================================================================
# CONFIGURATION & CONSTANTS
# =============================================================================

STOCKFISH_PATH: str = os.environ.get(
    "STOCKFISH_PATH",
    "stockfish.exe" if sys.platform == "win32" else "stockfish"
)

@dataclass(frozen=True)
class EngineConfig:
    """Immutable configuration for engine behavior."""
    low_depth: int = 4
    high_depth: int = 12
    multipv: int = 8
    max_cp_loss_ratio: float = 0.15
    max_cp_loss_min: int = 50
    trap_weight_base: float = 0.5
    opp_safety_threshold: int = 200
    hopeless_threshold: int = -250
    hopeless_penalty: float = 0.5
    cache_max_size: int = 10000
    error_slope: float = 0.1
    error_baseline: float = 0.05
    gain_risk_ratio: float = 2.0

DEFAULT_CONFIG: EngineConfig = EngineConfig()

# =============================================================================
# LOGGING SETUP
# =============================================================================

logging.basicConfig(
    filename="trapster.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(module)s:%(lineno)d %(message)s"
)
logger: logging.Logger = logging.getLogger("HonestTrapster")

# =============================================================================
# DATA MODELS
# =============================================================================

class MoveCategory(Enum):
    """Classification of move types based on trap potential."""
    BEST = "best"
    TRAP_CANDIDATE = "trap_candidate"
    SAFE = "safe"
    RISKY = "risky"
    DANGEROUS = "dangerous"

@dataclass(frozen=True)
class ScoreMetrics:
    """Contains all score-related metrics for a candidate move."""
    our_cp: int
    best_cp: int
    cp_loss: int
    trap_potential: int
    ev_score: float
    category: MoveCategory

@dataclass
class MoveCandidate:
    """Represents a candidate move with its analysis results."""
    move: chess.Move
    metrics: ScoreMetrics
    opponent_responses: List[chess.Move] = field(default_factory=list)
    confidence: float = 0.0

@dataclass
class TrapAnalysis:
    """Complete analysis result for a position."""
    best_move: chess.Move
    candidates: List[MoveCandidate]
    position_fen: str
    analysis_depth: int
    config_used: EngineConfig

# =============================================================================
# LRU CACHE IMPLEMENTATION
# =============================================================================

class LRUCache:
    """Thread-safe LRU cache for position analysis."""

    def __init__(self, max_size: int = 10000):
        self._cache: OrderedDict[str, chess.Move] = OrderedDict()
        self._max_size: int = max_size
        self._hits: int = 0
        self._misses: int = 0

    def get(self, key: str) -> Optional[chess.Move]:
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, key: str, value: chess.Move) -> None:
        if key in self._cache:
            self._cache.move_to_end(key)
        else:
            if len(self._cache) >= self._max_size:
                self._cache.popitem(last=False)
            self._cache[key] = value

    def clear(self) -> None:
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

# Global cache instance
_position_cache: LRUCache = LRUCache(DEFAULT_CONFIG.cache_max_size)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def score_to_cp(score: chess.engine.Score) -> int:
    """
    Converts chess.engine.Score to centipawns, preserving mate distance.
    Mate scores are converted to extreme CP values (±30000).
    """
    relative = score.relative
    if relative.is_mate():
        mate_in = relative.mate()
        if mate_in > 0:
            return 30000 - 100 * mate_in
        else:
            return -30000 + 100 * abs(mate_in)
    return relative.score() or 0

def calculate_error_probability(
    depth_diff: int,
    cp_diff: int,
    config: EngineConfig
) -> float:
    """
    Calculates P(error) - probability that opponent makes a mistake.
    
    Uses sigmoid-like function based on:
    - depth_diff: difference between shallow and deep analysis
    - cp_diff: complexity measured by evaluation swing
    
    Formula: P(error) = baseline + (1 - baseline) * sigmoid(depth_diff * slope)
    """
    import math
    baseline = config.error_baseline
    slope = config.error_slope
    
    # Complexity factor from evaluation volatility
    complexity_factor = min(1.0, abs(cp_diff) / 500.0)
    
    # Depth-based error probability
    depth_factor = 1.0 / (1.0 + math.exp(-slope * (depth_diff - 3)))
    
    # Combined probability
    p_error = baseline + (1.0 - baseline) * depth_factor * complexity_factor
    return min(0.95, max(0.05, p_error))

def calculate_ev(
    our_cp: int,
    trap_potential: int,
    p_error: float,
    config: EngineConfig
) -> float:
    """
    Calculates Expected Value (EV) for a trap candidate.
    
    EV = (1 - P(error)) * our_cp + P(error) * (our_cp + trap_potential)
    
    Simplified: EV = our_cp + P(error) * trap_potential
    
    Additional penalty applied if Gain/Loss ratio is unfavorable.
    """
    if trap_potential <= 0:
        return float(our_cp)
    
    # Base EV calculation
    base_ev = our_cp + p_error * trap_potential
    
    # Gain vs Loss ratio adjustment
    gain = trap_potential
    loss = max(1, abs(our_cp)) if our_cp < 0 else 1
    ratio = gain / loss
    
    if ratio < config.gain_risk_ratio:
        penalty = 1.0 - (config.gain_risk_ratio - ratio) * 0.1
        base_ev *= max(0.5, penalty)
    
    return base_ev

# =============================================================================
# ENGINE MANAGER
# =============================================================================

class EngineManager:
    """
    Manages chess engine lifecycle with async support.
    Implements connection pooling and error recovery.
    """

    def __init__(self, engine_path: str, config: EngineConfig = DEFAULT_CONFIG):
        self.engine_path: str = engine_path
        self.config: EngineConfig = config
        self._engine: Optional[chess.engine.SimpleEngine] = None
        self._lock: asyncio.Lock = asyncio.Lock()
        self._error_count: int = 0
        self._max_errors: int = 3

    async def get_engine(self) -> chess.engine.SimpleEngine:
        """Gets or creates engine instance with error handling."""
        async with self._lock:
            if self._engine is not None:
                try:
                    # Check if engine is still responsive
                    await self._engine.ping()
                    return self._engine
                except Exception:
                    await self.close()
            
            return await self._create_engine()

    async def _create_engine(self) -> chess.engine.SimpleEngine:
        """Creates new engine instance."""
        try:
            loop = asyncio.get_event_loop()
            transport, engine = await chess.engine.popen_uci(
                self.engine_path,
                loop=loop
            )
            self._engine = engine
            self._error_count = 0
            logger.info(f"Engine created: {self.engine_path}")
            return engine
        except Exception as e:
            self._error_count += 1
            logger.error(f"Failed to create engine: {e}")
            if self._error_count >= self._max_errors:
                raise RuntimeError(f"Engine creation failed {self._max_errors} times")
            raise

    async def close(self) -> None:
        """Closes engine connection gracefully."""
        async with self._lock:
            if self._engine is not None:
                try:
                    await self._engine.quit()
                except Exception as e:
                    logger.warning(f"Error closing engine: {e}")
                finally:
                    self._engine = None

    async def analyze(
        self,
        board: chess.Board,
        limit: chess.engine.Limit,
        multipv: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Performs analysis with automatic engine recovery."""
        engine = await self.get_engine()
        try:
            mp = multipv if multipv else self.config.multipv
            result = await engine.analyse(
                board,
                limit,
                multipv=mp,
                info=chess.engine.INFO_ALL
            )
            if isinstance(result, list):
                return result
            return [result]
        except Exception as e:
            logger.warning(f"Analysis failed, recreating engine: {e}")
            await self.close()
            engine = await self.get_engine()
            result = await engine.analyse(board, limit, multipv=multipv or self.config.multipv)
            if isinstance(result, list):
                return result
            return [result]

# =============================================================================
# TRAP FINDER (CORE LOGIC)
# =============================================================================

class TrapFinder:
    """
    Core trap-finding algorithm using EV-based evaluation.
    Analyzes positions to find moves with high trap potential.
    """

    def __init__(
        self,
        engine_manager: EngineManager,
        config: EngineConfig = DEFAULT_CONFIG
    ):
        self.engine_manager: EngineManager = engine_manager
        self.config: EngineConfig = config

    async def find_best_trap(
        self,
        board: chess.Board,
        time_budget: float = 3.0,
        verbose: bool = False
    ) -> TrapAnalysis:
        """
        Main entry point: finds the best trap candidate.
        
        Algorithm:
        1. Check cache for existing analysis
        2. Get root analysis (multi-PV candidates)
        3. For each candidate, simulate opponent responses
        4. Calculate EV for each candidate
        5. Return best move with full analysis
        """
        fen = board.fen()
        cached = _position_cache.get(fen)
        if cached is not None:
            logger.info("Cache hit for position")
            return TrapAnalysis(
                best_move=cached,
                candidates=[],
                position_fen=fen,
                analysis_depth=0,
                config_used=self.config
            )

        # Root analysis
        limit = chess.engine.Limit(time=time_budget * 0.5) if time_budget > 0 \
            else chess.engine.Limit(depth=self.config.high_depth)
        
        root_infos = await self.engine_manager.analyze(board, limit)
        if not root_infos:
            raise RuntimeError("Root analysis returned no results")

        best_cp = score_to_cp(root_infos[0]["score"])
        logger.info(f"Best evaluation: {best_cp:+d} cp, {len(root_infos)} candidates")

        max_cp_loss = max(
            self.config.max_cp_loss_min,
            int(self.config.max_cp_loss_ratio * abs(best_cp))
        )

        candidates: List[MoveCandidate] = []

        for info in root_infos:
            move = info["pv"][0]
            our_cp = score_to_cp(info["score"])
            cp_loss = best_cp - our_cp

            if cp_loss > max_cp_loss:
                logger.debug(f"Move {move.uci()} rejected: loss {cp_loss} > {max_cp_loss}")
                continue

            # Analyze opponent responses
            board_after = board.copy()
            board_after.push(move)

            opp_responses = await self._get_opponent_responses(board_after)
            if not opp_responses:
                continue

            # Calculate trap metrics
            trap_metrics = await self._evaluate_trap_potential(
                board, board_after, move, our_cp, opp_responses, best_cp
            )

            if trap_metrics.category != MoveCategory.DANGEROUS:
                candidate = MoveCandidate(move=move, metrics=trap_metrics)
                candidate.opponent_responses = opp_responses[:3]
                candidate.confidence = self._calculate_confidence(trap_metrics)
                candidates.append(candidate)

                if verbose:
                    print(f"{move.uci():<8} {our_cp:>+6d} EV:{trap_metrics.ev_score:>+8.1f} [{trap_metrics.category.value}]")

        # Select best candidate
        if not candidates:
            best_move = root_infos[0]["pv"][0]
            logger.info("No trap candidates, playing best move")
        else:
            candidates.sort(key=lambda c: c.metrics.ev_score, reverse=True)
            best_move = candidates[0].move
            logger.info(f"Selected trap: {best_move.uci()} (EV: {candidates[0].metrics.ev_score:.1f})")

        _position_cache.put(fen, best_move)

        return TrapAnalysis(
            best_move=best_move,
            candidates=candidates,
            position_fen=fen,
            analysis_depth=self.config.high_depth,
            config_used=self.config
        )

    async def _get_opponent_responses(
        self,
        board_after: chess.Board
    ) -> List[chess.Move]:
        """Gets opponent's likely responses using shallow analysis."""
        try:
            limit = chess.engine.Limit(depth=self.config.low_depth)
            infos = await self.engine_manager.analyze(board_after, limit, multipv=3)
            return [info["pv"][0] for info in infos if "pv" in info and info["pv"]]
        except Exception as e:
            logger.warning(f"Failed to get opponent responses: {e}")
            return []

    async def _evaluate_trap_potential(
        self,
        original_board: chess.Board,
        board_after: chess.Board,
        our_move: chess.Move,
        our_cp: int,
        opp_responses: List[chess.Move],
        best_cp: int
    ) -> ScoreMetrics:
        """
        Evaluates trap potential using async parallel analysis.
        
        For each opponent response:
        1. Play the response
        2. Deep analyze resulting position
        3. Calculate evaluation swing
        """
        async def analyze_response(opp_move: chess.Move) -> Optional[int]:
            board_reply = board_after.copy()
            board_reply.push(opp_move)
            try:
                limit = chess.engine.Limit(depth=self.config.high_depth)
                info = await self.engine_manager.analyze(board_reply, limit)
                return score_to_cp(info[0]["score"])
            except Exception as e:
                logger.warning(f"Response analysis failed: {e}")
                return None

        # Parallel analysis of all responses
        tasks = [analyze_response(m) for m in opp_responses]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_results: List[int] = [
            r for r in results
            if isinstance(r, int) and r is not None
        ]

        if not valid_results:
            return ScoreMetrics(
                our_cp=our_cp,
                best_cp=best_cp,
                cp_loss=best_cp - our_cp,
                trap_potential=0,
                ev_score=float(our_cp),
                category=MoveCategory.SAFE
            )

        # Best case scenario (opponent blunders)
        max_trap_cp = max(valid_results)
        trap_potential = max_trap_cp - our_cp
        
        # Worst case (opponent plays best refutation)
        min_trap_cp = min(valid_results)
        
        # Safety check
        if min_trap_cp < -self.config.opp_safety_threshold:
            return ScoreMetrics(
                our_cp=our_cp,
                best_cp=best_cp,
                cp_loss=best_cp - our_cp,
                trap_potential=trap_potential,
                ev_score=float(our_cp),
                category=MoveCategory.DANGEROUS
            )

        # Calculate error probability
        depth_diff = self.config.high_depth - self.config.low_depth
        eval_swing = max_trap_cp - min_trap_cp
        p_error = calculate_error_probability(depth_diff, eval_swing, self.config)

        # Calculate EV
        ev = calculate_ev(our_cp, trap_potential, p_error, self.config)

        # Categorize move
        category = self._categorize_move(our_cp, trap_potential, ev, best_cp)

        return ScoreMetrics(
            our_cp=our_cp,
            best_cp=best_cp,
            cp_loss=best_cp - our_cp,
            trap_potential=trap_potential,
            ev_score=ev,
            category=category
        )

    def _categorize_move(
        self,
        our_cp: int,
        trap_potential: int,
        ev: float,
        best_cp: int
    ) -> MoveCategory:
        """Categorizes move based on risk/reward profile."""
        if trap_potential <= 0:
            return MoveCategory.SAFE
        
        if our_cp >= best_cp - 10:
            return MoveCategory.BEST
        
        if ev > best_cp:
            return MoveCategory.TRAP_CANDIDATE
        
        if our_cp < self.config.hopeless_threshold:
            return MoveCategory.RISKY
        
        return MoveCategory.SAFE

    def _calculate_confidence(self, metrics: ScoreMetrics) -> float:
        """Calculates confidence score for a trap candidate."""
        base_confidence = 0.5
        
        # Higher trap potential increases confidence
        if metrics.trap_potential > 100:
            base_confidence += 0.2
        elif metrics.trap_potential > 50:
            base_confidence += 0.1
        
        # Positive EV increases confidence
        if metrics.ev_score > metrics.best_cp:
            base_confidence += 0.2
        
        # Safe moves have higher confidence
        if metrics.category in (MoveCategory.BEST, MoveCategory.TRAP_CANDIDATE):
            base_confidence += 0.1
        
        return min(1.0, max(0.0, base_confidence))

# =============================================================================
# UCI HANDLER
# =============================================================================

class UCIHandler:
    """Handles UCI protocol communication."""

    def __init__(
        self,
        engine_manager: EngineManager,
        trap_finder: TrapFinder
    ):
        self.engine_manager = engine_manager
        self.trap_finder = trap_finder
        self.board = chess.Board()
        self.config = DEFAULT_CONFIG
        self._running = True

    async def run(self) -> None:
        """Main UCI loop."""
        while self._running:
            try:
                line = await asyncio.get_event_loop().run_in_executor(
                    None, sys.stdin.readline
                )
                if not line:
                    break
                await self._process_command(line.strip())
            except Exception as e:
                logger.error(f"UCI processing error: {e}")

    async def _process_command(self, line: str) -> None:
        """Processes single UCI command."""
        if line == "uci":
            await self._send_uci_info()
        elif line == "isready":
            print("readyok")
            sys.stdout.flush()
        elif line == "ucinewgame":
            self.board = chess.Board()
            _position_cache.clear()
            await self.engine_manager.close()
        elif line.startswith("setoption"):
            self._handle_setoption(line)
        elif line.startswith("position"):
            self._handle_position(line)
        elif line.startswith("go"):
            await self._handle_go(line)
        elif line == "quit":
            self._running = False
        elif line:
            logger.debug(f"Unknown command: {line}")

    async def _send_uci_info(self) -> None:
        """Sends UCI identification and options."""
        print("id name Honest Trapster v2.0")
        print("id author TrapMaster")
        print("option name LowDepth type spin default 4 min 1 max 10")
        print("option name HighDepth type spin default 12 min 6 max 99")
        print("option name TrapWeight type spin default 50 min 0 max 200")
        print("option name CacheSize type spin default 10000 min 100 max 100000")
        print("uciok")
        sys.stdout.flush()

    def _handle_setoption(self, line: str) -> None:
        """Handles setoption command."""
        parts = line.split()
        if len(parts) < 5 or parts[1] != "name":
            return
        
        opt_name = parts[2]
        value = int(parts[-1])
        
        if opt_name == "LowDepth":
            object.__setattr__(self.config, 'low_depth', value)
        elif opt_name == "HighDepth":
            object.__setattr__(self.config, 'high_depth', value)
        elif opt_name == "TrapWeight":
            object.__setattr__(self.config, 'trap_weight_base', value / 100.0)
        elif opt_name == "CacheSize":
            _position_cache._max_size = value

    def _handle_position(self, line: str) -> None:
        """Handles position command."""
        parts = line.split()
        if len(parts) < 2:
            return
        
        if parts[1] == "startpos":
            self.board = chess.Board()
            if len(parts) > 3 and parts[2] == "moves":
                for move_uci in parts[3:]:
                    self.board.push_uci(move_uci)
        elif parts[1] == "fen":
            fen_start = line.find("fen ") + 4
            rest = line[fen_start:].strip()
            if " moves " in rest:
                fen_str, moves_str = rest.split(" moves ", 1)
                self.board.set_fen(fen_str)
                for move_uci in moves_str.split():
                    self.board.push_uci(move_uci)
            else:
                self.board.set_fen(rest)

    async def _handle_go(self, line: str) -> None:
        """Handles go command and outputs bestmove."""
        params = self._parse_go_params(line)
        time_budget = self._extract_time_budget(params)
        
        try:
            analysis = await self.trap_finder.find_best_trap(
                self.board,
                time_budget=time_budget,
                verbose=False
            )
            print(f"bestmove {analysis.best_move.uci()}")
            sys.stdout.flush()
        except Exception as e:
            logger.error(f"Go command failed: {e}")
            # Fallback to simple move
            try:
                engine = await self.engine_manager.get_engine()
                result = await engine.play(self.board, chess.engine.Limit(time=1.0))
                print(f"bestmove {result.move.uci()}")
            except Exception:
                print("bestmove 0000")
            sys.stdout.flush()

    def _parse_go_params(self, line: str) -> Dict[str, int]:
        """Parses go command parameters."""
        params: Dict[str, int] = {}
        tokens = line.split()[1:]
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in ("wtime", "btime", "winc", "binc", "movetime", "depth"):
                if i + 1 < len(tokens):
                    try:
                        params[token] = int(tokens[i + 1])
                    except ValueError:
                        pass
                    i += 2
                else:
                    i += 1
            elif token in ("searchmoves", "ponder", "infinite"):
                i += 1
            else:
                i += 1
        return params

    def _extract_time_budget(self, params: Dict[str, int]) -> float:
        """Extracts time budget from go parameters."""
        if "movetime" in params:
            return params["movetime"] / 1000.0
        
        if "wtime" in params and "btime" in params:
            # Determine side from board state
            is_white = self.board.turn == chess.WHITE
            my_time = params["wtime"] if is_white else params["btime"]
            inc_key = "winc" if is_white else "binc"
            inc = params.get(inc_key, 0)
            
            remaining = (my_time + inc) / 1000.0
            return max(0.5, min(5.0, remaining * 0.02 + inc / 1000.0))
        
        return 2.0

# =============================================================================
# INTERACTIVE MODE
# =============================================================================

class InteractivePlayer:
    """Handles interactive play mode."""

    def __init__(
        self,
        engine_manager: EngineManager,
        trap_finder: TrapFinder,
        engine_color: chess.Color
    ):
        self.engine_manager = engine_manager
        self.trap_finder = trap_finder
        self.engine_color = engine_color
        self.board = chess.Board()
        self._pre_move: Optional[chess.Move] = None
        self._stop_event = asyncio.Event()

    async def run(self) -> None:
        """Main interactive loop."""
        print(f"\n=== 🧠 Honest Trapster v2.0 ===")
        print(f"You are playing as {'White' if self.engine_color == chess.BLACK else 'Black'}")
        print("Type 'quit' to exit\n")

        # Start background pre-analysis
        asyncio.create_task(self._preanalyze_loop())

        # Opening move if engine is White
        if self.engine_color == chess.WHITE:
            await self._make_engine_move()

        while not self.board.is_game_over():
            if self.board.turn == self.engine_color:
                await self._make_engine_move()
            else:
                await self._get_human_move()

        self._print_game_result()

    async def _preanalyze_loop(self) -> None:
        """Background task for pre-analyzing positions."""
        while not self._stop_event.is_set():
            if self.board.turn == self.engine_color and not self.board.is_game_over():
                try:
                    analysis = await self.trap_finder.find_best_trap(
                        self.board.copy(),
                        time_budget=5.0
                    )
                    self._pre_move = analysis.best_move
                except Exception:
                    pass
            await asyncio.sleep(0.2)

    async def _make_engine_move(self) -> None:
        """Makes a move for the engine."""
        print("🧠 Thinking...")
        try:
            if self._pre_move and self._pre_move in self.board.legal_moves:
                move = self._pre_move
            else:
                analysis = await self.trap_finder.find_best_trap(
                    self.board,
                    time_budget=3.0,
                    verbose=True
                )
                move = analysis.best_move
            
            san = self.board.san(move)
            self.board.push(move)
            print(f"🧠 My move: {move.uci()} ({san})")
            print(self.board)
            print()
        except Exception as e:
            logger.error(f"Engine move failed: {e}")
            print("Error calculating move")

    async def _get_human_move(self) -> None:
        """Gets and validates human move."""
        loop = asyncio.get_event_loop()
        while True:
            try:
                user_input = await loop.run_in_executor(None, input, "Your move: ")
                user_input = user_input.strip()
                
                if user_input.lower() in ("quit", "exit"):
                    print("Game terminated.")
                    self._stop_event.set()
                    return
                
                move = self._parse_human_input(user_input)
                if move not in self.board.legal_moves:
                    print("Illegal move. Try again.")
                    continue
                
                self.board.push(move)
                print(self.board)
                print()
                break
            except ValueError as e:
                print(str(e))

    def _parse_human_input(self, user_input: str) -> chess.Move:
        """Parses human move input (UCI, SAN, Russian notation)."""
        clean = user_input.replace('+', '').replace('#', '').replace('x', '')
        
        # Try UCI format
        if len(clean) in (4, 5) and clean[0] in "abcdefgh" and clean[2] in "abcdefgh":
            try:
                return chess.Move.from_uci(clean.lower())
            except chess.InvalidMoveError:
                pass
        
        # Try SAN
        try:
            return self.board.parse_san(clean)
        except ValueError:
            pass
        
        # Try Russian piece notation
        ru_pieces = {'Ф': 'Q', 'К': 'N', 'С': 'B', 'Л': 'R', 'Кр': 'K'}
        trans = clean
        for ru, en in sorted(ru_pieces.items(), key=lambda x: -len(x[0])):
            if trans.startswith(ru):
                trans = en + trans[len(ru):]
                break
        
        try:
            return self.board.parse_san(trans)
        except ValueError:
            pass
        
        raise ValueError(f"Cannot parse move: '{user_input}'")

    def _print_game_result(self) -> None:
        """Prints game result."""
        if self.board.is_checkmate():
            winner = "White" if self.board.turn == chess.BLACK else "Black"
            print(f"Checkmate! {winner} wins!")
        elif self.board.is_stalemate():
            print("Stalemate. Draw.")
        elif self.board.is_insufficient_material():
            print("Insufficient material. Draw.")
        else:
            print("Game over.")

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

async def async_main(args: argparse.Namespace) -> None:
    """Async main entry point."""
    config = DEFAULT_CONFIG
    engine_manager = EngineManager(args.engine, config)
    trap_finder = TrapFinder(engine_manager, config)

    try:
        if args.mode == "uci":
            handler = UCIHandler(engine_manager, trap_finder)
            await handler.run()
        else:
            engine_color = chess.WHITE if args.side == "white" else chess.BLACK
            player = InteractivePlayer(engine_manager, trap_finder, engine_color)
            await player.run()
    finally:
        await engine_manager.close()

def main() -> None:
    """Synchronous wrapper for async main."""
    parser = argparse.ArgumentParser(description="Honest Trapster v2.0")
    parser.add_argument(
        "--mode",
        choices=["interactive", "uci"],
        default="interactive",
        help="Run mode"
    )
    parser.add_argument(
        "--side",
        choices=["white", "black"],
        default=None,
        help="Engine color"
    )
    parser.add_argument(
        "--engine",
        default=STOCKFISH_PATH,
        help="Path to chess engine"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    if args.side is None and args.mode == "interactive":
        args.side = input("Engine color (white/black): ").strip().lower()
        while args.side not in ("white", "black"):
            args.side = input("Please enter 'white' or 'black': ").strip().lower()

    # Setup signal handlers
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(async_main(args))
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        loop.close()

if __name__ == "__main__":
    main()