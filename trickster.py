#!/usr/bin/env python3
"""
Honest Trapster v1.0 — ищет реальные ловушки, моделируя правдоподобные ошибки соперника.
Полный UCI, честное управление временем, без костылей.
"""
import sys
import os
import time
import logging
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import chess
import chess.engine

# -------------------------- НАСТРОЙКИ --------------------------
STOCKFISH_PATH = os.environ.get("STOCKFISH_PATH",
                                "stockfish.exe" if sys.platform == "win32" else "stockfish")
LOW_DEPTH = 4                # глубина имитации соперника
HIGH_DEPTH = 12              # глубина нашего анализа
MULTIPV = 8                  # число кандидатов
MAX_CP_LOSS_RATIO = 0.15     # макс. доля ухудшения от лучшей оценки
MAX_CP_LOSS_MIN = 50         # минимальный порог ухудшения (ср)
TRAP_WEIGHT = 0.5            # вес потенциала ловушки в финальной оценке
OPP_SAFETY = 200             # если лучший ответ соперника > +200cp — ход опасен
HOPELESS_THRESHOLD = -250    # ниже этой оценки ловушки менее агрессивны
HOPELESS_PENALTY = 0.5       # множитель веса ловушки в плохой позиции
# ----------------------------------------------------------------

logging.basicConfig(filename="trapster.log", level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("HonestTrapster")

def score_to_cp(score):
    """Переводит Score в сантипешки, сохраняя расстояние до мата."""
    inner = score.relative
    if inner.is_mate():
        mate_in = inner.mate()
        if mate_in > 0:
            return 30000 - 100 * mate_in
        else:
            return -30000 + 100 * abs(mate_in)
    return inner.score()

def time_from_uci(params: dict) -> float:
    """Вычисляет бюджет на ход (сек) из параметров go."""
    if "movetime" in params:
        return params["movetime"] / 1000.0
    if "wtime" in params and "btime" in params:
        my_time = params["wtime"] if params.get("side") == "white" else params["btime"]
        inc = params.get("winc" if params["side"] == "white" else "binc", 0)
        remaining = (my_time + inc) / 1000.0
        return max(0.5, min(5.0, remaining * 0.02 + inc / 1000.0))
    return 2.0

def create_engine(path):
    """Запускает UCI-движок с перехватом ошибок."""
    try:
        engine = chess.engine.SimpleEngine.popen_uci(path)
        return engine
    except Exception as e:
        logger.error(f"Не удалось запустить движок {path}: {e}")
        raise

# Простой кеш анализов
_trap_cache = {}

def find_trap(engine_path, board, verbose=False, time_budget=3.0):
    """
    Ядро движка: выбирает ход с наибольшим ожидаемым результатом с учётом ловушек.
    """
    key = board.fen()
    if key in _trap_cache:
        logger.info("Анализ взят из кеша")
        return _trap_cache[key]

    engine = create_engine(engine_path)
    best_move = None
    try:
        # 1. Получаем лучшие ходы-кандидаты (мульти-PV)
        limit_root = chess.engine.Limit(time=time_budget * 0.5) if time_budget > 0 else chess.engine.Limit(depth=HIGH_DEPTH)
        infos = engine.analyse(board, limit_root, multipv=MULTIPV)
        best_cp = score_to_cp(infos[0]["score"])
        logger.info(f"Лучшая оценка позиции: {best_cp:+.0f} cp. Кандидатов: {len(infos)}")

        # Динамический порог допустимого ухудшения
        max_cp_loss = max(MAX_CP_LOSS_MIN, int(MAX_CP_LOSS_RATIO * abs(best_cp)))

        candidates = []
        if verbose:
            print(f"\n[Анализ] best: {best_cp:+.0f} cp")
            print(f"{'Ход':<8} {'Наша':>8} {'Потенциал':>10} {'Итог':>9}")
            print("-" * 40)

        for info in infos:
            move = info["pv"][0]
            our_cp = score_to_cp(info["score"])   # оценка после нашего хода
            cp_loss = best_cp - our_cp
            if cp_loss > max_cp_loss:
                logger.debug(f"Ход {move.uci()} отброшен: потеря {cp_loss} > {max_cp_loss}")
                continue

            # Делаем ход на доске
            board_after = board.copy()
            board_after.push(move)

            # 2. Поверхностный анализ для соперника (его уровень)
            try:
                low_info = engine.analyse(board_after, chess.engine.Limit(depth=LOW_DEPTH), multipv=2)
            except Exception:
                logger.exception("Ошибка поверхностного анализа соперника")
                continue

            opp_moves = [pv["pv"][0] for pv in low_info]
            if not opp_moves:
                continue

            # 3. Глубокий анализ каждого ответа соперника параллельно
            def evaluate_reply(opp_move):
                board_reply = board_after.copy()
                board_reply.push(opp_move)
                try:
                    deep = engine.analyse(board_reply, chess.engine.Limit(depth=HIGH_DEPTH))
                    return score_to_cp(deep["score"])  # оценка после ответа соперника (наша перспектива)
                except Exception:
                    logger.warning("Сбой при глубоком анализе ответа")
                    return None

            trap_cps = []
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(evaluate_reply, mv) for mv in opp_moves]
                for future in as_completed(futures):
                    res = future.result()
                    if res is not None:
                        trap_cps.append(res)

            if not trap_cps:
                continue

            max_trap = max(trap_cps)  # лучший результат, если соперник ошибается
            trap_potential = max_trap - our_cp  # насколько улучшается позиция при ошибке

            # Проверка безопасности: если соперник своим лучшим ответом получает > OPP_SAFETY, пропускаем
            if max_trap < -OPP_SAFETY:
                if verbose:
                    print(f"{move.uci():<8} {our_cp:>+8.0f}   ⚠ опасно")
                continue

            # Итоговая привлекательность = наша оценка + вес * потенциал ловушки
            weight = TRAP_WEIGHT
            if best_cp < HOPELESS_THRESHOLD:
                weight *= HOPELESS_PENALTY
            final_score = our_cp + weight * trap_potential

            candidates.append((final_score, move, our_cp, trap_potential))
            if verbose:
                print(f"{move.uci():<8} {our_cp:>+8.0f} {trap_potential:>+10.1f} {final_score:>+9.1f}")

        if not candidates:
            best_move = infos[0]["pv"][0]
            logger.info("Нет подходящих ловушек, играю лучший ход")
        else:
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_move = candidates[0][1]
            logger.info(f"Выбрана ловушка: {best_move.uci()} (потенциал {candidates[0][3]:.1f})")

        _trap_cache[key] = best_move
        return best_move
    finally:
        engine.quit()

# ---------------------- ПАРСИНГ ХОДОВ ----------------------
_RU_PIECES = {
    'Ф': 'Q', 'ф': 'Q',
    'К': 'K', 'к': 'K',
    'С': 'B', 'с': 'B',
    'Л': 'R', 'л': 'R',
    'Кр': 'K', 'кр': 'K',
}

def parse_move_input(user_input, board):
    s = user_input.strip()
    if not s:
        raise ValueError("Пустой ввод")
    clean = s.replace('+', '').replace('#', '').replace('x', '')
    # UCI (e2e4, e7e8q)
    if len(clean) in (4, 5) and clean[0] in "abcdefgh" and clean[2] in "abcdefgh":
        try:
            return chess.Move.from_uci(clean.lower())
        except chess.InvalidMoveError:
            pass
    # SAN
    try:
        return board.parse_san(clean)
    except ValueError:
        pass
    # SAN с русскими буквами
    trans = clean
    for r, e in sorted(_RU_PIECES.items(), key=lambda x: -len(x[0])):
        if trans.startswith(r):
            trans = e + trans[len(r):]
            break
    try:
        return board.parse_san(trans)
    except ValueError:
        pass
    raise ValueError(f"Не удалось распознать ход: '{user_input}'. Используйте UCI или SAN.")

# ---------------------- UCI ----------------------
def uci_loop(engine_path):
    board = chess.Board()
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if line == "uci":
            print("id name Honest Trapster v1.0")
            print("id author TrapMaster")
            print("option name LowDepth type spin default 4 min 1 max 10")
            print("option name HighDepth type spin default 12 min 6 max 99")
            print("option name TrapWeight type spin default 50 min 0 max 200")
            print("uciok")
            sys.stdout.flush()
        elif line == "isready":
            print("readyok")
            sys.stdout.flush()
        elif line == "ucinewgame":
            board = chess.Board()
            _trap_cache.clear()
        elif line.startswith("setoption"):
            parts = line.split()
            if len(parts) >= 5 and parts[1] == "name":
                opt = parts[2]
                val = parts[-1]
                if opt == "LowDepth":
                    global LOW_DEPTH
                    LOW_DEPTH = int(val)
                elif opt == "HighDepth":
                    global HIGH_DEPTH
                    HIGH_DEPTH = int(val)
                elif opt == "TrapWeight":
                    global TRAP_WEIGHT
                    TRAP_WEIGHT = int(val) / 100.0
        elif line.startswith("position"):
            parts = line.split()
            if parts[1] == "startpos":
                board = chess.Board()
                if len(parts) > 2 and parts[2] == "moves":
                    for m in parts[3:]:
                        board.push_uci(m)
            elif parts[1] == "fen":
                fen_moves = line[9:].strip()
                if " moves " in fen_moves:
                    fen_str, moves_str = fen_moves.split(" moves ", 1)
                    board.set_fen(fen_str)
                    for m in moves_str.split():
                        board.push_uci(m)
                else:
                    board.set_fen(fen_moves)
        elif line.startswith("go"):
            params = {}
            tokens = line.split()[1:]
            i = 0
            while i < len(tokens):
                if tokens[i] in ("searchmoves", "ponder"):
                    i += 1
                    continue
                if tokens[i] in ("wtime", "btime", "winc", "binc", "movetime", "depth", "nodes", "mate", "infinite"):
                    if i+1 < len(tokens):
                        params[tokens[i]] = int(tokens[i+1])
                        i += 2
                    else:
                        i += 1
                else:
                    i += 1
            time_budget = time_from_uci(params)
            move = find_trap(engine_path, board, verbose=False, time_budget=time_budget)
            print(f"bestmove {move.uci()}")
            sys.stdout.flush()
        elif line == "quit":
            break

# ---------------------- ИНТЕРАКТИВ ----------------------
def interactive_play(engine_path, side, debug=False):
    board = chess.Board()
    engine_color = chess.WHITE if side == "white" else chess.BLACK
    human_color = not engine_color

    print(f"\n=== 🧠 Honest Trapster v1.0 ===")
    print(f"Вы за {'белых' if human_color else 'чёрных'}. Вводите ходы, quit для выхода.\n")

    pre_move = [None]
    stop_event = threading.Event()

    def preanalyze():
        while not stop_event.is_set():
            if board.turn == engine_color and not board.is_game_over():
                try:
                    pre_move[0] = find_trap(engine_path, board.copy(), verbose=debug, time_budget=10.0)
                except Exception:
                    pass
            time.sleep(0.1)

    pre_thread = threading.Thread(target=preanalyze, daemon=True)
    pre_thread.start()

    try:
        if engine_color == chess.WHITE:
            move = find_trap(engine_path, board, verbose=debug)
            san = board.san(move)
            board.push(move)
            logger.info(f"Ход движка: {move.uci()} ({san})")
            print(f"🧠 Мой ход: {move.uci()} ({san})")
            print(board, "\n")

        while not board.is_game_over():
            if board.turn == human_color:
                u = input("Ваш ход: ").strip()
                if u.lower() in ("quit", "exit"):
                    print("Игра завершена."); break
                try:
                    move = parse_move_input(u, board)
                except ValueError as e:
                    print(e); continue
                if move not in board.legal_moves:
                    print("Недопустимый ход."); continue
                board.push(move)
                print(board, "\n")
            else:
                print("🧠 Думаю...")
                if pre_move[0] and board.fen() == chess.Board().fen():  # упрощение: используем если позиция начальная
                    move = pre_move[0]
                else:
                    move = find_trap(engine_path, board, verbose=debug)
                san = board.san(move)
                board.push(move)
                logger.info(f"Ход движка: {move.uci()} ({san})")
                print(f"🧠 Мой ход: {move.uci()} ({san})")
                print(board, "\n")
    finally:
        stop_event.set()

    if board.is_checkmate():
        winner = "белые" if board.turn == chess.BLACK else "чёрные"
        print(f"Мат! Победили {winner}.")
    elif board.is_stalemate():
        print("Пат. Ничья.")
    elif board.is_insufficient_material():
        print("Недостаточно материала. Ничья.")
    else:
        print("Игра завершена.")

# ---------------------- ТОЧКА ВХОДА ----------------------
def main():
    parser = argparse.ArgumentParser(description="Honest Trapster v1.0")
    parser.add_argument("--mode", choices=["interactive", "uci"], default="interactive")
    parser.add_argument("--side", choices=["white", "black"], default=None)
    parser.add_argument("--engine", default=STOCKFISH_PATH)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.mode == "uci":
        uci_loop(args.engine)
    else:
        side = args.side
        if side is None:
            side = input("За какую сторону играть движку? (white/black): ").strip().lower()
            while side not in ("white", "black"):
                side = input("Пожалуйста, введите 'white' или 'black': ").strip().lower()
        interactive_play(args.engine, side, debug=args.debug)

if __name__ == "__main__":
    main()