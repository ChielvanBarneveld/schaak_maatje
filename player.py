import re
import random
import torch
import chess
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from chess_tournament.players import Player

class TransformerPlayer(Player):
    UCI_REGEX = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)

    def __init__(
        self,
        name: str = "Schaakmaatje-Elite",
        model_id: str = "Chiel399/Schaakmaatje_smol_V_0307_2157", 
        temperature: float = 0.0,
        max_new_tokens: int = 8,
    ):
        super().__init__(name)
        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = None
        self.model = None

    def _load_model(self):
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model = AutoModelForCausalLM.from_pretrained(self.model_id)
            self.model.to(self.device)
            self.model.eval()

    def _build_prompt(self, fen: str) -> str:
        return f"FEN: {fen}\nMove:"

    def _extract_move(self, text: str) -> Optional[str]:
        match = self.UCI_REGEX.search(text)
        return match.group(1).lower() if match else None

    def _random_legal(self, fen: str) -> Optional[str]:
        board = chess.Board(fen)
        moves = list(board.legal_moves)
        return random.choice(moves).uci() if moves else None

    def _score_move(self, board: chess.Board, move_uci: str) -> int:
        move = chess.Move.from_uci(move_uci)
        score = 0
        board.push(move)
        is_mate = board.is_checkmate()
        is_stalemate = board.is_stalemate()
        board.pop()

        if is_mate:
            return 10000
        if is_stalemate:
            return -5000

        if board.is_capture(move):
            captured_piece = board.piece_at(move.to_square)
            if captured_piece:
                piece_values = {chess.PAWN: 10, chess.KNIGHT: 30, chess.BISHOP: 30, chess.ROOK: 50, chess.QUEEN: 90}
                score += piece_values.get(captured_piece.piece_type, 0)

        moving_piece = board.piece_at(move.from_square)
        if moving_piece:
            enemy_color = not board.turn
            if board.is_attacked_by(enemy_color, move.to_square):
                piece_values = {chess.PAWN: 10, chess.KNIGHT: 30, chess.BISHOP: 30, chess.ROOK: 50, chess.QUEEN: 90, chess.KING: 1000}
                score -= piece_values.get(moving_piece.piece_type, 0)
        return score

    def get_move(self, fen: str) -> Optional[str]:
        try:
            self._load_model()
            board = chess.Board(fen)
            legal_moves_uci = [m.uci() for m in board.legal_moves]
            if not legal_moves_uci:
                return None

            prompt = self._build_prompt(fen)
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=10,
                    num_return_sequences=10,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            valid_ai_ideas = []
            for out in outputs:
                decoded = self.tokenizer.decode(out, skip_special_tokens=True)
                if decoded.startswith(prompt):
                    decoded = decoded[len(prompt):]
                move = self._extract_move(decoded)
                if move and move in legal_moves_uci:
                    valid_ai_ideas.append(move)

            if valid_ai_ideas:
                best_move = valid_ai_ideas[0]
                best_score = -999999
                for idea in valid_ai_ideas:
                    score = self._score_move(board, idea)
                    if score > best_score:
                        best_score = score
                        best_move = idea
                return best_move

        except Exception:
            pass # Als er iets misgaat, vallen we terug op de return hieronder

        return self._random_legal(fen)