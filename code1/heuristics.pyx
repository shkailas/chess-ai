""" Collection of heuristic related evaluation - move sorting,
board evaluation """
import random
from enum import Enum
from typing import Dict, List, Union

import chess  # type: ignore



# simple move ordering algorithm that prioritizes captures
def MVV_LVA(board: chess.Board) -> List[chess.Move]:
    ConventionalPieceValues = {}
    ConventionalPieceValues["P"] = 100
    ConventionalPieceValues["N"] = 350
    ConventionalPieceValues["B"] = 350
    ConventionalPieceValues["R"] = 525
    ConventionalPieceValues["Q"] = 1000
    ConventionalPieceValues["K"] = 999999
    available_captures: Dict[int, List[chess.Move]] = {}
    move_list = list(board.legal_moves)

    for move in move_list:
        if board.is_capture(move):
            aggressor_piece = str(move)[:2].upper()
            victim_piece = str(move)[2:].upper()

            # print(aggressor_piece)
            a_rank = (int) (aggressor_piece[1])
            v_rank = (int) (victim_piece[1])
            
            a_file = ord(aggressor_piece[0]) - 65
            v_file = ord(aggressor_piece[0]) - 65

            a_square = chess.SQUARES[a_file + (a_rank-1) * 8]
            v_square = chess.SQUARES[v_file + (v_rank-1) * 8]

            a_piece = board.piece_at(a_square)
            v_piece = board.piece_at(v_square)

            
            if (v_piece == None or a_piece == None):
                value_diff = 0
            else:
                value_diff = ConventionalPieceValues[v_piece.symbol().upper()]

            if value_diff not in available_captures:
                available_captures[value_diff] = [move]
            else:
                available_captures[value_diff].append(move)

        else:
            if 0 not in available_captures:
                available_captures[0] = [move]
            else:
                available_captures[0].append(move)

    if available_captures:
        move_list_sorted = []
        for val_diff in sorted(available_captures, reverse=True):
            move_list_sorted.extend(available_captures[val_diff])
        return move_list_sorted

    random.shuffle(move_list)
    return move_list
