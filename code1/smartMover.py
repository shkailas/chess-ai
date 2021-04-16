import random
import chess
import time
import numpy as np


class Player:
    depth = 2
    board = chess.Board()
    def __init__(self, board, color, time):
        pass

    def move(self, board, time):
        # return self.alBeMinMaxVal(board, 1, float("-inf"), float("inf"), True)[0]
        depth = 3
        bestMove = chess.Move.null()
        bestValue = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        for move in board.legal_moves:
            board.push(move)
            value = -self.negamax(board, -beta, -alpha, depth - 1)
            if value > bestValue:
                bestValue = value
                bestMove = move
            if value > alpha:
                alpha = value
            board.pop()
        return bestMove

    def evaluation(self, board):
        P = 100
        N = 320
        B = 330
        R = 500
        Q = 900
        K = 2000
        wp = len(board.pieces(chess.PAWN, chess.WHITE))
        bp = len(board.pieces(chess.PAWN, chess.BLACK))
        wn = len(board.pieces(chess.KNIGHT, chess.WHITE))
        bn = len(board.pieces(chess.KNIGHT, chess.BLACK))
        wb = len(board.pieces(chess.BISHOP, chess.WHITE))
        bb = len(board.pieces(chess.BISHOP, chess.BLACK))
        wr = len(board.pieces(chess.ROOK, chess.WHITE))
        br = len(board.pieces(chess.ROOK, chess.BLACK))
        wq = len(board.pieces(chess.QUEEN, chess.WHITE))
        bq = len(board.pieces(chess.QUEEN, chess.BLACK))
        wk = len(board.pieces(chess.KING, chess.WHITE))
        bk = len(board.pieces(chess.KING, chess.BLACK)) 
        eval = float(P * (wp - bp) + N * (wn - bn) + B * (wb - bb) * R * (wr - br) + Q * (wq - bq) + K * (wk - bk)) 

        #adapted from chess wikipedia page with piece square values
        pawntable = np.array([
        0,  0,  0,  0,  0,  0,  0,  0,
        5, 10, 10,-20,-20, 10, 10,  5,
        5, -5,-10,  0,  0,-10, -5,  5,
        0,  0,  0, 20, 20,  0,  0,  0,
        5,  5, 10, 25, 25, 10,  5,  5,
        10, 10, 20, 30, 30, 20, 10, 10,
        50, 50, 50, 50, 50, 50, 50, 50,
        0,  0,  0,  0,  0,  0,  0,  0])

        knightstable = np.array([
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50])

        bishopstable = np.array([
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -20,-10,-10,-10,-10,-10,-10,-20])

        rookstable = np.array([
        0,  0,  0,  5,  5,  0,  0,  0,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        5, 10, 10, 10, 10, 10, 10,  5,
        0,  0,  0,  0,  0,  0,  0,  0])

        queenstable = np.array([
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  5,  5,  5,  5,  5,  0,-10,
        0,  0,  5,  5,  5,  5,  0, -5,
        -5,  0,  5,  5,  5,  5,  0, -5,
        -10,  0,  5,  5,  5,  5,  0,-10,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20])

        kingstable = np.array([
        20, 30, 10,  0,  0, 10, 30, 20,
        20, 20,  0,  0,  0,  0, 20, 20,
        -10,-20,-20,-20,-20,-20,-20,-10,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30])
        pawnsq = sum([pawntable[i] for i in board.pieces(chess.PAWN, chess.WHITE)])
        pawnsq = pawnsq + sum([-pawntable[chess.square_mirror(i)] for i in board.pieces(chess.PAWN, chess.BLACK)])
        knightsq = sum([knightstable[i] for i in board.pieces(chess.KNIGHT, chess.WHITE)])
        knightsq = knightsq + sum([-knightstable[chess.square_mirror(i)] for i in board.pieces(chess.KNIGHT, chess.BLACK)])
        bishopsq= sum([bishopstable[i] for i in board.pieces(chess.BISHOP, chess.WHITE)])
        bishopsq= bishopsq + sum([-bishopstable[chess.square_mirror(i)] for i in board.pieces(chess.BISHOP, chess.BLACK)])
        rooksq = sum([rookstable[i] for i in board.pieces(chess.ROOK, chess.WHITE)])
        rooksq = rooksq + sum([-rookstable[chess.square_mirror(i)] for i in board.pieces(chess.ROOK, chess.BLACK)])
        queensq = sum([queenstable[i] for i in board.pieces(chess.QUEEN, chess.WHITE)])
        queensq = queensq + sum([-queenstable[chess.square_mirror(i)] for i in board.pieces(chess.QUEEN, chess.BLACK)])
        kingsq = sum([kingstable[i] for i in board.pieces(chess.KING, chess.WHITE)])
        kingsq = kingsq + sum([-kingstable[chess.square_mirror(i)] for i in board.pieces(chess.KING, chess.BLACK)])

        eval += pawnsq + knightsq + bishopsq+ rooksq + queensq + kingsq
        if board.is_checkmate():
            return -1e6
        elif board.is_stalemate():
            return 0
        else:
            if board.turn:
                return eval
            return -eval

    def finalValueAlBeMinMax(self, board, depth, alpha, beta):
        if depth is self.depth or not bool(board.legal_moves):
            # if board.is_capture(board.peek()) or board.is_check():
            #     return self.quiescence(board, alpha, beta)
            return -self.evaluation(board)
        if depth % 2 == 0:
            return self.alBeMinMaxVal(board, depth + 1, alpha, beta, True)[1]
        else:
            return self.alBeMinMaxVal(board, depth + 1, alpha, beta, False)[1]

    def alBeMinMaxVal(self, board, depth, alpha, beta, isMax):
        if isMax:
            bestAction = ("max", float("-inf"))
        else:
            bestAction = ("min", float("inf"))

        moves = list(board.legal_moves)
        for move in moves:
            testBoard = board
            board.push(move)
            if isMax:
                maxAction = (move, self.finalValueAlBeMinMax(testBoard, depth, alpha, beta))
                bestAction = max(bestAction, maxAction, key = lambda x:x[1])
                testBoard.pop()
                if bestAction[1] > beta:
                    return bestAction
                else:
                    alpha = max(alpha, bestAction[1])
            else:
                minAction = (move, self.finalValueAlBeMinMax(testBoard, depth, alpha, beta))
                bestAction = min(bestAction, minAction, key = lambda x:x[1])
                testBoard.pop()
                if bestAction[1] < alpha:
                    return bestAction
                else:
                    beta = min(beta, bestAction[1])
        return bestAction

    def negamax(self, board, alpha, beta, depth):
        bestScore = float("-inf")
        if depth == 0:
            return self.quiescence(board, alpha, beta)
        for move in board.legal_moves:
            board.push(move)
            score = -self.negamax(board, -beta, -alpha, depth - 1)
            board.pop()
            if score >= beta:
                return score
            if score > bestScore:
                bestScore = score
            if score > alpha:
                alpha = score
        return bestScore 

    def quiescence(self, board, alpha, beta):
        standPat = self.evaluation(board) #trivial starting point number that's returned in case no captures can be made
        if standPat >= beta:
            return beta
        if alpha < standPat:
            alpha = standPat
        for move in board.legal_moves:
            if board.is_capture(move):
                board.push(move)
                score = -self.quiescence(board, -beta, -alpha)
                board.pop()
                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score
        return alpha