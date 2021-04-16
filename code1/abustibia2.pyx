import random
import chess
import keras
from keras import models
import os 
import tensorflow as tf
import numpy as np
import sys
from . import heuristics as hr
# import eval as ev
import concurrent.futures
import chess.polyglot
from multiprocessing import Process


# Player class for our chess AI

class Player:
    depth = 4 # depth to search
    opening_book = None
    opening = None
    
    # model loading
    def __init__(self, board, color, time):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        os.chdir(os.path.join(dir_path, ".."))
        os.chdir("storage")
        os.chdir(os.path.join("..", os.path.join("code1", "books")))
        self.opening_book = chess.polyglot.MemoryMappedReader("elo-2700.bin")
        self.opening = True

    def local_eval_func(self, board):
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
        pawntable = [
        0,  0,  0,  0,  0,  0,  0,  0,
        5, 10, 10,-20,-20, 10, 10,  5,
        5, -5,-10,  0,  0,-10, -5,  5,
        0,  0,  0, 20, 20,  0,  0,  0,
        5,  5, 10, 25, 25, 10,  5,  5,
        10, 10, 20, 30, 30, 20, 10, 10,
        50, 50, 50, 50, 50, 50, 50, 50,
        0,  0,  0,  0,  0,  0,  0,  0]

        knightstable = [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50]

        bishopstable = [
        -20,-10,-10,-10,-10,-10,-10,-20,
        -10,  5,  0,  0,  0,  0,  5,-10,
        -10, 10, 10, 10, 10, 10, 10,-10,
        -10,  0, 10, 10, 10, 10,  0,-10,
        -10,  5,  5, 10, 10,  5,  5,-10,
        -10,  0,  5, 10, 10,  5,  0,-10,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -20,-10,-10,-10,-10,-10,-10,-20]

        rookstable = [
        0,  0,  0,  5,  5,  0,  0,  0,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        -5,  0,  0,  0,  0,  0,  0, -5,
        5, 10, 10, 10, 10, 10, 10,  5,
        0,  0,  0,  0,  0,  0,  0,  0]

        queenstable = [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  5,  5,  5,  5,  5,  0,-10,
        0,  0,  5,  5,  5,  5,  0, -5,
        -5,  0,  5,  5,  5,  5,  0, -5,
        -10,  0,  5,  5,  5,  5,  0,-10,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20]

        kingstable = [
        20, 30, 10,  0,  0, 10, 30, 20,
        20, 20,  0,  0,  0,  0, 20, 20,
        -10,-20,-20,-20,-20,-20,-20,-10,
        -20,-30,-30,-40,-40,-30,-30,-20,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30,
        -30,-40,-40,-50,-50,-40,-40,-30]

        pawn_score = 0
        for piece in board.pieces(chess.PAWN,chess.WHITE):
            pawn_score += pawntable[piece]
        for piece in board.pieces(chess.PAWN, chess.BLACK):
            pawn_score -+ pawntable[chess.square_mirror(piece)]

        knight_score = 0
        for piece in board.pieces(chess.KNIGHT,chess.WHITE):
            knight_score += knightstable[piece]
        for piece in board.pieces(chess.KNIGHT, chess.BLACK):
            knight_score -+ knightstable[chess.square_mirror(piece)]

        bishop_score = 0
        for piece in board.pieces(chess.BISHOP,chess.WHITE):
            bishop_score += bishopstable[piece]
        for piece in board.pieces(chess.BISHOP, chess.BLACK):
            bishop_score -+ bishopstable[chess.square_mirror(piece)]
        
        rook_score = 0
        for piece in board.pieces(chess.ROOK,chess.WHITE):
            rook_score += rookstable[piece]
        for piece in board.pieces(chess.ROOK, chess.BLACK):
            rook_score -+ rookstable[chess.square_mirror(piece)]

        queen_score = 0
        for piece in board.pieces(chess.QUEEN,chess.WHITE):
            queen_score += queenstable[piece]
        for piece in board.pieces(chess.QUEEN, chess.BLACK):
            queen_score -+ queenstable[chess.square_mirror(piece)]

        king_score = 0
        for piece in board.pieces(chess.KING,chess.WHITE):
            king_score += kingstable[piece]
        for piece in board.pieces(chess.KING, chess.BLACK):
            king_score -+ kingstable[chess.square_mirror(piece)]
        
        
        eval += pawn_score + knight_score + bishop_score + rook_score + queen_score + king_score
        if board.is_checkmate():
            return -1e6
        elif board.is_stalemate():
            return 0
        else:
            if board.turn:
                return eval
            return -eval

    # calls minimax algorithm
    def move(self, board, time):
        #opening moves
        if(self.opening):
            try:
                #try to find current board in opening book
                ret_move = self.opening_book.find(board).move
                return ret_move
            except:
                self.opening = False
                # return self.maxval(board, 0, -float("inf"), float("inf"))[0]
                depth = 2
                a = float("-inf")
                b = float("inf")
                bestMove = chess.Move.null()
                bestValue = float("-inf")
                for move in hr.MVV_LVA(board):
                    board.push(move)
                    value = -self.negamax(board, -b, -a, depth - 1)
                    if value > bestValue:
                        bestValue = value
                        bestMove = move
                    if value > a:
                        a = value
                    board.pop()
                return bestMove
        #rest of game
        else:
            depth = 2
            a = float("-inf")
            b = float("inf")
            bestMove = chess.Move.null()
            bestValue = float("-inf")
            for move in hr.MVV_LVA(board):
                board.push(move)
                value = -self.negamax(board, -b, -a, depth - 1)
                if value > bestValue:
                    bestMove = move
                    bestValue = value
                if value > a:
                    a = value
                board.pop()
            return bestMove
         
    # alpha beta algorithm declaration
    def alphabeta(self, board, depth, alpha, beta):
        # legal_moves = hr.MVV_LVA(board)
        if depth is self.depth or not bool(board.legal_moves):
            return self.local_eval_func(board)
        if depth%2==0:
            return self.maxval(board, depth, alpha, beta)[1]
        else:
            return self.minval(board, depth, alpha, beta)[1]

    # maximizing method in minimax
    def maxval(self, board, depth, alpha, beta):
        
        bestAction = ("max",-float("inf"))
        
        # MVV_LVA is the method for getting a capture-prioritized move list
        legal_moves = hr.MVV_LVA(board)

        # attempts at multi-processing, could be useful to look at?

        # if (depth ==0):
        #     succboard = board
        #     results = []
        #     processes = []
        #     with concurrent.futures.ProcessPoolExecutor() as executor:
        #         legal_moves = hr.MVV_LVA(board)
        #         # for action in legal_moves:
        #         #     succboard = board
        #         #     succboard.push(action)
        #         #     results.append(executor.submit(self.alphabeta, succboard, depth+1, alpha, beta))
        #         #     succboard.pop()
        #         for action in legal_moves:
        #             r = multiprocessing.Process(target =self.alphabeta, args=[succboard, depth+1, alpha, beta])
        #             r.start()
        #             processes.append(r)
        #         for process in processes:
        #             process.join() 
        #         for f in concurrent.futures.as_completed(results):
        #             succAction = f.result()
        #             bestAction = max(bestAction,succAction,key=lambda x:x[1])
        #             if bestAction[1] > beta: 
        #                 return bestAction
        #             else: alpha = max(alpha,bestAction[1])

            # return bestAction
                #results = [executer.submit(self.alphabeta, succboard.push(move), depth+1, alpha, beta) for move in legal_moves]
                # for f in concurrent.futures.as_completed(results):
                #     succAction = f.result()
                #     bestAction = max(bestAction,succAction,key=lambda x:x[1])

            # for action in legal_moves:
            #     succboard = board#.copy()
            #     succboard.push(action)
                
            #     succAction = (action,self.alphabeta(succboard,depth+1, alpha, beta))
            #     bestAction = max(bestAction,succAction,key=lambda x:x[1])
            #     succboard.pop()
            #     # Prunning
            #     if bestAction[1] > beta: 
            #         return bestAction
            #     else: alpha = max(alpha,bestAction[1])

            # return bestAction
        
        for action in legal_moves:
            succboard = board#.copy()
            succboard.push(action)
            
            succAction = (action,self.alphabeta(succboard,depth+1, alpha, beta))
            bestAction = max(bestAction,succAction,key=lambda x:x[1])
            succboard.pop()
            # Prunning
            if bestAction[1] > beta: 
                return bestAction
            else: alpha = max(alpha,bestAction[1])

        return bestAction

    # minimizing method for minimax
    def minval(self, board, depth, alpha, beta):
        bestAction = ("min",float("inf"))
        legal_moves = hr.MVV_LVA(board)
        for action in legal_moves:
            succboard = board
            succboard.push(action)
            succAction = (action,self.alphabeta(succboard,depth+1, alpha, beta))
            bestAction = min(bestAction,succAction,key=lambda x:x[1])
            succboard.pop()

            # Prunning
            if bestAction[1] < alpha: return bestAction
            else: beta = min(beta, bestAction[1]) 

        return bestAction

    def negamax(self, board, alpha, beta, depth):
        bestScore = float("-inf")
        if depth == 0:
            return self.quiescence(board, alpha, beta)

        legal_moves = hr.MVV_LVA(board)
        for move in legal_moves:
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
        standPat = self.local_eval_func(board) #trivial starting point number that's returned in case no captures can be made
        if standPat >= beta:
            return beta
        if alpha < standPat:
            alpha = standPat
        legal_moves = hr. MVV_LVA(board)
        for move in legal_moves:
            if board.is_capture(move):
                board.push(move)
                score = -self.quiescence(board, -beta, -alpha)
                board.pop()
                if score >= beta:
                    return beta
                if score > alpha:
                    alpha = score
        return alpha


            

