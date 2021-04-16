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
    loaded_model = None
    opening_book = None
    opening = None
    
    # model loading
    def __init__(self, board, color, time):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        os.chdir(os.path.join(dir_path, ".."))
        os.chdir("storage")
        self.loaded_model = models.load_model("chess_model.h5")
        os.chdir(os.path.join("..", os.path.join("code1", "books")))
        self.opening_book = chess.polyglot.MemoryMappedReader("elo-2700.bin")
        self.opening = True


    # get a matrix representation given a board
    def matrix_rep(self, board): 
        board_epd = board.epd()
        mat = []  
        pieces = board_epd.split(" ", 1)[0]
        rows = pieces.split("/")
        for r in rows:
            sub_mat = []  
            for pic in r:
                if pic.isdigit():
                    for i in range(0, int(pic)):
                        sub_mat.append('.')
                else:
                    sub_mat.append(pic)
            mat.append(sub_mat)
        return mat


    # encode the matrix using some dictionary
    def trans_code(self, matrix,pieces_dict):
        rows = []
        for row in matrix:
            terms = []
            for term in row:
                terms.append(pieces_dict[term])
            rows.append(terms)
        return rows

    # dictionary used to encode pieces
    one_hot_pieces_dict = {
        'p' : [1,0,0,0,0,0,0,0,0,0,0,0],
        'P' : [0,0,0,0,0,0,1,0,0,0,0,0],
        'n' : [0,1,0,0,0,0,0,0,0,0,0,0],
        'N' : [0,0,0,0,0,0,0,1,0,0,0,0],
        'b' : [0,0,1,0,0,0,0,0,0,0,0,0],
        'B' : [0,0,0,0,0,0,0,0,1,0,0,0],
        'r' : [0,0,0,1,0,0,0,0,0,0,0,0],
        'R' : [0,0,0,0,0,0,0,0,0,1,0,0],
        'q' : [0,0,0,0,1,0,0,0,0,0,0,0],
        'Q' : [0,0,0,0,0,0,0,0,0,0,1,0],
        'k' : [0,0,0,0,0,1,0,0,0,0,0,0],
        'K' : [0,0,0,0,0,0,0,0,0,0,0,1],
        '.' : [0,0,0,0,0,0,0,0,0,0,0,0],
    }

    
        # print("done loading")

    # creates a Tensorflow function using the model we loaded in the constructor
    # preferrable over to simply calling self.loaded_model(matrix) due to some Tensorflow voodoo magic

    @tf.function
    def pre_eval(self, trans_mat_reshape):
        
        return self.loaded_model(trans_mat_reshape)

    def local_eval_func(self, board):
        mat = self.matrix_rep(board)
        trans_mat = self.trans_code(mat, self.one_hot_pieces_dict)
        trans_mat_reshape = np.array(trans_mat).reshape((1,8,8,12))

        # return self.loaded_model.predict(trans_mat_reshape).reshape((1))[0]
        return (self.pre_eval(trans_mat_reshape)[0][0])

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


            

