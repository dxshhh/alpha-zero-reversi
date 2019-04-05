# -*- coding: utf-8 -*-
"""
Monte Carlo Tree Search in AlphaGo Zero style, which uses a policy-value
network to guide the tree search and evaluate the leaf nodes

@author: Junxiao Song
"""
from __future__ import print_function
from operator import itemgetter
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import json
import numpy
import random

DIR = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)) # 方向向量
use_gpu = torch.cuda.is_available()


def rollout_policy_fn(board):
    """a coarse, fast version of policy_fn used in the rollout phase."""
    # rollout randomly 
    if len(board.availables)== 0:
        return -100
    action_probs = np.random.rand(len(board.availables))
    return zip(board.availables, action_probs)


def policy_value_fn(board):
    """a function that takes in a state and outputs a list of (action, probability)
    tuples and a score for the state"""
    # return uniform probabilities and 0 score for pure MCTS
    action_probs = np.ones(len(board.availables))/len(board.availables)
    return zip(board.availables, action_probs), 0


class TreeNode(object):
    """A node in the MCTS tree. Each node keeps track of its own value Q,
    prior probability P, and its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self, action_priors):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        for action, prob in action_priors:
            if action not in self._children:
                self._children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._u = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded).
        """
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """A simple implementation of Monte Carlo Tree Search."""

    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout

    def _playout(self, state):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while(1):
            if node.is_leaf():
                break
            # Greedily select next move.
            action, node = node.select(self._c_puct)
            state.do_move(action)

        action_probs, _ = self._policy(state)
        # Check for end of game
        end, winner = state.game_end()
        leaf_value = 0
        if not end:
            node.expand(action_probs)
            leaf_value = self._evaluate_rollout(state)
        else:
            if winner == -1:
                leaf_value = 0.0
            else:
                leaf_value = (1.0 if winner == state.get_current_player() else -1.0)
        
        node.update_recursive(-leaf_value)

    def _evaluate_rollout(self, state, limit=1000):
        """Use the rollout policy to play until the end of the game,
        returning +1 if the current player wins, -1 if the opponent wins,
        and 0 if it is a tie.
        """
        
        """player = state.get_current_player()
        for i in range(limit):
            end, winner = state.game_end()
            if end:
                break
            action_probs = rollout_policy_fn(state)
            max_action = max(action_probs, key=itemgetter(1))[0]
            state.do_move(max_action)
        else:
            # If no break from the loop, issue a warning.
            print("WARNING: rollout reached move limit")
        if winner == -1:  # tie
            return 0
        else:
            return 1 if winner == player else -1"""
        evl = 0
        player = state.get_current_player()
        board_value=[i[:] for i in state.board_value]
        for i in range(8):
            for j in range(8):
                board_value[i][j] = (board_value[i][j]*3-5)*board_value[i][j]

        for i in range(2,6):
            evl+=board_value[i][0]
        for i in range(2,6):
            evl+=board_value[i][7]
        for i in range(2,6):
            evl+=board_value[0][i]
        for i in range(2,6):
            evl+=board_value[7][i]

        evl+= board_value[0][0]*4
        evl+= board_value[0][7]*4
        evl+= board_value[7][0]*4
        evl+= board_value[7][7]*4
        
        if not board_value[0][0]:
            evl -= board_value[1][0]
            evl -= board_value[0][1]
            evl -= board_value[1][1]
        if not board_value[7][7]:
            evl -= board_value[6][6]
            evl -= board_value[7][6]
            evl -= board_value[6][7]
        if not board_value[7][0]:
            evl -= board_value[6][1]
            evl -= board_value[6][0]
            evl -= board_value[7][1]
        if not board_value[0][7]:
            evl -= board_value[1][6]
            evl -= board_value[0][6]
            evl -= board_value[1][7]

        if evl ==0:
            return 0
        if (player == 1 and evl < 0) or (player == 2 and evl > 0):
            return 1
        return -1




    def get_move(self, state):
        """Runs all playouts sequentially and returns the most visited action.
        state: the current game state

        Return: the selected action
        """
        for n in range(self._n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)
        return max(self._root._children.items(),
                   key=lambda act_node: act_node[1]._n_visits)[0]

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"


class MCTSPlayer(object):
    """AI player based on MCTS"""
    def __init__(self, c_puct=5, n_playout=2000):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)

    def set_player_ind(self, p):
        self.player = p

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        sensible_moves = board.availables
        if len(sensible_moves) > 0:
            move = self.mcts.get_move(board)
            self.mcts.update_with_move(-1)
            return move
        else:
            return -1
          
    def __str__(self):
        return "MCTS {}".format(self.player)




# -*- coding: utf-8 -*-
"""
@author: Junxiao Song
"""

Init_board = [[0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0],
              [0,0,0,2,1,0,0,0],
              [0,0,0,1,2,0,0,0],
              [0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0],
              [0,0,0,0,0,0,0,0]]

class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}                                   
        self.players = [1, 2]

    def init_board(self, start_player=0, board_value = Init_board):
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = []
        self.states = {}
        self.current_player = self.players[start_player]
        self.board_value = [i[:] for i in board_value]
        for i in range(8):
            for j in range(8):
                if self.board_value[i][j]==-1:
                    self.board_value[i][j]=2
        self.update_available()
        self.last_move = -1

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """
        square_state = np.zeros((4, self.width, self.height))
        if self.states:
            for i in range(8):
                for j in range(8):
                    if self.board_value[i][j]==self.current_player:
                        square_state[0][i][j]=1
                    elif self.board_value[i][j]!=self.current_player and self.board_value[i][j]!= 0:
                        square_state[1][i][j]=1
            # indicate the last move location
            square_state[2][self.last_move // self.width, self.last_move % self.height] = 1.0
        if len(self.states) % 2 == 0:
            square_state[3][:, :] = 1.0  # indicate the colour to play
        return square_state[:, ::-1, :]


    def change_board(self,move):
        h = move // self.width
        w = move % self.width

        self.board_value[h][w] = self.current_player
        oppo_player = (self.players[0] if self.current_player == self.players[1] 
                    else self.players[1])
        list_dic = [[0,1],[0,-1],[1,0],[-1,0],[1,1],[-1,-1],[1,-1],[-1,1]]
        for i in range(8):
            list_temp_dic = list_dic[i]

            temp_h = h + list_temp_dic[0]
            temp_w = w + list_temp_dic[1]
            flag = 0
            num = 0
            while temp_h>=0 and temp_w>=0 and temp_h<8 and temp_w<8:
                if self.board_value[temp_h][temp_w] == oppo_player:
                    num+=1
                    temp_h += list_temp_dic[0]
                    temp_w += list_temp_dic[1]
                elif self.board_value[temp_h][temp_w] == self.current_player:
                    flag = 1
                    break
                else:
                    break
            #print(num)
            if flag == 1:
                for j in range(num):
                    self.board_value[h+(j+1)*list_temp_dic[0]][w+(j+1)*list_temp_dic[1]]=self.current_player

    def update_available(self):
        oppo_player = (self.players[0] if self.current_player == self.players[1] 
                    else self.players[1])
        self.availables = []
        for temp_i in range(8):
            for temp_j in range(8):
                if self.board_value[temp_i][temp_j]== 0:
                    can_it = 0
                    list_dic = [[0,1],[0,-1],[1,0],[-1,0],[1,1],[-1,-1],[1,-1],[-1,1]]
                    for i in range(8):
                        list_temp_dic = list_dic[i]
                        temp_h = temp_i + list_temp_dic[0]
                        temp_w = temp_j + list_temp_dic[1]
                        flag = 0
                        num = 0
                        while temp_h>=0 and temp_w>=0 and temp_h<8 and temp_w<8:
                            if self.board_value[temp_h][temp_w] == oppo_player:
                                num+=1
                                temp_h += list_temp_dic[0]
                                temp_w += list_temp_dic[1]
                            elif self.board_value[temp_h][temp_w] == self.current_player:
                                flag = 1
                                break
                            else:
                                break
                        if flag == 1 and num > 0:
                            can_it = 1
                            break
                    if can_it == 1:
                        self.availables.append(temp_i*8+temp_j)
                        


    def do_move(self, move):
        if move == -1:
            self.current_player = (self.players[0] if self.current_player == self.players[1] else self.players[1])
            return 

        self.states[move] = self.current_player
        self.change_board(move)
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )

        self.update_available()
        self.last_move = move
    
    def who_win(self):
        num_1=0
        num_2=0
        for i in range(8):
            for j in range(8):
                if self.board_value[i][j]==1:
                    num_1+=1
                if self.board_value[i][j]==2:
                    num_2+=1
        if num_1 > num_2:
            return 1
        if num_1 < num_2:
            return 2
        return -1

    def game_end(self):
        if len(self.availables) != 0:
            return False, -1
        flag=0

        temp_player=self.current_player
        temp_a = self.availables[:]
        self.current_player = (self.players[0] if self.current_player == self.players[1] 
                    else self.players[1])
        self.update_available()
        if len(self.availables) == 0:
            flag=1
        self.current_player = temp_player
        self.availables = temp_a
 
        if flag == 0:
            return False, -1
        else:
            return True, self.who_win()


    def get_current_player(self):
        return self.current_player

    def location_to_move(self, location):
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = h * self.width + w
        if move not in range(self.width * self.height):
            return -1
        return move

class Game(object):
    """game server"""

    def __init__(self, board, **kwargs):
        self.board = board

    def graphic(self, board, player1, player2):
        """Draw the board and show game info"""
        width = board.width
        height = board.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                p = board.board_value[i][j]
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """start a game between two players"""
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board.init_board(start_player)
        p1, p2 = self.board.players
        player1.set_player_ind(p1)
        player2.set_player_ind(p2)
        players = {p1: player1, p2: player2}
        if is_shown:
            self.graphic(self.board, player1.player, player2.player)
        while True:
            move = -1
            current_player = self.board.current_player
            player_in_turn = players[current_player]
            if len(self.board.availables) != 0:
                move = player_in_turn.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", players[winner])
                    else:
                        print("Game end. Tie")
                return winner

    def start_self_play(self, player, is_shown=0, temp=1e-3):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        self.board.init_board()
        p1, p2 = self.board.players
        states, mcts_probs, current_players = [], [], []

        mem_for_win = 0
        

        while True:
            if is_shown:
                self.graphic(self.board, 1, 2)

            move = -1
            move_probs = 1
            if len(self.board.availables)!=0:
                mem_for_win = 0
                move, move_probs = player.get_action(self.board,
                                                 temp=temp,
                                                 return_prob=1)
                states.append(self.board.current_state())
                mcts_probs.append(move_probs)
                current_players.append(self.board.current_player)    
            else:
                mem_for_win+=1
                if mem_for_win == 2:
                    winner = self.board.who_win()
                    winners_z = np.zeros(len(current_players))
                    if winner != -1:
                        winners_z[np.array(current_players) == winner] = 1.0
                        winners_z[np.array(current_players) != winner] = -1.0
                # reset MCTS root node
                    player.reset_player()
                    if is_shown:
                        if winner != -1:
                            print("Game end. Winner is player:", winner)
                        else:
                            print("Game end. Tie")
                    return winner, zip(states, mcts_probs, winners_z)
            self.board.do_move(move)




# 放置棋子，计算新局面
def place(board, x, y, color):
    if x < 0:
        return False
    board[x][y] = color
    valid = False
    for d in range(8):
        i = x + DIR[d][0]
        j = y + DIR[d][1]
        while 0 <= i and i < 8 and 0 <= j and j < 8 and board[i][j] == -color:
            i += DIR[d][0]
            j += DIR[d][1]
        if 0 <= i and i < 8 and 0 <= j and j < 8 and board[i][j] == color:
            while True:
                i -= DIR[d][0]
                j -= DIR[d][1]
                if i == x and j == y:
                    break
                valid = True
                board[i][j] = color
    return valid

def my_place(board, color):
    mcts_player = MCTSPlayer(5,2000)
    current_board = Board(width=8, height=8)
    if color == -1:
        current_board.init_board(1,board)
    else:
        current_board.init_board(0,board)
    move = mcts_player.get_action(current_board)
    moves = (-1,-1)
    if move!=-1:
        moves = (int(move // 8), int(move % 8))
    return moves


def initBoard():
    fullInput = json.loads(input())
    requests = fullInput["requests"]
    responses = fullInput["responses"]
    board = numpy.zeros((8, 8), dtype=numpy.int)
    board[3][4] = board[4][3] = 1
    board[3][3] = board[4][4] = -1
    myColor = 1
    if requests[0]["x"] >= 0:
        myColor = -1
        place(board, requests[0]["x"], requests[0]["y"], -myColor)
    turn = len(responses)
    for i in range(turn):
        place(board, responses[i]["x"], responses[i]["y"], myColor)
        place(board, requests[i + 1]["x"], requests[i + 1]["y"], -myColor)
    return board, myColor


board, myColor = initBoard()
x, y = my_place(board, myColor)

#print(x,y)

print(json.dumps({"response": {"x": x, "y": y}}))

















