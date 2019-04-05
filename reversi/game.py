# -*- coding: utf-8 -*-
"""
@author: Junxiao Song
"""

from __future__ import print_function
import numpy as np

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
        self.update_available()
        self.last_move = -1

    def current_state(self):
        """return the board state from the perspective of the current player.
        state shape: 4*width*height
        """
        square_state = np.zeros((2, self.width, self.height))
        if self.states:
            for i in range(8):
                for j in range(8):
                    if self.board_value[i][j]==self.current_player:
                        square_state[0][i][j]=1
                    elif self.board_value[i][j]!=self.current_player and self.board_value[i][j]!= 0:
                        square_state[1][i][j]=1
            # indicate the last move location
       #    square_state[2][self.last_move // self.width, self.last_move % self.height] = 1.0
        #if len(self.states) % 2 == 0:
          #  square_state[3][:, :] = 1.0  # indicate the colour to play
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
                            if self.board_value[temp_h][temp_w] == 0:
                                break
                            elif self.board_value[temp_h][temp_w] == self.current_player:
                                flag = 1
                                break
                            else:
                                num+=1
                                temp_h += list_temp_dic[0]
                                temp_w += list_temp_dic[1]
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
        flag=1
        sum1 = 0
        sum2 = 0
        for i in range(8):
            for j in range(8):
                if self.board_value[i][j]==1:
                    sum1+=1
                elif self.board_value[i][j]==2:
                    sum2+=1
                else:
                    flag = 0
        if flag == 1:
            if sum1 > sum2:
                return True,1
            if sum1 < sum2:
                return True,2
            return True,-1
        if sum1==0:
            return True,2
        if sum2==0:
            return True,1
        return False, -1
         
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
