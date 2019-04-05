
class Board(object):
    """board for the game"""

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 8))
        self.height = int(kwargs.get('height', 8))
        # board states stored as a dict,
        # key: move as location on the board,
        # value: player as pieces type
        self.states = {}
        # need how many pieces in a row to win                                              
        self.players = [1, 2]  # player1 and player2

    def init_board(self, start_player=0):
        self.current_player = self.players[start_player]  # start player
        # keep available moves in a list
        self.availables = []
        self.states = {}
        self.current_player = self.players[start_player]
        self.board_value = [[0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0],
                            [0,0,0,2,1,0,0,0],
                            [0,0,0,1,2,0,0,0],
                            [0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0],
                            [0,0,0,0,0,0,0,0]]
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
            print(num)
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

    def graphic(self, player1, player2):
        """Draw the board and show game info"""
        width = self.width
        height = self.height

        print("Player", player1, "with X".rjust(3))
        print("Player", player2, "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x), end='')
        print('\r\n')
        for i in range(height - 1, -1, -1):
            print("{0:4d}".format(i), end='')
            for j in range(width):
                p = self.board_value[i][j]
                if p == player1:
                    print('X'.center(8), end='')
                elif p == player2:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

board = Board()
board.init_board()
board.update_available()
board.do_move(19)
board.graphic(1,2)
print(board.availables)

