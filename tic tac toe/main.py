# Imports
from time import sleep
from random import choice
from joblib import load
import pandas as pd



# Load Model
next_move_model = load('./models/joblib/next_move_model.joblib')
game_status_model = load('./models/joblib/game_status_model.joblib')


# Functions
def print_board(board):
    """
        this function create the table of the game and print that

        :param board: the table of game
        :return: nothing just print the table
    """
    symbols = {0: " ", 1: "X", 2: "O"}
    print("\n-------------")
    for i in range(3):
        row = " | ".join(symbols[board[j]] for j in range(i*3, i*3+3))
        print(f"| {row} |")
        print("-------------")

def choose_column(num, symbol, board):
    """
        this function should change the table of the game with the params

        :param num: the number of columns
        :param symbol: the shape of in game
        :param board: the table of game
        :return: nothing
    """
    index = num - 1
    board[index] = 1 if symbol == 'X' else 2

def computer_move(board):
    """
        this function should return what computer want to choose

        :param board: the table of game
        :return: the number of columns which is computer choice
    """
    df_board = pd.DataFrame([board], columns=[f"c{i}" for i in range(1,10)])
    next_move = next_move_model.predict(df_board)
    return int(next_move[0])

def check_who_win(board):
    """
        this function should to check the table and tell us who win the game

        :param board: the table of game
        :return: the number of columns which is computer choice
    """
    df_board = pd.DataFrame([board], columns=[f"c{i}" for i in range(1, 10)])
    status_array = game_status_model.predict(df_board)
    return int(status_array[0])

def check_column(board, num):
    """
        this function should check the column which is user chose and tell us that it is fill or no

        :param num: the number of columns
        :param board: the table of game
        :return: a Boolean
    """
    index = num -1
    if board[index] != 0:
        return False
    return True


# Game Variables
main_board = [0] * 9
computer_symbol = choice(['X', 'O'])
user_symbol = 'X' if computer_symbol == 'O' else 'O'
counter = 1

# Run
if __name__ == '__main__':
    print('Hello Wellcome To Tic Tac Toe Game')
    sleep(1)
    print('Tips : For Choose a Columns You Should Pass Me The Number Of Columns')
    print_board(main_board)
    sleep(1)
    print('This Is The Game Board')
    print(f'This Is Your Symbol {user_symbol}')
    print(f'This Is Computer Symbol {computer_symbol}')
    while True :
        try :
            number = int(input(f'Type The Number Of Columns ( Try {counter} ): '))
        except ValueError:
            print('Type Number Please :)')
            continue

        if check_column(main_board, number):
            choose_column(number, user_symbol, main_board)
            move = computer_move(main_board)
            choose_column(move + 1, computer_symbol, main_board)
            print_board(main_board)
            counter = counter + 1
            winner_number = check_who_win(main_board)

            match winner_number:
                case 0:
                    continue
                case 1:
                    print('X win')
                case 2:
                    print('O win')
                case -1:
                    print('Draw')
        else:
            print('The Number You Passed To Me Is Fill Please Try an Other Number :)')
            continue
        break


