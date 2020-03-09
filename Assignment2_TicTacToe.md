
# Python Script to Play Tic-Tac-Toe!

### Just Run this Script to Start a 2-Player Game


```python
### Read in Libraries and Set Initial, Empty Game Board
import numpy as np
board_original=[['A1','B1','C1'],['A2','B2','C2'],['A3','B3','C3']]
```


```python
### Function to Initialize Board
def initialboard(boardlist):

    print(boardlist[0])
    print(boardlist[1])
    print(boardlist[2])
    
### Function to For Choosing Correct Player (X or O) Each Turn
def chooseplayer(turn):
    if (turn+1)%2==1:
        player='X'
    else:
        player='O'
    return player

### Function to Take and Ensure Valid Move Each Turn
def getmove(startboard,taken):
    while True:
        
        position = input( "Choose a position: " )
        
        if position in taken:
            print("Position already taken, please try again")
        
        elif position not in np.array(startboard):
            print("Invalid position, please try again")
        
        else:
            taken.append(position)
            break
    
    return position


### Function to Determine if a Player Has Won any Given Turn
def win(boardlist,index,token):
    if boardlist[0][index]==token and boardlist[1][index]==token and boardlist[2][index]==token:
        return True
    
    elif len( set( boardlist[0] ) ) == 1 or len( set( boardlist[1] ) ) == 1 or len( set( boardlist[2] ) )==1:
        return True
        
    elif (boardlist[0][0]==token and boardlist[1][1]==token and boardlist[2][2]==token) | (boardlist[0][2]==token and boardlist[1][1]=='X' and boardlist[2][0]==token):
        return True


  

```


```python
### Function for Playing
def tic_tac_toe(startboard):
    
    # Initialize Empty Board
    board=[['A1','B1','C1'],['A2','B2','C2'],['A3','B3','C3']]
    initialboard(board)
    
    # Run through each turn (9 maximum) until someone wins or there is a tie
    position_taken=[]
    for play in range(9):
        
        # Set X or O
        player=chooseplayer(play)
        
        # Take Position/Move
        position=getmove(startboard,position_taken)
    
        i1=0
        for row in board:
            for i in range(len(row)):
                if row[i]==position:
                    row[i]=player
                    i1=i  
            print(row)
        
        # Check if the player has won and, if so, end the game
        if win(board,i1,player)==True:
            print('Player' + ' ' + str(player) + ' ' + 'Wins!')
            break
        
        # End the game if the board is full and there is a tie
        elif play==8:
            print('Tie!')
            break

```


```python
### Play the Game!
tic_tac_toe(board_original)
```

    ['A1', 'B1', 'C1']
    ['A2', 'B2', 'C2']
    ['A3', 'B3', 'C3']
    Choose a position: A1
    ['X', 'B1', 'C1']
    ['A2', 'B2', 'C2']
    ['A3', 'B3', 'C3']
    Choose a position: C1
    ['X', 'B1', 'O']
    ['A2', 'B2', 'C2']
    ['A3', 'B3', 'C3']
    Choose a position: B2
    ['X', 'B1', 'O']
    ['A2', 'X', 'C2']
    ['A3', 'B3', 'C3']
    Choose a position: C2
    ['X', 'B1', 'O']
    ['A2', 'X', 'O']
    ['A3', 'B3', 'C3']
    Choose a position: C3
    ['X', 'B1', 'O']
    ['A2', 'X', 'O']
    ['A3', 'B3', 'X']
    Player X Wins!



