chess = {'King': 1, 'Queen' : 1, 'Rook': 2, 'Bishop': 2, 'Knight': 2, 'Pawn': 8}
chess_input = input().split()

chess_input = list(map(int, chess_input))

for n, s in zip(chess_input, chess.values()):
    print(s - int(n), end=' ')