#!/usr/bin/env python

# app.py
# Authors: Matthew Chan and Jeremy Savarin
# CS440: Project 1 - Heuristic Search

import sys

from heuristic_search.grid import run_search

'''
usage: python app.py {mode pairs}
'file'   [filename]         - generate grid from file
    default: False
'gen'    [int(seed)|'None'] - generate grid from seed (None for random)
    default: None
'rows'   [int > 0]          - number of rows
    default: 100
'cols'   [int > 0]          - number of cols
    default: 100
'plen'   [int > 0]          - length of path
    default: 100
'''


if __name__ == '__main__':
    filename = None
    rand_state = None
    mode = True  # true=gen, false=file
    rows = 100
    cols = 100
    pathlength = 100


    if len(sys.argv) < 3:
        print("====================================")
        print(" usage: python app.py {mode pairs}\n 'file'   [filename]         - generate grid from file\n default: False\n 'gen'    [int(seed)|'None'] - generate grid from seed (None for random)\n default: None\n 'rows'   [int > 0]          - number of rows\n default: 100\n 'cols'   [int > 0]          - number of cols\n default: 100\n 'plen'   [int > 0]          - length of path\n default: 100 ")
        print("====================================")

        if len(sys.argv)==2 and sys.argv[1]=='help':
            sys.exit()

    print("====================================")
    print("Commands:\n\
          s - step once\n\
          c - continuous step (hold down)\n\
          h - heatmap toggle\n\
          a - toggle most-likely explanation\n\
          g - toggle ground truth\n\
          b - goto step 0\n\
          e - goto last step\n\
          v - show 10 most-likely explanations\n\
          w - print grid to a file (uses current time)\n\
          esc - exit")
    print("====================================")


    for i in range(len(sys.argv)):
        if i == (len(sys.argv) - 1):
            break

        if sys.argv[i] == 'file':
            filename = sys.argv[i+1]
            mode = False

        if sys.argv[i] == 'gen':
            if sys.argv[i+1] != 'None':
                rand_state = int(sys.argv[i+1])

        if sys.argv[i] == 'rows':
            if 0 < int(sys.argv[i+1]):  # valid row
                rows = int(sys.argv[i+1])

        if sys.argv[i] == 'cols':
            if 0 < int(sys.argv[i+1]):  # valid col
                cols = int(sys.argv[i+1])

        if sys.argv[i] == 'plen':
            if 0 < int(sys.argv[i+1]):  # valid length
                pathlength = int(sys.argv[i+1])



    run_search(rows, cols, pathlength, mode, filename, rand_state, existing=None)
    # if want existing, just write it to a file
