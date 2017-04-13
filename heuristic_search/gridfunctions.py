#!/usr/bin/env python

# heuristic_search/gridfunctions.py
# CS440: Project 1 - Heuristic Search
# Authors: Matthew Chan and Jeremy Savarin

import random
import copy
import time
#from my_pq import My_PQ
from heuristic_search.my_pq import My_PQ

'''
class Grid - class to encompass data representation of grid world and build
grid world from scratch. The following represents the type of squares in
the grid:

    * 0 - blocked cell
    * 1 - unblocked cell
    * 2 - hard-to-traverse cell
    * a - highway cell (regular)
    * b - highway cell (hard-to-traverse)
'''


class Grid:

    def __init__(self, rows=120, cols=160, rand_state=None, existing=None, path=None, pathlength=100, start_cell=None):
        # Choose a seed value for repeatable results
        if rand_state != None: # without none, misses 0
            random.seed(rand_state)

        if existing:
            self.ROWS = len(existing)
            self.COLS = len(existing[0])
            self.grid = [[existing[r][c] for c in range(self.COLS)] for r in range(self.ROWS)]
        else:
            self.ROWS = rows
            self.COLS = cols
            self.grid = self.create_grid()

        if path != None:
            self.actions = ''.join(path[0])
            self.observs = ''.join(path[1])
            self.pathlength = len(path[0])

            self.start_cell = None
            self.gt_a = None
            self.gt_o = None
        else:
            # creating a path
            sc, a, o, gta, gto = self.create_a_path(pathlength, start_cell)
            self.actions = a
            self.observs = o
            self.pathlength = pathlength

            self.start_cell = tuple(sc)
            self.gt_a = gta
            self.gt_o = gto

        self.probs = self.initialize_probabilities()
        self.most_likely = [["" for c in range(self.COLS)] for r in range(self.ROWS)]
        self.step = 0
        self.most_likely_cell = None

        return


    def create_grid(self):
        grid = [['1' for c in range(self.COLS)] for r in range(self.ROWS)]

        for r in range(self.ROWS):
            for c in range(self.COLS):

                if self.generate_probability(.5): # not normal
                    if self.generate_probability(.2):
                        grid[r][c] = '0'  # blocked
                    elif self.generate_probability(.5):
                        grid[r][c] = 'a'  # highway
                    else:
                        grid[r][c] = '2'  # htt

        return grid

    def viterbi(trans, obs, model_t, model_o):
        #TODO: find most likely for each cell
        # only using observation model
        most_likely = []

        return

    def find_most_likely_paths(self, paths=10):
        pq = My_PQ()
        most_probable = []
        for r in range(self.ROWS):
            for c in range(self.COLS):
                pq.put( (-self.probs[r][c], (r,c)) )

        for i in range(paths):
            if pq.qsize() > 0:
                most_probable.append(pq.get()[1])
        
        return most_probable



    def step_grid(self):
        if self.step > self.pathlength - 1:
            print("Already at last step")
            return 1

        self.eval_transition( self.actions[self.step])
        self.eval_observation(self.observs[self.step])
        self.normalize()
        #print(self.most_likely_cell)
        self.step += 1

        return 0

    def reset_steps(self):
        self.probs = self.initialize_probabilities()
        self.most_likely = [["" for c in range(self.COLS)] for r in range(self.ROWS)]
        self.step = 0

        return


    def oob(self, cell):
        if cell[0] >= self.ROWS or cell[1] >= self.COLS or cell[0] < 0 or cell[1] < 0:
            return True
        else:
            return False


    def can_move(self, cell): # checks can move from/to
        if self.oob(cell):
            return False
        if self.grid[cell[0]][cell[1]] == '0':
            return False
        return True


    def initialize_probabilities(self):
        num_blocked = 0;
        probs = [[0 for c in range(self.COLS)] for r in range(self.ROWS)]
        for r in range(self.ROWS):
            for c in range(self.COLS):
                if self.grid[r][c] == '0':
                    num_blocked += 1

        prob = 1.0/(self.ROWS * self.COLS - num_blocked)

        for r in range(self.ROWS):
            for c in range(self.COLS):
                if self.grid[r][c] != '0':
                    probs[r][c] = prob

        return probs


    # directions: 'r', 'l', 'u', 'd'
    def eval_transition(self, direction, success_model = None):
        if not success_model:
            success_model = [.9, .1] # success, failure

        s = success_model[0]
        f = success_model[1]
        prob_map = [[0 for c in range(self.COLS)] for r in range(self.ROWS)]



        ro = 0  # row and col offset
        co = 0

        if   direction == 'r': co = 1
        elif direction == 'l': co = -1
        elif direction == 'u': ro = -1
        elif direction == 'd': ro = 1

        # calculating transition probs for each cell
        for r in range(self.ROWS):
            for c in range(self.COLS):

                # for most lkely explanation
                p_this = -1
                p_from = -1

                # check blocked
                if self.grid[r][c] == '0': continue

                # calculating move failure probability
                p_this = f * self.probs[r][c]

                # checking this cell if success=fail (on border/blocked movement)
                if not self.can_move( (r+ro, c+co) ):
                    p_this += s * self.probs[r][c]

                prob_map[r][c] = p_this


                # checking move to this cell success
                if self.can_move( (r-ro,c-co) ):
                    p_from = s * self.probs[r-ro][c-co]
                    prob_map[r][c] += p_from

                # book-keeping for most likely explanation
                if p_this < p_from:
                    self.most_likely[r][c] += direction
                else:
                    self.most_likely[r][c] += '.'

        self.probs = prob_map

        return prob_map


    def eval_observation(self, obs, obs_model=None):
        if not obs_model:
            obs_model = [.9, .05]

        s = obs_model[0]
        f = obs_model[1]
        obs_map = [[0 for c in range(self.COLS)] for r in range(self.ROWS)]

        p_this = 0
        p_from = 0

        for r in range(self.ROWS):
            for c in range(self.COLS):
                if self.grid[r][c] == obs:
                    self.probs[r][c] *= s
                    obs_map[r][c] = s
                else:
                    self.probs[r][c] *= f
                    obs_map[r][c] = f

        return obs_map


    def normalize(self, EPSILON=-1):
        prob_sum = 0
        max_prob = -1
        max_cell = None
        #EPSILON = 1e-16

        for r in range(self.ROWS):
            for c in range(self.COLS):
                prob_sum += self.probs[r][c]

                if self.probs[r][c] > max_prob:
                    max_prob = self.probs[r][c]
                    max_cell = (r,c)
        self.most_likely_cell = max_cell

        for r in range(self.ROWS):
            for c in range(self.COLS):
                self.probs[r][c] = self.probs[r][c] / prob_sum
                if self.probs[r][c] < EPSILON:
                    self.probs[r][c] = 0 # for ease of use

        return prob_sum

    def find_mle(self, coord=None):
        r = -1
        c = -1

        if coord == None:
            m = -1
            for i in range(self.ROWS):
                for j in range(self.COLS):
                    if self.probs[i][j] > m:
                        m = self.probs[i][j]
                        r = i
                        c = j
        else:
            # coord should be 0-indexed
            r = coord[0]
            c = coord[1]

        path = []
        if len(self.most_likely[0][0]) == 0:
            return path

        for i in range(len(self.most_likely[0][0])-1, -1, -1):
            if len(self.most_likely[r][c]) == 0: continue  # blocked cell
            prev = self.most_likely[r][c][i]
            path.append( (r,c) )

            # direction that moved us to this cell
            if prev == '.': continue
            elif prev == 'r': c -= 1
            elif prev == 'l': c += 1
            elif prev == 'u': r += 1
            elif prev == 'd': r -= 1

        return path[::-1] # reverse


    '''
    fn generate_probability(p) - returns True with probability p, False
    with probability 1-p.
    '''
    def generate_probability(self, p):
        if (p < 0 or p > 1):
            raise ValueError("p can only be between 0 and 1.")

        return random.random() < p


    '''
    fn print_grid() - prints data representation of grid world to stdout
    '''
    def print_grid(self, pretty_print=False):
        vals = {
            '0': 'X', # blocked
            '1': '.', # normal
            '2': '^', # htt
            'a': 'H', # highway
            'b': '#', # htt-highway
            'S': 'S', # start
            'G': 'G'  # goal
        }

        print('\t+' + '-' * self.COLS + '+')

        for r in range(self.ROWS):
            print(str(r) + '\t|', end='')

            for c in range(self.COLS):
                if pretty_print: print(vals[self.grid[r][c]], end='')
                else:            print(self.grid[r][c], end='')

            print('|')

        print('\t+' + '-' * self.COLS + '+')
        if pretty_print:
            print('Size (r, c):', (self.ROWS, self.COLS))
            print('start:', self.start_cell)
            print('act:', self.actions)
            print('obs:', self.observs)
            print()
            print('gta:', self.gt_a)
            print('gto:', self.gt_o)


    def create_a_path(self, length, start_cell=None, interesting=True):
        # TODO: possibly create paths that don't include dir that came from, aka 'interesting paths'
        if start_cell == None:
            start_cell = (random.randrange(self.ROWS-1), random.randrange(self.COLS-1))

        while self.grid[start_cell[0]][start_cell[1]] == '0':  # blocked
            start_cell = [random.randrange(self.ROWS-1), random.randrange(self.COLS-1)]

        cell = [start_cell[0], start_cell[1]]
        actions = ""
        #act_map = {0:'l', 1:'r', 2:'u', 3:'d'}
        act_map = {0:'u', 1:'r', 2:'d', 3:'l'}
        observs = ""
        obs_map = {'1':0, '2':1, 'a':2}  # normal, htt, highway
        obs_map2= ['1', '2', 'a']

        gt_a = ""  # ground truth - actions
        gt_o = ""  # ground truth - observations
        prev_action = -1

        for i in range(length):
            action = random.randrange(4)
            if(interesting):
                while prev_action == ((action+2)%4):
                    action = random.randrange(4)

            prev_action = action
            actions += act_map[action]
            ro = 0
            co = 0
            
            if self.generate_probability(.9):  # calculating offsets
                if   action == 0: ro -= 1
                elif action == 1: co += 1
                elif action == 2: ro += 1
                elif action == 3: co -= 1
                if self.can_move( (cell[0]+ro, cell[1]+co) ):  # testing action success
                    cell[0] += ro
                    cell[1] += co
                    gt_a += act_map[action]
                else:
                    gt_a += '.'
            else:
                gt_a += '.'

            if self.generate_probability(.9):  # testing observation success
                observs += self.grid[cell[0]][cell[1]]
            else:
                obs = ( obs_map[self.grid[cell[0]][cell[1]]] + random.randint(1,2) ) % 3
                observs += obs_map2[obs]

            gt_o += self.grid[cell[0]][cell[1]]

        return (start_cell, actions, observs, gt_a, gt_o)


    def create_a_new_path(self, length=100, start_cell=None):
        sc, a, o, gta, gto = self.create_a_path(length, start_cell)

        self.start_cell = sc
        self.actions = a
        self.observs = o
        self.gt_a = gta
        self.gt_o = gto

        self.step = 0
        self.probs = self.initialize_probabilities()
        self.most_likely = [["" for c in range(self.COLS)] for r in range(self.ROWS)]
        self.most_likely_cell = None


    '''
    fn print_grid_to_file(filename) - prints grid data to file given by
    filename
    '''
    def print_grid_to_file(self, filename):
        with open(filename, 'w') as file:
            # Print start and goal coordinates
            if self.start_cell == None:
                file.write(str((-1,-1)) + '\n')
            else:
                file.write(str(self.start_cell) + '\n')

            # Print ground truth
            file.write('ground_truth:' + '\n')
            if self.gt_a == None:
                file.write('\n')
            else:
                file.write(self.gt_a + '\n')


            # Print actions
            file.write('actions:' + '\n')
            file.write(self.actions + '\n')

            # Print observations
            file.write('observations:' + '\n')
            file.write(self.observs + '\n')


            # Print data representation of the grid
            file.write('grid:' + '\n')

            for r in range(self.ROWS):
                for c in range(self.COLS):
                    file.write(self.grid[r][c])
                file.write('\n')

        return

    def read_grid_from_file(filename):
        lines = []

        with open(filename, 'r') as file:
            for line in file:
                lines.append(line.strip())

        # Read in start and goal
        st = lines[0].replace('(', '').replace(')', '').split(',')

        start = (int(st[0]), int(st[1]))

        gt_str = "ground_truth:"
        act_str = "actions:"
        obs_str = "observations:"
        grid_str = "grid:"

        # taking ground truth
        gt_a = ''
        for i in range(lines.index(gt_str)+1, lines.index(act_str)):
            line = lines[i]
            gt_a = line
            #htt = line.replace('(', '').replace(')', '').split(',')
            #htt_cells.append((int(htt[0]), int(htt[1])))
        actions = ''
        for i in range(lines.index(act_str)+1, lines.index(obs_str)):
            line = lines[i]
            actions = line
        observs = ''
        for i in range(lines.index(obs_str)+1, lines.index(grid_str)):
            line = lines[i]
            observs = line

        # Only take lines relating to grid
        grid_lines = lines[(lines.index(grid_str)+1):]
        ROWS = len(grid_lines)
        COLS = len(grid_lines[0])
        terr = [[grid_lines[r][c] for c in range(COLS)] \
                for r in range(ROWS)]

        grid = Grid(ROWS,COLS, existing = [])

        grid.grid = terr
        if start != (-1,-1): grid.start_cell = start
        grid.gt_a = gt_a
        grid.gt_o = None
        grid.actions = actions
        grid.observs = observs

        grid.pathlength = len(actions)
        grid.probs = grid.initialize_probabilities()

        return grid


    def gt_to_coords(self):
        if self.start_cell == None:
            print("No start cell found, use mle instead")
            return []
        
        path = []
        cell = list(self.start_cell)
        path.append(self.start_cell)

        for act in self.gt_a:
            ro = 0
            co = 0
            if   act == '.': pass
            elif act == 'u': ro -= 1
            elif act == 'r': co += 1
            elif act == 'd': ro += 1
            elif act == 'l': co -= 1

            cell[0] += ro
            cell[1] += co
            path.append(tuple(cell))

        return path

if __name__ == "__main__":

    gridA = [['a','a','2'],
             ['1','1','1'],
             ['1','0','a']]

    trs = ['r', 'r', 'd', 'd']
    obs = ['1', '1', 'a', 'a']

    g = Grid(existing = gridA, path=(trs,obs))
    g.print_grid_to_file("existing_grid.txt")
    g = Grid.read_grid_from_file('existing_grid.txt')

    g.print_grid(pretty_print=True)
    print(g.probs, "\n")

    for i in range(5):
        g.step_grid()

        print(i, g.probs)
        print(g.most_likely)

    res = g.find_most_likely_paths()
    print(res)

    #g2 = Grid(rows=120, cols=160, rand_state=10, pathlength=100)
    #g2.print_grid_to_file("test_grid.txt")
    #g2 = Grid.read_grid_from_file("test_grid.txt")
    #g2.print_grid(pretty_print=True)
    #print(g2.gt_to_coords())



    #print()
    #print()
    #g2 = Grid(rows=100, cols=100, rand_state=0, pathlength=100, start_cell=(50,50))
    #g2.print_grid(pretty_print=True)




    #import sys  # testing size of node
    #print("Size of tuple (r,c):", sys.getsizeof((999,999)))
    #print("Size of fringe obj (priority,(r,c)):", sys.getsizeof((999,(999,999))))
    #print("Size of int:", sys.getsizeof(1))



