#!/usr/bin/env python

# heuristic_search/gridfunctions.py
# CS440: Project 1 - Heuristic Search
# Authors: Matthew Chan and Jeremy Savarin

import random
import copy
import time

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

    def __init__(self, rows=120, cols=160, rand_state=None, add_blocked=True,
                 p_b=.2, add_htt=True, n_anchors=8, p_a=0.5, add_highways=True,
                 n_highways=4, h_length=100, h_segment=20, existing=None):

        self.isnew = False
        if existing:
            self.isnew = True
            self.ROWS = len(existing)
            self.COLS = len(existing[0])

            self.grid = [[existing[r][c] for c in range(self.COLS)] for r in range(self.ROWS)]
            self.initialize_probabilities()
            self.likelihood = [[[0,0] for c in range(self.COLS)] for r in range(self.ROWS)]
            self.most_likely = [["" for c in range(self.COLS)] for r in range(self.ROWS)]

            self.htt_cells = None
            self.highwaylist = None
            self.htt_cells = None
            self.parents = None
            self.values = None
            self.cells = None
            return

        
        # Choose a seed value for repeatable results
        if rand_state != None: # without none, misses 0
            random.seed(rand_state)

        self.ROWS = rows
        self.COLS = cols

        # Initialize all cells as unblocked
        self.grid = [['1' for c in range(cols)] for r in range(rows)]

        self.htt_cells = self.add_hardtraverse(n_anchors, p_a) \
            if add_htt else []
        self.highwaylist = self.add_highways(n_highways, h_length, h_segment) \
            if add_highways else []
        if add_blocked:
            self.add_blocked_cells(p_b)
        self.cells = self.add_start_goal_cells()

        INF = float('inf')
        # grid containing traversal parents (r,c tuples)
        self.parents = [[None for c in range(cols)] for r in range(rows)]
        # grid containing g, h, and f values from a* traversal
        self.values = [[[INF for c in range(cols)] for r in range(rows)] for val in range(3)]


# $$$$ New functions $$$$

    def viterbi(trans, obs, model_t, model_o):
        # only using observation model
        most_likely = []

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
        self.probs = [[0 for c in range(self.COLS)] for r in range(self.ROWS)]
        for r in range(self.ROWS):
            for c in range(self.COLS):
                if self.grid[r][c] == '0':
                    num_blocked += 1

        prob = 1.0/(self.ROWS * self.COLS - num_blocked)

        for r in range(self.ROWS):
            for c in range(self.COLS):
                if self.grid[r][c] != '0':
                    self.probs[r][c] = prob

        return prob


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

                ## book-keeping for most likely explanation
                #self.likelihood[r][c][0] = p_this
                #self.likelihood[r][c][1] = p_from
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


    def normalize(self):
        prob_sum = 0
        EPSILON = 1e-16

        for r in range(self.ROWS):
            for c in range(self.COLS):
                prob_sum += self.probs[r][c]

        for r in range(self.ROWS):
            for c in range(self.COLS):
                self.probs[r][c] = self.probs[r][c] / prob_sum
                if self.probs[r][c] < EPSILON:
                    self.probs[r][c] = 0 # for ease of use

        return prob_sum

    def find_mle(self, coord):
        # coord should be 0-indexed
        r = coord[0]
        c = coord[1]

        path = []

        for i in range(-1, -len(self.most_likely)-2, -1):
            prev = self.most_likely[r][c][i]
            path.append( (r,c) )

            # direction that moved us to this cell
            if prev == '.': continue
            elif prev == 'r': c -= 1
            elif prev == 'l': c += 1
            elif prev == 'u': r += 1
            elif prev == 'd': r -= 1

        return path[::-1] # reverse

    



# %%%% Old functions %%%%

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
                if pretty_print:
                    print(vals[self.grid[r][c]], end='')
                    if self.htt_cells and (r, c) in self.htt_cells:
                        print("\b*", end='')
                else:
                    print(self.grid[r][c], end='')

            print('|')

        print('\t+' + '-' * self.COLS + '+')
        if pretty_print:
            print('Size (r, c):', (self.ROWS, self.COLS))
            print('start, goal:', self.cells)

    '''
    fn add_hardtraverse(n_anchors=8, p=0.5) - adds hard-to-traverse cells to
    grid. Returns list of tuples representing the anchors (row,col).
    '''
    def add_hardtraverse(self, n_anchors=8, p=0.5):
        htt_cells = []

        i = 0
        while i < n_anchors:
            # Select random anchor points
            while(1):
                r = random.randrange(self.ROWS)
                c = random.randrange(self.COLS)
                if not (r, c) in htt_cells:
                    htt_cells.append((r, c))
                    i += 1
                    break

            # Consider 31x31 region centered around anchors, accounting for
            # hitting the boundaries prematurely
            rmin = max(0, r - 15)
            rmax = min(self.ROWS, r + 15)
            cmin = max(0, c - 15)
            cmax = min(self.COLS, c + 15)

            # Mark cells in region with 50% probability
            for r in range(rmin, rmax):
                for c in range(cmin, cmax):
                    if self.generate_probability(p):
                        self.grid[r][c] = '2'

        return htt_cells

    '''
    fn add_highways(n_highways=4) - adds highway cells to the grid to allow for
    "faster" movement. Returns a list of tuples containing highway vertex
    coordinates ((begin_row, begin_col),(end_row, end_col)).
    '''
    def add_highways(self, n_highways=4, h_length=100, h_segment=20):
        highwaylist = []
        attempts = 0
        MAX_ATTEMPTS = 10

        # Try to build highways
        while len(highwaylist) < n_highways:
            # Restart highway building process after MAX_ATTEMPTS
            if attempts > MAX_ATTEMPTS:
                highwaylist = []  # restarting

            new_highway = self.build_highway(h_length, h_segment)

            # Check for conflicts in built highway before appending
            if not self.has_conflicts(new_highway, highwaylist):
                highwaylist.append(new_highway)
                attempts = 0

            attempts += 1

        # Add highways to grid
        for highway in highwaylist:
            # Add individual segments
            for segment in highway:
                b = segment[0]  # begin
                e = segment[1]  # end

                if e[1] == b[1]:  # vertical
                    direction = (1 if (e[0]-b[0] > 0) else -1)
                    for disp in range(b[0], e[0] + direction, direction):
                        r = disp
                        c = b[1]

                        if self.is_htt_cell(r, c):
                            self.grid[r][c] = 'b'
                        else:
                            self.grid[r][c] = 'a'

                else:  # horizontal
                    direction = (1 if (e[1]-b[1] > 0) else -1)
                    for disp in range(b[1], e[1] + direction, direction):
                        r = b[0]
                        c = disp

                        if self.is_htt_cell(r, c):
                            self.grid[r][c] = 'b'
                        else:
                            self.grid[r][c] = 'a'

        return highwaylist


    def build_highway(self, h_length, h_segment):
        highway = []
        # a highway contains: ( (start-coords), (stop-coords) ) for each
        # segment of the highway

        startside = random.randrange(4)  # 0 is top, going clockwise

        if startside % 2 == 0:
            sidelength = self.COLS
        else:
            sidelength = self.ROWS

        cell_coord = random.randrange(1, sidelength-1)  # no corners

        row = -1
        col = -1
        direction = -1
        #   0
        # 3 + 1
        #   2

        if startside == 0:
            row = 0
            col = cell_coord
        elif startside == 1:
            row = cell_coord
            col = self.COLS - 1
        elif startside == 2:
            row = self.ROWS - 1
            col = cell_coord
        elif startside == 3:
            row = cell_coord
            col = 0
        direction = (startside + 2) % 4

        done = False
        start = (row, col)

        while not done:
            if direction == 0:
                stop = (start[0] - h_segment, start[1])
            if direction == 1:
                stop = (start[0], start[1] + h_segment)
            if direction == 2:
                stop = (start[0] + h_segment, start[1])
            if direction == 3:
                stop = (start[0], start[1] - h_segment)

            # checking in bounds
            if (0 < stop[1] < (self.COLS-1)) and (0 < stop[0] < (self.ROWS-1)):
                highway.append((start, stop))
                start = stop  # for the next segment
                if self.generate_probability(.4):  # change direction
                    if self.generate_probability(.5):
                        direction = (direction + 1) % 4
                    else:
                        direction = (direction - 1) % 4

            else:  # at/past an edge
                if stop[0] <= 0:
                    stop = (0, stop[1])
                elif stop[0] >= self.ROWS:
                    stop = (self.ROWS-1, stop[1])
                if stop[1] <= 0:
                    stop = (stop[0], 0)
                elif stop[1] >= self.COLS:
                    stop = (stop[0], self.COLS-1)
                highway.append((start, stop))
                # TODO: check conflicts  & length req
                if not self.has_conflicts(highway, h_length=h_length):
                    done = True

            # checking for self-intersections every iteration
            for i in range(len(highway) - 2):
                if self.check_conflict(highway[-1], highway[i]):
                    highway = []
                    direction = (startside + 2) % 4
                    start = (row, col)
                    done = False
                    break

        return highway

    def has_conflicts(self, highway, highwaylist=None, h_length=100):
        # selfcheck will also do length check
        selfcheck = False

        if highwaylist is None:
            highwaylist = [highway]
            selfcheck = True

        if selfcheck:  # length requirement only
            h_len = 0
            for segment in highway:
                h_len += abs(segment[1][0] - segment[0][0]) + \
                         abs(segment[1][1] - segment[0][1])
            if h_len < h_length:
                return True
            else:
                return False

        for i in range(len(highwaylist)):
            for s1 in range(len(highwaylist[i])):
                for s2 in range(len(highway)):
                    if self.check_conflict(highwaylist[i][s1], highway[s2]):
                        return True

        return False

    def check_conflict(self, segment1, segment2):
        s1 = segment1
        s2 = segment2
        c_cross = False
        r_cross = False
        diff1r = s1[1][0] - s1[0][0]  # vert
        diff1c = s1[1][1] - s1[0][1]  # horiz
        diff2r = s2[1][0] - s2[0][0]  # vert
        diff2c = s2[1][1] - s2[0][1]  # horiz

        # checking for row overlap
        # check if vertexes of s2 are in s1
        if min(s1[0][0], s1[1][0]) <= s2[0][0] <= max(s1[0][0], s1[1][0]):
            r_cross = True
        if min(s1[0][0], s1[1][0]) <= s2[1][0] <= max(s1[0][0], s1[1][0]):
            r_cross = True
        if min(s2[0][0], s2[1][0]) <= s1[0][0] <= max(s2[0][0], s2[1][0]):
            r_cross = True
        if min(s2[0][0], s2[1][0]) <= s1[1][0] <= max(s2[0][0], s2[1][0]):
            r_cross = True

        # checking for column overlap
        # check if vertexes of s2 are in s1
        if min(s1[0][1], s1[1][1]) <= s2[0][1] <= max(s1[0][1], s1[1][1]):
            c_cross = True
        if min(s1[0][1], s1[1][1]) <= s2[1][1] <= max(s1[0][1], s1[1][1]):
            c_cross = True
        if min(s2[0][1], s2[1][1]) <= s1[0][1] <= max(s2[0][1], s2[1][1]):
            c_cross = True
        if min(s2[0][1], s2[1][1]) <= s1[1][1] <= max(s2[0][1], s2[1][1]):
            c_cross = True

        return (r_cross and c_cross)

    '''
    fn is_highway_cell(row, col) - Returns True if grid[row][col] is a highway
    cell ,and False otherwise.
    '''
    def is_highway_cell(self, row, col):
        cell = self.grid[row][col]
        return cell == 'a' or cell == 'b'

    '''
    fn is_htt_cell(row, col) - Returns True if grid[row][col] is a
    hard-to-traverse cell, and False otherwise.
    '''
    def is_htt_cell(self, row, col):
        cell = self.grid[row][col]
        return cell == '2'

    '''
    fn is_blocked_cell(row, col) - Returns True if grid[row][col] is a
    blocked cell, and False otherwise.
    '''
    def is_blocked_cell(self, row, col):
        cell = self.grid[row][col]
        return cell == '0'

    def add_blocked_cells(self, p=.2):
        # 20% of cells are blocked
        BLOCKED_CELLS = p * self.ROWS * self.COLS

        i = 0

        while i < BLOCKED_CELLS:
            # Generate random cell
            r = random.randint(0, self.ROWS-1)
            c = random.randint(0, self.COLS-1)

            # Check if highway
            if not self.is_highway_cell(r, c):
                self.grid[r][c] = '0'
                i += 1

    '''
    fn add_start_goal_cells() - adds start and goal cells to the grid. Returns
    a list of tuples representing their coordinates in the grid.
    '''
    def add_start_goal_cells(self):
        # Try to place start cell
        is_start_placed = False
        on_sides = False

        cells = []

        while not is_start_placed:
            # Place in top left of grid
            # Guarantees correct spacing
            if self.generate_probability(.5):
                r = random.randint(0, 19)
                c = random.randint(0, self.COLS-1)
            else:
                r = random.randint(0, self.ROWS-1)
                c = random.randint(0, 19)
                on_sides = True

            if not self.is_blocked_cell(r, c):
                cells.append((r, c))
                is_start_placed = True

        # Try to place goal cell
        is_goal_placed = False

        while not is_goal_placed:
            # Place opposite of start
            if on_sides:
                r = random.randint(0, self.ROWS-1)
                c = random.randint(self.COLS-20, self.COLS-1)
            else:
                r = random.randint(self.ROWS-20, self.ROWS-1)
                c = random.randint(0, self.COLS-1)

            if not self.is_blocked_cell(r, c):
                cells.append((r, c))
                is_goal_placed = True

        return cells

    '''
    fn print_grid_to_file(filename) - prints grid data to file given by
    filename
    '''
    def print_grid_to_file(self, filename):
        with open(filename, 'w') as file:
            # Print start and goal coordinates
            file.write(str(self.cells[0]) + '\n')
            file.write(str(self.cells[1]) + '\n\n')

            # Print anchors in hard-to-traverse cells
            file.write('hard-to-traverse centers: ' + '\n')

            for point in self.htt_cells:
                file.write(str(point) + '\n')

            # Print data representation of the grid
            file.write('grid: ' + '\n')

            for r in range(self.ROWS):
                for c in range(self.COLS):
                    file.write(self.grid[r][c])

                file.write('\n')

    def read_grid_from_file(filename):
        lines = []

        with open(filename, 'r') as file:
            for line in file:
                lines.append(line.strip())

        # Read in start and goal
        st = lines[0].replace('(', '').replace(')', '').split(',')
        gl = lines[1].replace('(', '').replace(')', '').split(',')

        start = (int(st[0]), int(st[1]))
        goal = (int(gl[0]), int(gl[1]))
        cells = (start, goal)

        htt_str = "hard-to-traverse centers:"
        grid_str = "grid:"

        # taking htt centers
        htt_cells = []
        for i in range(lines.index(htt_str)+1, lines.index(grid_str)):
            line = lines[i]
            htt = line.replace('(', '').replace(')', '').split(',')
            htt_cells.append((int(htt[0]), int(htt[1])))

        # Only take lines relating to grid
        grid_lines = lines[(lines.index(grid_str)+1):]
        ROWS = len(grid_lines)
        COLS = len(grid_lines[0])
        terr = [[grid_lines[r][c] for c in range(COLS)] \
                for r in range(ROWS)]

        grid = Grid(ROWS,COLS,add_blocked=False, add_htt=False, add_highways=False)

        grid.grid = terr
        grid.cells = cells
        grid.htt_cells = htt_cells

        return grid

    def replace_start_and_goal(self):
        new_cells = self.add_start_goal_cells()
        self.cells = new_cells

        return new_cells


if __name__ == "__main__":

    gridA = [['a','a','2'],
             ['1','1','1'],
             ['1','0','a']]

    trs = ['r', 'r', 'd', 'd']
    obs = ['1', '1', 'a', 'a']

    g = Grid(existing = gridA)
    g.print_grid(pretty_print=True)
    print(g.probs, "\n")

    for i in range(4):

        g.eval_transition(trs[i])
        g.eval_observation(obs[i])
        #g.eval_transition('d')
        #g.eval_observation('a')
        g.normalize()
        print(i, g.probs)
        print(g.most_likely)

        print()
    print(g.find_mle((2,2)))




    #g = Grid(120, 160, rand_state=0)
    #g.print_grid(pretty_print=True)
    #g.print_grid_to_file('test.txt')
    #g2 = Grid.read_grid_from_file('test.txt')
    #g2.print_grid(pretty_print=True)
    #

    #print(g.replace_start_and_goal())
    #print(g.replace_start_and_goal())
    #print(g.replace_start_and_goal())


    #import sys  # testing size of node
    #print("Size of tuple (r,c):", sys.getsizeof((999,999)))
    #print("Size of fringe obj (priority,(r,c)):", sys.getsizeof((999,(999,999))))
    #print("Size of int:", sys.getsizeof(1))



