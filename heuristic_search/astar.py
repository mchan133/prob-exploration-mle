#!/usr/bin/env python

# heuristic_search/astar.py
# CS440: Project 1 - Heuristic Search
# Authors: Matthew Chan and Jeremy Savarin

#import random
#import queue  # contains priority queue for astar
import sys
import time
# temporary workaround for 'module not found'
if sys.argv[0]=="app.py" or sys.argv[0]=="../app.py":
    from heuristic_search.gridfunctions import Grid
    from heuristic_search.my_pq import My_PQ
else:
    from gridfunctions import Grid
    from my_pq import My_PQ


def calc_cost(terrain, current, neighbor):
    v1 = current
    v2 = neighbor
    t1 = terrain[current[0]][current[1]]
    t2 = terrain[neighbor[0]][neighbor[1]]
    diagonal = ((v2[0]-v1[0]) != 0) and ((v2[1]-v1[1]) != 0)
    return cost_of([t1, t2], diagonal)


# helper for cost
def cost_of(charlist, diagonal):
    if len(charlist) > 2:
        return None

    cost = 0

    for char in charlist:
        if char == '0':
            cost += float('inf')
        if char == '1':
            cost += 2**.5 if diagonal else 1.0
        if char == '2':
            cost += 8**.5 if diagonal else 2.0
        if char == 'a':
            cost += .125**.5 if diagonal else .25
        if char == 'b':
            cost += .5**.5 if diagonal else .5

    if cost != 0:
        return cost / 2.0
    else:
        return None


# path length from start
# inefficient, no longer in use
'''
def cost_function(terrain, parent_grid, neighbor):
    current = neighbor
    parent = parent_grid[current[0]][current[1]]
    cost = 0

    while parent != current:
        v1 = current
        v2 = parent
        t1 = terrain[current[0]][current[1]]
        t2 = terrain[parent[0]][parent[1]]
        diagonal = ((v2[0]-v1[0]) != 0) and ((v2[1]-v1[1]) != 0)
        cost += cost_of([t1, t2], diagonal)

        current = parent
        parent = parent_grid[current[0]][current[1]]

    return cost
'''


# returns if a cell is in bounds
def in_bounds(point, r, c):
    if 0 <= point[0] < r:
        if 0 <= point[1] < c:
            return True

    return False


# traverses path back to start from goal (at end of astar)
def traverse_path(parent_grid, goal):
    current = goal
    parent = parent_grid[current[0]][current[1]]
    pathlist = []

    while parent != current:
        pathlist.append(current)
        current = parent
        parent = parent_grid[current[0]][current[1]]

    pathlist.append(current)  # starting node
    pathlist.reverse()
    return pathlist

def integ_astar_search(grid, heuristics=[], weight=1, w2=1, test_stats=False):
    # initializing variables
    terrain = grid.grid
    cols = grid.COLS
    rows = grid.ROWS
    start = grid.cells[0]
    goal = grid.cells[1]
    s = start  # placeholder variable
    g = goal

    if len(heuristics) < 2:
        print("not enough heuristics")
        return
    num_heurs = len(heuristics)
    inf = float('inf')

    # for test_stats
    fringe_size = []

    p_grids = []
    g_grids = []
    h_grids = []
    f_grids = []
    expanded = []
    fringe = []

    # one grid shared between searches (wrapped for compatibility)
    p_grids.append([[None for c in range(cols)] for r in range(rows)])
    g_grids.append([[inf  for c in range(cols)] for r in range(rows)])
    p_grids[0][s[0]][s[1]] = s
    g_grids[0][s[0]][s[1]] = 0

    # only 2 of these (admissible/inadmissable)
    for i in range(2):
        expanded.append(set())

    # initializing variables for all heuristics
    for i in range(num_heurs):
        h_grids.append([[inf for c in range(cols)] for r in range(rows)])
        f_grids.append([[inf for c in range(cols)] for r in range(rows)])
        compute_heuristic(h_grids[i], heuristics[i], goal, weight)

        fringe.append(My_PQ())

        f_grids[i][s[0]][s[1]] = h_grids[i][s[0]][s[1]]

        # fringe stored as (f, (r,c))
        fringe[i].put( (f_grids[i][s[0]][s[1]], s) )

        # testing
        fringe_size.append(0)

    start_t = time.clock()

    while not fringe[0].empty():
        minkey_0 = fringe[0].peek()  # gets from fringe (not pop)
        while minkey_0[1] in expanded[0]:
            print("already expanded", minkey_0)
            fringe[0].get()
            minkey_0 = fringe[0].peek()

        if test_stats:
            fringe_size[0] = max(fringe_size[0], fringe[0].qsize())

        for i in range(1,num_heurs):
            if test_stats:
                fringe_size[i] = max(fringe_size[i], fringe[i].qsize())

            minkey_i = fringe[i].peek()
            while minkey_i and minkey_i[1] in expanded[1]:
                print("already expanded", minkey_i)
                fringe[i].get()
                minkey_i = fringe[i].peek()

            if minkey_i and minkey_i[0] < w2*minkey_0[0]:
                if g_grids[0][g[0]][g[1]] < minkey_i[0]:
                    if g_grids[0][g[0]][g[1]] < inf:
                        # found a path in search i
                        found_path = traverse_path(p_grids[0], goal)
                        grids = (h_grids, g_grids, f_grids, p_grids)

                        if test_stats:
                            search_t = time.clock() - start_t
                            stats = (fringe_size, expanded, search_t, \
                                    g_grids[0][g[0]][g[1]])
                            return (found_path, grids, stats, 0)
                        else:
                            return (found_path, grids, 0)
                else:  # expand current node
                    #print("expanding", i, minkey_0, minkey_i)
                    cur = minkey_i[1] #current cell
                    expanded[1].add(cur)

                    # removing from all fringes
                    for j in range(num_heurs):
                        fringe[j].remove(cur)

                    for r in range(-1, 2):
                        for c in range(-1, 2):
                            # neighboring cell
                            n = (cur[0]+r, cur[1]+c)

                            if not in_bounds(n, rows, cols):
                                continue
                            if n in expanded[1]:  # also catches self
                                continue
                            if terrain[n[0]][n[1]] == '0':  # blocked
                                continue

                            temp_cost = g_grids[0][cur[0]][cur[1]] + \
                                    calc_cost(terrain, cur, n)

                            #if fringe[i].contains(n):  # a path already known
                            if g_grids[0][n[0]][n[1]] < float('inf'):
                                if temp_cost >= g_grids[0][n[0]][n[1]]:
                                    continue

                            p_grids[0][n[0]][n[1]] = cur
                            g_grids[0][n[0]][n[1]] = temp_cost

                            if n not in expanded[0]:
                                key0 = f_fcn(g_grids[0], h_grids[0], n)
                                f_grids[0][n[0]][n[1]] = key0
                                fringe[0].put((key0, n))
                                
                                if n not in expanded[1]:
                                    for j in range(1, num_heurs):
                                        keyi = f_fcn(g_grids[0], h_grids[j], n)
                                        if keyi < w2*key0:
                                            f_grids[j][n[0]][n[1]] = keyi
                                            fringe[j].put((keyi, n))

            else:  # expanding state in admissible heuristic
                if g_grids[0][g[0]][g[1]] < minkey_0[0]:
                    if g_grids[0][g[0]][g[1]] < inf:
                        # found a path in search 0
                        found_path = traverse_path(p_grids[0], goal)
                        grids = (h_grids, g_grids, f_grids, p_grids)

                        if test_stats:
                            search_t = time.clock() - start_t
                            stats = (fringe_size, expanded, \
                                    search_t, g_grids[0][g[0]][g[1]])
                            return (found_path, grids, stats, 0)
                        else:
                            return (found_path, grids, 0)
                else:  # expand current node
                    #print("expanding", i, minkey_0, minkey_i)
                    cur = fringe[0].get()[1] #current cell
                    expanded[0].add(cur)

                    # removing from all fringes
                    for j in range(num_heurs):
                        fringe[j].remove(cur)

                    for r in range(-1, 2):
                        for c in range(-1, 2):
                            # neighboring cell
                            n = (cur[0]+r, cur[1]+c)

                            if not in_bounds(n, rows, cols):
                                continue
                            if n in expanded[0]:  # also catches self
                                continue
                            if terrain[n[0]][n[1]] == '0':  # blocked
                                continue

                            temp_cost = g_grids[0][cur[0]][cur[1]] + \
                                    calc_cost(terrain, cur, n)

                            #if fringe[0].contains(n):  # a path already known
                            if g_grids[0][n[0]][n[1]] < float('inf'):
                                if temp_cost >= g_grids[0][n[0]][n[1]]:
                                    continue

                            p_grids[0][n[0]][n[1]] = cur
                            g_grids[0][n[0]][n[1]] = temp_cost

                            if n not in expanded[0]:
                                key0 = f_fcn(g_grids[0], h_grids[0], n)
                                f_grids[0][n[0]][n[1]] = key0
                                fringe[0].put((key0, n))
                                
                                if n not in expanded[1]:
                                    for j in range(1, num_heurs):
                                        keyi = f_fcn(g_grids[0], h_grids[j], n)
                                        if keyi < w2*key0:
                                            f_grids[j][n[0]][n[1]] = keyi
                                            fringe[j].put((keyi, n))

    print("reached here")
    return None


def seq_astar_search(grid, heuristics=[], weight=1, w2=1, test_stats=False):
    # initializing variables
    terrain = grid.grid
    cols = grid.COLS
    rows = grid.ROWS
    start = grid.cells[0]
    goal = grid.cells[1]
    s = start  # placeholder variable
    g = goal

    if len(heuristics) < 2:
        print("not enough heuristics")
        return
    num_heurs = len(heuristics)
    inf = float('inf')

    # for test_stats
    fringe_size = []

    p_grids = []
    g_grids = []
    h_grids = []
    f_grids = []
    expanded = []
    fringe = []

    # initializing variables for all heuristics
    for i in range(num_heurs):
        p_grids.append([[None for c in range(cols)] for r in range(rows)])
        g_grids.append([[inf  for c in range(cols)] for r in range(rows)])
        h_grids.append([[inf  for c in range(cols)] for r in range(rows)])
        f_grids.append([[inf  for c in range(cols)] for r in range(rows)])
        compute_heuristic(h_grids[i], heuristics[i], goal, weight)

        expanded.append(set())
        fringe.append(My_PQ())

        p_grids[i][s[0]][s[1]] = s
        g_grids[i][s[0]][s[1]] = 0
        f_grids[i][s[0]][s[1]] = h_grids[i][s[0]][s[1]]

        # fringe stored as (f, (r,c))
        fringe[i].put( (f_grids[i][s[0]][s[1]], s) )

        # testing
        fringe_size.append(0)

    start_t = time.clock()

    while not fringe[0].empty():
        minkey_0 = fringe[0].peek()  # gets from fringe (not pop)
        while minkey_0[1] in expanded[0]:
            #print("already expanded", minkey_0)
            fringe[0].get()
            minkey_0 = fringe[0].peek()

        if test_stats:
            fringe_size[0] = max(fringe_size[0], fringe[0].qsize())

        for i in range(1,num_heurs):
            if test_stats:
                fringe_size[i] = max(fringe_size[i], fringe[i].qsize())

            minkey_i = fringe[i].peek()
            while minkey_i and minkey_i[1] in expanded[i]:
                #print("already expanded", minkey_i)
                fringe[i].get()
                minkey_i = fringe[i].peek()

            if minkey_i and minkey_i[0] < w2*minkey_0[0]:
                if g_grids[i][g[0]][g[1]] < minkey_i[0]:
                    if g_grids[i][g[0]][g[1]] < inf:
                        # found a path in search i
                        found_path = traverse_path(p_grids[i], goal)
                        grids = (h_grids, g_grids, f_grids, p_grids)

                        if test_stats:
                            search_t = time.clock() - start_t
                            stats = (fringe_size, expanded, search_t, \
                                    g_grids[i][g[0]][g[1]])
                            return (found_path, grids, stats, i)
                        else:
                            return (found_path, grids, i)
                else:  # expand current node
                    #print("expanding",i, minkey_0, minkey_i)
                    cur = minkey_i[1] #current cell
                    expanded[i].add(cur)

                    for r in range(-1, 2):
                        for c in range(-1, 2):
                            # neighboring cell
                            n = (cur[0]+r, cur[1]+c)

                            if not in_bounds(n, rows, cols):
                                continue
                            if n in expanded[i]:  # also catches self
                                continue
                            if terrain[n[0]][n[1]] == '0':  # blocked
                                continue

                            temp_cost = g_grids[i][cur[0]][cur[1]] + \
                                    calc_cost(terrain, cur, n)

                            if fringe[i].contains(n):  # a path already known
                                if temp_cost >= g_grids[i][n[0]][n[1]]:
                                    continue

                            p_grids[i][n[0]][n[1]] = cur
                            g_grids[i][n[0]][n[1]] = temp_cost
                            f_grids[i][n[0]][n[1]] = f_fcn(g_grids[i], h_grids[i], n)

                            fringe[i].put((f_grids[i][n[0]][n[1]], n))

            else:  # expanding state in admissible heuristic
                if g_grids[0][g[0]][g[1]] < minkey_0[0]:
                    if g_grids[0][g[0]][g[1]] < inf:
                        # found a path in search 0
                        found_path = traverse_path(p_grids[0], goal)
                        grids = (h_grids, g_grids, f_grids, p_grids)

                        if test_stats:
                            search_t = time.clock() - start_t
                            stats = (fringe_size, expanded, \
                                    search_t, g_grids[0][g[0]][g[1]])
                            return (found_path, grids, stats, 0)
                        else:
                            return (found_path, grids, 0)
                else:  # expand current node
                    #print("expanding", i, minkey_0, minkey_i)
                    cur = fringe[0].get()[1] #current cell
                    expanded[0].add(cur)

                    for r in range(-1, 2):
                        for c in range(-1, 2):
                            # neighboring cell
                            n = (cur[0]+r, cur[1]+c)

                            if not in_bounds(n, rows, cols):
                                continue
                            if n in expanded[0]:  # also catches self
                                continue
                            if terrain[n[0]][n[1]] == '0':  # blocked
                                continue

                            temp_cost = g_grids[0][cur[0]][cur[1]] + \
                                    calc_cost(terrain, cur, n)

                            if fringe[0].contains(n):  # a path already known
                                if temp_cost >= g_grids[0][n[0]][n[1]]:
                                    continue

                            p_grids[0][n[0]][n[1]] = cur
                            g_grids[0][n[0]][n[1]] = temp_cost
                            f_grids[0][n[0]][n[1]] = f_fcn(g_grids[0], h_grids[0], n)

                            fringe[0].put((f_grids[0][n[0]][n[1]], n))

    print("reached here")
    return None

def astar_search(grid, heuristic=None, weight=1, test_stats=False):
    # setup
    terrain = grid.grid
    cols = grid.COLS
    rows = grid.ROWS
    start = grid.cells[0]
    goal = grid.cells[1]
    s = start  # placeholder variables
    g = goal
    if not heuristic:
        heuristic = h_function0
    inf = float('inf')

    # for test_stats
    fringe_size = 0

    # initializing variables for a*
    p_grid = grid.parents
    g_grid = grid.values[0]
    h_grid = grid.values[1]
    f_grid = grid.values[2]

    compute_heuristic(h_grid, heuristic, g, weight)

    expanded = set()
    fringe = My_PQ()

    p_grid[s[0]][s[1]] = s
    g_grid[s[0]][s[1]] = 0
    f_grid[s[0]][s[1]] = h_grid[s[0]][s[1]]

    fringe.put((f_grid[s[0]][s[1]], s))

    start_t = time.clock()

    while not fringe.empty():
        # for testing
        if test_stats:
            fringe_size = max(fringe_size, fringe.qsize())
        # get from fringe and expand
        current = fringe.get()[1]  # strip off the f-value
        if current in expanded:
            continue
        expanded.add(current)

        # check for the goal (g)
        if current == goal:
            # wrap vals w/ list for compatibility with seq.
            found_path = traverse_path(p_grid, g)
            grids = ([h_grid], [g_grid], [f_grid], [p_grid])

            if test_stats:
                search_t = time.clock() - start_t
                stats = ([fringe_size], [len(expanded)], \
                        search_t, g_grid[g[0]][g[1]])
                return (found_path, grids, stats, 0)
            else:
                return (found_path, grids, 0)

        # get neighbors, add to fringe (calc f value)
        for i in range(-1, 2):
            for j in range(-1, 2):
                neighbor = (current[0]+i, current[1]+j)

                if not in_bounds(neighbor, rows, cols):
                    continue
                if neighbor in expanded:  # also catches self
                    continue
                if terrain[neighbor[0]][neighbor[1]] == '0':  # blocked
                    continue

                temp_cost = g_grid[current[0]][current[1]] + calc_cost(terrain, current, neighbor)

                #if neighbor in fringe_check:  # a path already known
                if fringe.contains(neighbor):
                    if temp_cost >= g_grid[neighbor[0]][neighbor[1]]:
                        continue

                p_grid[neighbor[0]][neighbor[1]] = current
                g_grid[neighbor[0]][neighbor[1]] = temp_cost
                f_grid[neighbor[0]][neighbor[1]] = f_fcn(g_grid, h_grid, neighbor)

                fringe.put((f_grid[neighbor[0]][neighbor[1]], neighbor))
                #fringe_check.add(neighbor)

    return None  #No path

def f_fcn(g, h, cell):
    return g[cell[0]][cell[1]] + h[cell[0]][cell[1]]

def compute_heuristic(h_grid, heuristic, goal, weight=1):
    for r in range(len(h_grid)):  # initializing h_grid w/ heuristic
        for c in range(len(h_grid[0])):
            h_grid[r][c] = weight * heuristic((r, c), goal)


# Heuristics
def h_function0(cell, goal):
    return 0  # uniform cost search (uninformed)


def h_function1(cell, goal):  # euclidean / 4
    return ((cell[0]-goal[0])**2 + (cell[1]-goal[1])**2)**.5 / 4.0  # highways


def h_function2(cell, goal):  # Manhattan distance
    return abs(cell[0]-goal[0]) + abs(cell[1]-goal[1])


def h_function3(cell, goal):  # Square root distance
    return (abs(cell[0]-goal[0]) + abs(cell[1]-goal[1]))**.5


def h_function4(cell, goal):  # Square distance
    return (cell[0]-goal[0])**2 + (cell[1]-goal[1])**2


def h_function5(cell, goal):  # radial (# cells)
    return max(abs(cell[0]-goal[0]), abs(cell[1]-goal[1]))


if __name__ == "__main__":

    g = Grid(120, 160, rand_state=2)
        
    import cProfile
    import pstats

    heurs = [h_function1, h_function2, h_function3, h_function4, h_function5]
    #heurs = [h_function1]

    pr = cProfile.Profile()
    pr.enable()
    #r = integ_astar_search(g, heurs, w2=1.5)
    r = seq_astar_search(g, heurs, w2=1)
    #r = astar_search(g)
    pr.disable()
    pstats.Stats(pr).print_stats()
    print(r[0])
    print(g.cells)


