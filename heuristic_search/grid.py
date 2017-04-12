#!/usr/bin/env python

# heuristic_search/grid.py
# Authors: Matthew Chan and Jeremy Savarin
# CS440: Project 1 - Heuristic Search

import pygame
import statistics

#from heuristic_search.gridfunctions import Grid
from gridfunctions import Grid


# Some global constants
# Set up colors for the grid
START = (255, 0, 0)
PATH = (0, 255, 0)
GOAL = (0, 200, 0)
BLOCKED = (0, 0, 0)
HARD_TO_TRAVERSE = (127, 127, 127)
UNBLOCKED = (255, 255, 255)
HIGHWAY_REGULAR = (24, 19, 178)
HIGHWAY_HTT = (114, 111, 198)
DEBUG = 1

# square size for grid display
SQ_SZ = 10
TOT_SIZE = 7 * 120

'''
fn create_grid - draws a 120x160 grid map of squares representing different
types of terrain: unblocked, partially-blocked and blocked cells. There is also
a representation of grids where motion can be "accelerated" (like a river or a
highway).
'''


def create_grid(grid_obj):

    global SQ_SZ

    ROWS = grid_obj.ROWS
    COLS = grid_obj.COLS
    SQ_SZ = TOT_SIZE // ROWS  # scaling squares, constant window size

    GRID_WIDTH = COLS * SQ_SZ
    GRID_HEIGHT = ROWS * SQ_SZ

    lines = grid_obj.grid

    # Intitialize pygame
    pygame.init()

    # Create grid
    grid = pygame.display.set_mode((GRID_WIDTH, GRID_HEIGHT + 200))

    # Check if pygame initialized
    is_init = pygame.display.get_init()

    # Set window title
    pygame.display.set_caption('Heuristic search simulation')


    # Color cells by type
    color_cells(grid, grid_obj, (GRID_WIDTH, GRID_HEIGHT))


    toggle_path = False
    path_step = -1
    mle_path = None

    toggle_gt = False
    gt_coords = grid_obj.gt_to_coords()

    toggle_v = False
    v_coords = []


    toggle_heatmap = False

    clock = pygame.time.Clock()
    median = statistics.median
    KEY_PRESSED = None  # for registering held-down keys

    while True:
        # Poll for user events
        event = pygame.event.poll()
        clicked = pygame.mouse.get_pressed()
        new_pos = pygame.mouse.get_pos()

        if event.type == pygame.QUIT:
            break

        # Normalize for square size, median for bounds checking
        c = median([0, new_pos[0]//SQ_SZ, COLS-1])
        r = median([0, new_pos[1]//SQ_SZ, ROWS-1])
        #r = new_pos[1] // SQ_SZ

        coord = (r, c)

        # Checking if key pressed to do stuff
        key = pygame.key.get_pressed()
        if KEY_PRESSED == None: #initializing key-is-pressed array
            if DEBUG: print("init key_press")
            KEY_PRESSED = [False for l in range(len(key))]

        result = do_key_events(key, grid_obj, KEY_PRESSED)
        if result == 0:  # esc
            break
        if result == 1:  # a
            toggle_path = False if toggle_path else True
        if result == 2:  # s
            grid_obj.step_grid() #TODO: show direction
        if result == 3:  # h
            toggle_heatmap = False if toggle_heatmap else True
        if result == 4:  # e
            while not grid_obj.step_grid(): pass
        if result == 5:  # b
            grid_obj.reset_steps()
        if result == 6:  # c
            grid_obj.step_grid() #TODO: show direction
        if result == 7:  # g
            toggle_gt = False if toggle_gt else True
        if result == 8:  # v
            toggle_v = False if toggle_v else True


        # coloring cells
        color_cells(grid, grid_obj, (GRID_WIDTH, GRID_HEIGHT), toggle_heatmap)
        if toggle_path:
            if grid_obj.step != path_step:
                mle_path = grid_obj.find_mle()
            draw_path(grid, mle_path)

        if toggle_gt:
            draw_path(grid, gt_coords[:(grid_obj.step+2)], color=START, lt=1)

        if toggle_v:
            res = grid_obj.find_most_likely_paths()
            first = True

            if grid_obj.step != path_step:
                v_coords = []
                for point in res:
                    v_coords.append(grid_obj.find_mle(point))

            for vc in v_coords:
                lt = 1
                if first:
                    first = False
                    lt = 3
                draw_path(grid, vc, lt=lt)


        grid.fill(BLOCKED, (0, GRID_HEIGHT, GRID_WIDTH, 200))
        font = pygame.font.SysFont('verdana', 20, 16)

        vals = "{0}, step:{2} | p:{1}".format(str(coord), str(grid_obj.probs[r][c]), str(grid_obj.step))

        fn_info = font.render(vals, False, UNBLOCKED)
        grid.blit(fn_info, (0, GRID_HEIGHT))

        pygame.display.flip()
        clock.tick(5)

    pygame.quit()

    return is_init
#### End grid loop


def do_key_events(key, grid, KEY_PRESSED):
    # keys monitored
    esc = pygame.K_ESCAPE  # 0
    a = pygame.K_a  # 1
    s = pygame.K_s  # 2
    h = pygame.K_h  # 3
    e = pygame.K_e  # 4
    b = pygame.K_b  # 5
    c = pygame.K_c  # 6
    g = pygame.K_g  # 7
    v = pygame.K_v  # 8


    
    if key[esc]: #end program with escape
        if DEBUG: print("esc press")
        return 0
 
    if key[a] and not KEY_PRESSED[a]:
        KEY_PRESSED[a] = True
        print("a press")
        return 1
    elif not key[a] and KEY_PRESSED[a]: #checks let go of a
        if DEBUG: print("a unpress")
        KEY_PRESSED[a] = False
        return -1

    if key[s] and not KEY_PRESSED[s]:
        KEY_PRESSED[s] = True
        print("stepping:", grid.step, "-->", grid.step+1)
        return 2
    elif not key[s] and KEY_PRESSED[s]: #checks let go of s
        if DEBUG: print("s unpress")
        KEY_PRESSED[s] = False
        return -2

    if key[h] and not KEY_PRESSED[h]:
        KEY_PRESSED[h] = True
        print("toggling heatmap")
        return 3
    elif not key[h] and KEY_PRESSED[h]: #checks let go of h
        if DEBUG: print("h unpress")
        KEY_PRESSED[h] = False
        return -3

    if key[e] and not KEY_PRESSED[e]:
        KEY_PRESSED[e] = True
        print("going to end")
        return 4
    elif not key[e] and KEY_PRESSED[e]: #checks let go of e
        if DEBUG: print("e unpress")
        KEY_PRESSED[e] = False
        return -4

    if key[b] and not KEY_PRESSED[b]:
        KEY_PRESSED[b] = True
        print("going to beginning")
        return 5
    elif not key[b] and KEY_PRESSED[b]: #checks let go of b
        if DEBUG: print("b unpress")
        KEY_PRESSED[b] = False
        return -5

    if key[c] and not KEY_PRESSED[c]:
        #KEY_PRESSED[c] = True
        print("c-stepping:", grid.step, "-->", grid.step+1)
        return 6
    elif not key[c] and KEY_PRESSED[c]: #checks let go of c
        if DEBUG: print("c unpress")
        #KEY_PRESSED[c] = False
        return -6

    if key[g] and not KEY_PRESSED[g]:
        KEY_PRESSED[g] = True
        print("toggle ground truth")
        return 7
    elif not key[g] and KEY_PRESSED[g]: #checks let go of g
        if DEBUG: print("g unpress")
        KEY_PRESSED[g] = False
        return -7

    if key[v] and not KEY_PRESSED[v]:
        KEY_PRESSED[v] = True
        print("toggle 10 mle")
        return 8
    elif not key[v] and KEY_PRESSED[v]: #checks let go of v
        if DEBUG: print("v unpress")
        KEY_PRESSED[v] = False
        return -8



'''
fn color_cells - colors in cells on the grid based on terrain type
'''

def color_cells(surf, grid, dims, heatmap=False):
    # dims = (WIDTH, HEIGHT)
    # for refreshing the grid
    lines = grid.grid
    probs = grid.probs
    start = None
    goal = None
    pygame.draw.rect(surf, BLOCKED, (0, 0, dims[0], dims[1]))

    if heatmap:
        for i in range(0, len(lines)):
            for j in range(0, len(lines[i])):
                cell_coord = (j*SQ_SZ, i*SQ_SZ, SQ_SZ, SQ_SZ)
                p = (probs[i][j])**.25  # shows small probabilities more

                fill_color = (int(p*255), 0, int((1-p)*128))

                surf.fill(fill_color, cell_coord)

    else: # Color cells by type
        for i in range(0, len(lines)):
            for j in range(0, len(lines[i])):
                cell_coord = (j*SQ_SZ, i*SQ_SZ, SQ_SZ, SQ_SZ)

                if lines[i][j] == 'S':
                    surf.fill(START, cell_coord)
                elif lines[i][j] == 'G':
                    surf.fill(GOAL, cell_coord)
                elif lines[i][j] == 'O':
                    surf.fill(BLOCKED, cell_coord)
                elif lines[i][j] == '1':
                    surf.fill(UNBLOCKED, cell_coord)
                elif lines[i][j] == '2':
                    surf.fill(HARD_TO_TRAVERSE, cell_coord)
                elif lines[i][j] == 'a':
                    surf.fill(HIGHWAY_REGULAR, cell_coord)
                elif lines[i][j] == 'b':
                    surf.fill(HIGHWAY_HTT, cell_coord)

    ## Fill in start and goal cells
    #if start and goal:
    #    start_coord = (start[1]*SQ_SZ, start[0]*SQ_SZ, SQ_SZ, SQ_SZ)
    #    surf.fill(START, start_coord)
    #    goal_coord = (goal[1]*SQ_SZ, goal[0]*SQ_SZ, SQ_SZ, SQ_SZ)
    #    surf.fill(GOAL, goal_coord)

'''
fn draw_line() - draws a line segment b/w neighborng cells
'''


def draw_line(surf, begin, end, color=START, lw=2):
    # offset, center of square
    ctr = SQ_SZ / 2

    # pygame draws in (x, y), args are in (r, c)
    b = (begin[1]*SQ_SZ + ctr, begin[0]*SQ_SZ + ctr)
    e = (end[1]*SQ_SZ + ctr, end[0]*SQ_SZ + ctr)

    pygame.draw.line(surf, color, b, e, lw)

'''
fn draw_path - draws path from starting cell to target cell
'''


def draw_path(surf, path, color=None, lt=3):
    if color == None:
        color = PATH
    if path == None:
        # no path exists
        # do something
        print("no path exists")
        return 0

    for i in range(1, len(path)):
        draw_line(surf, path[i-1], path[i], color, lt)


def run_search(mode=True, filename=None, rand_state=None, existing=None):
    print("Setting up...")


    #if mode: # not file
    #    g = Grid(120, 160, rand_state=rand_state)
    #    # gets the relevant parameters from the grid
    #else:
    #    g = Grid.read_grid_from_file(filename)

    if existing:
        g = existing
    else:
        g = Grid(rows=120, cols=160, pathlength=100, rand_state=rand_state)

    create_grid(g)


if __name__ == '__main__':

    gridA = [['a','a','2'],
             ['1','1','1'],
             ['1','0','a']]
    trs = ['r', 'r', 'd', 'd']
    obs = ['1', '1', 'a', 'a']

    #g = Grid(existing=gridA, path=(trs,obs))
    g = Grid(100, 100, rand_state=1)

    run_search(existing=g)
