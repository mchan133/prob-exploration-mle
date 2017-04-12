from gridfunctions import Grid
import astar
import sys

'''
Usage: python testbench.py > textfile
Deterministic stats, to be processed in Excel
'''

#print("Size of tuple (r,c):", sys.getsizeof((999,999)))
#print("Size of fringe obj (priority,(r,c)):", 2*sys.getsizeof((999,999)))
#print("Size of int:", sys.getsizeof(1))

# Node size
NS = 2*sys.getsizeof((999,999))
TEST_SETUP = 1

    
astar_search = astar.astar_search
# print("grid#\tst#\tstep\tml_c\tgt_c\tdist\tp(gt)") # header
w = 1 # heuristic weight
for i in range(10):  # grid
    g = Grid(100, 100, rand_state=2*(i+1), pathlength=100)

    for j in range(10):  # start path pairs

        gt_coords = g.gt_to_coords()

        while not g.step_grid():
            pass
            #ml_c = g.most_likely_cell
            #gt_c = gt_coords[g.step]
            #p = g.probs[gt_c[0]][gt_c[1]]
            #dist = ((ml_c[0] - gt_c[0])**2 + (ml_c[1] - gt_c[1])**2)**.5
            #print("%d\t%d\t%d\t%s\t%s\t%3.3f\t%3.3f" % \
            #      (i, j, g.step, ml_c, gt_c, dist, p))
        # part f, calculated avg. viterbi diff (vs gt) vs. iteration
        # done at end
        ml_c = g.find_mle()
        gt_c = gt_coords[1:] # get rid of start cell
        #print(len(ml_c), len(gt_c))
        dist = []
        for i in range(len(ml_c)):
            dist.append( ((ml_c[i][0] - gt_c[i][0])**2 + (ml_c[i][1] - gt_c[i][1])**2)**.5 )

        for v in dist:
            print(v, " ", end="")
        print()

        g.create_a_new_path()

