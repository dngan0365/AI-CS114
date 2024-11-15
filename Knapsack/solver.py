from ortools.algorithms.python import knapsack_solver
import sys
import glob
import os
import time
def load(level, solver, output_file):
    files=glob.glob(f"test/case{level}/*.kp")
    for f in files:
        values=[]
        weight=[]
        output_file.write(f"{f}\n")
        with open(f,'r') as readfile:
            rows = readfile.read().split('\n')
        readfile.close()
        i=1
        n=int(rows[i])
        i+=1
        capacities=[int(rows[i])]
        i+=2
        for j in range(i,n+4):
            v, w=list(map(int,rows[j].split(' ')))
            values.append(v)
            weight.append(w)
            #print(v,' ',w)
        #print(len(values))
        #print(weight)
        weights=[weight,]
        #print(weights)
    
        solver.init(values, weights, capacities)
        
        start=time.time()
        solver.set_time_limit(180)
        computed_value = solver.solve()
        
        end=time.time()


        packed_items = []
        packed_weights = []
        total_weight = 0
        output_file.write(f"Total value = {computed_value}\n")
        for i in range(len(values)):
            if solver.best_solution_contains(i):
                packed_items.append(i)
                packed_weights.append(weights[0][i])
                total_weight += weights[0][i]
        output_file.write(f"Total weight: {total_weight}\n")
        output_file.write(f"Packed items: {packed_items}\n")
        output_file.write(f"Packed_weights: {packed_weights}\n" )
        output_file.write(f"Time running: {end-start}\n")

        if solver.is_solution_optimal():
            output_file.write("optimal\n\n")
        else:
            output_file.write("not optimal\n\n")
def main():
    # Create the solver.
    
    solver = knapsack_solver.KnapsackSolver(
        knapsack_solver.SolverType.KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER,
        "KnapsackExample",
    )


    output_file=open('Output.txt','a')
    for i in range(12,13):
        load(i,solver, output_file)

if __name__ == "__main__":
    main()