import math
import sys
import numpy as np
import pandas as pd
import mysymnmf as mf

np.random.seed(0)

#this function calculates and returns the average of all entries of the normalized matrix W,
#which has num_of_elements rows and num_of_elements columns.
def calculate_average(W, num_of_elements):
    sum = 0.0
    for i in range(num_of_elements):
        for j in range(num_of_elements):
            sum += W[i][j]
    average = sum/(num_of_elements**2)
    return average

#this function randomly initializes the matrix H for the symNMF algorythm 
# from the normalized matrix W. k is the number of required clusters.
# W has num_of_elements rows and num_of_elements columns.
# the returned matrix has num_of_elements rows and k columns.
def initialize_H(W, num_of_elements, k):
    average = calculate_average(W, num_of_elements)
    H = np.random.uniform(low=0.0, high=2*math.sqrt(average/k), size=(num_of_elements, k))
    return H

def is_float(n):
    try:
        float(n)
        return True
    except:
        return False

#this function prints the elements of the given matrix, which has num_of_rows rows 
#and num_of_cols columns, separated by commas    
def printMatrix(matrix, num_of_rows, num_of_cols):
    for i in range(num_of_rows):
        for j in range(num_of_cols):
            print('%.4f' % matrix[i][j], end='')
            if j<num_of_cols-1:
                print(",", end='')
        print('')
    return

def main():
    k=0
    file_name=""
    elements = []
    num_of_elements=0
    d=0

    if len(sys.argv) == 4:
        if sys.argv[1].isdigit():
            k = int(sys.argv[1])
        else:
            print("Invalid number of clusters!")
            exit(0)
        file_name = str(sys.argv[3])
        goal = str(sys.argv[2])
    else:
        print("An Error Has Occurred")
        exit(0)

    file = pd.read_csv(file_name, header=None)

    elements = file.values #this is the array of the datapoints from the file

    num_of_elements=len(elements)
    d=len(elements[0])

    if k<1 or k>num_of_elements:
        print("An Error Has Occurred")
        exit(0)

    #performing the given goal
    if goal=="sym":
        printMatrix(mf.sym(elements.tolist(), num_of_elements, d), num_of_elements, num_of_elements)
    elif goal=="ddg":
        printMatrix(mf.ddg(elements.tolist(), num_of_elements, d), num_of_elements, num_of_elements)
    elif goal=="norm":
        printMatrix(mf.norm(elements.tolist(), num_of_elements, d), num_of_elements, num_of_elements)
    elif goal=="symnmf":
        W = mf.norm(elements.tolist(), num_of_elements, d)
        H = initialize_H(W, num_of_elements, k)
        printMatrix(mf.symnmf(H.tolist(), W, k, num_of_elements), num_of_elements, k)
    else:
        print("An Error Has Occurred")
        exit(0)

if __name__ == "__main__":
    main()
