import numpy as  np
import os
from scipy.spatial.transform import Rotation as R

np.random.seed(69)

def frobNorm(P1, P2, representation1, representation2):
    #np.set_printoptions(suppress=True)
    val = np.linalg.norm(P1 - P2, 'fro')
    print(f"The Frobenius Norm between the {representation1} and the {representation2} is: {val}")
    return val

def closedFormICP(pointsA, pointsB, pointsNeeded, method):

    pointsA_mean = np.mean(pointsA[:pointsNeeded], axis=0)
    pointsB_mean = np.mean(pointsB[:pointsNeeded], axis=0)
    pointsA_meanSubtracted = pointsA[:pointsNeeded] - pointsA_mean
    pointsB_meanSubtracted = pointsB[:pointsNeeded] - pointsB_mean
    #Taking points and subtracting the mean from them.

    W = np.zeros((3,3))
    #Cross covariance matrix
    for i in range(pointsNeeded):
        W += np.matmul(pointsB_meanSubtracted[i, np.newaxis].T, pointsA_meanSubtracted[i, np.newaxis])
        u, s, vh = np.linalg.svd(W, full_matrices=True)
        print(f"The Singular values are {str(s)}")
        M = np.diag([1,1, np.linalg.det(u) * np.linalg.det(vh)])

        if method == "Wahba":
            recoveredRotation = u @ M @ vh
            #The closed-form solution to find the rotation between two frames according to the Wahba algorithm
        else:
            recoveredRotation = u @ vh
            #The closed-form solution to find the rotation between two frames according to the Orthogonal Procrustes algorithm
            recoveredTranslation = pointsB_mean - recoveredRotation @ pointsA_mean
            recoveredTransform = np.hstack((recoveredRotation, np.array([[recoveredTranslation[0]], [recoveredTranslation[1]], [recoveredTranslation[2]]])))
            recoveredTransform = np.vstack((recoveredTransform, np.array([[0,0,0,1]])))

            return recoveredTransform


if __name__ == "__main__":
    linearlyDependantPoint = np.array(np.random.rand(3) * 10).astype('int32')
    pointsA = np.array([[0,9,0],[1,1,1],[1,2,3],linearlyDependantPoint])
    #First three points are hard-coded as they need to be Linearly Independant.

    # Homogenising
    pointsA_homogenous = np.hstack((pointsA, np.ones(pointsA.shape[0]).reshape((-1,1))))

    # A random transform between points in A and B
    Transform = np.eye(4).astype('float32')
    Transform[0:3, 0:3] = R.from_euler('zyx', np.random.rand(3) * 90, degrees = True).as_matrix()
    Transform[0,3], Transform[1,3], Transform[2,3] = np.random.rand(3) * 10

    pointsB_homogenous = np.matmul(Transform, pointsA_homogenous.T).T
    pointsB = np.delete(pointsB_homogenous, -1, 1)
    #Converting back to the cartesian coordinate system
    #We do not need to divide by the last element as it is already 1

    #Evaluating the algorithms with 2,3 and 4 correspondances

    method = "Orthogonal Procrustes"
    for p in range(1,5):
        recoveredTransform = closedFormICP(pointsA, pointsB, pointsNeeded := p, method)
        recovered = float(frobNorm(Transform, recoveredTransform, representation1="original transform", representation2="recovered transform")) < 1e-4
        print(f"With {p} point(s), the {method} algorithm can recover the original transform: {recovered} \n")

    method = "Wahba"
    for p in range(1,5):
        recoveredTransform = closedFormICP(pointsA, pointsB, pointsNeeded := p, method)
        recovered = float(frobNorm(Transform, recoveredTransform, representation1="original transform", representation2="recovered transform")) < 1e-4
        print(f"With {p} point(s), the {method} algorithm can recover the original transform: {recovered} \n")
