import numpy as np

def edge_edge_distance(P1,Q1,P2,Q2):
    """
    Computes the distance between two edges (segments) in 3D space.

    Parameters
    ----------
    P1 : (3,) numpy array
        start point of first edge
    Q1 : (3,) numpy array
        end point of first edge
    P2 : (3,) numpy array
        start point of second edge
    Q2 : (3,) numpy array
        end point of second edge

    Returns
    -------
    d : float
        The distance between the two edges.
    R1 : (3,) numpy array
        The closest point on the first edge to the second edge.
    R2 : (3,) numpy array
        The closest point on the second edge to the first edge.

    Notes
    -----
    This function is based on the algorithm from the "Proximity Query Pack" by Eric Larsen and Stefan Gottschalk

    Examples
    --------
    ```python
    import numpy as np
    from gpytoolbox import edge_edge_distance
    P1 = np.array([0.0,0.0,0.0])
    P2 = np.array([1.0,0.0,0.0])
    Q1 = np.array([0.0,1.0,0.0])
    Q2 = np.array([1.0,1.0,0.0])
    dist,R1,R2 = gpytoolbox.edge_edge_distance(P1,Q1,P2,Q2)
    ```
    """

    
    P = P1
    Q = P2
    A = Q1 - P1
    B = Q2 - P2
        #     VmV(T,Q,P);
    # P -> P1
    # Q -> P2
    T = Q - P
    # A_dot_A = VdotV(A,A);
    # B_dot_B = VdotV(B,B);
    # A_dot_B = VdotV(A,B);
    # A_dot_T = VdotV(A,T);
    # B_dot_T = VdotV(B,T);
    A_dot_A = np.dot(A,A)
    B_dot_B = np.dot(B,B)
    A_dot_B = np.dot(A,B)
    A_dot_T = np.dot(A,T)
    B_dot_T = np.dot(B,T)


    # // t parameterizes ray P,A 
    # // u parameterizes ray Q,B 

    # PQP_REAL t,u;

    # // compute t for the closest point on ray P,A to
    # // ray Q,B

    # PQP_REAL denom = A_dot_A*B_dot_B - A_dot_B*A_dot_B;

    # t = (A_dot_T*B_dot_B - B_dot_T*A_dot_B) / denom;

    # // clamp result so t is on the segment P,A

    # if ((t < 0) || isnan(t)) t = 0; else if (t > 1) t = 1;

    # // find u for point on ray Q,B closest to point at t

    # u = (t*A_dot_B - B_dot_T) / B_dot_B;

    denom = A_dot_A*B_dot_B - A_dot_B*A_dot_B
    if denom == 0:
        t = 0
    else:
        t = (A_dot_T*B_dot_B - B_dot_T*A_dot_B) / denom
    # print("t: ",t)
    if((t < 0) or np.isnan(t)):
        t = 0
    elif(t > 1):
        t = 1
    # print("t: ",t)
    if B_dot_B == 0:
        u = 0
    else:
        u = (t*A_dot_B - B_dot_T) / B_dot_B
    # print("u: ",u)
#     if ((u <= 0) || isnan(u)) {

#     VcV(Y, Q);

#     t = A_dot_T / A_dot_A;

#     if ((t <= 0) || isnan(t)) {
#       VcV(X, P);
#       VmV(VEC, Q, P);
#     }
#     else if (t >= 1) {
#       VpV(X, P, A);
#       VmV(VEC, Q, X);
#     }
#     else {
#       VpVxS(X, P, A, t);
#       VcrossV(TMP, T, A);
#       VcrossV(VEC, A, TMP);
#     }
#   }
    if ((u<=0) or np.isnan(u)):
        Y = Q.copy()
        t = A_dot_T / A_dot_A
        if ((t<=0) or np.isnan(t)):
            X = P.copy()
        elif (t>=1):
            X = P + A
        else:
            X = P + t*A
#   else if (u >= 1) {

#     VpV(Y, Q, B);

#     t = (A_dot_B + A_dot_T) / A_dot_A;

#     if ((t <= 0) || isnan(t)) {
#       VcV(X, P);
#       VmV(VEC, Y, P);
#     }
#     else if (t >= 1) {
#       VpV(X, P, A);
#       VmV(VEC, Y, X);
#     }
#     else {
#       VpVxS(X, P, A, t);
#       VmV(T, Y, P);
#       VcrossV(TMP, T, A);
#       VcrossV(VEC, A, TMP);
#     }
#   }

    elif (u>=1):
        Y = Q + B
        t = (A_dot_T + A_dot_B) / A_dot_A
        if ((t<=0) or np.isnan(t)):
            X = P.copy()
        elif (t>=1):
            X = P + A
        else:
            X = P + t*A
#   else {

#     VpVxS(Y, Q, B, u);

#     if ((t <= 0) || isnan(t)) {
#       VcV(X, P);
#       VcrossV(TMP, T, B);
#       VcrossV(VEC, B, TMP);
#     }
#     else if (t >= 1) {
#       VpV(X, P, A);
#       VmV(T, Q, X);
#       VcrossV(TMP, T, B);
#       VcrossV(VEC, B, TMP);
#     }
#     else {
#       VpVxS(X, P, A, t);
#       VcrossV(VEC, A, B);
#       if (VdotV(VEC, T) < 0) {
#         VxS(VEC, VEC, -1);
#       }
#     }
#   }
    else:
        Y = Q + u*B
        if ((t<=0) or np.isnan(t)):
            X = P.copy()
        elif (t>=1):
            # print("here")
            X = P + A
        else:
            X = P + t*A

    R1 = X
    R2 = Y
    dist = np.linalg.norm(R1-R2)
    return dist, R1, R2
# }




