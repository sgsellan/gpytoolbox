import numpy as np
from gpytoolbox.edge_edge_distance import edge_edge_distance

def triangle_triangle_distance(s0,s1,s2,t0,t1,t2):
    """Compute the distance between two triangles.
    Parameters
    ----------
    s0 : (3,) array
        First vertex of first triangle.
    s1 : (3,) array
        Second vertex of first triangle.
    s2 : (3,) array
        Third vertex of first triangle.
    t0 : (3,) array
        First vertex of second triangle.
    t1 : (3,) array
        Second vertex of second triangle.
    t2 : (3,) array
        Third vertex of second triangle.
    Returns
    -------
    d : float
        Distance between the two triangles.
    s : (3,) array
        Closest point on first triangle.
    t : (3,) array
        Closest point on second triangle.
    Notes
    -----
    This function is based on the algorithm from the "Proximity Query Pack" by Eric Larsen and Stefan Gottschalk. It would be nice to also output the closest point but this is hard in the case where the triangles intersect each other. Whenever we have a pure python triangle-triangle intersection function, we can use it here.
    
    Examples
    --------
    ```python
    import numpy as np
    from gpytoolbox import triangle_triangle_distance
    s0 = np.array([0.0,0.0,0.0])
    s1 = np.array([1.0,0.0,0.0])
    s2 = np.array([0.0,1.0,0.0])
    t0 = np.array([0.0,0.0,1.0])
    t1 = np.array([1.0,0.0,1.0])
    t2 = np.array([0.0,1.0,1.0])
    dist,s,t = triangle_triangle_distance(s0,s1,s2,t0,t1,t2)
    ```
    """
    shown_disjoint = False
    S = [s0,s1,s2]
    T = [t0,t1,t2]
    Sv = [s1 - s0, s2 - s1, s0 - s2]
    Tv = [t1 - t0, t2 - t1, t0 - t2]
    # Sv1 = s2 - s1
    # Sv2 = s0 - s2

    mindd = np.sum((S[0]-T[0])**2) + 10.0

    for i in range(3):
        for j in range(3):
            # print("S[i] : ", S[i])
            # print("Sv[i] : ", S[i]+Sv[i])
            # print("T[j] : ", T[j])
            # print("Tv[j] : ", T[j]+Tv[j])
            _,P,Q = edge_edge_distance(S[i],S[i]+Sv[i],T[j],T[j]+Tv[j])
            # print("P : ", P)
            # print("Q : ", Q)
            # # print(P)
            # # print(Q)
            VEC = Q - P
            V = Q - P
            dd = np.dot(V,V)
            # # print("BB")
            if dd <= mindd:
                minP = P.copy()
                minQ = Q.copy()
                mindd = dd.copy()
                Z = S[(i+2)%3] - P
                a = np.dot(Z,VEC)
                Z = T[(j+2)%3] - Q
                b = np.dot(Z,VEC)
                if ((a <= 0) and (b >= 0)):
                    # print("Here0")
                    # return np.sqrt(mindd), minP, minQ
                    return np.sqrt(mindd)
                p = np.dot(V,VEC)
                if a<0:
                    a = 0
                if b>0:
                    b = 0
                if ((p - a + b) > 0):
                    shown_disjoint = True
    # .....
    # # print("Sv[0] : ", Sv[0])
    # # print("Sv[1] : ", Sv[1])
    Sn = np.cross(Sv[0],Sv[1])
    # # print("Sn : ", Sn)
    Snl = np.dot(Sn,Sn)

    if (Snl > 1e-15):
        # V = S[0] - T[0]
        Tp = [np.dot(Sn,S[0] - T[0]), np.dot(Sn,S[0] - T[1]), np.dot(Sn,S[0] - T[2])]
        point = -1
        if ((Tp[0] > 0) and (Tp[1] > 0) and (Tp[2] > 0)):
            if (Tp[0] < Tp[1]):
                point = 0 
            else: 
                point = 1
            if (Tp[2] < Tp[point]):
                point = 2
        elif ((Tp[0] < 0) and (Tp[1] < 0) and (Tp[2] < 0)):
            if (Tp[0] > Tp[1]): 
                point = 0
            else:
                point = 1
            if (Tp[2] > Tp[point]):
                point = 2
        if (point >= 0):
            shown_disjoint = True
            V = T[point] - S[0]
            Z = np.cross(Sn,Sv[0])
            if (np.dot(V,Z) > 0):
                V = T[point] - S[1]
                Z = np.cross(Sn,Sv[1])
                if (np.dot(V,Z) > 0):
                    V = T[point] - S[2]
                    Z = np.cross(Sn,Sv[2])
                    if (np.dot(V,Z) > 0):
                        # # print("T[point] : ", T[point])
                        # # print("Tp[point] : ", Tp[point])
                        # # print("Sn : ", Sn)
                        # # print("Snl : ", Snl)
                        P = T[point] + Sn*(Tp[point])/Snl
                        Q = T[point].copy()
                        # print("Here1")
                        # return np.sqrt(np.dot(P-Q,P-Q)), P, Q
                        return np.sqrt(np.dot(P-Q,P-Q))
    
    Tn = np.cross(Tv[0],Tv[1])
    Tnl = np.dot(Tn,Tn)

    if (Tnl > 1e-15):
        # V = T[0] - S[0]
        # Sp = [T[0] - S[0], T[0] - S[1], T[0] - S[2]]
        Sp = [np.dot(Tn,T[0] - S[0]), np.dot(Tn,T[0] - S[1]), np.dot(Tn,T[0] - S[2])]
        point = -1
        if ((Sp[0] > 0) and (Sp[1] > 0) and (Sp[2] > 0)):
            if (Sp[0] < Sp[1]):
                point = 0
            else:
                point = 1
            if (Sp[2] < Sp[point]):
                point = 2
        elif ((Sp[0] < 0) and (Sp[1] < 0) and (Sp[2] < 0)):
            if (Sp[0] > Sp[1]):
                point = 0
            else:
                point = 1
            if (Sp[2] > Sp[point]):
                point = 2
        if (point >= 0):
            shown_disjoint = True
            V = S[point] - T[0]
            Z = np.cross(Tn,Tv[0])
            if (np.dot(V,Z) > 0):
                V = S[point] - T[1]
                Z = np.cross(Tn,Tv[1])
                if (np.dot(V,Z) > 0):
                    V = S[point] - T[2]
                    Z = np.cross(Tn,Tv[2])
                    if (np.dot(V,Z) > 0):
                        Q = S[point] + Tn*(Sp[point])/Tnl
                        P = S[point].copy()
                        # print("Here2")
                        # return np.sqrt(np.dot(P-Q,P-Q)), P, Q
                        return np.sqrt(np.dot(P-Q,P-Q))
    
    if(shown_disjoint):
        # print("Here3")
        # return np.sqrt(mindd), minP, minQ
        return np.sqrt(mindd)
    else:
        # print("Here4")
        return 0.0