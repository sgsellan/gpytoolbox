from .context import gpytoolbox as gpy
from .context import numpy as np
from .context import unittest
import scipy as sp

class TestMinQuadWithFixed(unittest.TestCase):

    def test_unconstrained_posdef(self):
        Qdense = np.array(
            [[1., 0.2, 0.3],
            [0.2, 2., -0.1],
            [0.3, -0.1, 1.]])
        Q = sp.sparse.csr_matrix(Qdense)

        test_mqwf(self, np.array([-1.29692833, -0.9556314 , -1.70648464]), Q, c=2.)
        test_mqwf(self, np.array([-1.86860068, -0.32423208, -0.221843]), Q, c=np.array([2., 1., 0.75]))
        test_mqwf(self, np.array([[-1.86860068, 0.34243458], [-0.32423208, -1.57224118], [-0.221843, -0.75995449]]), Q, c=np.array([[2.,0.2], [1.,3.], [0.75,0.5]]))

    def test_unconstrained_nonsymm(self):
        Qdense = np.array(
            [[1., 0.2, 0.3],
            [0.9, 2., -0.1],
            [5., 24., 1.]])
        Q = sp.sparse.csr_matrix(Qdense)

        test_mqwf(self, np.array([-1.80213314, 0.2427363, -0.12504597]), Q, c=2.)
        test_mqwf(self, np.array([-2.07398553, 0.40180213, -0.05547383]), Q, c=np.array([2., 1., 0.75]))
        test_mqwf(self, np.array([[-2.07398553, 0.54712912], [0.40180213, -0.14204499], [-0.05547383, -0.25245448]]), Q, c=np.array([[2.,0.2], [1.,3.], [0.75,0.5]]))


    def test_unconstrained_indefinite(self):
        Qdense = np.array(
            [[1., -2., 0.2],
            [-2., 0.5, -1.],
            [0.2, -1., 1.5]])
        Q = sp.sparse.csr_matrix(Qdense)

        test_mqwf(self, np.array([0.20338208, 0.22212066, -0.04570384]), Q, c=0.25)
        test_mqwf(self, np.array([0.07449726, 0.16563071, 0.03382084]), Q, c=np.array([0.25, 0.1, 0.1]))
        test_mqwf(self, np.array([[0.07449726, 0.14853748], [0.16563071, 0.13638026], [0.03382084, -0.12888483]]), Q, c=np.array([[0.25,0.15], [0.1,0.1], [0.1,0.3]]))
 
    def test_fix_definite(self):
        Qdense = np.array(
            [[1., 0.2, 0.3, -0.1],
            [0.2, 2., -0.1, 0.05],
            [0.3, -0.1, 1., 0.1],
            [-0.1, 0.05, 0.1, 1.5]])
        Q = sp.sparse.csr_matrix(Qdense)

        c = np.array([1., 2., 0.25, 0.5])
        k = np.array([0,2])

        test_mqwf(self, np.array([0., -0.99249374, 0., -0.30025021]), Q, c=c, k=k, y=None)
        test_mqwf(self, np.array([1., -1.04253545, 1., -0.29858215]), Q, c=c, k=k, y=1.)
        test_mqwf(self, np.array([0.5, 0.02668891, 1.5, -0.0675563]), Q, c=None, k=k, y=np.array([0.5, 1.5]))
        test_mqwf(self, np.array([0.5, -0.96580484, 1.5, -0.36780651]), Q, c=c, k=k, y=np.array([0.5, 1.5]))
        test_mqwf(self, np.array([[0.5, -1.], [-0.96580484, -0.78732277], [1.5, 2.], [-0.36780651, -0.50708924]]), Q, c=np.stack([c,c],axis=-1), k=k, y=np.array([[0.5, -1.], [1.5, 2.]]))


    def test_fix_indefinite(self):
        Qdense = np.array(
            [[-1., 0.2, 0.3, -0.1],
            [0.2, 2., -0.1, 0.05],
            [0.3, -0.1, 1., 0.1],
            [-0.1, 0.05, 0.1, -1.5]])
        Q = sp.sparse.csr_matrix(Qdense)

        c = np.array([1., 2., 0.25, 0.5])
        k = np.array([0,2])

        test_mqwf(self, np.array([0., -1.00749376, 0., 0.29975021]), Q, c=c, k=k, y=None)
        test_mqwf(self, np.array([1., -1.05745212, 1., 0.29808493]), Q, c=c, k=k, y=1.)
        test_mqwf(self, np.array([0.5, 0.02331391, 1.5, 0.0674438]), Q, c=None, k=k, y=np.array([0.5, 1.5]))
        test_mqwf(self, np.array([0.5, -0.98417985, 1.5, 0.367194]), Q, c=c, k=k, y=np.array([0.5, 1.5]))
        test_mqwf(self, np.array([[0.5, -1.], [-0.98417985, -0.81265612], [1.5, 2.], [0.367194, 0.5062448]]), Q, c=np.stack([c,c],axis=-1), k=k, y=np.array([[0.5, -1.], [1.5, 2.]]))


    def test_eq_constraint_definite(self):
        Qdense = np.array(
            [[1., 0.2, 0.3, -0.1],
            [0.2, 2., -0.1, 0.05],
            [0.3, -0.1, 1., 0.1],
            [-0.1, 0.05, 0.1, 1.5]])
        Q = sp.sparse.csr_matrix(Qdense)

        c = np.array([1., 2., 0.25, 0.5])

        Adense = np.array(
            [[1., 0., 0., -3.],
            [0., 0.2, -0.5, 1.]])
        A = sp.sparse.csr_matrix(Adense)

        test_mqwf(self, np.array([-0.42205769, -0.89801359, -0.64057723, -0.1406859]), Q, c=c, A=A, b=None)
        test_mqwf(self, np.array([0.98982889, -0.87231052, -2.35570495, -0.00339037]), Q, c=c, A=A, b=1.)
        test_mqwf(self, np.array([1.70479868, 0.03719266, -2.18192382, 0.40159956]), Q, c=None, A=A, b=np.array([0.5, 1.5]))
        test_mqwf(self, np.array([1.28274098, -0.86082093, -2.82250105, 0.26091366]), Q, c=c, A=A, b=np.array([0.5, 1.5]))
        test_mqwf(self, np.array([[1.28274098, 1.16262189], [-0.86082093, -0.85069329], [-2.82250105, -2.89852939], [0.26091366, 0.72087396]]), Q, c=np.stack([c,c],axis=-1), A=A, b=np.array([[0.5, -1.], [1.5, 2.]]))


    def test_eq_constraint_indefinite(self):
        Qdense = np.array(
            [[-1., 0.2, 0.3, -0.1],
            [0.2, 2., -0.1, 0.05],
            [0.3, -0.1, 1., 0.1],
            [-0.1, 0.05, 0.1, -1.5]])
        Q = sp.sparse.csr_matrix(Qdense)

        c = np.array([1., 2., 0.25, 0.5])

        Adense = np.array(
            [[1., 0., 0., -3.],
            [0., 0.2, -0.5, 1.]])
        A = sp.sparse.csr_matrix(Adense)

        test_mqwf(self, np.array([1.58824492, -1.42958399, 0.48699635, 0.52941497]), Q, c=c, A=A, b=None)
        test_mqwf(self, np.array([-3.04438144, 0.19442778, -4.61848318, -1.34812715]), Q, c=c, A=A, b=1.)
        test_mqwf(self, np.array([-6.07510496, 2.09437871, -6.54565182, -2.19170165]), Q, c=None, A=A, b=np.array([0.5, 1.5]))
        test_mqwf(self, np.array([-4.48686004, 0.66479472, -6.05865547, -1.66228668]), Q, c=c, A=A, b=np.array([0.5, 1.5]))
        test_mqwf(self, np.array([[-4.48686004, -5.05550406], [0.66479472, 0.79352271], [-6.05865547, -6.38626029], [-1.66228668, -1.35183469]]), Q, c=np.stack([c,c],axis=-1), A=A, b=np.array([[0.5, -1.], [1.5, 2.]]))


    def test_all_definite(self):
        Qdense = np.array(
            [[1., 0.2, 0.3, -0.1, 0.05, 0.025],
            [0.2, 2., -0.1, 0.05, 0.1, -0.05],
            [0.3, -0.1, 1., 0.1, 0.03, 0.01],
            [-0.1, 0.05, 0.1, 1.5, -0.05, 0.15],
            [0.5, 0.1, 0.03, -0.05, 3., 0.5],
            [0.025, -0.05, 0.01, 0.15, 0.5, 4.]])
        Q = sp.sparse.csr_matrix(Qdense)

        c = np.array([1., 2., 0.25, 0.5, 0.5, 4.])

        k = np.array([0,3])

        Adense = np.array(
            [[1., 0., 4., 0., -3., 0.],
            [0., 0.2, 0., -0.5, 1., 0.25]])
        A = sp.sparse.csr_matrix(Adense)

        test_mqwf(self, np.array([0., 0., 0., 0., 0., 0.]), Q, c=None, A=A, b=None, k=k, y=None)
        test_mqwf(self, np.array([0., -0.50731781, 1.23547555, 0., 1.3139674, -0.85001536]), Q, c=c, A=A, b=1., k=k, y=None)
        test_mqwf(self, np.array([1., -0.79752442, 0.41943775, 1., 0.89258367, -0.93231514]), Q, c=c, A=A, b=None, k=k, y=1.)
        test_mqwf(self, np.array([1., -0.4423561, 1.34574049, 1., 1.79432065, -0.82339773]), Q, c=c, A=A, b=1., k=k, y=1.)
        test_mqwf(self, np.array([0.2, -0.23986143, -0.31211128, 0.5, -0.68281504, -0.0768507]), Q, c=None, A=A, b=np.array([1., -1.]), k=k, y=np.array([0.2, 0.5]))
        test_mqwf(self, np.array([0.2, -1.10234756, -0.00293846, 0.5, -0.27058462, -1.03578347]), Q, c=c, A=A, b=np.array([1., -1.]), k=k, y=np.array([0.2, 0.5]))
        test_mqwf(self, np.array([0.2, -0.80690879, 0.43662672, 0.5, 0.64883563, -0.94981548]), Q, c=c, A=A, b=None, k=k, y=np.array([0.2, 0.5]))
        test_mqwf(self, np.array([0., -1.1579249, -0.13039237, 0., -0.50718983, -1.04490076]), Q, c=c, A=A, b=np.array([1., -1.]), k=k, y=None)
        test_mqwf(self, np.array([[0.2, -0.6], [-1.10234756, -0.36078195], [-0.00293846, 1.5709661], [ 0.5, 2.], [-0.27058462, 1.79462146], [-1.03578347, -0.88986029]]), Q, c=np.stack([c,c],axis=-1), A=A, b=np.array([[1.,0.3], [-1.,0.5]]), k=k, y=np.array([[0.2,-0.6], [0.5,2.]]))


    def test_all_indefinite(self):
        Qdense = np.array(
            [[1., 0.2, 0.3, -0.1, 0.05, 0.025],
            [0.2, 2., -0.1, 0.05, 0.1, -0.05],
            [0.3, -0.1, -1., 0.1, 0.03, 0.01],
            [-0.1, 0.05, 0.1, 1.5, -0.05, 0.15],
            [0.5, 0.1, 0.03, -0.05, -3., 0.5],
            [0.025, -0.05, 0.01, 0.15, 0.5, -4.]])
        Q = sp.sparse.csr_matrix(Qdense)

        c = np.array([1., 2., 0.25, 0.5, 0.5, 4.])

        k = np.array([0,3])

        Adense = np.array(
            [[1., 0., 4., 0., -3., 0.],
            [0., 0.2, 0., -0.5, 1., 0.25]])
        A = sp.sparse.csr_matrix(Adense)

        test_mqwf(self, np.array([0., 0., 0., 0., 0., 0.]), Q, c=None, A=A, b=None, k=k, y=None)
        test_mqwf(self, np.array([0., -1.17893715, 0.93979026, 0., 0.91972035, 1.26426833]), Q, c=c, A=A, b=1., k=k, y=None)
        test_mqwf(self, np.array([1., -1.07984649, 0.08252405, 1., 0.4433654, 1.09041557]), Q, c=c, A=A, b=None, k=k, y=1.)
        test_mqwf(self, np.array([1., -1.41734295, 1.06831087, 1., 1.4244145, 1.43621636]), Q, c=c, A=A, b=1., k=k, y=1.)
        test_mqwf(self, np.array([0.2, 0.2263423, -0.35345761, 0.5, -0.73794348, -0.22929992]), Q, c=None, A=A, b=np.array([1., -1.]), k=k, y=np.array([0.2, 0.5]))
        test_mqwf(self, np.array([0.2, -0.61509839, -0.39945417, 0.5, -0.79927223, 0.68916762]), Q, c=c, A=A, b=np.array([1., -1.]), k=k, y=np.array([0.2, 0.5]))
        test_mqwf(self, np.array([0.2, -0.94463659, 0.08923189, 0.5, 0.18564252, 1.01313919]), Q, c=c, A=A, b=None, k=k, y=np.array([0.2, 0.5]))
        test_mqwf(self, np.array([0., -0.51190249, -0.53468262, 0., -1.04624349, 0.59449597]), Q, c=c, A=A, b=np.array([1., -1.]), k=k, y=None)
        test_mqwf(self, np.array([[0.2, -0.6], [-0.61509839, -1.34747734], [-0.39945417, 1.26928525], [ 0.5, 2.], [-0.79927223, 1.39238033], [0.68916762, 1.50846054]]), Q, c=np.stack([c,c],axis=-1), A=A, b=np.array([[1.,0.3], [-1.,0.5]]), k=k, y=np.array([[0.2,-0.6], [0.5,2.]]))


def test_mqwf(test, u_gt, Q, c=None, A=None, b=None, k=None, y=None):
    u1 = gpy.min_quad_with_fixed(Q, c, A, b, k, y)
    if b is not None:
        test.assertTrue(np.isclose(A*u1, b).all())
    if y is not None:
        test.assertTrue(np.isclose(u1[k], y).all())
    if u_gt is not None:
        test.assertTrue(np.isclose(u1, u_gt).all())

    solver = gpy.min_quad_with_fixed_precompute(Q, A, k)
    u2 = solver.solve(c, b, y)
    test.assertTrue(np.isclose(u1, u2).all())
    u3 = solver.solve(c, b, y)
    test.assertTrue(np.isclose(u1, u3).all())


if __name__ == '__main__':
    unittest.main()