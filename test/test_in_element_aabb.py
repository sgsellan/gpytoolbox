from .context import gpytoolbox
from .context import numpy as np
from .context import unittest

class TestInElementAABB(unittest.TestCase):
    def test_synthetic_samples(self):
        # Generate triangle mesh
        V,F = gpytoolbox.regular_square_mesh(23)
        num_samples = 100
        B = np.random.rand(num_samples,3)
        B = B/np.tile(np.linalg.norm(B,axis=1,ord=1),(3,1)).transpose()
        # We generate from these triangles, using barycentric coordinates
        indeces = np.random.randint(F.shape[0],size=num_samples)
        queries = np.zeros((num_samples,2))
        queries[:,0] = B[:,0]*V[F[indeces,0],0] + B[:,1]*V[F[indeces,1],0] + B[:,2]*V[F[indeces,2],0]
        queries[:,1] = B[:,0]*V[F[indeces,0],1] + B[:,1]*V[F[indeces,1],1] + B[:,2]*V[F[indeces,2],1]

        I = gpytoolbox.in_element_aabb(queries,V,F.astype(np.int32))

        # We should find exactly the triangles we generated from
        assert((I==indeces).all())

        # Now let's do the same but in 3D, for a tet mesh
        # Generate tet mesh
        V,T= gpytoolbox.regular_cube_mesh(23)
        num_samples = 100
        B = np.random.rand(num_samples,4)
        B = B/np.tile(np.linalg.norm(B,axis=1,ord=1),(4,1)).transpose()
        # We generate from these triangles, using barycentric coordinates
        indeces = np.random.randint(T.shape[0],size=num_samples)
        queries = np.zeros((num_samples,3))
        queries[:,0] = B[:,0]*V[T[indeces,0],0] + B[:,1]*V[T[indeces,1],0] + B[:,2]*V[T[indeces,2],0] + B[:,3]*V[T[indeces,3],0]
        queries[:,1] = B[:,0]*V[T[indeces,0],1] + B[:,1]*V[T[indeces,1],1] + B[:,2]*V[T[indeces,2],1] + B[:,3]*V[T[indeces,3],1]
        queries[:,2] = B[:,0]*V[T[indeces,0],2] + B[:,1]*V[T[indeces,1],2] + B[:,2]*V[T[indeces,2],2] + B[:,3]*V[T[indeces,3],2]

        I = gpytoolbox.in_element_aabb(queries,V,T.astype(np.int32))

        # We should find exactly the triangles we generated from
        assert((I==indeces).all())

        # TODO: Assert that runtime is logarithmic (I kinda trust libigl that this is right, I have used this in the past)
    # def test_npe_type(self):
    #     V,T= gpytoolbox.regular_cube_mesh(23)
    #     num_samples = 100
    #     B = np.random.rand(num_samples,4)
    #     B = B/np.tile(np.linalg.norm(B,axis=1,ord=1),(4,1)).transpose()
    #     # We generate from these triangles, using barycentric coordinates
    #     indeces = np.random.randint(T.shape[0],size=num_samples)
    #     queries = np.zeros((num_samples,3))
    #     queries[:,0] = B[:,0]*V[T[indeces,0],0] + B[:,1]*V[T[indeces,1],0] + B[:,2]*V[T[indeces,2],0] + B[:,3]*V[T[indeces,3],0]
    #     queries[:,1] = B[:,0]*V[T[indeces,0],1] + B[:,1]*V[T[indeces,1],1] + B[:,2]*V[T[indeces,2],1] + B[:,3]*V[T[indeces,3],1]
    #     queries[:,2] = B[:,0]*V[T[indeces,0],2] + B[:,1]*V[T[indeces,1],2] + B[:,2]*V[T[indeces,2],2] + B[:,3]*V[T[indeces,3],2]

    #     # This breaks
    #     I = gpytoolbox.in_element_aabb(queries,V,T)

    #     # We should find exactly the triangles we generated from
    #     assert((I==indeces).all())

if __name__ == '__main__':
    unittest.main()