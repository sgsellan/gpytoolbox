import numpy as np

def regular_square_mesh(gs):
    x, y = np.meshgrid(np.linspace(0,1,gs),np.linspace(0,1,gs))
    v = np.concatenate((np.reshape(x,(-1, 1)),np.reshape(y,(-1, 1))),axis=1)

    f = np.zeros((2*(gs-1)*(gs-1),3),dtype=int)

    a = np.linspace(0,gs-2,gs-1,dtype=int)
    for i in range(0,gs-1):
        f[((gs-1)*i):((gs-1)*i + gs-1),0] = gs*i + a
        f[((gs-1)*i):((gs-1)*i + gs-1),1] = gs*i + a + gs + 1
        f[((gs-1)*i):((gs-1)*i + gs-1),2] = gs*i + a + gs
    for i in range(0,gs-1):
        f[((gs-1)*(i+gs-1)):((gs-1)*(i+gs-1) + gs-1),0] = gs*i + a
        f[((gs-1)*(i+gs-1)):((gs-1)*(i+gs-1) + gs-1),1] = gs*i + a + 1
        f[((gs-1)*(i+gs-1)):((gs-1)*(i+gs-1) + gs-1),2] = gs*i + a + gs + 1
    return v,f
