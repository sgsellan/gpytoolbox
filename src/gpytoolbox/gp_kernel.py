import numpy as np
import sys


class gp_kernel:
    def __init__(self,dim=2,length=0.2,scale=1,type='exponential'):
        self.scale = scale
        self.length = length
        self.dim = dim
        b = {
            'exponential' : 0,
            'thinplate' : 1,
            'wendland_compact' : 2,
            # 'compact_exponential' : 2,
            # 'compact_thinplate' : 3
            }
        self.var_type = b.get(type,-1)

    def __call__(self,P1,P2):
        if self.var_type==0:
            k = np.exp(-0.5*np.sum(((P1-P2) * (P1-P2)),axis=1)/(self.length**2.0))
        elif self.var_type==1:
            d = np.linalg.norm(P1-P2,axis=1)
            if self.dim==1:
                k = 2*(d**3.0) - 3*self.length*(d**2.0) + (self.length**3.0)
            elif self.dim==2:
                k = 2*(d**2.0)*(np.log(d)) - (1+2*np.log(self.length))*(d**2.0) + (self.length**2.0)
                k[d<1e-5] = (self.length**2.0)
            elif self.dim==3:
                k = 2*(d**3.0) - 3*self.length*(d**2.0) + (self.length**3.0)
        elif self.var_type==2:
            d = np.linalg.norm(P1-P2,axis=1)/self.length
            if self.dim==1:
                k = (np.abs(1-d)**3.0) * (3*d + 1)
                k[d>=1.0] = 0
            elif (self.dim==2) or (self.dim==3):
                k = (np.abs(1-d)**4.0) * (4*d + 1)
                k[d>=1.0] = 0
        return self.scale*k
    def partial_derivative(self,indi,indj):
        # print(self.var_type)
        # print(self.dim)

        if (indi==-1 and indj==-1):
            # This is just evaluating the kernel
            return self.__call__

        if self.var_type==0: # Exponential covariance
            gamma = 1/(self.length**2.0)         
            if (indj==-1 or indi==-1):
                ind = np.maximum(indi,indj)
                def fun(P1,P2):
                    # derivative of the kernel wrt x_ind
                    return -(P1[:,ind] - P2[:,ind])*self.__call__(P1,P2)/(self.length**2.0)
            else:
                def fun(P1,P2):
                     # derivative of the kernel wrt x_indi and x_indj
                    return ( (indi==indj)*gamma - gamma*gamma*(P1[:,indi] - P2[:,indi])*(P1[:,indj] - P2[:,indj]))*self.__call__(P1,P2)
        elif self.var_type==1: #Thinplate covariance
            sys.exit("Haven't written down these derivatives yet")
        
        elif self.var_type==2: #Wendland kernel
            if self.dim==1:
                if (indj==-1 or indi==-1): # derivative of the kernel wrt x_ind
                    ind = np.maximum(indi,indj)
                    def fun(P1,P2):
                        d = np.linalg.norm(P1-P2,axis=1)/self.length
                        s = np.squeeze(np.sign(P1-P2))
                        # derivative of the kernel wrt x_ind
                        k = s*self.scale*( -(3*(np.abs(1-d)**2.0) * (3*d + 1)) + (3*((1-d)**3.0)))/self.length
                        k[d>=1.0] = 0
                        return k
                else:
                    def fun(P1,P2):
                        d = np.linalg.norm(P1-P2,axis=1)/self.length
                        k = self.scale*( (6*(1-d) * (3*d + 1)) - (18*((1-d)**2.0) ))/(self.length**2.0)
                        k[d>=1] = 0
                        return k
            if (self.dim==2 or self.dim==3):
                if (indj==-1 or indi==-1): # derivative of the kernel wrt x_ind
                    ind = np.maximum(indi,indj)
                    def fun(P1,P2):
                        d = np.linalg.norm(P1-P2,axis=1)/self.length
                        # derivative of the kernel wrt x_ind
                        k = -4.0*(P1[:,ind] - P2[:,ind])**1.0*(1 - d)**3.0*(4*d + 1)/d + 4.0*(P1[:,ind] - P2[:,ind])**1.0*(1 - d)**4.0/d
                        k = (self.scale/(self.length**2.0))*k
                        k[d>=1.0] = 0
                        k[np.isnan(k)] = 0
                        return k
                else:
                    def fun(P1,P2):
                        d = np.linalg.norm(P1-P2,axis=1)/self.length
                        if indi==indj:
                            k = (-4.0*(1 - d)**4.0*((P1[:,indi] - P2[:,indi])**2.0/((d**2.0)*(self.length**2.0)) - 1)/(d*self.length) + (1 + 4*d)*(4.0*(P1[:,indi] - P2[:,indi])**2.0*(1 - d)**3.0/((d**2.0)*(self.length**2.0))**(3/2) - 4.0*(1 - d)**3.0/(d*self.length) + 12.0*(P1[:,indi] - P2[:,indi])**2.0*(1 - d)**2.0/(self.length*((d**2.0)*(self.length**2.0)))) - 32.0*(P1[:,indi] - P2[:,indi])**2.0*(1 - d)**3.0/(self.length*((d**2.0)*(self.length**2.0))))/self.length
                        else:
                            k = (P1[:,indi] - P2[:,indi])**1.0*(P1[:,indj] - P2[:,indj])**1.0*(4.0*(1 - d)**3.0*(1 + 4*d)/((d**2.0)*(self.length**2.0))**(3/2) - 4.0*(1 - d)**4.0/((d**2.0)*(self.length**2.0))**(3/2) + 12.0*(1 - d)**2.0*(1 + 4*d)/(self.length*((d**2.0)*(self.length**2.0))) - 32.0*(1 - d)**3.0/(self.length*((d**2.0)*(self.length**2.0))))/self.length

                        k = self.scale*k
                        k[d>=1.0] = 0
                        k[np.isnan(k)] = 0
                        return k
        return fun

        # elif self.var_type==2:
        #     k = gpytoolbox.compactly_supported_normal(P1-P2, n=3,center=np.zeros(self.dim),sigma=self.length)
        # elif self.var_type==3:
        #     d = np.linalg.norm(P1-P2,axis=1)
        #     if self.dim==1:
        #         k = 2*(d**3.0) - 3*self.length*(d**2.0) + (self.length**3.0)
        #     elif self.dim==2:
        #         k = 2*(d**2.0)*(np.log(d)) - (1+2*np.log(self.length))*(d**2.0) + (self.length**2.0)
        #         k[d<1e-5] = (self.length**2.0)
        #     elif self.dim==3:
        #         k = 2*(d**3.0) - 3*self.length*(d**2.0) + (self.length**3.0)
        #     k[d>self.length]=0.0
        
        # return self.scale*k