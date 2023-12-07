import numpy as np
import athena_read

class AthenaReader:
    """
        Customized script to read Athena++ snapshots

        Output is dict and contains the grid (x1v etc. for cell-centered coordinate, 
        x1f etc. for interface coordinate) and the physical variables ('rho', 'press', 
        'vel1' etc.)
    """

    def __init__(self, filename):
        self.filename = filename

    def get_data(self):
        return athena_read.athdf(
            self.filename, 
            face_func_2 = self.GetMeshGen(
                nth_lo=16, nth_hi=8, dth_pole=0.3, h_hi=0.05, reflect=False)
        )

    def MeshGenOnePoint(self, n, nth_lo, nth_hi, dth_pole, h_hi):
        dth_mid = h_hi/nth_hi
        gamma = (dth_pole*np.log(dth_pole/dth_mid) + dth_mid - dth_pole) / (nth_lo*dth_pole + nth_hi*dth_mid - np.pi/2)
        if n<=nth_hi:
            th=n*dth_mid
        elif n<=nth_hi+np.log(dth_pole/dth_mid)/gamma:
            th=nth_hi*dth_mid+1/gamma*(np.exp((n-nth_hi)*gamma)-1)*dth_mid
        else: 
            th = np.pi/2 - dth_pole*(nth_hi+nth_lo-n)
        return th

    def GetMeshGen(self, nth_lo=12, nth_hi=8, dth_pole=0.3, h_hi=0.05, reflect=False):
        def GetTh(x):
            if reflect:
                sign = 1
                x = 1-x
            else:
                sign = np.sign(0.5-x)
                x = np.abs(0.5-x)*2
            x *= nth_lo + nth_hi
            th = self.MeshGenOnePoint(x, nth_lo, nth_hi, dth_pole, h_hi)
            th = np.pi/2 - th*sign
            return th
        def MeshGen(a,b,c,n):
            x = np.linspace(0,1,n)
            return np.vectorize(GetTh)(x)
        return MeshGen
