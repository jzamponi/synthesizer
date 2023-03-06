import numpy as np

class VectorField():

    def __init__(self, x, y, z, morphology, normalize=True, a_eff=1):
        """ Create an object containing 3D vector field for a given morphology. 
            The field can be optionally normalized and consider an alignment
            efficiency factor. Morphologies supported are:
                - Uniform x: 'x'
                - Uniform y: 'y'
                - Uniform z: 'z'
                - Toroidal: 't'
                - Radial: 'r'
                - Hourglass: 'h'
                - Helicoidal: 'hel'
                - Dipole: 'd'
                - Quadrupole: 'q'
        """

        self.x = x
        self.y = y 
        self.z = z 
        self.vx = np.zeros((x.shape[0], y.shape[0], z.shape[0]))
        self.vy = np.zeros((x.shape[0], y.shape[0], z.shape[0]))
        self.vz = np.zeros((x.shape[0], y.shape[0], z.shape[0]))
        if morphology is None:
            return morphology
        else:
            self.morphology = morphology.lower()

        self.a_eff = a_eff
 
        if self.morphology == 'x':
            self.vx = np.ones(x.shape)
 
        elif self.morphology == 'y':
            self.vy = np.ones(y.shape)
 
        elif self.morphology == 'z':
            self.vz = np.ones(z.shape)
 
        elif self.morphology in ['t', 'toroidal']:
            r = np.sqrt(x**2 + y**2)
            self.vx = y / r
            self.vy = -x / r
 
        elif self.morphology in ['r', 'radial']:
            r = np.sqrt(x**2 + y**2 + z**2)
            self.vx = x / r
            self.vy = y / r
            self.vz = z / r
 
        elif self.morphology in ['h', 'hourglass']:
            a = 5e-34
            factor = 1 / np.sqrt(
                1 + (a*x*z)**2*np.exp(-2*a*z*z) + (a*y*z)**2*np.exp(-2*a*z*z))
            self.vx = a * x * z * np.exp(-a * z*z) * factor
            self.vy = a * y * z * np.exp(-a * z*z) * factor
            self.vz = factor 

        elif self.morphology in ['hel', 'helicoidal']:
            # Helicoidal = Superpoistion of Toroidal & Hourglass
            r = np.sqrt(x**2 + y**2)
            toro = np.array([y/r, -x/r, np.zeros(z.shape)])

            a = 5e-34
            factor = np.sqrt(
                1 + (a*x*z)**2*np.exp(-2*a*z*z) + (a*y*z)**2*np.exp(-2*a*z*z))
            hour = np.array([
                a * x * z * np.exp(-a * z*z) / factor, 
                a * y * z * np.exp(-a * z*z) / factor,
                np.ones(z.shape)
            ])
            heli = (toro + hour) / np.linalg.norm(toro + hour)
            self.vx = heli[0]
            self.vy = heli[1]
            self.vz = heli[2]

        elif self.morphology in ['d', 'dipole']:
            r5 = np.sqrt(x**2 + y**2 + z**2)**5
            self.vx = (3 * x * z) / r5
            self.vy = (3 * y * z) / r5
            self.vz = (2*z*z - x*x - y*y) / r5
 
        elif self.morphology in ['q', 'quadrupole']:
            r7 = 2 * np.sqrt(x**2 + y**2 + z**2)**7 
            self.vx = (-3*x*(x**2 + y**2 - 4*z**2)) / r7
            self.vy = (-3*y*(x**2 + y**2 - 4*z**2)) / r7
            self.vz = (3*z*(-3*x**2 - 3*y**2 + 2*z**2)) / r7

        elif self.morphology == 'custom':
            normalize = False

            # Feel free to write here your own vector field components.
            # If you don't they'll just be zeros
        


        # Normalize the field
        if normalize:
            self.r = np.sqrt(self.vx**2 + self.vy**2 + self.vz**2)
            self.vx = np.squeeze(self.vx) / self.r
            self.vy = np.squeeze(self.vy) / self.r
            self.vz = np.squeeze(self.vz) / self.r
 
        # Assume perfect alignment (a_eff = 1). We can change it in the future
        self.vx *= a_eff
        self.vy *= a_eff
        self.vz *= a_eff

        self.rxy = np.sqrt(self.x**2 + self.y**2)
        self.rxz = np.sqrt(self.x**2 + self.z**2)
        self.ryz = np.sqrt(self.y**2 + self.z**2)
