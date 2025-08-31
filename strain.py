import numpy as np


def _strain(self,x,t,k,e0,phi):
        if self.exc == "const":
             e = e0
        elif self.exc == "SAW_sin":
            e =  self._sin_strain(x,t,k,e0)
        elif self.exc == "SAW_Gauss":
            e =  self._gaussian_alt_strain(x,t)[0][0]
        elif self.exc == "lin":
            e =  self._lin_strain(e0,x)
        elif self.exc == 'oscill':
            w = self.of
            e = self._oscill_strain(x,t,e0,w)
        elif self.exc =="dsin":
            e = self._dual_sin_wave_strain(x,t,k,e0)
        else:
            raise ValueError(f"Unknown excitation type: {self.exc}")
        
        if self.strain_type == 'x':
            e = -e*np.cos(phi)**2
        if self.strain_type =="xz":
            #e = 2*e
            e = e + e*np.cos(phi)**2
        return e

def _strainx(self,x,t,k,e0,phi):
        if self.exc == "const":
             e = e0
        elif self.exc == "SAW_sin":
            e =  self._sin_strain(x,t,k,e0)
        elif self.exc == "SAW_Gauss":
            e =  self._gaussian_alt_strain(x,t)[0][0]
        elif self.exc == "lin":
            e =  self._lin_strain(e0,x)
        elif self.exc == 'oscill':
            w = self.of
            e = self._oscill_strain(x,t,e0,w)
        elif self.exc =="dsin":
            e = self._dual_sin_wave_strain(x,t,k,e0)
        else:
            raise ValueError(f"Unknown excitation type: {self.exc}")
        if self.strain_type == 'x':
            e = e
        if self.strain_type == 'xz':
            e = -e
        return e

def _deriv_strain(self,e0,x,t,k,phi):
        if self.exc == "const":
             dedx = 0
        if self.exc == "SAW_sin":
            dedx = self._dsin_strain(e0,x,t,k)
        elif self.exc == "SAW_Gauss":
            dedx = self._gaussian_alt_strain(x,t)[0][1]
        elif self.exc == "lin":
            dedx =  self._dlin_strain(e0)
        elif self.exc == 'oscill':
            w = self.of
            dedx = self._doscill_strain(e0,w,t)
        elif self.exc == 'dsin':
             dedx = self._ddual_sin_wave_strain(x,t,k,e0)
        if self.strain_type == "x":
            dedx = -dedx*np.cos(phi)**2
        elif self.strain_type =="xz":
            dedx += dedx*np.cos(phi)**2
        return dedx


def _lin_strain(self,e0,x):
        return self._dlin_strain(e0)*(x) + self.strain0 

def _dlin_strain(self,e0):
        return (e0-self.strain0)/self.L

def _oscill_strain(self,x,t,e0,w):
        return self._dlin_strain(e0)*x*np.sin(2*np.pi*w*t)

def _doscill_strain(self,e0,w,t):
        return self._dlin_strain(e0)*np.sin(2*np.pi*w*t)

def _sin_strain(self,x,t,k,e0):
        return e0 * np.sin(k * (x - self.vsaw * t) + self.phi)

def _dsin_strain(self,e0,x,t,k):
        return e0 * np.cos(k * (x - self.vsaw * t) + self.phi) * k
    
def _gaussian_alt_strain(self,x,t):
        """
        Compute the wave amplitude at a single space-time point (x, t),
        for a series of alternating Gaussian pulses from opposite sides.

        Parameters:
            x : float
                Spatial coordinate.
            t : float
                Time.
            f : float
                Pulse repetition frequency (Hz).
            c : float
                Wave propagation speed.
            sigma : float
                Width of the Gaussian pulse.
            A : float
                Amplitude of each pulse.
            x_bounds : tuple
                (x_min, x_max), domain bounds for alternating source locations.

        Returns:
            psi : float
                Wave amplitude at (x, t).
        """
        FWHM = self.FWHM
        sigma = FWHM/2.355
        A = self.strain
        c = self.vsaw
        x_bounds=[0,self.L]
        if t < 0:
            return 0.0  # No pulses before time zero
        Tg = 1.0 / self.gf                      # Period between pulses
        n_pulses = int(t // Tg) + 1       # Number of pulses emitted so far
        psi = 0.0
        dpsi = 0.0
        for n in range(n_pulses):
            t_emit = n * Tg
            dtt = t - t_emit
            if dtt < 0:
                continue

            # Alternate side and direction
            if n % 2 == 0:
                x0 = x_bounds[0]
                direction = +1
                Af = A
                cf =c 
                sigmaf = sigma
            else:
                x0 = x_bounds[1]
                direction = -1
                fac = self.fac
                cf = c * self.fac
                Af = A#*self.fac
                sigmaf = sigma/ self.fac
            x_c = x0 + direction * cf * dtt

            # Gaussian pulse centered at x_c
            psi += Af * np.exp(-((x - x_c)**2) / (2 * sigmaf**2))
            dpsi += -Af*(x-x_c)/(sigmaf**2) * np.exp(-((x - x_c)**2) / (2 * sigmaf**2))
        return [psi, dpsi],

def _dual_sin_wave_strain(self,x,t,k,e0):
        f0 = self._f0()
        df = self.df
        k1 = 2*np.pi*(f0 - df)/self.vsaw
        k2 = 2*np.pi*(f0 + df)/self.vsaw
        wave1 = self._sin_strain(x,t,k1,e0)
        wave2 = self.fac*self._sin_strain(x,-t,k2,e0)
        return wave1 + wave2

def _ddual_sin_wave_strain(self,x,t,k,e0):
        f0 = self._f0()
        df = self.df
        k1 = 2*np.pi*(f0 + df)/self.vsaw
        k2 = 2*np.pi*(f0 - df)/self.vsaw
        dwave1 = self._dsin_strain(e0,x,-t,k1)
        dwave2 = self.fac*self._dsin_strain(e0,x,t,k2)
        return dwave1 + dwave2