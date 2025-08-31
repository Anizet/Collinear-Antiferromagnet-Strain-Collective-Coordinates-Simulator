import numpy as np
import re

def _calculate_tau(self, x, e0, k, delta, Meff,t,phi):
        Fform = self.Ffact * delta * self._deriv_strain(e0,x,t,k,phi)
        return np.sqrt(np.abs(Fform) / (Meff * 2 * delta)) #/ 10

def _calculate_Bdyn(self, f, Bs, Binf, tau):
        nume = (1 + (f**2/ (tau+1e-12)**2)**((1-self.alph)*2))**(-self.beta/2)*np.cos(self.beta*np.arctan(f/(tau+1e-12))**(1-self.alph))
        return Bs #Binf/2 + (Bs - Binf/2) * nume

def _calculate_Bdyn_f(self, f, tau):
        nume = (1 + (f**2 / (tau+1e-15)**2)**((1 - self.alph) * 2))**(-self.beta / 2) * np.cos(self.beta * np.arctan(f / (tau+1e-15))**(1 - self.alph))
        return self.B1

def _calculate_Keff(self,f, Bs, Binf, tau, strain):
        if self.K - (self._calculate_Bdyn(f, Bs, Binf, tau) * strain) < 0:
              raise ValueError("Keff cannot be negative, check parameters or strain.")
        if self.strain_type == "z":
            return (self.K - (self._calculate_Bdyn(f, Bs, Binf, tau) * strain))
        elif self.strain_type == "x": 
            return (self.K - (self._calculate_Bdyn(f, Bs, Binf, tau) * strain))
        elif self.strain_type == "xz":
            return (self.K - (self._calculate_Bdyn(f, Bs, Binf, tau) * strain))

def _dyn_freq(self,v):
        if self.exc == "const":
              f = 0
        elif self.exc=="lin":
            f=0
        elif self.exc=="oscill":
            f=self.of
        elif re.search(r"SAW",self.exc):
            f = (np.abs(self.vsaw + v))/self.strain_length()
        elif self.exc == 'dsin':
              f = np.abs(v + self._v_dsin())/self.strain_length()
        return f