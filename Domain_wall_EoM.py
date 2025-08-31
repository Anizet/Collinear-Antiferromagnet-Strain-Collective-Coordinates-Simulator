import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename

import plot_utils as _PLT
import strain as _STA
import dynamic_coupling as _DYN

class DomainWallSimulator:
    def __init__(self):
        # Geometry parameters
        self.L = 2048e-9 * 2
        self.ty = 64e-9
        self.tz = 1e-9
        self.dx = 1e-9 # Cell size #
        self.x0 = self.L / 2  # Initial position of the domain wall
        self.p0 = 0
        self.wnotch = 10e-9  # Width of the notch
        self.dnotch = 13e-9  # Depth of the notch
        self.pin0 = self.L/2
        self.notch = False

        # Model parameters
            # Dispersion #
        self.alph = 0.00 # 0.00 for SAW strain
        self.beta = 0.63 # 0.63 for SAW strain
        self.Fcorr = True  # Force correction flag
        self.deltaForce = True
            # Potential #
        self.aa = -1.6e-4*60  # J/m^2
        self.bb = 6e-7 # J/m^4
        self.potential = False

        # Material parameters
        self.d = 0.42e-9
        self.C11 = 1
        self.C12 = 1
        self.C44 = 1

            # Coupling #
        self.B1 = -3e7/2
        self.B2 = -1.77e7/2
        self.Binf = self.B1/2

            # Magnetic #
        self.Ms = 4.25e5
        self.alpha = 2.1e-4
        self.A = 5e-12
        self.A0 = -5e-12
        self.A12 = -2*self.A
        self.Alambda = -4*self.A0/self.d**2
        self.K = 85.7e3
        self.DMI = (0.7e-3)/2

        # Constants
        self.mu0 = 4 * np.pi * 1e-7
        self.gyr2 = 2.211e5
        self.vmax = 0.5*self.gyr2 * np.sqrt(4*2*self.A*2*(self.Alambda) / (self.mu0 * self.Ms)**2) # Max Spin wave velocity #
        self.Ffact = 2 * self.B1 * self.ty * self.tz # Static Magnetoelastic force parameter #

        # Simulation setup
            # Time #
        self.Time = 1e-9
        self.timestep = 1e-13
        self.dynamic_wall = True  # Dynamic wall flag
        self.lim = True
        self.simple_phi = False
        self.deltastart = None
        self.boundary = False
            # Strain #
        self.strain = 1e-4
        self.strain0 = 0 # defines the slope along with self.strain
        self.strain_type = "z"  # default applied strain direction
        self.exc = "lin" #Default strain excitation type
        self.vsaw = 4e3
        self.lamb = 5e-7
        self.df = 0.00e9 #Hz frequency difference between the two sin waves
        self.gf = 5e9 #Hz pulse frequency
        self.of = 5e9 #Hz oscill frequency
        self.FWHM = 800e-9
        self.fac = 1.0  # Amplitude Asymmetry Factor between counterpropagating waves
        self.phi = 0  # Phase shift for SAW strain
        self.Rayleigh = False  # Rayleigh strain flag, should not be used anymore
    
    def _k(self):
        return 2*np.pi/self.lamb

    def _f0(self):
        return self.vsaw/self.lamb

    def _v_dsin(self):
        return self.vsaw * self.df/self._f0()

    def run(self):
        # Integration parameters
        T, dt = self.Time, self.timestep
        N = int(T / dt)

        # Simulation parameters
        t, v, x, p, ap, w = 0, 0, self.x0, self.p0, 0, 0
        
        e0 = self.strain
        k = self._k()
        strain = self._strain(x,t,k,e0,p)
        strainx = self._strainx(x,t,k,e0,p)

        # Initial conditions
        delta = np.sqrt(2 * self.A / (2 * self.K))
        Meff = self._Effective_mass(delta,v)

        # Recalculate initial values : Dynamics
        Bs, Binf = self.B1, self.Binf
        f = self._dyn_freq(v)
        tau = self._calculate_tau(x, e0, k, delta, Meff,t,p)
        
        K_eff = self._calculate_Keff(f, Bs, Binf, tau, strain)
        delta = np.sqrt(2 * self.A / (2 * K_eff)) * self._lorentz(v)
        Meff = self._Effective_mass(delta,v)
        
        if self.dynamic_wall == True and self.simple_phi == False:
            Tform = self._phi_torque(delta,x,p,t,k,e0)
            ap = Tform / Meff

        if self.simple_phi == True:
            val = np.pi * self.DMI / (2 * delta * self.B1 * self._strainx(x, t, k, e0, p) + 1e-12)
            # Clamp val into valid arccos domain [-1, 1]
            if np.isnan(val) or val < -1 or val > 1:
                if p > np.pi/2:
                    p = np.pi
                else:
                    p = 0
            else:
                p = np.arccos(val)
        
        strain = self._strain(x,t,k,e0,p)
        K_eff = self._calculate_Keff(f, Bs, Binf, tau, strain)
        delta = np.sqrt(2 * self.A / (2 * K_eff)) * self._lorentz(v)
        Meff = self._Effective_mass(delta,v)

        if self.dynamic_wall == True: #and self.simple_phi == False
            Tform = self._phi_torque(delta,x,p,t,k,e0)
            ap = Tform / Meff

        if self.simple_phi == True:
            val = np.pi * self.DMI / (2 * delta * self.B1 * self._strainx(x, t, k, e0, p) + 1e-12)
            # Clamp val into valid arccos domain [-1, 1]
            if np.isnan(val) or val < -1 or val > 1:
                if p > np.pi/2:
                    p = np.pi
                else:
                    p = 0
            else:
                p = np.arccos(val)
        
        strain = self._strain(x,t,k,e0,p)
        K_eff = self._calculate_Keff(f, Bs, Binf, tau, strain)
        delta = np.sqrt(2 * self.A / (2 * K_eff)) * self._lorentz(v)
        #if self.deltastart is not None:
        #    delta = self.delta1()
        Meff = self._Effective_mass(delta,v)

        # Initial Acceleration
        Fform = self._calculate_Fform(x, e0, k, delta, dt, tau,t, p)
        
        Ftot = Fform
        #if self.Fcorr == True: Fform = Fform /self.B1 * self._calculate_Bdyn_f(f, tau)
        #if self.potential == True: Ftot += self._potential_force(self.aa, self.bb, x)[1]
        if self.boundary == True: Ftot += self._calculate_BoundForce(x,delta,K_eff)
        if self.notch == True: Ftot += self._calculate_triang_pinning_force(x,delta,K_eff)
        #if self.strain_type == "xz":
        #    if self.Rayleigh == True:
        #        Fform += self._calculate_Fxz(x, e0, k, delta, dt, tau,t, p)

        a = Ftot / Meff

        # Storage lists
        self.time, self.positions, self.velocities, self.accelerations = [], [], [], []
        self.angle, self.wstore, self.paccstore = [],[],[]
        self.Fformstore, self.Fvstore, self.Tformstore, self.Fwstore = [],[],[],[]
        self.deltastore, self.strainstore, self.Kstore = [], [], []
        self.Bdynstore, self.taustore, self.fstore,  self.vtermstore = [], [], [], []
        self.Fdeltastore, self.Fdeltastore_test, self.Fpinstore = [], [], []

        for i in tqdm(range(N)):
            t = i * dt

            vterm = self.v_term0(e0,x,t,k,f,tau,p,delta = delta)
            Bdyn = self._calculate_Bdyn_f(f, tau)
            self.store_data(f,t,x,v,a,delta,tau,Fform,strain,K_eff,vterm,Bdyn,ap,w,p)

            v += a * dt
            x += v * dt

            if self.dynamic_wall == True :#and self.simple_phi == False:
                w += ap * dt
                p += w * dt

            strain = self._strain(x,t,k,e0,p)
            strainx = self._strainx(x,t,k,e0,p)

            f = self._dyn_freq(v)
            K_eff = self._calculate_Keff(f, Bs, Binf, tau, strain)

            delta = np.sqrt(2 * self.A / (2 * K_eff)) * self._lorentz(v) #/ np.sqrt(1 - f**2/(self.vmax*np.sqrt(2*K_eff/(2*self.A)))**2)

            Fform = self._calculate_Fform(x, e0, k, delta, dt, tau,t,p)
            #if self.Fcorr == True: Fform = Fform /self.B1 * self._calculate_Bdyn_f(f, tau)
            #if self.potential == True: Fform += self._potential_force(self.aa, self.bb, x)[1]
            #if self.boundary == True: Fform += self._calculate_BoundForce(x,delta,K_eff)
            #if self.notch == True: Fform += self._calculate_triang_pinning_force(x,delta,K_eff)
            #if self.strain_type == "xz":
            #    if self.Rayleigh == True:
            #        Fform += self._calculate_Fxz(x, e0, k, delta, dt, tau,t,p)
           
            Fv = self._Fv_force(v, delta)
            self.Fvstore.append(Fv)

            tau = self._calculate_tau(x, e0, k, delta, Meff,t,p)

            Meff = self._Effective_mass(delta,v)

            Fvdelta = self._deriv_delta(delta, Meff, v)
            self.Fdeltastore.append(Fvdelta)
            Ftot = Fform + Fv + Fvdelta
            if self.boundary == True: Ftot += self._calculate_BoundForce(x,delta,K_eff)
            if self.notch == True: Ftot += self._calculate_triang_pinning_force(x,delta,K_eff)

            a = (Ftot) / Meff
            
            if self.dynamic_wall == True :#and self.simple_phi == False:
                Tform = self._phi_torque(delta,x,p,t,k,e0)
                Fw = self._Fv_force(w,delta)
                self.Fwstore.append(Fw)
                ap = (Tform + Fw) / Meff
            
            if self.simple_phi == True:
                val = np.pi * self.DMI / (2 * delta * self.B1 * self._strainx(x, t, k, e0, p) + 1e-12)
                # Clamp val into valid arccos domain [-1, 1]
                if np.isnan(val) or val < -1 or val > 1:
                    if p > np.pi/2:
                        p = np.pi
                    else:
                        p = 0
                else:
                    p = np.arccos(val)
        
            if self.lim == True:
                if np.abs(x) > 2*self.L or np.isnan(x):
                    print("Simulation stopped: domain wall out of bounds or diverged.")
                    break


    def store_data(self,f,t,x,v,a,delta,tau,Fform,strain,K_eff,vterm,Bdyn,ap,w,p):
        self.fstore.append(f)
        self.time.append(t)
        self.positions.append(x)
        self.velocities.append(v)
        self.accelerations.append(a)
        self.deltastore.append(delta)
        self.taustore.append(tau)
        self.Fformstore.append(Fform)
        self.Bdynstore.append(Bdyn)
        self.strainstore.append(strain)
        self.Kstore.append(K_eff)
        self.vtermstore.append(vterm)
        self.paccstore.append(ap)
        self.angle.append(p)
        self.wstore.append(w)
        if self.notch == True : self.Fpinstore.append(self._calculate_triang_pinning_force(x, delta, K_eff))

    def DW_profile(self, x, phase=0):
        delta = self.delta1()
        theta = 2 * np.arctan(np.exp((x) / delta) )#* np.tan((np.pi - phase)/2))
        phi = 2 * np.arctan(np.exp((x) / delta) * np.tan(phase/2) + np.pi/2)
        return [theta, phi] 

    def delta0(self):
        return np.sqrt(2 * self.A / (2 * (self.K)))
    def delta1(self):
        Keff = self._calculate_Keff(0,self.B1,self.B1/2,0,self.strain)
        return np.sqrt(2 * self.A / (2 * (Keff)))

    def _lorentz(self, v):
        return np.sqrt(1 - (v/self.vmax)**2 + 1e-12)

    def _Effective_mass(self, delta,v):
        """
        Calculate the effective mass of the domain wall.
        """
        return self.mu0**2 * self.Ms**2 * self.ty * self.tz * 2/delta / ((2*self.Alambda) * self.gyr2**2)

    def _Fv_force(self, v, delta):
        """
        Calculate the viscous force acting on the domain wall.
        """
        self.GAMMA = -self.alpha * self.mu0 * self.Ms * self.ty * self.tz * 2 / delta / self.gyr2
        return self.GAMMA * v 

    def v_term0(self,e0,x,t,k,f,tau,phi, delta = None):
        if delta == None: delta = self.delta0()
        vterm0 = self.gyr2*self._calculate_Bdyn_f(f,tau)*(np.array(delta))**2/(self.alpha*self.mu0*self.Ms) * self._deriv_strain(e0,x,t,k,phi)
        return (-1 + np.sqrt(1 + 4*vterm0**2/self.vmax**2))/(2*vterm0/self.vmax**2)



    ################### STRAIN ######################

    def _strain(self,x,t,k,e0,phi):
        return _STA._strain(self,x,t,k,e0,phi)

    def _strainx(self,x,t,k,e0,phi):
        return _STA._strainx(self,x,t,k,e0,phi)

    def strain_length(self):
        return self.lamb if self.strain_type=="SAW_sin" else self.FWHM

    def _lin_strain(self,e0,x):
        return _STA._lin_strain(self,e0,x)

    def _dlin_strain(self,e0):
        return _STA._dlin_strain(self,e0)

    def _oscill_strain(self,x,t,e0,w):
        return _STA._oscill_strain(self,x,t,e0,w)

    def _doscill_strain(self,e0,w,t):
        return _STA._doscill_strain(self,e0,w,t)

    def _sin_strain(self,x,t,k,e0):
        return _STA._sin_strain(self,x,t,k,e0)

    def _dsin_strain(self,e0,x,t,k):
        return _STA._dsin_strain(self,e0,x,t,k)
    
    def _deriv_strain(self,e0,x,t,k,phi):
        return _STA._deriv_strain(self,e0,x,t,k,phi)

    def _gaussian_alt_strain(self,x,t):
        return _STA._gaussian_alt_strain(self,x,t)

    def _dual_sin_wave_strain(self,x,t,k,e0):
        return _STA._dual_sin_wave_strain(self,x,t,k,e0)

    def _ddual_sin_wave_strain(self,x,t,k,e0):
        return _STA._ddual_sin_wave_strain(self,x,t,k,e0)


    ############# DYNAMIC COUPLING #######################
    def _calculate_tau(self, x, e0, k, delta, Meff,t,phi):
        return _DYN._calculate_tau(self, x, e0, k, delta, Meff,t,phi)

    def _calculate_Bdyn(self, f, Bs, Binf, tau):
        return _DYN._calculate_Bdyn(self, f, Bs, Binf, tau)

    def _calculate_Bdyn_f(self, f, tau):
        return _DYN._calculate_Bdyn_f(self, f, tau)

    def _calculate_Keff(self,f, Bs, Binf, tau, strain):
        return _DYN._calculate_Keff(self,f, Bs, Binf, tau, strain)

    def _dyn_freq(self,v):
        return _DYN._dyn_freq(self,v)


    ################### FORCES ##########################

    def _calculate_Fform(self, x, e0, k, delta, dt, tau,t,phi):
            return self.Ffact * delta * self._deriv_strain(e0,x,t,k,phi)

    def _calculate_BoundForce(self,x,delta,K_eff):
        Fb = 2*self.DMI/delta / np.cosh(x/delta) - (2*self.A/delta + 2*K_eff*delta)/delta / np.cosh(x/delta)**2
        Fb2 = 2*self.DMI/delta / np.cosh((self.L-x)/delta) - (2*self.A/delta + 2*K_eff*delta)/delta / np.cosh((self.L-x)/delta)**2
        return (Fb - Fb2)*self.ty*self.tz
    
    def _calculate_triang_pinning_force(self,x,delta,K_eff):
        R = self.pin0 - x
        a = self.wnotch
        d = self.dnotch
         # Return zero if x is outside the notch region
        if not (self.pin0 - a <= x <= self.pin0 + a):
            return 0.0
        sig_DW = 4*np.sqrt(2*self.A*K_eff) - np.pi*self.DMI
        def S(g):
            return 1/(np.cosh((g)/delta))**2
        Fnotch = -sig_DW * self.tz * d/(2*delta) * (S(R+a) - S(R-a) + R/a * (S(R+a) -2*S(R) + S(R-a)))
        return Fnotch
    
    #def _calculate_Fxz(self, x, e0, k, delta, dt, tau,t,phi):
    #    """ Calculate the shear xz force component for a RAYLEIGH (3/8 factor) strain
    #    """
    #    if self.Fcorr == True:
    #        B = self._calculate_Bdyn_f(self._dyn_freq(0), tau)
    #    else:
    #        B = self.B1
    #    return self.Ffact * delta * self._deriv_strain(e0,x,t,k,phi) * 3/8 * self.B2/B

    def _potential_force(self,aa,bb,x):
        """
        Calculate the potential force acting on the domain wall.
        For pinning sites of boundary forces for example.
        """
        xp = x - self.x0
        energy = (aa*(xp*1e9)**2 + bb*(xp*1e9)**4) * 1e-18
        energy_force = -(2*aa*(xp*1e9) + 4*bb*(xp*1e9)**3) * 1e-18
        return energy,energy_force

    def _deriv_delta(self, delta, Meff,v):
        if self.deltaForce == False : 
            return 0
        else:
            return (delta - self.deltastore[-1])/self.timestep * Meff/delta * v #- Meff/delta**2 * v**2 * (delta - self.deltastore[-1])/self.timestep

    #def _deriv_delta_test(self,F,delta,Meff,v,a,K_eff,e0,x,t,k,phi):
    #    X1 = Meff * v**2
    #    X2 = self._lorentz(v)*(2*self.A)*self.B1*self._deriv_strain(e0,x,t,k,phi)/(delta*4*K_eff**2 * np.sqrt(2*self.A/(2*K_eff)))
    #    X3 = (self._lorentz(v)**2*self.vmax**2)
    #    X4 = Meff
    #    self.X1 = X1
    #    self.X2= X2
    #    self.X3 = X3
    #    self.X4 = X4
    #    return delta/(self._lorentz(v)**2) * v/self.vmax**2 * a * Meff/delta *v + self.alpha*Meff*v**3*delta/self._lorentz(v)**2/self.vmax**2
    #    return X1 * (X2 + F/(X3*X4))/(1 - X1/(X3*X4)) 

    def _phi_torque(self, delta, x, phi,t,k,e0):
        DMI_torque = 2*np.pi*self.DMI*self.ty*self.tz*np.sin(phi)/(delta**2)
        if self.strain_type == "x" : 
            Strain_torque = -8/2*self.B1*self.ty*self.tz*self._strainx(x,t,k,e0,phi)/delta * np.cos(phi) * np.sin(phi)
            return DMI_torque + Strain_torque
        if self.strain_type == "xz" : 
            Strain_torque = 8/2*self.B1*self.ty*self.tz*self._strainx(x,t,k,e0,phi)/delta * np.cos(phi) * np.sin(phi)
            return DMI_torque + Strain_torque
        else: return DMI_torque

    ################# PLOT UTILS ###########################

    def single_plot(self,color="black", x = "x", label = None):
        return _PLT.single_plot(self, color=color, x=x, label=label)

    def x_v_a_plot(self, data = False, offset = 0.0, vterm = False, sk = 5, oscF = None):
        return _PLT.x_v_a_plot(self, data = data, offset = offset, vterm = vterm, sk = sk, oscF = oscF)

    def x_v_phase_plot(self):
        return _PLT.x_v_phase_plot(self)

    def _1D_anim_SAW(self,data=False, save = False, follow = False):
        return _PLT._1D_anim_SAW(self,data=data, save = save, follow = follow)

    def plot_fft(self, omega, map=None, label = None,color = "k"):
        fs = self.timestep
        return _PLT.plot_fft(self,omega,fs,color,label, map)

    #################  MISC #######################
    def store_pos(self):
        np.save("SAW_model_test.npy",self.positions)

    def compute_fft(self):
        N = int(np.floor(self.Time/self.timestep))
        fft_vals = np.fft.fft(self.positions - np.mean(self.positions))  # remove DC offset
        fft_freq = np.fft.fftfreq(N, d=self.timestep)
        # Only keep positive frequencies
        pos_mask = fft_freq > 0
        return fft_freq[pos_mask], np.abs(fft_vals[pos_mask])

    def fft_plot(self):
        freq,spectrum = self.compute_fft()
        plt.figure(figsize=(8, 4))
        plt.plot(freq, spectrum)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.title('FFT of Domain Wall Position')
        plt.grid(True)
        plt.tight_layout()
        plt.show()