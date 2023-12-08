import numpy as np
import qutip as qt
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class vdp:
    def __init__(self, N, omega_0 = 1, omega = 1, Omega = 1, gamma_1 = 1, gamma_2 = 0.1):
        self.name = "vdP"
        self.N = N
        self.omega_0 = omega_0
        self.omega = omega
        self.Delta = self.omega - self.omega_0
        self.Omega = Omega
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
    
    def dynamics(self, Liouvillian = False, b = None):
        if not(b):
            b = qt.destroy(self.N)
            
        Ham = -self.Delta * b.dag() * b + 1j * self.Omega * (b - b.dag())
        c_ops = [np.sqrt(self.gamma_1) * b.dag(), np.sqrt(self.gamma_2) * b**2]
        
        if Liouvillian:
            return qt.liouvillian(Ham, c_ops)
        else:
            return Ham, c_ops
    
    def couple(self, vdp2, D, Liouvillian = False, antiphase = False):
        b1 = qt.tensor(qt.destroy(self.N), qt.qeye(self.N))
        b2 = qt.tensor(qt.qeye(self.N), qt.destroy(self.N))
        
        Ham1, c_ops1 = self.dynamics(b = b1)
        Ham2, c_ops2 = vdp2.dynamics(b = b2)
        
        mul = -1
        if antiphase:
            mul = 1
        
        Ham = Ham1 + Ham2
        c_ops = c_ops1 + c_ops2 + [np.sqrt(D) * (b1 + mul * b2)]
        
        if Liouvillian:
            return qt.liouvillian(Ham, c_ops)
        else:
            return Ham, c_ops

    def adler(self, t_end = 1e2, t_eval = 10, timepoints_returned = 100, method = "Radau",
                init_polar = [1, 0], plot = False, overlap_with = None, color = "k",
                one_cycle = False):
        '''
        Solve the equation of motion for the expectation value of the annihilation operator
        using the Adler equations (https://doi.org/10.1103/PhysRevLett.112.094102).
        
        ----------
        Returns
        ----------
        A tuple containing the time list, ``r``, ``phi``, and ``beta = r*exp(1j*phi)``, in that order, for the chosen
        evaluation interval.
        
        ----------
        Parameters
        ----------
        
        ``t_end``       : ``100``
            Limit of solution computation.
            
        ``t_eval``      : ``10``
            How far from ``t_end`` the result is returned, i.e. the function returns the result from ``t_end-t_eval`` until ``t_end``.
        
        ``timepoints_returned`` : ``100``
            Number of timepoints returned.

        ``method``  : ``Radau``
            Method for ``scipy.integrate.solve_ivp`` evaluation.
            
        ``init_polar``  : ``[1,0]``
            Initial value for ``r`` and ``phi``, respectively.
        
        ``plot``    : ``False``
            Put in ``x`` and ``y`` obtained from ``r`` and  ``phi`` for plotting.
            
        ``overlap_with``    : None
            Plot in the given axis. If ``None``, then make a new matplotlib figure and axes object.
            
        ``color``   :   ``"k"``
            Color of the curve.
            
        ``one_cycle``   :   ``False``
            Return only one cycle of the motion evaluated from ``t_end-t_eval``.
            
        '''
        
        t_return = np.linspace(t_end-t_eval, t_end, timepoints_returned)
        
        def adler_eq(t, y):
            r, phi = y
            return [(self.gamma_1/2-self.gamma_2*r**2)*r-self.Omega*np.cos(phi), self.Delta + self.Omega / r * np.sin(phi)]
            
        sol = solve_ivp(adler_eq, t_span = [0, t_end], y0 = init_polar, dense_output=True, method = method, rtol = 1e-9)
        sol_vals = sol.sol(t_return)
        
        r = sol_vals[0] * np.sqrt(2) # to overlap correctly with the wigner function.
        phi = sol_vals[1]
        
        for i in range(timepoints_returned):
            if r[i]<0:
                r[i] *= -1
                phi[i] += np.pi
            phi[i] = phi[i]%(2*np.pi)
        
        if one_cycle:
            r, phi = self._get_cycle(r, phi)
        
        beta = r * np.exp(1j * phi)

        if plot:
            if overlap_with:
                ax = overlap_with
            else:
                fig, ax = plt.subplots(1, figsize = (5,5))
            
            if abs(r[0]-r[1]) < 1e-6 and abs(phi[0]-phi[1]) < 1e-6:
                marker = "o"
            else:
                marker = None
                
            ax.plot(r * np.cos(phi), r * np.sin(phi), marker = marker, mec = color, mfc = color, c = color, ms = 4)
                    
        return t_return, r, phi, beta
    
    def _get_cycle(self, r, phi):
        '''
        Given two arrays of [r] and [phi] for an oscillator in the phase space,
        slice the arrays to get the values over one period of oscillation.
        '''
        
        increase = False
        if phi[1]>phi[0]:
            increase = True
        mark = 0
        j = 0
        for i in range(1, len(phi) - 1):
            j += 1
            if (phi[i-1]-phi[i])*(phi[i]-phi[i+1]) < 0:
                mark += 1
            if mark == 2 and increase and phi[i+1]>phi[0]:
                break
            if mark == 2 and not(increase) and phi[i+1]<phi[0]:
                break
        return r[:j+1], phi[:j+1]
