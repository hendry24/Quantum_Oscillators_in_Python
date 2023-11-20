import numpy as np
import qutip as qt
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class vdp_params:
    def __init__(self, N, Delta = 16, Omega = 1, gamma_1 = 1, gamma_2 = 0.1):
        self.name = "vdP"
        self.N = N
        self.Delta = Delta
        self.Omega = Omega
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        
    def get_params(self):
        return self.Delta, self.Omega, self.gamma_1, self.gamma_2
        
def vdp_lindblad(vdp_params):
    b = qt.destroy(vdp_params.N)

    Delta = vdp_params.Delta
    Omega = vdp_params.Omega
    gamma_1 = vdp_params.gamma_1
    gamma_2 = vdp_params.gamma_2

    return -Delta * b.dag() * b + 1j * Omega * (b - b.dag()), [np.sqrt(gamma_1) * b.dag(), np.sqrt(gamma_2) * b ** 2]

def vdp_expvalb(vdp_params, t_end = 1e2, t_eval = 10, timepoints_returned = 100, method = "Radau",
                init_polar = [1, 0], plot = False, overlap_with = None,
                color = "k"):
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
    ``vdp_params``  :
        Parameters for vdP oscillators, made with ``vdp_params``.
    
    ``t_end``       : ``100``
        Limit of solution computation.
        
    ``t_eval``      : ``10``
        How far from ``t_end`` the result is returned, i.e. this function returns from ``t_eval`` until ``t_end``.
    
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
        
    '''
    
    t_return = np.linspace(t_end-t_eval, t_end, timepoints_returned)
    
    Delta, Omega, gamma_1, gamma_2 = vdp_params.get_params()
    
    def adler(t, y):
        r, phi = y
        return [(gamma_1/2-gamma_2*r**2)*r-Omega*np.cos(phi), Delta + Omega / r * np.sin(phi)]
        
    sol = solve_ivp(adler, t_span = [0, t_end], y0 = init_polar, dense_output=True, method = method)
    sol_vals = sol.sol(t_return)
    
    r = sol_vals[0] * np.sqrt(2) # to overlap correctly with the wigner function.
    phi = sol_vals[1]
    
    beta = r * np.exp(1j * phi)
    
    for i in range(timepoints_returned):
        if r[i]<0:
            r[i] *= -1
            phi[i] += np.pi
        phi[i] = phi[i]%(2*np.pi)
    
    if plot:
        if overlap_with:
            ax = overlap_with
        else:
            fig, ax = plt.subplots(1, figsize = (5,5))
        
        if abs(r[0]-r[1]) < 1e-6 and abs(phi[0]-phi[1]) < 1e-6:
            marker = "o"
        else:
            marker = None
            
        ax.plot(r * np.cos(phi), r * np.sin(phi), marker = marker, mec = color, mfc = color, ms = 4)
            
    return t_return, r, phi, beta
