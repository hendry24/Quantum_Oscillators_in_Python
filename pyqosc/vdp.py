import numpy as np
import qutip as qt
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class vdp:
    def __init__(self, N, omega_0 = 1, omega = 1, Omega_1 = 0, Omega_2 = 1, gamma_1 = 1, gamma_2 = 0.1):
        self.name = "vdP"
        self.N = N
        self.omega_0 = omega_0
        self.omega = omega
        self.Delta = self.omega - self.omega_0
        self.Omega_1 = Omega_1
        self.Omega_2 = Omega_2
        self.gamma_1 = gamma_1
        self.gamma_2 = gamma_2
        
        self.Ham = None
        self.c_ops = None
    
    def dynamics(self, Liouvillian = False, b = None):
        
        if not(b):
            b = qt.destroy(self.N)
            
        cond_A = isinstance(self.Omega_1, np.ndarray)
        cond_B = isinstance(self.Omega_2, np.ndarray)
        if cond_A or cond_B:
            self.Ham = [-self.Delta * b.dag() * b]
            
            if not(cond_A):
                self.Ham[0] += (b+b.dag()) * self.Omega_1
            else:
                self.Ham.append([(b+b.dag()), self.Omega_1])
                
            if not(cond_B):
                self.Ham[0] += 1j * (b - b.dag()) * self.Omega_2
            else:
                self.Ham.append([1j*(b-b.dag()), self.Omega_2])
        else:
            self.Ham = -self.Delta * b.dag() * b + (b+b.dag()) * self.Omega_1 + 1j * (b - b.dag()) * self.Omega_2
            
        self.c_ops = [np.sqrt(self.gamma_1) * b.dag(), np.sqrt(self.gamma_2) * b**2]
    
        if Liouvillian:
            return qt.liouvillian(self.Ham, self.c_ops)
        else:
            return self.Ham, self.c_ops
            
    def couple(self, vdp2, D, Liouvillian = False, antiphase = False):
        b1 = qt.tensor(qt.destroy(self.N), qt.qeye(self.N))
        b2 = qt.tensor(qt.qeye(self.N), qt.destroy(self.N))
        
        Ham1, c_ops1 = self.dynamics(b = b1)
        Ham2, c_ops2 = vdp2.dynamics(b = b2)
        
        mul = -1
        if antiphase:
            mul = 1
        
        self.Ham = Ham1 + Ham2
        self.c_ops = c_ops1 + c_ops2 + [np.sqrt(D) * (b1 + mul * b2)]
        
        if Liouvillian:
            return qt.liouvillian(self.Ham, self.c_ops)
        else:
            return self.Ham, self.c_ops

    def evolve(self, rho_0, timelst, expect = []):
        
        if not(self.Ham):
            self.dynamics()
        
        res = qt.mesolve(self.Ham, rho_0, timelst, self.c_ops, expect, options = qt.Options(nsteps = int(1e8)))
        
        if expect:
            out = res.expect
        else:
            out = res.states
        
        return out

    def adler(self, t_end = 1e2, t_eval = 10, timepoints_returned = 100, method = "Radau",
                init_polar = [1, 0], plot = False, overlap_with = None, color = "k",
                one_cycle = False):
        '''
        Solve the equation of motion for the expectation value of the annihilation operator
        using the Adler equations (https://doi.org/10.1103/PhysRevLett.112.094102). 
        
        This method does NOT work with time-dependent ``Omega_i``. If ``Omega_i`` is time 
        dependent, then only the first entry is taken as the value for evaluation. 
        
        ----------
        Returns
        ----------
        A tuple containing the time list, ``r``, ``phi``, and ``beta = r*exp(1j*phi)``, in that order, for the chosen
        evaluation interval.
        
        ----------
        Parameters
        ----------
        
        ``t_end``       : ``100``
            End of solution computation. 
            
        ``t_eval``      : ``10``
            How far from ``t_end`` the result is returned, i.e. the function returns the result from ``t_end-t_eval`` until ``t_end``.
        
        ``timepoints_returned`` : ``100``
            Number of timepoints returned.

        ``method``  : ``Radau``
            Method for ``scipy.integrate.solve_ivp`` evaluation.
            
        ``init_polar``  : ``[1,0]``
            Initial value for ``r`` and ``phi``, respectively. ``r=0`` is an invalid initial condition.
        
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
            return [(self.gamma_1/2-self.gamma_2*r**2)*r-self.Omega_1[0]*np.sin(phi)-self.Omega_2[0]*np.cos(phi), 
                    self.Delta - self.Omega_1[0]/r*np.cos(phi) + self.Omega_2[0]/r*np.sin(phi)]             
                
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

    def shortcut_impens(self, rho_0, tau, timepoints = 101, err_tol_b = 1e-3, err_tol_bbb = 1, dy_lst = None, maxiter = 100, report = True):
        
        t_firsthalf = np.linspace(0, tau/2, timepoints//2+1)
        t_secondhalf = np.linspace(tau/2, tau, timepoints//2+1)
        timelst = np.concatenate((t_firsthalf, t_secondhalf[1:]))
        
        Ham, c_ops = self.dynamics()
        
        b = qt.destroy(self.N)
        bbx = b.dag()*b*(b+b.dag())/2
        bby = b.dag()*b*(b-b.dag())/(2*1j)
        
        b_0 = qt.expect(b, rho_0)
        
        rho_ss = qt.steadystate(Ham, c_ops)
        b_ss = qt.expect(b, rho_ss)
        bbx_ss = qt.expect(bbx, rho_ss)
        bby_ss = qt.expect(bby, rho_ss)
            
        if not(dy_lst.all()):
            dy_lst = [0, -0.5, 0.5]
        
        Delta_b_OK = False
        Delta_bbb_OK = False
        b_target = b_ss
        i = 0
        iters = 0
        while True:
            iters += 1
            b_i = 0.5*(b_target+b_0) + 1j*dy_lst[i]
            
            db_firsthalf = 2/tau*(b_i-b_0)
            b_firsthalf = b_0 + db_firsthalf * t_firsthalf
            
            db_secondhalf = 2/tau*(b_target-b_i)
            b_secondhalf = b_i + db_secondhalf * t_secondhalf
            
            Omega_firsthalf = -db_firsthalf + 1j*self.Delta*b_firsthalf + self.gamma_1/2*b_firsthalf - self.gamma_2*np.abs(b_firsthalf)**2*b_firsthalf
            Omega_secondhalf = -db_secondhalf + 1j*self.Delta*b_secondhalf + self.gamma_1/2*b_secondhalf - self.gamma_2*np.abs(b_secondhalf)**2*b_secondhalf
            
            Omega = np.concatenate((Omega_firsthalf, Omega_secondhalf[1:]))
            self.Omega_1 = np.imag(Omega)
            self.Omega_2 = np.real(Omega)
            
            rho_tau = self.evolve(rho_0, timelst)[-1]
            bbx_tau = qt.expect(bbx, rho_tau)
            bby_tau = qt.expect(bby, rho_tau)
            b_tau = qt.expect(b, rho_tau)
            
            Delta_b = np.abs(b_tau - b_ss)
            Delta_bbb = np.sqrt((bbx_tau-bbx_ss)**2+(bby_tau-bby_ss)**2)
            
            if Delta_b < err_tol_b:
                Delta_b_OK = True
            else:
                offset_b = b_tau - b_target
                b_target -= offset_b
            
            if Delta_bbb < err_tol_bbb:
                Delta_bbb_OK = True
            elif Delta_b_OK:    # TODO: check if this condition is necessary.
                while i < len(dy_lst)-1:
                    i += 1
                
            if (Delta_b_OK and Delta_bbb_OK) or (iters == maxiter):
                break
        
        if report:
            s = "== Shortcut finished == \n\n"
            if iters < maxiter:
                s += f"Result obtained with {iters} iterations. \n\n"
            else:
                s += f"Maximum iterations reached. Not all dy may have been checked. \n\n"
            s += "Specifications: \n"
            s += "   Method: Impens2023 \n"
            s += f"   tau = {tau} \n"
            s += f"   Delta_y = {dy_lst[i]} \n\n"
            s += f"Calculated metrics: \n"
            s += f"   Delta <b> = {Delta_b} \n"
            s += f"   Delta <b*bb> = {Delta_bbb} \n\n"
            s += f"Omega_1 and Omega_2 is saved into the [vdp] class. Check with [vdp.Omega_1] and [vdp.Omega_2]"
            print(s)
        
        return