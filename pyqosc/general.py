import numpy as np
import qutip as qt
import matplotlib.pyplot as plt
import scipy as sp

options = qt.Options(nsteps = int(1e12))
plt.rcParams.update({"font.size" : 13})

################################################################################################################################################################
################################################################################################################################################################

def steady_state(lindblad, plot_wigner = False, xlim = 6, ylim = 6, overlap_with = None):
    Ham, c_ops = lindblad
    
    rho_ss = qt.steadystate(Ham, c_ops)
    
    if plot_wigner:
        if overlap_with:
            ax = overlap_with
        else:
            fig, ax = plt.subplots(1, figsize = (5, 5))
        x = np.linspace(-xlim, xlim, 31)
        y = np.linspace(-ylim, ylim, 31)
        plot = ax.contourf(x, y, qt.wigner(rho_ss, x, y), 100, cmap = "viridis")
        fig.colorbar(plot, ax = ax)
        ax.set_aspect("equal")
    
    return rho_ss

################################################################################################################################################################
################################################################################################################################################################

def fidelity_ss(lindblad, rho0, timelst, plot = False):
    Ham, c_ops = lindblad
    
    rho_ss = steady_state(lindblad)
    
    rho = qt.mesolve(Ham, rho0, timelst, c_ops).states
    
    fid_lst = []
    xx = True
    for i in range(len(timelst)):
        fid = qt.fidelity(rho[i], rho_ss)
        fid_lst.append(fid)
        if xx and abs(fid - 1.0) < 1e-6:
            xx = False
            ax.axvline(timelst[i], ls = ":", c = 'r', label = f"t = {round(timelst[i], 2)}")
    
    if plot:
        fig, ax = plt.subplots(1, figsize = (4, 4))
        ax.plot(timelst, fid_lst, c = 'b')
        ax.axhline(1.0, c = "k", ls = ":", alpha = 0.5)
        ax.set_xlabel("time")
        ax.set_ylabel("fidelity wrt. steady-state")
        ax.set_xlim(0, timelst[-1])
        ax.set_ylim(0, 1.05)
        ax.legend(loc = "lower right")
        
    return fid_lst

################################################################################################################################################################
################################################################################################################################################################

def _get_cycle(r, phi):
    '''
    Given two arrays of [r] and [phi] for an oscillator in the phase space,
    slice the arrays to get the values over one period of oscillation.
    '''
    mark = False
    j = 0
    for i in range(len(phi) - 1):
        j += 1
        if phi[i+1]<phi[1]:
            mark = True
        if phi[i+1]>phi[0] and mark:
            break
    return r[:j], phi[:j]

################################################################################################################################################################
################################################################################################################################################################

def ss_expval_phasedist(rho_ss, late_r, late_phi, num_bins = 36, plot = False, overlap_with = None):
    '''
    Plot the probability histogram corresponding ot the expectation value of the oscillator
    given by the late-time dynamics (``r_over_cycle``, ``phi_over_cycle``). The probability
    is taken with respect to the steady-state Wigner function. 
    
    If there are mre than one points over a given interval of phi, then the positions are 
    averaged, then the Wigner function is evaluated with respect to that point.
    
    ----------
    Returns
    ----------
    A matplotlib plot.
    
    ----------
    Parameters
    ----------
    
    ``rho_ss``  :
        The steady-state density matrix.
        
    ``late_r``  :
        Late-time values for r. Get from e.g. [vdp_expvalb].
        
    ``late_phi``    :
        Late-time values for phi. Get from e.g. [vdp_expvalb].
        
    ``num_bins``    :   36
        Number of histogram bins.
        
    ``overlap_with``    : ``matplotlib.axes.Axes`` object
        Plot in an existing axis, useful for comparisons.
        
    '''
    
    r_cycle, phi_cycle = _get_cycle(late_r, late_phi)
    
    phi_hist = np.linspace(0, 2*np.pi, num_bins+1)
    phi_bin_midpoints = np.linspace(phi_hist[1]/2, 2*np.pi-phi_hist[1]/2, num_bins)
    
    hist_data = []
    
    ignore_lst = []
    for i in range(num_bins):
        x = y = 0
        get_index = []
        for j in range(len(phi_cycle)):
            if j in ignore_lst:
                continue
            
            if phi_hist[i] < phi_cycle[j] < phi_hist[i+1]:            
                get_index.append(j)
                ignore_lst.append(j)
        
        if get_index:
            for index in get_index:
                x += r_cycle[index] * np.cos(phi_cycle[index])
                y += r_cycle[index] * np.sin(phi_cycle[index])
            x /= len(get_index)
            y /= len(get_index)
        
            hist_data.append(qt.wigner(rho_ss, x, y)[0][0])
        else:
            hist_data.append(0.0)
    
    # normalize
    hist_data = np.array(hist_data)
    hist_data /= np.sum(hist_data)
    
    if plot:
        if overlap_with:
            ax = overlap_with
        else:
            fig, ax = plt.subplots(1, figsize = (10, 5))
        
        ax.bar(phi_bin_midpoints, hist_data, width = phi_bin_midpoints[1]-phi_bin_midpoints[0], ec = "k", color = "b", label = "cl")
        
        ax.set_xticks([0, np.pi, 2*np.pi])
        ax.set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    
    return phi_bin_midpoints, hist_data

################################################################################################################################################################
################################################################################################################################################################

def ss_q_phasedist(rho_ss, num_bins, plot = False, overlap_with = None):
    N = rho_ss.dims[0][0]

    bin_width = 2*np.pi / num_bins
    phi_hist_midpoints = np.linspace(bin_width/2, 2*np.pi - bin_width/2, num_bins)
    
    hist_data = []
    for i in range(num_bins):
        phi = phi_hist_midpoints[i]
        phi_ket = 0
        for n in range(N):
            phi_ket += np.exp(1j * n * phi) * qt.basis(N, n)
        hist_data.append(qt.expect(rho_ss, phi_ket))    # No need to divide by 2pi as the array will be normalized below.
    
    hist_data = np.array(hist_data)
    hist_data /= np.sum(hist_data)
    
    if plot:
        if overlap_with:
            ax = overlap_with
        else:
            fig, ax = plt.subplots(1, figsize = (10, 5))
        
        ax.bar(phi_hist_midpoints, hist_data, width = bin_width, ec = "k", color = "r", label = "qm")
        
        ax.set_xticks([0, np.pi, 2*np.pi])
        ax.set_xticklabels([r"$0$", r"$\pi$", r"$2\pi$"])
    
    return phi_hist_midpoints, hist_data

################################################################################################################################################################
################################################################################################################################################################

def ss_q_spectrum(lindblad, omega = np.linspace(-1, 1, 101), 
               plot = False, overlap_with = None, label = r"qm"):
    Ham, c_ops = lindblad
    N = Ham.dims[0][0]
    b = qt.destroy(N)
    
    spect = qt.spectrum(Ham, omega, c_ops, b, b.dag())
    spect /= np.max(spect)
    
    if plot:
        if overlap_with:
            ax = overlap_with
        else:
            fig, ax = plt.subplots(1, figsize = (5, 4)) 
        
        ax.plot(omega, spect, label = label, ls = "--")
        ax.legend(loc = "best")
        ax.set_ylabel(r"$S(\omega)$")
    
    return omega, spect, omega[spect == np.max(spect)]

################################################################################################################################################################
################################################################################################################################################################

def ss_c_spectrum(timelst_ss, beta_ss, omega_lim = 1.0, plot = False, plot_bar = False, 
                  overlap_with = None, **plot_kwargs):
    
    n = len(timelst_ss)
    nT = timelst_ss[-1] - timelst_ss[0] # assume the beginning of the steady-state time list to be zero.
    T = nT/n
    
    acf = sp.signal.correlate(beta_ss, beta_ss, mode = "same")

    spect = np.abs(sp.fft.fft(acf))
    spect /= np.max(spect)
    
    omega = sp.fft.fftfreq(n, T) * 2 * np.pi
    
    if plot:
        if overlap_with:
            ax = overlap_with
        else:
            fig, ax = plt.subplots(1, figsize = (5, 4))
        
        if plot_bar:
            ax.bar(omega, spect, **plot_kwargs)
        else:
            ax.plot(omega, spect, **plot_kwargs)
            
        ax.legend(loc = "best")
        ax.set_ylabel(r"$S(\omega)$")
        ax.set_xlim(-omega_lim, omega_lim)

    return omega, spect, omega[np.where(spect==np.max(spect))]