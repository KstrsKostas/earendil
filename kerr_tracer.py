"""
Eärendil 
--------------------------------------------------------------------------
Real-time gravitational lensing visualization around a spinning black hole
using real infrared sky survey data from 2MASS.

Named after the star of high hope in Tolkien's legendarium.

Created by K.Kostaros
"""

import jax
import jax.numpy as jnp
from jax import jit, vmap
import diffrax

jax.config.update("jax_enable_x64", True)

R_CELESTIAL = 5000.0

@jit
def initial_conditions(r0, th0, b, al, m_val, a_val, E=1.0, inward=True):
    t0 = 0.0
    phi0 = 0.0
    
    L = -b * E * jnp.sin(th0)
    vth0 = al / (r0 ** 2)
    
    Sigma = r0**2 + a_val**2 * jnp.cos(th0)**2
    Delta = r0**2 - 2*m_val*r0 + a_val**2
    
    g_tt = -(1 - (2*m_val*r0)/Sigma)
    g_tphi = -((2*m_val*a_val*r0)/Sigma) * jnp.sin(th0)**2
    g_phiphi = (r0**2 + a_val**2 + (2*m_val*a_val**2*r0*jnp.sin(th0)**2)/Sigma) * jnp.sin(th0)**2
    g_rr = Sigma / Delta
    g_thth = Sigma
    
    D = g_tphi**2 - g_tt * g_phiphi
    
    vt0 = (E * g_phiphi + L * g_tphi) / D
    vth0_sq = vth0 ** 2
    
    numerator = (
        (L ** 2 * g_tt + 2 * E * L * g_tphi + E ** 2 * g_phiphi) / D
        - vth0_sq * g_thth
    )
    vr0_sq = numerator / g_rr
    vr0 = jnp.sqrt(jnp.abs(vr0_sq))
    vr0 = jnp.where(inward, -vr0, vr0)
    
    pr0 = g_rr * vr0
    pth0 = g_thth * vth0
    pt0 = -E
    pphi0 = L
    
    return jnp.array([t0, r0, th0, phi0, pr0, pth0, pt0, pphi0])

@jit 
def eom_system(t, y, args):
    t_var, r_val, th_val, phi_val, pr_val, pth_val, pt_val, pphi_val = y
    m_val, a_val = args
    

    Sigma = r_val**2 + a_val**2 * jnp.cos(th_val)**2
    
    # dt/dλ
    dt_dl = -0.5*(4*a_val*jnp.sin(th_val)**2*m_val*pphi_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + 2*jnp.sin(th_val)**2*pt_val*(2*a_val**2*jnp.sin(th_val)**2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + a_val**2 + r_val**2))/(4*a_val**2*jnp.sin(th_val)**4*m_val**2*r_val**2/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 - jnp.sin(th_val)**2*(2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) - 1)*(2*a_val**2*jnp.sin(th_val)**2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + a_val**2 + r_val**2))
    
    # dr/dλ
    dr_dl = 1.0*pr_val*(a_val**2 - 2*m_val*r_val + r_val**2)/(a_val**2*jnp.cos(th_val)**2 + r_val**2)
    
    # dθ/dλ
    dth_dl = 1.0*pth_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2)
    
    # dφ/dλ
    dphi_dl = -0.5*(4*a_val*jnp.sin(th_val)**2*m_val*pt_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + 2*pphi_val*(2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) - 1))/(4*a_val**2*jnp.sin(th_val)**4*m_val**2*r_val**2/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 - jnp.sin(th_val)**2*(2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) - 1)*(2*a_val**2*jnp.sin(th_val)**2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + a_val**2 + r_val**2))
    
    # dpr/dλ
    dpr_dl = 1.0*pr_val**2*r_val*(a_val**2 - 2*m_val*r_val + r_val**2)/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 - 0.5*pr_val**2*(-2*m_val + 2*r_val)/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + 1.0*pth_val**2*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 + 0.5*(-8*a_val*jnp.sin(th_val)**2*m_val*pphi_val*pt_val*r_val**2/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 + 4*a_val*jnp.sin(th_val)**2*m_val*pphi_val*pt_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + jnp.sin(th_val)**2*pt_val**2*(-4*a_val**2*jnp.sin(th_val)**2*m_val*r_val**2/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 + 2*a_val**2*jnp.sin(th_val)**2*m_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + 2*r_val) + pphi_val**2*(-4*m_val*r_val**2/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 + 2*m_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2)))/(4*a_val**2*jnp.sin(th_val)**4*m_val**2*r_val**2/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 - jnp.sin(th_val)**2*(2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) - 1)*(2*a_val**2*jnp.sin(th_val)**2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + a_val**2 + r_val**2)) + 0.5*(4*a_val*jnp.sin(th_val)**2*m_val*pphi_val*pt_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + jnp.sin(th_val)**2*pt_val**2*(2*a_val**2*jnp.sin(th_val)**2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + a_val**2 + r_val**2) + pphi_val**2*(2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) - 1))*(16*a_val**2*jnp.sin(th_val)**4*m_val**2*r_val**3/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**3 - 8*a_val**2*jnp.sin(th_val)**4*m_val**2*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 + jnp.sin(th_val)**2*(2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) - 1)*(-4*a_val**2*jnp.sin(th_val)**2*m_val*r_val**2/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 + 2*a_val**2*jnp.sin(th_val)**2*m_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + 2*r_val) + jnp.sin(th_val)**2*(-4*m_val*r_val**2/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 + 2*m_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2))*(2*a_val**2*jnp.sin(th_val)**2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + a_val**2 + r_val**2))/(4*a_val**2*jnp.sin(th_val)**4*m_val**2*r_val**2/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 - jnp.sin(th_val)**2*(2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) - 1)*(2*a_val**2*jnp.sin(th_val)**2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + a_val**2 + r_val**2))**2
    
    # dpth/dλ
    dpth_dl = -1.0*a_val**2*jnp.cos(th_val)*jnp.sin(th_val)*pr_val**2*(a_val**2 - 2*m_val*r_val + r_val**2)/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 - 1.0*a_val**2*jnp.cos(th_val)*jnp.sin(th_val)*pth_val**2/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 + 0.5*(8*a_val**3*jnp.cos(th_val)*jnp.sin(th_val)**3*m_val*pphi_val*pt_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 + 4*a_val**2*jnp.cos(th_val)*jnp.sin(th_val)*m_val*pphi_val**2*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 + 8*a_val*jnp.cos(th_val)*jnp.sin(th_val)*m_val*pphi_val*pt_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + 2*jnp.cos(th_val)*jnp.sin(th_val)*pt_val**2*(2*a_val**2*jnp.sin(th_val)**2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + a_val**2 + r_val**2) + jnp.sin(th_val)**2*pt_val**2*(4*a_val**4*jnp.cos(th_val)*jnp.sin(th_val)**3*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 + 4*a_val**2*jnp.cos(th_val)*jnp.sin(th_val)*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2)))/(4*a_val**2*jnp.sin(th_val)**4*m_val**2*r_val**2/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 - jnp.sin(th_val)**2*(2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) - 1)*(2*a_val**2*jnp.sin(th_val)**2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + a_val**2 + r_val**2)) + 0.5*(4*a_val*jnp.sin(th_val)**2*m_val*pphi_val*pt_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + jnp.sin(th_val)**2*pt_val**2*(2*a_val**2*jnp.sin(th_val)**2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + a_val**2 + r_val**2) + pphi_val**2*(2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) - 1))*(-16*a_val**4*jnp.cos(th_val)*jnp.sin(th_val)**5*m_val**2*r_val**2/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**3 - 16*a_val**2*jnp.cos(th_val)*jnp.sin(th_val)**3*m_val**2*r_val**2/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 + 4*a_val**2*jnp.cos(th_val)*jnp.sin(th_val)**3*m_val*r_val*(2*a_val**2*jnp.sin(th_val)**2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + a_val**2 + r_val**2)/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 + 2*jnp.cos(th_val)*jnp.sin(th_val)*(2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) - 1)*(2*a_val**2*jnp.sin(th_val)**2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + a_val**2 + r_val**2) + jnp.sin(th_val)**2*(2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) - 1)*(4*a_val**4*jnp.cos(th_val)*jnp.sin(th_val)**3*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 + 4*a_val**2*jnp.cos(th_val)*jnp.sin(th_val)*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2)))/(4*a_val**2*jnp.sin(th_val)**4*m_val**2*r_val**2/(a_val**2*jnp.cos(th_val)**2 + r_val**2)**2 - jnp.sin(th_val)**2*(2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) - 1)*(2*a_val**2*jnp.sin(th_val)**2*m_val*r_val/(a_val**2*jnp.cos(th_val)**2 + r_val**2) + a_val**2 + r_val**2))**2
    
    # conserved quantities
    dpt_dl = 0.0
    dpphi_dl = 0.0
    
    return jnp.array([dt_dl, dr_dl, dth_dl, dphi_dl, dpr_dl, dpth_dl, dpt_dl, dpphi_dl])

def event_horizon(t, y, args, **kwargs):
    r_val = y[1]
    m_val, a_val = args
    r_horizon = m_val + jnp.sqrt(m_val**2 - a_val**2)
    return r_val - r_horizon * 1.1

def solve_single_ray(r0, th0, b, al, m_val, a_val, t0, t1, dt0):
    y0 = initial_conditions(r0, th0, b, al, m_val, a_val, E=1.0, inward=True)
    args = (m_val, a_val)
    
    ode_term = diffrax.ODETerm(eom_system)
    solver = diffrax.Tsit5()
    controller = diffrax.PIDController(rtol=1e-7, atol=1e-8, dtmin=0.01, force_dtmin=True)
    event = diffrax.Event(cond_fn=event_horizon)
    
    solution = diffrax.diffeqsolve(
        terms=ode_term,
        solver=solver,
        t0=t0,
        t1=t1,
        dt0=dt0,
        y0=y0,
        args=args,
        max_steps=100000,
        event=event,
        stepsize_controller=controller,
        saveat=diffrax.SaveAt(t1=True)
    )
    
    return solution.ys[-1]

def trace_rays_batch(b_array, al_array, r0, th0, m_val, a_val, t0, t1, dt0):
    @jit
    def _trace_batch(b_arr, al_arr):
        solve_fn = vmap(lambda b, al: solve_single_ray(r0, th0, b, al, m_val, a_val, t0, t1, dt0))
        return solve_fn(b_arr, al_arr)
    
    return _trace_batch(b_array, al_array)

@jit
def get_celestial_coords(final_states, m_val, a_val, r_celestial=R_CELESTIAL):
    r_vals = final_states[:, 1]
    theta_vals = final_states[:, 2]
    phi_vals = final_states[:, 3]
    pr_vals = final_states[:, 4]
    pth_vals = final_states[:, 5]
    r_horizon = m_val + jnp.sqrt(m_val**2 - a_val**2)
    r_horizon_buffer = r_horizon * 1.5
    theta_mod = jnp.mod(theta_vals, 2 * jnp.pi)
    theta_normalized = jnp.where(theta_mod > jnp.pi, 2 * jnp.pi - theta_mod, theta_mod)
    n_crossings = jnp.floor(theta_vals / jnp.pi)
    phi_adjusted = jnp.where(n_crossings % 2 == 1, phi_vals + jnp.pi, phi_vals)
    fell_in = r_vals < r_horizon_buffer
    didnt_escape = r_vals < r_celestial * 0.95
    any_nan = jnp.isnan(r_vals) | jnp.isnan(theta_vals) | jnp.isnan(phi_vals)
    any_inf = jnp.isinf(r_vals) | jnp.isinf(theta_vals) | jnp.isinf(phi_vals)
    r_bad = (r_vals < 0) | (r_vals > 1e10)
    momentum_bad = (jnp.abs(pr_vals) > 1e10) | (jnp.abs(pth_vals) > 1e10)
    too_close = r_vals < 3 * m_val
    
    is_bad = fell_in | didnt_escape | any_nan | any_inf | r_bad | momentum_bad | too_close
    
    thetas = jnp.where(is_bad, jnp.nan, theta_normalized)
    phis = jnp.mod(jnp.where(is_bad, jnp.nan, phi_adjusted), 2 * jnp.pi)
    
    return thetas, phis, is_bad
