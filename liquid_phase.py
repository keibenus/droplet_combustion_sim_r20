# =========================================
#        liquid_phase.py (r4)
# =========================================
# liquid_phase.py
"""Functions for liquid phase calculations (FVM)."""

import numpy as np
import config
from properties import LiquidProperties
from grid import face_areas
from numerics import harmonic_mean

def build_diffusion_coefficients_liquid(
    r_l_centers: np.ndarray,
    r_l_nodes: np.ndarray,
    volumes_l: np.ndarray,
    liquid_props: LiquidProperties,
    Nl: int,
    dt: float,
    T_l_for_props: np.ndarray # Temperature array (e.g., T_l_n) for evaluating properties
    ):
    """
    Builds the tridiagonal matrix coefficients (a, b, c) and diagonal term (diag_coeff)
    for implicit heat diffusion in the liquid phase (Backward Euler).
    Assumes FVM discretization.
    Boundary conditions are NOT applied here.
    Requires temperatures (T_l_for_props) to evaluate properties.
    """
    if Nl <= 0: return None, None, None, None

    # --- Get Properties based on T_l_for_props ---
    rho_l = np.zeros(Nl); cp_l = np.zeros(Nl); lambda_l = np.zeros(Nl)
    valid_props = True
    for j in range(Nl):
        T_val = T_l_for_props[j].item() if isinstance(T_l_for_props[j], np.ndarray) else T_l_for_props[j]
        props = liquid_props.get_properties(T_val) # get_properties handles clipping
        rho_l[j] = props.get('density', np.nan)
        cp_l[j] = props.get('specific_heat', np.nan)
        lambda_l[j] = props.get('thermal_conductivity', np.nan)
        if np.isnan(rho_l[j]) or rho_l[j] <= 0 or np.isnan(cp_l[j]) or cp_l[j] <= 0 or np.isnan(lambda_l[j]) or lambda_l[j] < 0:
             print(f"Error: Invalid liquid property at cell {j}, T={T_val:.1f} (rho={rho_l[j]}, cp={cp_l[j]}, lam={lambda_l[j]})")
             valid_props=False; break
    if not valid_props: print("Error: Invalid props in build_diffusion_coefficients_liquid"); return None, None, None, None

    A_l_faces = face_areas(r_l_nodes) # Areas of faces 0 to Nl

    # --- Calculate Matrix Coefficients (a, b, c) ---
    a = np.zeros(Nl); b = np.zeros(Nl); c = np.zeros(Nl)

    # Diagonal term from time derivative: (rho*cp*V)/dt
    diag_coeff = rho_l * cp_l * volumes_l / dt
    # Add safety check for diag_coeff
    if np.any(np.isnan(diag_coeff)) or np.any(diag_coeff <= 0):
        print(f"Error: Invalid diag_coeff in liquid phase: {diag_coeff}")
        return None, None, None, None

    # --- Diffusion terms ---
    # Face j=0 (center): Symmetry -> Zero flux -> No contribution to a[0], c[0], b[0] diffusion part
    # Faces j=1 to Nl-1 (internal faces)
    if Nl >= 2:
        lambda_f_internal = harmonic_mean(lambda_l[:-1], lambda_l[1:]) # lambda at faces 1 to Nl-1
        dr_c_internal = r_l_centers[1:] - r_l_centers[:-1] # distance between centers j and j-1
        dr_c_internal = np.maximum(dr_c_internal, 1e-15)
        # Diffusion coeff = lambda * A / dr
        D_internal = lambda_f_internal * A_l_faces[1:Nl] / dr_c_internal
        D_internal = np.nan_to_num(D_internal)

        a[1:] = -D_internal # Coefficient for T_{j-1} in equation for T_j
        c[:-1] = -D_internal # Coefficient for T_{j+1} in equation for T_j
        b[1:] += D_internal  # Contribution to b_j from face j
        b[:-1] += D_internal # Contribution to b_j from face j+1

    # Face j=Nl (surface): Boundary condition handled in main loop RHS (d vector)
    # Need diffusion coefficient for face Nl to correctly add BC flux later?
    # Yes, need D_surf = lambda_face_Nl * A_face_Nl / dr_centers_Nl
    #if Nl >= 1:
    #    lambda_face_Nl = lambda_l[Nl-1] # Approx lambda at face Nl with lambda in cell Nl-1
    #    dr_centers_Nl = r_l_nodes[Nl] - r_l_centers[Nl-1] # Distance from last cell center to surface
    #    dr_centers_Nl = max(dr_centers_Nl, 1e-15)
    #    D_surf = lambda_face_Nl * A_l_faces[Nl] / dr_centers_Nl
    #    D_surf = np.nan_to_num(D_surf)
    #    b[Nl-1] += D_surf # Contribution to b_{Nl-1} from face Nl (surface)

    # Final assembly of b
    b[:] += diag_coeff[:]

    # --- Check for NaN ---
    if np.any(np.isnan(a)) or np.any(np.isnan(b)) or np.any(np.isnan(c)):
        print("ERROR: NaN detected in liquid diffusion coefficients (a,b,c).")
        return None, None, None, None

    return a, b, c, diag_coeff

def calculate_dR_dt(mdot_double_prime, rho_l_s):
     """Calculates dR/dt based on surface mass flux (mdot'' > 0 for evaporation)."""
     if rho_l_s > 1e-6:
         # mdot'' = rho_l * (-dR/dt) => dR/dt = - mdot'' / rho_l
         dRdt = -mdot_double_prime / rho_l_s
         # dRdt should be negative or zero for evaporation/no evaporation
         return min(dRdt, 0.0)
     else:
         print("Warning: Zero or invalid liquid density at surface in calculate_dR_dt.")
         return 0.0