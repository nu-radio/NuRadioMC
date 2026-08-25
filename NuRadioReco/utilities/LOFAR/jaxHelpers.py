"""
JAX utility functions for the LOFAR IFT reconstruction: coordinate transforms,
B-spline LDF evaluation, and static-data path helpers.

.. moduleauthor:: Karen Terveer <karen.terveer@fau.de>
"""
import json
import os
import pickle
from functools import partial
from pathlib import Path
import numpy as np

try:
    import jax
    import jax.numpy as jnp
    import jax.numpy.linalg as jla
    import jax.scipy.special
    import nifty.re as jft
    from jax import lax, vmap
    from jax.typing import ArrayLike
except ImportError:
    jax = None
    jnp = None
    jla = None
    jft = None
    lax = None
    vmap = None
    ArrayLike = None

from NuRadioReco.modules.LOFAR.utilities import atmosphere as atm

_PACKAGE_IFT_DATA_DIR = Path(__file__).resolve().parents[3] / "utilities" / "data" / "LOFAR" / "ift"


def get_ift_data_path(filename, required=True, data_directory=None):
    """Return an IFT static-data path from an override or package-data location."""
    search_dirs = []
    if data_directory is not None:
        search_dirs.append(Path(data_directory))
    search_dirs.append(_PACKAGE_IFT_DATA_DIR)

    for directory in search_dirs:
        candidate = directory / filename
        if candidate.exists():
            return str(candidate)

    if required:
        raise FileNotFoundError(
            f"LOFAR IFT data file '{filename}' was not found in "
            f"{', '.join(str(path) for path in search_dirs)}"
        )
    return None

def hp_get_angle(v1: jax.Array, v2: jax.Array) -> jax.Array:
    """Return the angle (radians) between two vectors or batches of vectors."""
    # Normalize vectors to avoid numerical issues with dot product
    norm_v1 = jnp.linalg.norm(v1, axis=-1, keepdims=True)
    norm_v2 = jnp.linalg.norm(v2, axis=-1, keepdims=True)

    # Avoid division by zero for zero vectors
    safe_norm_v1 = jnp.where(norm_v1 == 0.0, 1.0, norm_v1)
    safe_norm_v2 = jnp.where(norm_v2 == 0.0, 1.0, norm_v2)
    unit_v1 = v1 / safe_norm_v1
    unit_v2 = v2 / safe_norm_v2

    # Calculate dot product
    dot_product = jnp.sum(unit_v1 * unit_v2, axis=-1)
    clipped_dot_product = jnp.clip(dot_product, -1.0 + 1e-7, 1.0 - 1e-7)
    angle = jnp.arccos(clipped_dot_product)
    is_zero_norm = jnp.logical_or(norm_v1.squeeze(-1) == 0.0, norm_v2.squeeze(-1) == 0.0)
    return jnp.where(is_zero_norm, 0.0, angle)


def hp_get_angle_to_magnetic_field_vector(
        zenith: jax.Array, azimuth: jax.Array,
        site: str = 'lofar', magnetic_field_vector: jax.Array | None = None) -> jax.Array:
    """Return the angle between the shower axis and the magnetic field vector."""
    if magnetic_field_vector is None:
        magnetic_field = hp_get_magnetic_field_vector(site=site)
    else:
        magnetic_field = magnetic_field_vector
    v = hp_spherical_to_cartesian(zenith, azimuth)
    return hp_get_angle(magnetic_field, v)


def hp_spherical_to_cartesian(zenith: jax.Array, azimuth: jax.Array) -> jax.Array:
    """Convert (zenith, azimuth) to a Cartesian unit vector of shape (..., 3)."""
    sinZenith = jnp.sin(zenith)
    x = sinZenith * jnp.cos(azimuth)
    y = sinZenith * jnp.sin(azimuth)
    z = jnp.cos(zenith)
    return jnp.stack([x, y, z], axis=-1)


def hp_cartesian_to_spherical(cartesian_coords: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Convert a Cartesian array (..., 3) to (zenith, azimuth) in radians."""
    x = cartesian_coords[..., 0]
    y = cartesian_coords[..., 1]
    z = cartesian_coords[..., 2]

    r = jnp.linalg.norm(cartesian_coords, axis=-1)
    safe_z_div_r = jnp.clip(z / jnp.where(r == 0, 1.0, r), -1.0 + 1e-7, 1.0 - 1e-7)
    zenith = jnp.arccos(safe_z_div_r)
    zenith = jnp.where(r == 0, 0.0, zenith)
    azimuth = jnp.arctan2(y, x)
    return zenith, azimuth


def hp_get_magnetic_field_vector(site: str = 'lofar') -> jax.Array:
    """Return the geomagnetic field vector in Gauss for the given site.

    The *site* argument must be a static Python string (not a JAX tracer).
    """
    magnetic_fields = {
        'auger': jnp.array([0.00871198, 0.19693423, 0.1413841]),
        'mooresbay': jnp.array([0.058457, -0.09042, 0.61439]),
        'summit': jnp.array([-.037467, 0.075575, -0.539887]),
        'southpole': jnp.array([-0.14390398, 0.08590658, 0.52081228]),
        'lofar': jnp.array([0.004675, 0.186270, -0.456412]),
    }
    return magnetic_fields[site.lower()]


def hp_get_declination(magnetic_field_vector: jax.Array) -> jax.Array:
    """Return the magnetic declination angle (radians) from the field vector."""
    # x points East, y points North
    b_horizontal = magnetic_field_vector[..., :2]
    norm_horizontal = jnp.linalg.norm(b_horizontal, axis=-1)
    safe_norm_horizontal = jnp.where(norm_horizontal == 0.0, 1.0, norm_horizontal)
    unit_b_horizontal = b_horizontal / safe_norm_horizontal
    cos_declination = unit_b_horizontal[..., 1]
    cos_declination_clipped = jnp.clip(cos_declination, -1.0 + 1e-7, 1.0 - 1e-7)
    declination = jnp.arccos(cos_declination_clipped)

    return jnp.where(norm_horizontal == 0.0, 0.0, declination)


class CStrafoJAX:
    """ JAX-compatible class for coordinate transformations in air shower radio detection.
    JAX-compatible radiotools version

    Handles JAX tracers for zenith and azimuth.
    Assumes input vectors/positions are jax.numpy arrays.
    """

    def __init__(self, zenith, azimuth, magnetic_field_vector=None, site=None):
        """ Initialization with JAX-compatible parameters.

        Parameters
        -
        zenith : JAX scalar or float
            Zenith angle (0=zenith).
        azimuth : JAX scalar or float
            Azimuth angle (0=North, 90=East - careful, original comment said 90=South).
        magnetic_field_vector : JAX array (3,) or None
            Magnetic field vector [Bx(East), By(North), Bz(Up)]. Uses default if None.
        site : str or None
            Site name to get default magnetic field (if magnetic_field_vector is None).
        """
        # Ensure inputs are JAX arrays if they might be tracers
        zenith = jnp.asarray(zenith)
        azimuth = jnp.asarray(azimuth)

        # Shower axis points *from* the direction (zenith, azimuth)
        # Original used hp convention, ensure hp_spherical_to_cartesian matches.
        showeraxis = -1.0 * hp_spherical_to_cartesian(zenith, azimuth)

        if magnetic_field_vector is None:
            magnetic_field_vector = hp_get_magnetic_field_vector(site=site)
        else:
            magnetic_field_vector = jnp.asarray(magnetic_field_vector)

        # Normalize B field
        magnetic_field_norm = jla.norm(magnetic_field_vector)
        # Add epsilon to prevent division by zero if B is zero (though unlikely)
        magnetic_field_normalized = magnetic_field_vector / (magnetic_field_norm)

        # Calculate vxB and vx(vxB) basis vectors (Lorentz force and acceleration)
        # Ensure showeraxis and B are treated as 3-vectors
        showeraxis = showeraxis.reshape(3)
        magnetic_field_normalized = magnetic_field_normalized.reshape(3)

        vxB = jnp.cross(showeraxis, magnetic_field_normalized)
        vxB_norm = jla.norm(vxB)
        e1 = vxB / (vxB_norm + 1e-12)  # Normalize vxB (e1)

        vxvxB = jnp.cross(showeraxis, e1) # Use normalized e1 here
        vxvxB_norm = jla.norm(vxvxB)
        e2 = vxvxB / (vxvxB_norm + 1e-12) # Normalize vx(vxB) (e2)

        # e3 should be parallel to shower axis
        # e3 = jnp.cross(e1, e2) # Or simply e3 = showeraxis
        e3 = showeraxis # Shower axis direction

        # Store transformation matrix (rows are the new basis vectors)
        # Ground -> v x B system
        self.transformation_matrix_vBvvB = jnp.array([e1, e2, e3])
        # v x B -> Ground (inverse)

        self.inverse_transformation_matrix_vBvvB = self.transformation_matrix_vBvvB.T


        # Transformation matrix to on-sky coordinate system (eR, eTheta, ePhi)
        # eR: Radial direction (points *from* the source) = -showeraxis
        # eTheta: Increasing zenith angle
        # ePhi: Increasing azimuth angle
        ct = jnp.cos(zenith)
        st = jnp.sin(zenith)
        cp = jnp.cos(azimuth)
        sp = jnp.sin(azimuth)

        e1_onsky = jnp.array([st * cp, st * sp, ct])
        e2_onsky = jnp.array([ct * cp, ct * sp, -st])
        e3_onsky = jnp.array([-sp, cp, jnp.zeros_like(sp)])


        self.transformation_matrix_onsky = jnp.stack([e1_onsky, e2_onsky, e3_onsky], axis=-2)
        self.inverse_transformation_matrix_onsky = self.transformation_matrix_onsky.T

        declination = hp_get_declination(magnetic_field_vector)
        
        c = jnp.cos(declination)
        s = jnp.sin(declination)
        self.transformation_matrix_magnetic = jnp.array([ # Mag -> Geo
            [c, -s, 0],
            [s,  c, 0],
            [0,  0, 1]
        ])
        # Geo -> Mag is the inverse (or transpose)
        self.inverse_transformation_matrix_magnetic = self.transformation_matrix_magnetic.T
        angle = -azimuth
        c = jnp.cos(angle)
        s = jnp.sin(angle)
        # This matrix transforms Geo -> Azimuth CS
        self.transformation_matrix_azimuth = jnp.array([
            [c, -s, jnp.zeros_like(s)],
            [s,  c, jnp.zeros_like(s)],
            [jnp.zeros_like(s), jnp.zeros_like(s), jnp.ones_like(s)]
        ])
        # Azimuth CS -> Geo
        self.inverse_transformation_matrix_azimuth = self.transformation_matrix_azimuth.T


        # Transformation matrix from ground (geographic) CS to shower plane (early-late) CS
        # z-axis is parallel to shower axis, shower axis projection on ground is in yz-plane
        # Step 1: Rotate around Z by (-azimuth + pi/2) -> aligns Y axis with ground projection
        angle1 = -azimuth + jnp.pi / 2.0
        c1 = jnp.cos(angle1)
        s1 = jnp.sin(angle1)
        rotZ = jnp.array([
            [c1, -s1,  jnp.zeros_like(s1)],
            [s1,  c1,  jnp.zeros_like(s1)],
            [jnp.zeros_like(s1), jnp.zeros_like(s1), jnp.ones_like(s1)]
        ])
        # Step 2: Rotate around X by zenith -> aligns Z axis with shower axis
        angle2 = zenith
        c2 = jnp.cos(angle2)
        s2 = jnp.sin(angle2)
        rotX = jnp.array([
            [jnp.ones_like(c2),  jnp.zeros_like(c2),  jnp.zeros_like(c2)],
            [jnp.zeros_like(c2), c2, -s2],
            [jnp.zeros_like(c2), s2,  c2]
        ])
        # Combined transformation: Apply rotZ first, then rotX
        # Transforms coordinates from Geo -> Shower Plane

        rotX = rotX.reshape(3, 3)
        rotZ = rotZ.reshape(3, 3)

        self.transformation_matrix_early_late = rotX @ rotZ

        self.inverse_transformation_matrix_early_late = self.transformation_matrix_early_late.T

    # Private helper for applying transformation (vectorized)
    def _transform(self, positions, matrix):
        """ Applies transformation matrix to positions.
            Handles single (3,) vector or batch (N, 3) vectors.
        """
        positions = jnp.asarray(positions)
        original_shape = positions.shape
        is_single_vector = positions.ndim == 1

        # Reshape to at least 2D (N, 3) for matmul
        if is_single_vector:
            positions = positions.reshape(1, 3) # Shape (1, 3)
        elif positions.ndim != 2 or positions.shape[-1] != 3:
             # Handle potential time series input (Polarizations, Time samples)
             # Assumes matrix transforms the polarization components [Ex, Ey, Ez]
             if positions.ndim == 2 and positions.shape[0] == 3 and positions.shape[1] != 3:
                 # Input shape (3, Nsamples) -> treat as 3D vectors over time
                 # Apply matrix to each time sample's vector: matrix @ positions
                 transformed_positions = matrix @ positions # (3,3) @ (3, N) -> (3, N)
                 return transformed_positions # Keep shape (3, Nsamples)
             else:
                raise ValueError(f"Input positions must have shape (3,) or (N, 3), or (3, Nsamples > 3). Got {original_shape}")

        # Perform transformation: (N, 3) @ (3, 3).T = (N, 3)
        # Or: matrix @ positions.T -> (3, 3) @ (3, N) = (3, N) -> transpose back
        transformed_positions = (matrix @ positions.T).T

        # Reshape back if original was a single vector
        if is_single_vector:
            return transformed_positions.reshape(3)
        else:
            return transformed_positions

    #  Public Transformation Methods 

    # On-Sky Transformations
    def transform_from_ground_to_onsky(self, positions):
        """ Ground (East, North, Up) to On-sky (eR, eTheta, ePhi) """
        return self._transform(positions, self.transformation_matrix_onsky)

    def transform_from_onsky_to_ground(self, positions):
        """ On-sky (eR, eTheta, ePhi) to Ground (East, North, Up) """
        return self._transform(positions, self.inverse_transformation_matrix_onsky)

    # Magnetic North Transformations
    def transform_from_magnetic_to_geographic(self, positions):
        """ Magnetic CS (x, y aligned MagNorth, z) to Geographic CS (East, North, Up) """
        return self._transform(positions, self.transformation_matrix_magnetic)

    def transform_from_geographic_to_magnetic(self, positions):
        """ Geographic CS (East, North, Up) to Magnetic CS (x, y aligned MagNorth, z) """
        return self._transform(positions, self.inverse_transformation_matrix_magnetic)

    # Azimuth-aligned Ground CS Transformations
    def transform_from_azimuth_to_geographic(self, positions):
        """ Azimuth-aligned CS to Geographic CS (East, North, Up) """
        # Note: Azimuth CS definition depends on the matrix derived in __init__
        return self._transform(positions, self.inverse_transformation_matrix_azimuth)

    def transform_from_geographic_to_azimuth(self, positions):
        """ Geographic CS (East, North, Up) to Azimuth-aligned CS """
        # Note: Azimuth CS definition depends on the matrix derived in __init__
        return self._transform(positions, self.transformation_matrix_azimuth)


    # Shower Plane (Early-Late) Transformations
    def transform_from_early_late(self, positions, core=None):
        """ Shower Plane (Early-Late) CS to Geographic CS (East, North, Up) """
        positions = jnp.asarray(positions)
        if core is not None:
            core = jnp.asarray(core)

        transformed_pos = self._transform(positions, self.inverse_transformation_matrix_early_late)

        if core is not None:
            # Add core position back (broadcasts if needed)
            transformed_pos += core

        return transformed_pos

    def transform_to_early_late(self, positions, core=None):
        """ Geographic CS (East, North, Up) to Shower Plane (Early-Late) CS """
        positions = jnp.asarray(positions)
        if core is not None:
            core = jnp.asarray(core)
            # Subtract core position first (broadcasts if needed)
            positions -= core

        transformed_pos = self._transform(positions, self.transformation_matrix_early_late)
        return transformed_pos


    # vxB / vx(vxB) Transformations
    def transform_to_vxB_vxvxB(self, positions, core=None):
        """ Geographic CS (East, North, Up) to vxB / vx(vxB) CS """
        positions = jnp.asarray(positions)
        if core is not None:
            core = jnp.asarray(core)
            # Subtract core position first (broadcasts if needed)
            positions -= core

        # _transform handles vectors (N,3), batches (N,3), and time series (3, Nsamples)
        transformed_pos = self._transform(positions, self.transformation_matrix_vBvvB)
        return transformed_pos

    def transform_from_vxB_vxvxB(self, positions, core=None):
        """ vxB / vx(vxB) CS to Geographic CS (East, North, Up) """
        positions = jnp.asarray(positions)
        if core is not None:
            core = jnp.asarray(core)

        # _transform handles vectors (N,3), batches (N,3), and time series (3, Nsamples)
        transformed_pos = self._transform(positions, self.inverse_transformation_matrix_vBvvB)

        if core is not None:
             # Add core position back (broadcasts if needed)
             # Need to be careful if transformed_pos is (3, Nsamples) and core is (3,)
             if transformed_pos.ndim == 2 and transformed_pos.shape[0] == 3 and core.shape == (3,):
                 transformed_pos += core[:, jnp.newaxis] # Add core to each time sample's vector
             else:
                 transformed_pos += core # Add core to each position vector

        return transformed_pos

    #  Other Methods 

    def get_height_in_showerplane(self, x_sp, y_sp):
        """ Calculates the Z coordinate in geographic CS for a point (x_sp, y_sp)
            given in the shower plane (vxB, vxvxB system) assuming Z_sp = 0.
            This solves Z_geo from the transformation equations, setting Z_sp=0.
        """
        x_sp = jnp.asarray(x_sp)
        y_sp = jnp.asarray(y_sp)
        InvM = self.inverse_transformation_matrix_vBvvB
        # Add epsilon for numerical stability if InvM[2,2] is close to zero
        z_sp = - (InvM[2, 0] * x_sp + InvM[2, 1] * y_sp) / (InvM[2, 2] + 1e-12)
        return z_sp

    def transform_from_vxB_vxvxB_2D(self, positions_2d, core=None):
        """ Transforms a list of 2D positions (x_sp, y_sp) from the vxB/vxvxB plane
            back to 3D geographic CS, assuming the points lie on the ground (z_geo=0).
        """
        positions_2d = jnp.asarray(positions_2d) # Shape (N, 2) or (2,)
        if core is not None:
            core = jnp.asarray(core)

        is_single = positions_2d.ndim == 1
        if is_single:
            positions_2d = positions_2d.reshape(1, 2) # (1, 2)

        x_sp = positions_2d[:, 0]
        y_sp = positions_2d[:, 1]
        z_sp = self.get_height_in_showerplane(x_sp, y_sp) # Calculate corresponding z_sp

        # Stack to get 3D positions in shower plane CS
        positions_3d_sp = jnp.stack([x_sp, y_sp, z_sp], axis=-1) # (N, 3)

        # Transform these 3D points back to geographic CS
        positions_3d_geo = self._transform(positions_3d_sp, self.inverse_transformation_matrix_vBvvB)

        if core is not None:
            positions_3d_geo += core # Add core position

        if is_single:
            return positions_3d_geo.reshape(3)
        else:
            return positions_3d_geo


    def get_euler_angles(self):
        """Calculate Euler angles (psi, theta, phi) for the vxB/vxvxB rotation matrix (Geographic → vxB).

        Uses ZXZ convention. Handles gimbal lock via jax.lax.cond for JIT compatibility.
        """
        R = self.transformation_matrix_vBvvB # Geo -> vBvvB

        cos_theta = R[2, 2] 

        is_gimbal_lock = jnp.abs(R[2, 0]) > 0.99999 # Check for near gimbal lock

        # Define functions for regular and gimbal lock cases
        def regular_case(R_):
            theta_1 = -jnp.arcsin(R_[2, 0])
            # theta_2 = jnp.pi - theta_1 # Second solution often ignored

            cos_theta_1 = jnp.cos(theta_1)
            # Add epsilon to prevent division by zero, although |R[2,0]|!=1 implies cos(theta)!=0
            cos_theta_1_safe = jnp.where(jnp.abs(cos_theta_1) < 1e-9, 1e-9, cos_theta_1)

            psi_1 = jnp.arctan2(R_[2, 1] / cos_theta_1_safe, R_[2, 2] / cos_theta_1_safe)
            phi_1 = jnp.arctan2(R_[1, 0] / cos_theta_1_safe, R_[0, 0] / cos_theta_1_safe)
            return psi_1, theta_1, phi_1

        def gimbal_lock_case(R_):
            phi_1 = 0. # Convention choice in gimbal lock
            # Check which pole: R[2,0] = -1 or +1
            # R[2, 0] == -1 case (theta = pi/2)
            theta_1_neg = jnp.pi * 0.5
            psi_1_neg = phi_1 + jnp.arctan2(R_[0, 1], R_[0, 2])

            # R[2, 0] == +1 case (theta = -pi/2)
            theta_1_pos = -jnp.pi * 0.5
            psi_1_pos = -phi_1 + jnp.arctan2(-R_[0, 1], -R_[0, 2]) # Note signs in original

            # Use jnp.where to select based on the sign of R[2,0]
            theta_1 = jnp.where(R_[2, 0] < 0, theta_1_neg, theta_1_pos)
            psi_1 = jnp.where(R_[2, 0] < 0, psi_1_neg, psi_1_pos)

            return psi_1, theta_1, phi_1

        # Use jax.lax.cond for safe branching with tracers
        psi, theta, phi = jax.lax.cond(
            is_gimbal_lock,
            gimbal_lock_case, # operand for true branch
            regular_case,     # operand for false branch
            R                 # argument to pass to the selected function
        )

        return psi, theta, phi


def _pad_knots(t, k):
    """Clamp-pad a knot vector so it spans [t[0], t[-1]] with multiplicity k."""
    t = np.asarray(t)
    return np.concatenate([np.full(k, t[0]), t, np.full(k, t[-1])])


jax_spline_data = {}

with open(get_ift_data_path("geo_rcut_b_splines.pickle"), "rb") as fin:
    u = pickle._Unpickler(fin)
    u.encoding = 'latin1'
    (t_rcut, c_rcut, k_rcut), (t_b, c_b, k_b) = u.load()
    jax_spline_data['rcut_geo'] = (jnp.array(_pad_knots(t_rcut, k_rcut)), jnp.array(c_rcut), k_rcut)
    jax_spline_data['b_geo'] = (jnp.array(_pad_knots(t_b, k_b)), jnp.array(c_b), k_b)

with open(get_ift_data_path("geo_sigmaR_spl.pickle"), "rb") as fin:
    u = pickle._Unpickler(fin)
    u.encoding = 'latin1'
    data = u.load()
    for _key in ('geo_R_0m', 'geo_sigma_0m'):
        t, c, k = data[_key]
        jax_spline_data[_key] = (jnp.array(_pad_knots(t, k)), jnp.array(c), k)

with open(get_ift_data_path("ce_sigma_spl.pickle"), "rb") as fin:
    u = pickle._Unpickler(fin)
    u.encoding = 'latin1'
    data = u.load()
    t, c, k = data['ce_sigma_0m']
    jax_spline_data['ce_sigma_0m'] = (jnp.array(_pad_knots(t, k)), jnp.array(c), k)

with open(get_ift_data_path("Ecorr.pickle"), "rb") as fin:
    u = pickle._Unpickler(fin)
    u.encoding = 'latin1'
    data = u.load()
    for _key in ('geo_Ecorr_0m', 'ce_Ecorr_0m'):
        t, c, k = data[_key]
        jax_spline_data[_key] = (jnp.array(_pad_knots(t, k)), jnp.array(c), k)

with open(get_ift_data_path("ce_b_rcut_spl.pickle"), "rb") as fin:
    u = pickle._Unpickler(fin)
    u.encoding = 'latin1'
    _spl = u.load()
for _key in ('ce_b_0m', 'ce_rcut_0m'):
    t, c, k = _spl[_key]
    t = np.asarray(t); c = np.asarray(c); k = int(k)
    jax_spline_data[_key] = (jnp.array(_pad_knots(t, k)), jnp.array(c), k)

with open(get_ift_data_path("ce_analytic_params.json")) as fin:
    _ce_klog = json.load(fin)["k_logistic"]
CE_K_LOGISTIC = (float(_ce_klog["a"]), float(_ce_klog["b"]),
                 float(_ce_klog["c"]), float(_ce_klog["d"]))


# b-spline evaluation function 
def _evaluate_bspline_scalar(xx, t, c, k, i):
    # i is already clipped to [k, n-1]
    start = i - k                              # guaranteed >= 0
    d = lax.dynamic_slice(c, (start,), (k + 1,))   # shape (k+1,)

    for r in range(1, k + 1):
        for j in range(k, r - 1, -1):
            left_idx  = i - k + j
            right_idx = i + j + 1 - r               # simplified: i - k + j + k + 1 - r
            denom = t[right_idx] - t[left_idx]

            alpha = jnp.where(denom > 1e-8,
                              (xx - t[left_idx]) / denom,
                              0.0)

            new_val = (1.0 - alpha) * d[j - 1] + alpha * d[j]
            d = d.at[j].set(new_val)

    return d[k]

def evaluate_bspline(x: ArrayLike, t: jnp.ndarray, c: jnp.ndarray, k: int) -> jnp.ndarray:
    """
    Evaluate univariate B-spline at point(s) x.
    Vectorised over x, degree k static → tiny, fast XLA code.
    """
    x = jnp.atleast_1d(jnp.asarray(x))
    t = jnp.asarray(t)
    c = jnp.asarray(c)
    n = c.shape[0]

    # Valid domain for padded knots: [t[k], t[-(k+1)] ]
    x = jnp.clip(x, t[k], t[-(k + 1)])

    def eval_single(xx):
        i = jnp.searchsorted(t, xx, side='right') - 1
        i = jnp.clip(i, k, n - 1)
        return _evaluate_bspline_scalar(xx, t, c, k, i)

    return jax.vmap(eval_single)(x)



# Helper function to get JAX spline data (t, c, k) based on key
def get_jax_spline_tck(key):
    """Retrieves JAX-compatible (t, c, k) tuple from the global dict."""
    if key in jax_spline_data:
        return jax_spline_data[key]
    raise ValueError(f"Spline data not found for key: '{key}'")

# Physics helper functions

def spherical_to_cartesian(zenith: ArrayLike, azimuth: ArrayLike) -> jnp.ndarray:
    """
    Converts zenith and azimuth angle into a cartesian 3-vector (JAX version).
    Handles scalar or array inputs. Output shape is (3,) or (N, 3).
    """
    zenith = jnp.asarray(zenith)
    azimuth = jnp.asarray(azimuth)
    sinZenith = jnp.sin(zenith)
    x = sinZenith * jnp.cos(azimuth)
    y = sinZenith * jnp.sin(azimuth)
    z = jnp.cos(zenith)
    cart = jnp.stack([x, y, z], axis=-1)
    return cart

def get_lorentz_force_vector(zenith: ArrayLike, azimuth: ArrayLike,
                             magnetic_field_vector: ArrayLike) -> jnp.ndarray:
    """Get the Lorentz force as a cartesian 3-vector (JAX version)."""
    magnetic_field_vector = jnp.asarray(magnetic_field_vector)
    showerAxis = spherical_to_cartesian(zenith, azimuth)
    norm = jnp.linalg.norm(magnetic_field_vector, axis=-1, keepdims=True)
    magnetic_field_vector_normalized = jnp.where(norm > 1e-9, magnetic_field_vector / norm, magnetic_field_vector * 0.0) # Zero vector if norm is zero
    return jnp.cross(showerAxis, magnetic_field_vector_normalized)

def get_sine_angle_to_lorentz_force(zenith: ArrayLike, azimuth: ArrayLike,
                                    magnetic_field_vector: ArrayLike = None) -> ArrayLike:
    """Returns the sine of the angle between shower axis and Lorentz force vector (JAX version)."""
    if magnetic_field_vector is None:
         raise ValueError("magnetic_field_vector must be provided")
    magnetic_field_vector = jnp.asarray(magnetic_field_vector)
    lorentz_force = get_lorentz_force_vector(zenith, azimuth, magnetic_field_vector)
    return jnp.linalg.norm(lorentz_force, axis=-1)


def get_a(rho: ArrayLike, magnetic_field_strength: ArrayLike = 0.243, magnetic_field_vector = None) -> ArrayLike:
    """Relative charge-excess fraction 'a' (JAX version)."""
    rho = jnp.asarray(rho)
    if magnetic_field_vector is not None:
        magnetic_field_strength = np.linalg.norm(magnetic_field_vector)
    else:
        magnetic_field_strength = jnp.asarray(magnetic_field_strength)
    average_density = 648.18353008270035 # Constant
    a_calc = -0.23604683 + 0.43426141 * jnp.exp(1.11141046e-3 * (rho - average_density))
    norm_factor = jnp.where(magnetic_field_strength > 1e-9, (magnetic_field_strength / 0.243) ** 0.9, 1.0)
    # Handle case where B=0, should norm_factor be infinity or 1? If B=0, Egeo=0 anyway. Let's use 1.
    return a_calc / norm_factor


def get_k_ce(dxmax: ArrayLike) -> ArrayLike:
    dxmax = jnp.asarray(dxmax)
    a, b, c, d = CE_K_LOGISTIC
    return jnp.maximum(b + (c - b) / (1.0 + jnp.exp(-d * (dxmax - a))), 0.0)

def get_b_ce(k: ArrayLike, dxmax: ArrayLike) -> ArrayLike:
    t, c, kk = get_jax_spline_tck('ce_b_0m')
    return evaluate_bspline(jnp.asarray(dxmax), t, c, kk)

def get_rcut_ce(k: ArrayLike, dxmax: ArrayLike) -> ArrayLike:
    t, c, kk = get_jax_spline_tck('ce_rcut_0m')
    return jnp.maximum(0.0, evaluate_bspline(jnp.asarray(dxmax), t, c, kk))


# Use the JAX spline data loaded into jax_spline_data dict
def get_b_geo_spl(dxmax: ArrayLike) -> ArrayLike:
    """Get parameter 'b' for Geo LDF exponent from spline (JAX version)."""
    t, c, k = get_jax_spline_tck('b_geo')
    return evaluate_bspline(dxmax, t, c, k)

def get_rcut_geo_spl(dxmax: ArrayLike) -> ArrayLike:
    """Get parameter 'rcut' for Geo LDF exponent from spline (JAX version)."""
    t, c, k = get_jax_spline_tck('rcut_geo')
    return evaluate_bspline(dxmax, t, c, k)


def get_p(r: ArrayLike, rcut: ArrayLike, b_param: ArrayLike) -> ArrayLike:
    """Parametrization of the exponent 'p' (JAX version)."""
    r = jnp.abs(jnp.asarray(r))
    rcut = jnp.maximum(1.0, jnp.abs(jnp.asarray(rcut)))
    b_param = 1e-3 * jnp.asarray(b_param)

    # Calculate p_geo safely: rcut^b_param
    # If rcut=1, p_geo=2. If b_param=0, p_geo=2.
    # Use jnp.power which handles non-integer exponents
    p_geo_base = jnp.power(rcut, b_param)
    p_geo = 2.0 * jnp.nan_to_num(p_geo_base, nan=1.0, posinf=1.0, neginf=1.0) # If pow fails, default to factor 1 -> p_geo=2

    # Calculate r^(-b_param) safely
    r_pow_neg_b = jnp.power(jnp.maximum(r, 1e-9), -b_param) # Avoid r=0
    r_pow_neg_b = jnp.nan_to_num(r_pow_neg_b, nan=0.0, posinf=0.0, neginf=0.0) # If pow fails, default to 0?

    # Use jnp.where for the conditional based on r vs rcut
    p = jnp.where(r <= rcut,
                  2.0,
                  p_geo * r_pow_neg_b)
    return p


def LDF_vB_parts(r: ArrayLike, sigma: ArrayLike, R_val: ArrayLike, p: ArrayLike = 2.0) -> ArrayLike:
    """Helper function for LDF_vB (JAX version)."""
    r = jnp.asarray(r)
    sigma = jnp.maximum(jnp.asarray(sigma), 1e-9) # Avoid sigma=0
    R_val = jnp.asarray(R_val)
    p = jnp.asarray(p)

    abs_r_minus_R = jnp.abs(r - R_val)
    base = abs_r_minus_R / (jnp.sqrt(2.0) * sigma)

    # Calculate base^p safely
    exponent_term = jnp.power(base, p)
    exponent_term = jnp.nan_to_num(exponent_term, nan=jnp.inf, posinf=jnp.inf) # Map NaNs/Infs from pow to Inf

    # Calculate exp(-term), exp(-inf) = 0
    return jnp.exp(-1.0 * exponent_term)


def LDF_vB(x: ArrayLike, y: ArrayLike, sigma: ArrayLike, R_val: ArrayLike, E: ArrayLike, p: ArrayLike = 2.0) -> ArrayLike:
    """Geomagnetic LDF component base function (JAX version)."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    sigma = jnp.maximum(jnp.asarray(sigma), 1e-12) # Avoid sigma=0, use smaller epsilon for stability
    R_val = jnp.asarray(R_val)
    E = jnp.asarray(E)
    p = jnp.asarray(p)

    r = jnp.sqrt(x**2 + y**2)

    #  Calculate Normalization (using jax.scipy.special) 
    sqrt2 = jnp.sqrt(2.0)
    sqrtpi = jnp.sqrt(jnp.pi)

    # Term 1 (Original R < 0 case)
    arg_erfc = -R_val * sqrt2 / (2.0 * sigma)
    exp_term_norm1 = jnp.exp(-R_val**2 / (2.0 * sigma**2))
    norm1_denom_term = (jax.scipy.special.erfc(arg_erfc) * sqrtpi * R_val) + \
                       (sqrt2 * sigma * exp_term_norm1)
    norm1 = jnp.abs(sigma * jnp.pi * sqrt2 * norm1_denom_term)
    norm1 = jnp.maximum(norm1, 1e-12) # Avoid division by zero
    value1 = (E / norm1) * LDF_vB_parts(r, sigma, R_val, p)

    # Term 2 (Original R >= 0 case)
    arg_erf = 0.5 * R_val * sqrt2 / sigma
    exp_term_norm2 = jnp.exp(0.5 * R_val**2 / sigma**2)
    denom_inner = (jax.scipy.special.erf(arg_erf) * sqrtpi * sqrt2 * exp_term_norm2 * R_val + 2.0 * sigma) * jnp.pi
    denom_inner = jnp.where(jnp.abs(denom_inner) > 1e-12, denom_inner, 1e-12) # Avoid division by zero
    # Original norm factor was complex, let's re-check:
    # norm_factor2 = 1. / sigma ** 2 * 0.5 * exp_term_norm2 * sigma / denom_inner # Original calculation
    norm_factor2 = 0.5 * exp_term_norm2 / (sigma * denom_inner) # Simplified
    norm_factor2 = jnp.nan_to_num(norm_factor2, nan=0.0, posinf=0.0, neginf=0.0) # Handle potential issues

    value2_part1 = LDF_vB_parts(r, sigma, R_val, p)
    value2_part2 = LDF_vB_parts(r, sigma, -R_val, p)
    value2 = E * norm_factor2 * (value2_part1 + value2_part2)

    # Combine using jnp.where based on R_val
    result = jnp.where(R_val < 0, value1, value2)

    return jnp.maximum(0.0, result) # Ensure non-negative


def my_gamma2(xx: ArrayLike, E: ArrayLike, sigma: ArrayLike, k: ArrayLike,
              rcut: ArrayLike, b_param: ArrayLike, p: ArrayLike = None, k_limit: float = 0) -> ArrayLike:
    """Charge Excess LDF component base function (JAX version)."""
    xx = jnp.asarray(xx)
    E = jnp.asarray(E)
    sigma = jnp.maximum(jnp.asarray(sigma), 1e-12) # Avoid sigma=0
    k = jnp.asarray(k)
    rcut = jnp.asarray(rcut)
    b_param = jnp.asarray(b_param) # Renamed from 'b' for clarity

    r_abs = jnp.abs(xx)

    if p is None:
        p = get_p(r_abs, rcut, b_param)
    else:
        p = jnp.asarray(p)

    #  Calculate Normalization 
    # Ensure arguments to gamma are positive
    gamma_arg = 0.5 * k + 1.0
    gamma_val = jax.scipy.special.gamma(jnp.maximum(gamma_arg, 1e-6)) # Avoid gamma(<=0)

    # Calculate powers safely
    k_plus_1 = k + 1.0
    sigma_pow_k_plus_2 = jnp.power(sigma, k_plus_1 + 1.0) # sigma^(k+2)
    term_pow_neg_half_k_base = 2.0 * k_plus_1 # 2k+2
    term_pow_neg_half_k = jnp.power(jnp.maximum(term_pow_neg_half_k_base, 1e-9), -0.5 * k)
    pow_2_k = jnp.power(2.0, k)

    # Combine norm parts
    norm_numerator = k_plus_1 / (pow_2_k * term_pow_neg_half_k)
    norm_denominator = sigma_pow_k_plus_2 * (2.0 * jnp.pi * gamma_val)
    norm_denominator = jnp.maximum(jnp.abs(norm_denominator), 1e-12) # Avoid division by zero
    norm = norm_numerator / norm_denominator
    norm = jnp.nan_to_num(norm, nan=0.0, posinf=0.0, neginf=0.0)

    #  Calculate Exponential Term 
    # exp(-(abs(xx)**p / (p/(k+1) * sigma**p)))
    exp_exponent_denom_factor = p / jnp.maximum(k_plus_1, 1e-9)
    sigma_pow_p = jnp.power(sigma, p)
    exp_exponent_denom = exp_exponent_denom_factor * sigma_pow_p
    exp_exponent_denom = jnp.maximum(jnp.abs(exp_exponent_denom), 1e-12) # Avoid division by zero

    r_abs_pow_p = jnp.power(r_abs, p)
    exp_exponent = -r_abs_pow_p / exp_exponent_denom
    exp_term = jnp.exp(exp_exponent) # exp(-large) -> 0 is fine

    # Calculate r^k term 
    # Handle r_abs=0 separately if k<0, although usually k>=0 here
    r_pow_k = jnp.power(r_abs, k)
    r_pow_k = jnp.where(r_abs < 1e-12, 0.0, r_pow_k) # Ensure 0^k is handled (gives 0 if k>0)
    r_pow_k = jnp.nan_to_num(r_pow_k, nan=0.0)

    # Combine terms
    fluence = norm * E * r_pow_k * exp_term

    # Apply k_limit condition using where (propagate NaN if k<limit)
    result = jnp.where(k < k_limit, jnp.nan, fluence)

    # Ensure non-negative fluence (unless NaN)
    return jnp.where(jnp.isnan(result), jnp.nan, jnp.maximum(0.0, result))


# LDF Functions using Splines 

def LDF_geo_dxmax(x: ArrayLike, y: ArrayLike, dxmax: ArrayLike, E: ArrayLike) -> ArrayLike:
    """Geomagnetic LDF using dxmax parametrization (JAX version)."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    dxmax = jnp.asarray(dxmax)
    E = jnp.asarray(E)

    # Geo R spline
    t_R, c_R, k_R = get_jax_spline_tck('geo_R_0m')
    R = evaluate_bspline(dxmax, t_R, c_R, k_R)

    # Geo sigma spline
    t_sig, c_sig, k_sig = get_jax_spline_tck('geo_sigma_0m')
    sigma = evaluate_bspline(dxmax, t_sig, c_sig, k_sig)

    # Geo Ecorr spline
    t_Ecorr, c_Ecorr, k_Ecorr = get_jax_spline_tck('geo_Ecorr_0m')
    Ecorr = evaluate_bspline(dxmax, t_Ecorr, c_Ecorr, k_Ecorr)
    Ecorr = jnp.where(jnp.abs(Ecorr) < 1e-9, 1.0, Ecorr) # Avoid division by zero

    # Get exponent parameters
    rcut = get_rcut_geo_spl(dxmax)
    b = get_b_geo_spl(dxmax) # This is b_param for get_p

    r = jnp.sqrt(x**2 + y**2)
    p = get_p(r, rcut, b)

    # Calculate base LDF
    fluence = LDF_vB(x, y, sigma, R, E, p)

    return fluence / Ecorr


def LDF_ce_dxmax(x: ArrayLike, y: ArrayLike, dxmax: ArrayLike, E: ArrayLike) -> ArrayLike:
    """Charge Excess LDF using dxmax parametrization (JAX version)."""
    x = jnp.asarray(x)
    y = jnp.asarray(y)
    dxmax = jnp.asarray(dxmax)
    E = jnp.asarray(E)

    # CE sigma spline
    t_sig, c_sig, k_sig = get_jax_spline_tck('ce_sigma_0m')
    sigma = evaluate_bspline(dxmax, t_sig, c_sig, k_sig)

    # CE Ecorr spline
    t_Ecorr, c_Ecorr, k_Ecorr = get_jax_spline_tck('ce_Ecorr_0m')
    Ecorr = evaluate_bspline(dxmax, t_Ecorr, c_Ecorr, k_Ecorr)
    Ecorr = jnp.where(jnp.abs(Ecorr) < 1e-9, 1.0, Ecorr)

    # Get exponent parameters (which depend on k, which depends on dxmax)
    k = get_k_ce(dxmax)
    rcut = get_rcut_ce(k, dxmax)
    b = get_b_ce(k, dxmax) # Note: Parameter 'b' for exponent, not spline 'b'

    # Calculate base LDF using my_gamma2
    r = jnp.sqrt(x**2 + y**2)
    # my_gamma2 calculates p internally using get_p(r, rcut, b_param=b)
    fluence = my_gamma2(r, E=E, sigma=sigma, k=k, rcut=rcut, b_param=b)

    return fluence / Ecorr


# Top-Level Functions

# JIT-compiled core computation (atmosphere params pre-computed)
@jax.jit
def _LDF_core(x: ArrayLike, y: ArrayLike, Erad: ArrayLike, xmax: ArrayLike,
              zenith: ArrayLike, azimuth: ArrayLike, core: ArrayLike,
              magnetic_field_vector: ArrayLike,
              Xatm_obs_gm2: ArrayLike, rho_xmax: ArrayLike) -> tuple:
    """
    JIT-compiled core LDF calculation. All tracing-compatible inputs.
    """
    # Calculate dxmax with proper units
    slant_depth_obs_gm2 = Xatm_obs_gm2 / jnp.cos(zenith)
    xmax_gm2 = xmax * 1e4
    dxmax_gm2 = slant_depth_obs_gm2 - xmax_gm2
    dxmax = dxmax_gm2 * 1e-4

    x_in, y_in, Erad_in, dxmax_in = jnp.asarray(x), jnp.asarray(y), jnp.asarray(Erad), jnp.asarray(dxmax)
    zenith_in, azimuth_in = jnp.asarray(zenith), jnp.asarray(azimuth)
    core_in, magnetic_field_vector_in = jnp.asarray(core), jnp.asarray(magnetic_field_vector)

    # Egeo/Ece split
    magnetic_field_strength = jnp.linalg.norm(magnetic_field_vector_in)
    magnetic_field_strength_safe = jnp.maximum(magnetic_field_strength, 1e-9)
    a = get_a(rho_xmax, magnetic_field_strength_safe)
    sinalpha = get_sine_angle_to_lorentz_force(zenith_in, azimuth_in, magnetic_field_vector_in)
    sinalpha_safe = jnp.maximum(sinalpha, 1e-9)
    a_over_sinalpha_sq = jnp.power(a / sinalpha_safe, 2)
    Egeo = jnp.where(sinalpha > 1e-9, Erad_in / (1.0 + a_over_sinalpha_sq), 0.0)
    Ece = Erad_in - Egeo

    # Relative coordinates
    x2 = x_in - core_in[0]
    y2 = y_in - core_in[1]

    # Fluences
    fce = LDF_ce_dxmax(x2, y2, dxmax_in, Ece)
    fgeo = LDF_geo_dxmax(x2, y2, dxmax_in, Egeo)

    # Combine fluences for components
    az = jnp.arctan2(y2, x2)
    cos_az = jnp.cos(az)
    sin_az = jnp.sin(az)
    sqrt_fgeo = jnp.sqrt(jnp.maximum(fgeo, 1e-12))
    sqrt_fce = jnp.sqrt(jnp.maximum(fce, 1e-12))
    fvB = jnp.power(sqrt_fgeo + sqrt_fce * cos_az, 2)
    fvvB = fce * jnp.power(sin_az, 2)
    f = fvB + fvvB

    return f, fvB, fvvB, fgeo, fce


def LDF(x: ArrayLike, y: ArrayLike, Erad: ArrayLike, xmax: ArrayLike,
        zenith: ArrayLike, azimuth: ArrayLike, core: ArrayLike = None,
        magnetic_field_vector: ArrayLike = None,
        atmosphere_path: str | None = None) -> tuple:
    """
    Combined LDF calculation (JAX version).
    Calls JIT-compiled _LDF_core for performance.
    """
    atmosphere = atm.Atmosphere(gdas_file=atmosphere_path)
    Xatm_obs_gm2 = atmosphere.get_atmosphere(0.)  # in g/m^2, LOFAR 0 m
    rho_xmax = atmosphere.get_density(xmax, zenith)

    if core is None:
        core = jnp.array([0.0, 0.0], dtype=jnp.float64)
    if magnetic_field_vector is None:
        magnetic_field_vector = jnp.array([0.004675, 0.186270, -0.456412], dtype=jnp.float64)

    return _LDF_core(x, y, Erad, xmax, zenith, azimuth, core,
                     magnetic_field_vector, Xatm_obs_gm2, rho_xmax)


@jax.jit
def LDF_geo_ce2(x: ArrayLike, y: ArrayLike, Egeo: ArrayLike, Ece: ArrayLike,
                dxmax: ArrayLike, core: ArrayLike = None) -> tuple:
    """
    Combined LDF calculation taking Egeo, Ece directly (JAX version).
    Returns total fluence, vB fluence, vvB fluence, geo fluence, ce fluence.
    """
    if core is None: core = jnp.array([0.0, 0.0], dtype=jnp.float64)
    x_in=jnp.asarray(x); y_in=jnp.asarray(y); Egeo_in=jnp.asarray(Egeo); Ece_in=jnp.asarray(Ece)
    dxmax_in=jnp.asarray(dxmax); core_in=jnp.asarray(core)

    x2 = x_in - core_in[0]; y2 = y_in - core_in[1]
    fce = LDF_ce_dxmax(x2, y2, dxmax_in, Ece_in)
    fgeo = LDF_geo_dxmax(x2, y2, dxmax_in, Egeo_in)
    az = jnp.arctan2(y2, x2); cos_az = jnp.cos(az); sin_az = jnp.sin(az)
    sqrt_fgeo = jnp.sqrt(jnp.maximum(fgeo, 1e-12))
    sqrt_fce = jnp.sqrt(jnp.maximum(fce, 1e-12))
    fvB = jnp.power(sqrt_fgeo + sqrt_fce * cos_az, 2)
    fvvB = fce * jnp.power(sin_az, 2)
    f = fvB + fvvB
    return f, fvB, fvvB, fgeo, fce