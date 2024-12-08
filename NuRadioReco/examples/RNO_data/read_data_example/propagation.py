import pykonal, copy
import numpy as np

class TravelTimeCalculator:

    # All coordinates here are 2d (r, z) in natural feet

    @classmethod
    def FromDict(cls, indict):
        obj = cls(**indict)
        return obj
    
    def __init__(self, tx_z, z_range, r_max, num_pts_z, num_pts_r, travel_time_maps = {}):

        self.tx_z = tx_z
        self.tx_pos = [0.0, self.tx_z]
        
        self.num_pts_z = num_pts_z
        self.num_pts_r = num_pts_r
        
        self.z_range = z_range
        self.r_max = r_max

        self.domain_start = np.array([0.0, self.z_range[0]])
        self.domain_end = np.array([self.r_max, self.z_range[1]])
        self.domain_shape = np.array([self.num_pts_r, self.num_pts_z])    

        # determine voxel size
        self.delta_r = self.r_max / self.num_pts_r
        self.delta_z = (self.z_range[1] - self.z_range[0]) / self.num_pts_z
        
        self.travel_time_maps = travel_time_maps
        self.tangent_vectors = {}

        self._build_tangent_vectors()

    def to_dict(self):        
        return copy.deepcopy({
            "tx_z": self.tx_z,
            "z_range": self.z_range,
            "r_max": self.r_max,
            "num_pts_z": self.num_pts_z,
            "num_pts_r": self.num_pts_r,
            "travel_time_maps": self.travel_time_maps
        })

    def _build_tangent_vectors(self):
        for comp_name, comp_map in self.travel_time_maps.items():            
            grad_r, grad_z = np.gradient(comp_map[:,:,0], self.delta_r, self.delta_z)
            grad_vec = np.stack([grad_r, grad_z], axis = -1)
            self.tangent_vectors[comp_name] = -grad_vec # keep the negative to make it point towards the antenna
    
    def set_ior_and_solve(self, ior, reflection_at_z = 0.0):

        def _get_solver(iordata):
            veldata = 1.0 / iordata  # c = 1 when distance measured in natural feet
            solver = pykonal.EikonalSolver(coord_sys = "cartesian")
            solver.velocity.min_coords = 0, 0, 0
            solver.velocity.npts = self.num_pts_r, self.num_pts_z, 1
            solver.velocity.node_intervals = self.delta_r, self.delta_z, 1        
            solver.velocity.values = veldata
            return solver
        
        # Build the IOR distribution
        zvals = np.linspace(self.z_range[0], self.z_range[1], self.num_pts_z)
        iorslice = ior(zvals)
        iordata = np.expand_dims(np.tile(iorslice, reps = (self.num_pts_r, 1)), axis = -1)
        
        boundary_z_ind = self._coord_to_pykonal([[0, reflection_at_z]])[0][1]

        # Calculate rays transmitted into the air
        solver = _get_solver(iordata)
        src_ind = self._coord_to_pykonal([self.tx_pos])[0]
        
        solver.traveltime.values[*src_ind] = 0 # Place a point source at the transmitter
        solver.unknown[*src_ind] = False    
        solver.trial.push(*src_ind)
        solver.solve()

        self.travel_time_maps["direct_air"] = np.copy(solver.traveltime.values)
        self.travel_time_maps["direct_air"][:, :boundary_z_ind, :] = np.nan # this is now unphysical in the ice, as in part of the volume
                                                                            # head-waves will overtake the direct bending modes
        
        # Calculate direct rays in the ice
        iordata[:, boundary_z_ind:, :] = 10.0 # assign a spuriously large IOR to the air to make sure there are no head waves
                                              # that can overtake the bulk-bending modes that we want
        solver = _get_solver(iordata)
        src_ind = self._coord_to_pykonal([self.tx_pos])[0]
        
        solver.traveltime.values[*src_ind] = 0 # Place a point source at the transmitter
        solver.unknown[*src_ind] = False    
        solver.trial.push(*src_ind)
        solver.solve()

        self.travel_time_maps["direct_ice"] = np.copy(solver.traveltime.values)
        self.travel_time_maps["direct_ice"][:, boundary_z_ind+1:, :] = np.nan # this is now unphysical in the air
        
        # Calculate reflected rays: place a line source at the air/ice boundary
        solver = _get_solver(iordata)
        solver.traveltime.values[:, boundary_z_ind, :] = self.travel_time_maps["direct_ice"][:, boundary_z_ind, :]
        solver.unknown[:, boundary_z_ind, :] = False
        for r_ind in range(self.num_pts_r):
            solver.trial.push(r_ind, boundary_z_ind, 0)
        solver.solve()

        self.travel_time_maps["reflected"] = np.copy(solver.traveltime.values)
        self.travel_time_maps["reflected"][:, boundary_z_ind:, :] = np.nan # this is now unphysical in the air

        self._build_tangent_vectors()

    def get_travel_time(self, coord, comp = "direct_ice"):

        if comp not in self.travel_time_maps:
            raise RuntimeError(f"Error: map for component '{comp}' not available!")
        
        ind = np.transpose(self._coord_to_pixel(coord))
        return self.travel_time_maps[comp][*ind]

    def get_tangent_vector(self, coord, comp = "direct_ice"):

        if comp not in self.travel_time_maps:
            raise RuntimeError(f"Error: map for component '{comp}' not available!")

        ind_r, ind_z, _ = np.transpose(self._coord_to_pixel(coord))
        return self.tangent_vectors[comp][ind_r, ind_z]
    
    def _coord_to_pykonal(self, coord):
        return tuple(self._coord_to_pixel(coord))
        
    def _coord_to_pixel(self, coord):
        return self._coord_to_frac_pixel(coord).astype(int)
        
    def _coord_to_frac_pixel(self, coord):        
        if isinstance(coord, list):
            coord = np.array(coord)
            
        pixel_2d = (coord - self.domain_start) / (self.domain_end - self.domain_start) * self.domain_shape
        pixel_3d = np.append(pixel_2d, np.zeros((len(coord), 1)), axis = 1)
        return pixel_3d
