import numpy as np
from NuRadioReco.utilities import units
from matplotlib import pyplot as plt
from radiotools import helper as hp
from NuRadioMC.utilities import cross_sections as cs
from NuRadioMC.utilities import earth_attenuation
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from NuRadioMC.simulation.simulation import pretty_time_delta
from NuRadioMC.EvtGen.generator import write_events_to_hdf5
from NuRadioMC.utilities import inelasticities
import pickle
import os
import time
import logging
import NuRadioMC
from NuRadioMC.utilities import version
# np.random.seed(10)  # just for testing

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

R_earth = 6357390 * units.m
earth = earth_attenuation.PREM()


def generate_eventlist_cylinder(filename, n_events, Emin, Emax,
                                full_rmin=None, full_rmax=None, full_zmin=None, full_zmax=None,
                                thetamin=0.*units.rad, thetamax=np.pi * units.rad,
                                phimin=0.*units.rad, phimax=2 * np.pi * units.rad,
                                start_event_id=1,
                                flavor=[12, -12, 14, -14, 16, -16],
                                n_events_per_file=None,
                                spectrum='log_uniform',
                                start_file_id=0):
    """
    Event generator

    Generates neutrino interactions, i.e., vertex positions, neutrino directions,
    neutrino flavor, charged currend/neutral current and inelastiviy distributions.
    All events are saved in an hdf5 file.

    Parameters
    ----------
    filename: string
        the output filename of the hdf5 file
    n_events: int
        number of events to generate
    Emin: float
        the minimum neutrino energy (energies are randomly chosen assuming a
        uniform distribution in the logarithm of the energy)
    Emax: float
        the maximum neutrino energy (energies are randomly chosen assuming a
        uniform distribution in the logarithm of the energy)

    full_rmin: float (default None)
        lower r coordinate of simulated volume (if None it is set to 1/3 of the fiducial volume, if second vertices are not activated it is set to the fiducial volume)
    full_rmax: float (default None)
        upper r coordinate of simulated volume (if None it is set to 5x the fiducial volume, if second vertices are not activated it is set to the fiducial volume)
    full_zmin: float (default None)
        lower z coordinate of simulated volume (if None it is set to 1/3 of the fiducial volume, if second vertices are not activated it is set to the fiducial volume)
    full_zmax: float (default None)
        upper z coordinate of simulated volume (if None it is set to 5x the fiducial volume, if second vertices are not activated it is set to the fiducial volume)
    thetamin: float
        lower zenith angle for neutrino arrival direction
    thetamax: float
        upper zenith angle for neutrino arrival direction
    phimin: float
        lower azimuth angle for neutrino arrival direction
    phimax: float
         upper azimuth angle for neutrino arrival direction
    start_event: int
        default: 1
        event number of first event
    flavor: array of ints
        default: [12, -12, 14, -14, 16, -16]
        specify which neutrino flavors to generate. A uniform distribution of
        all specified flavors is assumed.
        The neutrino flavor (integer) encoded as using PDF numbering scheme,
        particles have positive sign, anti-particles have negative sign,
        relevant for us are:
        * 12: electron neutrino
        * 14: muon neutrino
        * 16: tau neutrino
    n_events_per_file: int or None
        the maximum number of events per output files. Default is None, which
        means that all events are saved in one file. If 'n_events_per_file' is
        smaller than 'n_events' the event list is split up into multiple files.
        This is useful to split up the computing on multiple cores.
    spectrum: string
        defines the probability distribution for which the neutrino energies are generated
        * 'log_uniform': uniformly distributed in the logarithm of energy
    start_file_id: int (default 0)
        in case the data set is distributed over several files, this number specifies the id of the first file
        (useful if an existing data set is extended)
        if True, generate deposited energies instead of primary neutrino energies
    """

    attributes = {}
    n_events = int(n_events)

    # save current NuRadioMC version as attribute
    # save NuRadioMC and NuRadioReco versions
    attributes['NuRadioMC_EvtGen_version'] = NuRadioMC.__version__
    attributes['NuRadioMC_EvtGen_version_hash'] = version.get_NuRadioMC_commit_hash()

    attributes['start_event_id'] = start_event_id

    attributes['fiducial_rmin'] = full_rmin
    attributes['fiducial_rmax'] = full_rmax
    attributes['fiducial_zmin'] = full_zmin
    attributes['fiducial_zmax'] = full_zmax
    attributes['rmin'] = full_rmin
    attributes['rmax'] = full_rmax
    attributes['zmin'] = full_zmin
    attributes['zmax'] = full_zmax
    attributes['flavors'] = flavor
    attributes['Emin'] = Emin
    attributes['Emax'] = Emax
    attributes['thetamin'] = thetamin
    attributes['thetamax'] = thetamax
    attributes['phimin'] = phimin
    attributes['phimax'] = phimax

    # define cylinder by two points and the radius
    h_cylinder = full_zmax - full_zmin
    r_cylinder = full_rmax
    pt1 = np.array([0, 0, R_earth + full_zmax])
    pt2 = np.array([0, 0, R_earth + full_zmin])

    data_sets = {}

    # calculate maximum width of projected area
    theta_max = np.arctan(h_cylinder / 2 / r_cylinder)
    d = 2 * r_cylinder * np.cos(theta_max) + h_cylinder * np.sin(theta_max)  # width of area

    print(f"cylinder r = {r_cylinder/units.km:.1f}km, h = {h_cylinder/units.km:.1f}km -> dmax = {d/units.km:.1f}km")

    def perp(a) :
        b = np.empty_like(a)
        b[0] = -a[1]
        b[1] = a[0]
        return b

    # line segment a given by endpoints a1, a2
    # line segment b given by endpoints b1, b2
    # return
    def seg_intersect(a1, a2, b1, b2) :
        da = a2 - a1
        db = b2 - b1
        dp = a1 - b1
        dap = perp(da)
        denom = np.dot(dap, db)
        num = np.dot(dap, dp)
        return (num / denom) * db + b1

    def get_R(t, v, X):
        """"
        calculate distance to center of Earth as a function of travel distance
        
        Parameters
        -----------
        t: 3dim array
            travel distance
        v: 3dim array
            direction
        X: 3dim array
            start point
        """
        return np.linalg.norm(v * t + X)

    def get_density(t, v, X):
        """
        calculates density as a function of travel distance
        
        Parameters
        -----------
        t: 3dim array
            travel distance
        v: 3dim array
            direction
        X: 3dim array
            start point
        """
        return  earth.density(get_R(t, v, X))

    def slant_depth(t, v, X):
        """
        calculates slant depth (grammage) as a function of travel distance
        
        Parameters
        -----------
        t: 3dim array
            travel distance
        v: 3dim array
            direction
        X: 3dim array
            start point
        """
        res = quad(get_density, 0, t, args=(v, X), limit=200)
        return res[0]

    def slant_depth_num(t, v, X, step=50 * units.m):
        """
        calculates slant depth (grammage) as a function of travel distance
        
        Parameters
        -----------
        t: 3dim array
            travel distance
        v: 3dim array
            direction
        X: 3dim array
            start point
        """
        tt = np.linspace(0, t, t / step)
        rr = np.linalg.norm(X + np.outer(tt, v), axis=1)
        res = np.trapz(earth.density(rr), tt)
        return res

    def obj_dist_to_surface(t, v, X):
        return get_R(t, v, X) - R_earth

    def obj(t, v, X, Lint):
        """
        objective function to determine at which travel distance we reached the interaction point
        """
        return slant_depth(t, v, X) - Lint

    def points_in_cylinder(pt1, pt2, r, q):
        """
        determines if point lies within a cylinder
        
        Parameters
        -----------
        pt1: 3dim array
            lowest point on cylinder axis
        pt2: 3dim array
            highest point on cylinder axis
        r: float
            radius of cylinder
        q: 3dim array
            point under test
        
        Returns True/False
        """
        vec = pt2 - pt1
        const = r * np.linalg.norm(vec)
        return len(np.where(np.dot(q - pt1, vec) >= 0 and np.dot(q - pt2, vec) <= 0 and np.linalg.norm(np.cross(q - pt1, vec)) <= const)[0]) > 0

    # precalculate the maximum slant depth to the detector
    if(not os.path.exists("buffer_Llimit.pkl")):
        zens = np.arange(0, 180.1 * units.deg, 2 * units.deg)
        Lint_max = np.zeros_like(zens)
        Lint_min = np.zeros_like(zens)
        Xs = np.array([[0, -0.5 * r_cylinder, -h_cylinder + R_earth],
                       [0, -0.5 * r_cylinder, R_earth],
                       [0, 0.5 * r_cylinder, -h_cylinder + R_earth],
                       [0, 0.5 * r_cylinder, R_earth]]
                       )
        v_tmps = -hp.spherical_to_cartesian(zens, np.zeros_like(zens))  # neutrino direction
        for i in range(len(v_tmps)):
            v = v_tmps[i]
            sdepth_tmp = np.zeros(4)
            for j, X in enumerate(Xs):
                if((X[2] == R_earth) and (zens[i] <= 90 * units.deg)):
                    t = 0
                    sdepth_tmp[j] = 0
                else:
                    t = brentq(obj_dist_to_surface, 100, 2 * R_earth, args=(-v, X))
                    sdepth_tmp[j] = slant_depth_num(t, -v, X)
        #         print(i, zens[i] / units.deg, X, sdepth_tmp[j])
        #     enter_point = X + (-v * t)
            Lint_max[i] = np.max(sdepth_tmp)
            Lint_min[i] = np.min(sdepth_tmp)
        pickle.dump([zens, Lint_max, Lint_min], open("buffer_Llimit.pkl", "wb"), protocol=4)
    else:
        zens, Lint_max, Lint_min = pickle.load(open("buffer_Llimit.pkl", "rb"))

    get_Lmax = interp1d(zens, Lint_max, kind='next')
    get_Lmin = interp1d(zens, Lint_min, kind='previous')

    if 0:
        fig, a = plt.subplots(1, 1)
        ztmp = np.linspace(0, 180 * units.deg, 10000)
        a.plot(ztmp / units.deg, get_Lmax(ztmp) / units.g * units.cm ** 2, 'C0-', label="max possible Lint")
    #     a.plot(zens / units.deg, Lint_max / units.g * units.cm ** 2, 'oC0')
        a.plot(ztmp / units.deg, get_Lmin(ztmp) / units.g * units.cm ** 2, 'C1-', label="min possible Lint")
    #     a.plot(zens / units.deg, Lint_min / units.g * units.cm ** 2, 'dC1')
        a.hlines(cs.get_interaction_length(.1 * units.EeV, 1, 12, "total") / units.g * units.cm ** 2, 0, 180, label="0.1 EeV", colors='C2')
        a.hlines(cs.get_interaction_length(1 * units.EeV, 1, 12, "total") / units.g * units.cm ** 2, 0, 180 , label="1 EeV", colors='C3')
        a.hlines(cs.get_interaction_length(10 * units.EeV, 1, 12, "total") / units.g * units.cm ** 2, 0, 180 , label="10 EeV", colors='C4')
        a.set_xlabel("zenith angle [deg]")
        a.set_ylabel("slant depth [g/cm^2]")
        a.semilogy(True)
        a.set_xticks(np.arange(0, 181, 10))
        a.set_ylim(5e5)
        a.legend()

        fig.tight_layout()
        fig.savefig("Lvszen.png")
        plt.show()

    failed = 0
    if(spectrum == 'log_uniform'):
        Enu = 10 ** np.random.uniform(np.log10(Emin), np.log10(Emax), n_events)
    flavors = np.array([flavor[i] for i in np.random.randint(0, high=len(flavor), size=n_events)])
    az = np.random.uniform(phimin, phimax, n_events)
    zen = np.arccos(np.random.uniform(np.cos(thetamax), np.cos(thetamin), n_events))
    # generate random positions on an area perpendicular do neutrino direction
    ax, ay = np.random.uniform(-0.5 * d, 0.5 * d, (2, n_events))
    # az = np.ones(n_events) * (R_earth - .5 * h_cylinder)  # move plane to the center of the cylinder

    # calculate grammage (g/cm^2) after which neutrino interacted
    Lint = np.random.exponential(cs.get_interaction_length(Enu, 1, flavors, "total"), n_events)

    mask = (Lint < get_Lmax(zen)) & (Lint > get_Lmin(zen))
    print(f"{np.sum(mask)}/{n_events} = {np.sum(mask)/n_events:.2g} can potentially interact in simulation volume")

    # calculate position where neutrino interacts
    data_sets = {'xx': [],
                'yy': [],
                'zz': [],
                'azimuths': [],
                'zeniths': [],
                'flavors': [],
                'energies': []}
    # calculate rotation matrix to transform position on area to 3D
    mask_int = np.zeros_like(mask, dtype=np.bool)
    t0 = time.perf_counter()
    n_cylinder = 0
    for j, i in enumerate(np.arange(n_events, dtype=np.int)[mask]):
        if(j % 1000 == 0 and i > 0):
            eta = (time.perf_counter() - t0) * (n_events - i) / i
            logger.info(f"{i}/{n_events} interacting = {np.sum(mask_int)}, failed = {failed}, n_cylinder = {n_cylinder}, eta = {pretty_time_delta(eta)}")
    #     print(f"calculating interaction point of event {i}"),
        c, s = np.cos(az[i]), np.sin(az[i])
        Raz = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
        c, s = np.cos(zen[i]), np.sin(zen[i])
    #     Rzen = np.array(((c, 1, -s), (0, 1, 0), (s, 0, c)))
        Rzen = hp.get_rotation(hp.spherical_to_cartesian(0, az[i]), hp.spherical_to_cartesian(zen[i], az[i]))
    #     R = hp.get_rotation(np.array([0, 0, 1]), hp.spherical_to_cartesian(zen[i], az[i]))
        R = np.matmul(Rzen, Raz)
        v = -hp.spherical_to_cartesian(zen[i], az[i])  # neutrino direction
        X = np.matmul(R, np.array([ax[i], ay[i], 0])) + np.array([0, 0, -0.5 * h_cylinder])
        if 0:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from mpl_toolkits.mplot3d.art3d import Poly3DCollection
            fig = plt.figure()
            a = fig.add_subplot(111, projection='3d')
            # Cylinder
            x = np.linspace(-r_cylinder, r_cylinder, 100)
            z = np.linspace(0, -h_cylinder, 100)
            Xc, Zc = np.meshgrid(x, z)
            Yc = np.sqrt(r_cylinder ** 2 - Xc ** 2)

            # Draw parameters
            rstride = 20
            cstride = 10
            a.plot_surface(Xc, Yc, Zc, alpha=0.2, rstride=rstride, cstride=cstride)
            a.plot_surface(Xc, -Yc, Zc, alpha=0.2, rstride=rstride, cstride=cstride)
            a.set_title(f"zenith = {zen[i]/units.deg:.0f}")
            xx = []
            yy = []
            zz = []
            for vert in np.array([[-0.5 * d, -0.5 * d, 0], [0.5 * d, -0.5 * d, 0], [0.5 * d, 0.5 * d, 0], [-0.5 * d, 0.5 * d, 0]]):
                t = np.matmul(Raz, vert) + np.array([0, 0, -0.5 * h_cylinder])
                xx.append(t[0])
                yy.append(t[1])
                zz.append(t[2])
            verts = [list(zip(xx, yy, zz))]
            a.add_collection3d(Poly3DCollection(verts, alpha=0.5))

            xx = []
            yy = []
            zz = []
            for vert in np.array([[-0.5 * d, -0.5 * d, 0], [0.5 * d, -0.5 * d, 0], [0.5 * d, 0.5 * d, 0], [-0.5 * d, 0.5 * d, 0]]):
                t = np.matmul(Rzen, np.matmul(Raz, vert)) + np.array([0, 0, -0.5 * h_cylinder])
                xx.append(t[0])
                yy.append(t[1])
                zz.append(t[2])
            verts = [list(zip(xx, yy, zz))]
            a.add_collection3d(Poly3DCollection(verts, alpha=0.5))

            s = np.array([0, 0, -0.5 * h_cylinder])
            t = v * 10 * units.km + s
            a.plot([s[0], t[0]], [s[1], t[1]], [s[2], t[2]], '-d')

            s = np.array([0, 0, -0.5 * h_cylinder])
            t = -hp.spherical_to_cartesian(90 * units.deg, az[i]) * 10 * units.km + s
            a.plot([s[0], t[0]], [s[1], t[1]], [s[2], t[2]], '--d')

            # check if neutrino axis is perpendicular
            t = np.array(verts[0][0]) - np.array(verts[0][1])
            t /= np.linalg.norm(t)
            print(np.dot(t, v))

            a.set_xlabel("x")
            a.set_zlabel("z")
            a.set_ylabel("y")
            plt.show()

        # check if trajectory passes through cylinder
    #     if(not points_in_cylinder(pt1, pt2, r_cylinder, X)):
        # we rotate everything in the plane defined by z and the propagration direction (such that v_y = 0)
        Xaz = np.matmul(Raz.T, X)
        rmin = Xaz[1]  # the closest distance to the z axis (center of cyllinder)
        if(abs(rmin) >= r_cylinder):
            continue
        # define the projected square of the cylinder
        # the two endpoints of the two horizontal lines are
        xtmp = (r_cylinder ** 2 - rmin ** 2) ** 0.5
        Lh1 = np.array([[-xtmp, 0], [xtmp, 0]])
        Lh2 = np.array([[-xtmp, -h_cylinder], [xtmp, -h_cylinder]])
        # the two endpoints of the two vertical lines are
        Lv1 = np.array([[-xtmp, 0], [-xtmp, -h_cylinder]])
        Lv2 = np.array([[xtmp, 0], [xtmp, -h_cylinder]])

        # define line of neutrino propagation by two points
        vaz = np.matmul(Raz.T, v)
        if(abs(vaz[1]) > 1e-10):
            a = 1 / 0
        v2d = np.array([vaz[0], vaz[2]])
        X2d = np.array([Xaz[0], Xaz[2]])
        t = 2 * d
        Paz = np.array([X2d + -t * v2d, X2d + t * v2d])

        # calculate points that intersect any of the 4 area (projected cylinder) boundaries
        intersects = []
        for k, (a1, a2) in enumerate([Lh1, Lh2]):
            tmp = seg_intersect(a1, a2, Paz[0], Paz[1])
            if((tmp[0] >= a1[0]) and (tmp[0] <= a2[0])):
                intersects.append(tmp)
        for k, (a1, a2) in enumerate([Lv1, Lv2]):
            tmp = seg_intersect(a1, a2, Paz[0], Paz[1])
            if((tmp[1] <= a1[1]) and (tmp[1] >= a2[1])):
                intersects.append(tmp)
        intersects = np.array(intersects)
        if(len(intersects) != 2):
            if 0:
                print(len(intersects))
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure()
                a = fig.add_subplot(111, projection='3d')
                # Cylinder
                x = np.linspace(-r_cylinder, r_cylinder, 100)
                z = np.linspace(0, -h_cylinder, 100)
                Xc, Zc = np.meshgrid(x, z)
                Yc = np.sqrt(r_cylinder ** 2 - Xc ** 2)

                # Draw parameters
                rstride = 20
                cstride = 10
                a.plot_surface(Xc, Yc, Zc, alpha=0.2, rstride=rstride, cstride=cstride)
                a.plot_surface(Xc, -Yc, Zc, alpha=0.2, rstride=rstride, cstride=cstride)
                X_enter = X + 10 * units.km * v
                X_leave = X - 10 * units.km * v
                a.plot([X_enter[0], X_leave[0]], [X_enter[1], X_leave[1]], [X_enter[2], X_leave[2]], '-o')

                a.set_xlabel("x")
                a.set_zlabel("z")
                a.set_ylabel("y")
                a.legend()
                a.set_title("no intersection")

                fig = plt.figure()
                a = fig.add_subplot(111, projection='3d')
                for (a1, a2) in [Lh1, Lh2, Lv1, Lv2]:
                    a.plot([a1[0], a2[0]], [rmin, rmin], [a1[1], a2[1]])
                a.plot([Paz[0][0], Paz[1][0]], [rmin, rmin], [Paz[0][1], Paz[1][1]])
                a.set_xlabel("x")
                a.set_zlabel("z")
                a.set_ylabel("y")
                a.legend()
                plt.show()
            continue  # neutrino is not passing through cylinder
        n_cylinder += 1
        ss = []
        for tmp in intersects:
            ss.append(np.dot(tmp - X2d, v2d.T))
        argsort = np.argsort(np.array(ss))  # check which intersection happens first along neutrino path
        if 0:
            if(len(intersects)):
                import matplotlib.pyplot as plt
                from mpl_toolkits.mplot3d import Axes3D
                fig = plt.figure()
                a = fig.add_subplot(111, projection='3d')
                for (a1, a2) in [Lh1, Lh2, Lv1, Lv2]:
                    a.plot([a1[0], a2[0]], [rmin, rmin], [a1[1], a2[1]])
                a.plot([Paz[0][0], Paz[1][0]], [rmin, rmin], [Paz[0][1], Paz[1][1]])
                for k, tmp in enumerate(intersects):
                    a.plot([tmp[0]], [rmin], [tmp[1]], 'o', label=f"s = {ss[k]:.0f}")
                a.set_xlabel("x")
                a.set_zlabel("z")
                a.set_ylabel("y")
                a.legend()
    #             plt.ion()
                plt.show()

        # calculate the 3D points where the neutrino enters/leaves the cylinder and transform to outside Earth
        X_enter = np.matmul(Raz, np.array([intersects[argsort][0][0], rmin, intersects[argsort][0][1]])) + np.array([0, 0, R_earth])
        X_leave = np.matmul(Raz, np.array([intersects[argsort][1][0], rmin, intersects[argsort][1][1]])) + np.array([0, 0, R_earth])
        X += np.array([0, 0, R_earth])

        # calculate point where neutrino enters Earth
        if(np.linalg.norm(X_enter) > R_earth):  # if enter point is outside of Earth (can happen because cylinder does not account for Earth curvature)
            if(np.linalg.norm(X_leave) > R_earth):  # check if leave point is also outside of Earth (can also happen because cylinder does not account for Earth curvature)
                continue
            t = brentq(obj_dist_to_surface, 0, 5 * d, args=(-v, X_leave))
            enter_point = X_leave + (-v * t)
            X_enter = enter_point  # define point where neutrino enters the cylinder as the point where it enters the Earth
        else:
            t = brentq(obj_dist_to_surface, 0, 2 * R_earth, args=(-v, X_enter))
            enter_point = X_enter + (-v * t)
    #     logger.debug(f"zen = {zen[i]/units.deg:.0f}deg, trajectory enters Earth at {enter_point[0]:.1f}, {enter_point[0]:.1f}, {enter_point[0]:.1f}. Dist to core = {np.linalg.norm(enter_point)/R_earth:.5f}, dist to (0,0,R) = {np.linalg.norm(enter_point - np.array([0,0,R_earth]))/R_earth:.4f}")

        # check if event interacts at all
        # calcualte slant depth to point of entering cylinder
        t = np.linalg.norm(enter_point - X_enter)
        slant_depth_min = slant_depth(t, v, enter_point)
        if(t == 0):
            slant_depth_min = 0
        # calculate slant depth through the cylinder
        s = np.linalg.norm(X_leave - X_enter)
        slant_depth_max = slant_depth(s, v, X_enter) + slant_depth_min  # full slant depth from outside Earth to point when it leaves the cylinder

        if 0:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            fig = plt.figure()
            a = fig.add_subplot(111, projection='3d')
            # Cylinder
            x = np.linspace(-r_cylinder, r_cylinder, 100)
            z = np.linspace(0, -h_cylinder, 100)
            Xc, Zc = np.meshgrid(x, z)
            Yc = np.sqrt(r_cylinder ** 2 - Xc ** 2)

            # Draw parameters
            rstride = 20
            cstride = 10
            a.plot_surface(Xc, Yc, Zc, alpha=0.2, rstride=rstride, cstride=cstride)
            a.plot_surface(Xc, -Yc, Zc, alpha=0.2, rstride=rstride, cstride=cstride)
            a.plot([X_enter[0], X_leave[0]], [X_enter[1], X_leave[1]], [X_enter[2] - R_earth, X_leave[2] - R_earth], '-o')

            a.set_xlabel("x")
            a.set_zlabel("z")
            a.set_ylabel("y")
            print(f"Lmin = {slant_depth_min:.2g}, Lmax = {slant_depth_max:.2g}, Lnu = {Lint[i]:.2g}")
            a.legend()
            plt.show()
    #         a = 1 / 0
        if((Lint[i] <= slant_depth_min) or (Lint[i] >= slant_depth_max)):
            logger.debug("neutrino does not interact in cylinder, skipping to next event")
            continue

        try:
            # calculate interaction point by inegrating the density of Earth along the neutrino path until we wind the interaction length
            t = brentq(obj, 0, s, args=(v, X_enter, Lint[i] - slant_depth_min), maxiter=500)
        except:
            logger.warning("failed to converge, skipping event")
            failed += 1
            continue
        Xint = X_enter + v * t  # calculate interaction point

        is_in_cylinder = points_in_cylinder(pt1, pt2, r_cylinder, Xint)
        mask_int[i] = is_in_cylinder
        if(is_in_cylinder):
            logger.debug(f"event {i}, interaction point ({Xint[0]:.1f}, {Xint[1]:.1f}, {Xint[2]-R_earth:.1f}), in cylinder {is_in_cylinder}")
            data_sets['xx'].append(Xint[0])
            data_sets['yy'].append(Xint[1])
            data_sets['zz'].append(Xint[2] - R_earth)
            data_sets['zeniths'].append(zen[i])
            data_sets['azimuths'].append(az[i])
            data_sets['flavors'].append(flavors[i])
            data_sets['energies'].append(Enu[i])

        else:
            logger.error("interaction is not in cylinder but it should be")
            a = 1 / 0

    data_sets['event_ids'] = range(np.sum(mask_int))
    data_sets['flavors'] = np.ones(np.sum(mask_int))
    data_sets["event_ids"] = np.arange(np.sum(mask_int)) + start_event_id
    data_sets["n_interaction"] = np.ones(np.sum(mask_int), dtype=np.int)
    data_sets["vertex_times"] = np.zeros(np.sum(mask_int), dtype=np.float)

    data_sets["interaction_type"] = inelasticities.get_ccnc(np.sum(mask_int))
    data_sets["inelasticity"] = inelasticities.get_neutrino_inelasticity(np.sum(mask_int))

    attributes['n_events'] = n_events

    logger.info(f"{np.sum(mask_int)} event interacted in simulation volume")

    write_events_to_hdf5(filename, data_sets, attributes,
                         n_events_per_file=n_events_per_file, start_file_id=start_file_id)


if __name__ == "__main__":
    generate_eventlist_cylinder("test2.hdf5", 1e4, 1e18 * units.eV, 1e18 * units.eV, full_rmin=0, full_rmax=5 * units.km,
                                full_zmin=-2.7 * units.km, full_zmax=0, thetamin=0, thetamax=180 * units.deg,
                                phimin=0, phimax=360 * units.deg, start_event_id=0)
