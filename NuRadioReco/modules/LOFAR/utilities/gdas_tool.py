#!/usr/bin/env python
# Be ready for python 3, supported since python 2.6
from __future__ import print_function, division, unicode_literals # gdastool requires at least python version 2.7
import sys
if sys.version_info < (2,7):
    print("You are using python version: ")
    print(" ", sys.version)
    print("gdastool supports python versions 2.7 or later.")
    sys.exit(-1)

from argparse import ArgumentParser, RawTextHelpFormatter
import subprocess
import logging
import os.path
import struct

try:
    import numpy as np
    from scipy.optimize import curve_fit
    from scipy.interpolate import InterpolatedUnivariateSpline
except ImportError as E:
    print("\n!!! ERROR: gdastool requires the scipy and nump packages.\n")
    raise E

try:
    import matplotlib.pyplot as plt
    matplotlibAvailable = True
except:
    matplotlibAvailable = False

# predefined observatory locations
observatories = {'lofar': {'latitude': 52.9, 'longitude': 6.9},
                 'tunka-rex': {'latitude': 51.8, 'longitude': 103.1},
                 'ska-low': {'latitude': -26.7, 'longitude': 116.6},
                 'grand': {'latitude': 42.9, 'longitude': 86.7},
                 'aera': {'latitude': -35.2, 'longitude': -69.3}}

# parsing inputs
parser = ArgumentParser(description='''Creates an atmosphere profile for CORSIKA/CoREAS from GDAS data.

Downloads GDAS model data for the defined location and time and fits
a 5-layer model of the atmosphere to the data. Based on the fit, a
table for the refractive index is created for usage in CoREAS.''',
        epilog='''Please contact
  Pragati Mitra <pragati9163@gmail.com>,
  Arthur Corstanje <a.corstanje@astro.ru.nl> or
  Tobias Winchen <tobias.winchen@rwth-aachen.de>
in case of questions or bugs.''',
        formatter_class=RawTextHelpFormatter)
parser.add_argument("-t", "--utctimestamp", help="UTC time stamp of the event " )
parser.add_argument("-o", "--output", default="ATMOSPHERE.DAT", help="Name of the outputfile. Default is ATMOSPHERE.DAT")
cogroup = parser.add_mutually_exclusive_group(required=True)
cogroup.add_argument('--observatory', type=str, choices=observatories.keys(),
        help="Preset of observatory coordinates.")
cogroup.add_argument('-c', '--coordinates', nargs=2, type=float, help='Coordinates of the observatory lat=-90..90 lon=0..360 in deg, e.g. --coordinates 50.85 4.25 for Brussels.')
parser.add_argument('-m', '--minheight', type=float, default=-1E3, help='Mimimum height for the interpolation. Default is -1.0 km.')
parser.add_argument('-M', '--maxheight', type=float, help='Maximum height for the refractivity interpolation. Default is GDAS-provided maximum height.')
parser.add_argument('-s', '--interpolationSteps', default=1.0, type=float, help='Step length for interpolation. Default is 1 m. ')

parser.add_argument("-p", "--gdaspath", default='.', help="Path to local gdas file directory. If required file is not there, it will be downloaded.")
parser.add_argument("--cleanup", action='store_true', help="Delete the large GDAS file after the atmosphere profile is generated.")
parser.add_argument('-v', '--verbose', default=0, action='count', help='Set log level, -vv for debug.')
if matplotlibAvailable:
    parser.add_argument('-g', '--createplot', default=False, action='store_true', help='plot density profile.')


def generate_atmosphere(utctimestamp, output="ATMOSPHERE.DAT", observatory=None, coordinates=None, 
                        minheight=-1E3, maxheight=None, interpolationSteps=1.0, 
                        gdaspath='.', verbose=0, createplot=False, cleanup=False):
    """
    Creates an atmosphere profile for CORSIKA/CoREAS from GDAS data.
    """
    from datetime import datetime, timedelta
    import dateutil.parser

    utcstring = str(datetime.utcfromtimestamp(int(utctimestamp)))
    try:
        date_time = dateutil.parser.parse(utcstring)
    except Exception as e:
        print(f'ERROR: Cannot parse date from timestamp {utctimestamp}: {e}')
        return False

    if observatory in observatories:
        coordinates = [observatories[observatory]['latitude'],
                       observatories[observatory]['longitude']]
    
    if coordinates is None:
        print("ERROR: No coordinates or observatory provided.")
        return False

    print('')
    print('    Coordinates: lat = {:+.2f} deg, lon = {:+.2f} deg'.format(coordinates[0], coordinates[1]))
    print('     Time [UTC]: {}'.format(date_time.ctime()))

    # Round-off of time to nearest 3-hour time
    seconds = date_time.hour * 3600 + date_time.minute * 60 + date_time.second
    delta_seconds = round(seconds / (3.0*3600)) * 3*3600 - seconds
    delta_seconds = int(delta_seconds) # Number of seconds to add to get onto a 3-hour grid
    date_time = date_time + timedelta(seconds=delta_seconds) # Using this date&time in what follows
    print(' ')
    print('     Time on 3-hour grid [UTC]: {}'.format(date_time.ctime()))

    month_name = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
    part = 5
    if (date_time.day < 29):
        part -= 1
    if (date_time.day < 22):
        part -= 1
    if (date_time.day < 15):
        part -= 1
    if (date_time.day < 8):
        part -= 1

    year_gdas = str(date_time.year)
    gdasname = "gdas1.{0}{1}.w{2}".format(month_name[int(date_time.month) - 1], year_gdas[2:], part)
    print('Using GDAS file:', gdasname)

    gdaspath_abs = os.path.abspath(gdaspath)
    if not os.path.isdir(gdaspath_abs):
        logging.info("Creating nonexisting local gdas directory {}.".format(gdaspath_abs))
        os.makedirs(gdaspath_abs, exist_ok=True)

    if os.path.isfile(os.path.join(gdaspath_abs, gdasname)):
        print("Found {} in {}, no download.".format(gdasname, gdaspath_abs))
    else:
        gdasurl = 'ftp://arlftp.arlhq.noaa.gov/pub/archives/gdas1'
        print("File not found in {}, download from {}".format(gdaspath_abs, gdasurl))
        cmd = ['wget', '--directory-prefix={}'.format(gdaspath_abs), '{}/{}'.format(gdasurl, gdasname)]
        v = subprocess.call(cmd)
        if not v == 0:
            print('ERROR DOWNLOADING {}/{} -- ABORTING!'.format(gdasurl, gdasname))
            return False

    def parseGDAS_File_local(gdaspath, gdasname, date_time, coordinates):
        time_gdas = int(date_time.hour)
        day_gdas = int(date_time.day)
        month_gdas = int(date_time.month)
        lat = 90 + int(round(coordinates[0]))
        lon = int(round(coordinates[1]))

        altitude = np.zeros([24])
        temp = np.zeros([24])
        relh = np.zeros([24])
        pressure = np.array([0, 1000, 975, 950, 925, 900, 850, 800, 750, 700,
                            650, 600, 550, 500, 450, 400, 350, 300, 250, 200, 150, 100, 50, 20])
        h1 = int(np.floor(int(time_gdas) / 3.0) * 3)
        h2 = (h1 % 2) * 3
        string_id = str(int(year_gdas[2:])).rjust(2) + str(month_gdas).rjust(2) + str(day_gdas).rjust(2) + str(h1).rjust(2) + str(h2).rjust(2)
        logging.debug("string id = {}".format(string_id))
        f = open(os.path.join(gdaspath, gdasname), 'rb')
        s = f.read().decode('latin_1')
        f.close()
        skip = 0
        start = skip + s[skip:].find("INDX")
        block = s[start:start + 2000]
        nx, ny, _ = int(block[129:132]), int(block[132:135]), int(block[136:138])
        nxy = nx * ny

        tmp_data = np.zeros([ny, nx])
        skip = 0
        found = True

        for j in range(250):
            if j % 25 == 0:
                print('\rParsing gdas file: [{:10}]'.format(''.join(['.' for i in range(j // 25 + 1)])), end='')
                sys.stdout.flush()
            next = s[skip:].find(str(string_id))
            if (next < 0):
                found = False
            start = skip + next
            block = s[start:start + 1000]
            if (found):
                lvl, keyword, nexp, precision, value = int(block[10:12]), block[
                    14:18], int(block[20:23]), float(block[23:36]), float(block[36:50])
                scale = 2.0 ** (7 - nexp)
                if (keyword == 'PRSS' or keyword == 'RH2M' or keyword == 'SHGT' or keyword == 'T02M'
                   or keyword == 'HGTS' or keyword == 'TEMP' or keyword == 'RELH'):
                    datablock = bytes(s[start + 50:start + nxy + 50].encode('latin_1'))
                    diffs = (
                        (np.array([struct.unpack('65160B', datablock)]) - 127) / scale)[0]
                    vold = value
                    indx = 0
                    for k in np.arange(0, ny):
                        for l in np.arange(0, nx):
                            tmp_data[k, l] = vold + diffs[indx]
                            indx += 1
                            vold = tmp_data[k, l]
                        vold = tmp_data[k, 0]
                if (keyword == 'PRSS'):
                    pressure[0] = tmp_data[lat, lon]
                elif (keyword == 'SHGT'):
                    altitude[0] = tmp_data[lat, lon]
                elif (keyword == 'RH2M'):
                    relh[0] = tmp_data[lat, lon]
                elif (keyword == 'T02M'):
                    temp[0] = tmp_data[lat, lon]
                elif (keyword == 'HGTS'):
                    altitude[lvl] = tmp_data[lat, lon]
                elif (keyword == 'TEMP'):
                    temp[lvl] = tmp_data[lat, lon]
                elif (keyword == 'RELH'):
                    relh[lvl] = tmp_data[lat, lon]
                skip = start + 100
        print('')

        def geopot_to_geometric(lat,h):
            z = (1+0.002644*np.cos(2*lat*np.pi/ 180. )) * h + (1+0.0089*np.cos(2*lat*np.pi/180.)) * (h * h / 6245000.)
            return z

        altitude = geopot_to_geometric(lat-90, altitude)
        pressure = 100 * pressure
        tempC = temp - 273.15
        part_press = np.zeros([24])
        for j in np.arange(24):
            if (tempC[j] < 0):
                part_press[j] = (relh[j] / 100.) * 100 * 6.1064 * \
                    np.exp(21.88 * tempC[j] / (265.5 + tempC[j]))
            else:
                part_press[j] = (relh[j] / 100.) * 100 * 6.1070 * \
                    np.exp(17.15 * tempC[j] / (234.9 + tempC[j]))

        M_dry, M_water, M_CO2 = 0.02897, 0.01802, 0.04401
        phi_CO2 = 385.0 * 1e-6
        phi_water = part_press / pressure
        phi_dry = 1 - phi_water - phi_CO2
        M_air = phi_dry * M_dry + phi_water * M_water + phi_CO2 * M_CO2
        density = (pressure * M_air / temp / 8.31451) / 1000
        pressure_dry = pressure - part_press
        RI = (77.689 * (pressure_dry / temp) + 71.2952 * (part_press / temp) + 375463 * (part_press / (temp * temp))) / 100

        alt_ground, RI_ground, density_ground = altitude[0], RI[0], density[0]
        alt_max_local = maxheight if maxheight is not None else altitude[23]
        alt_values = np.arange(alt_ground, alt_max_local, interpolationSteps)
        try:
            idx = altitude.argsort()
            interpolation = InterpolatedUnivariateSpline(altitude[idx], np.log(RI[idx]), k=1)
        except Exception as e:
            print('Error in interpolation of values')
            raise e
        refractiveIndex = (np.exp(interpolation(alt_values)) * 1e-6 + 1)
        return altitude, alt_ground, alt_values, RI_ground, refractiveIndex, density, density_ground

    altitude, alt_ground, alt_values, RI_ground, refractiveIndex, density, density_ground = parseGDAS_File_local(gdaspath_abs, gdasname, date_time, coordinates)

    def fn_atm_depth(x, par1, par2, par3):
        return par1 + par2 * np.exp(-1e5 * x / par3)

    def a_from_atmdep(atmdep, x, b, c):
        return atmdep - b * np.exp(-1e5 * x / c)

    def Density(x, b, c):
        return b / c * np.exp(-1e5 * x / c)

    def find_par_b(rho, c, x0):
        return rho * c * np.exp(1e5 * x0 / c)

    def fit_lay(x, rho, x0, c):
        return Density(x, find_par_b(rho, c, x0), c)

    def rms(x):
        return np.sqrt(np.dot(x,x) / len(x))

    boun1, boun2 = 10, 17
    altitude = altitude / 1000.  # to km
    alt_bc1, alt_bc2, alt_bc3 = altitude[boun1], altitude[boun2], altitude[23]
    x_lay1, x_lay2, x_lay3 = altitude[:boun1+1], altitude[boun1:boun2+1], altitude[boun2:]
    x_lay4 = [alt_bc3, 100]
    den_lay1, den_lay2, den_lay3 = density[:boun1+1], density[boun1:boun2+1], density[boun2:]
    den_lay4 = [density[23], 1e-09]

    mat1 = np.array([[altitude[1], 1], [altitude[boun1], 1]])
    cons1 = [np.log(density[1]), np.log(density[boun1])]
    ans1 = np.linalg.solve(mat1, cons1)
    mat2 = np.array([[altitude[boun1], 1], [altitude[boun2], 1]])
    cons2 = [np.log(density[boun1]), np.log(density[boun2])]
    ans2 = np.linalg.solve(mat2, cons2)
    mat3 = np.array([[altitude[boun2], 1], [altitude[23], 1]])
    cons3 = [np.log(density[boun2]), np.log(density[23])]
    ans3 = np.linalg.solve(mat3, cons3)
    mat4 = np.array([[altitude[23], 1], [100, 1]])
    cons4 = [np.log(density[23]), np.log(1e-09)]
    ans4 = np.linalg.solve(mat4, cons4)

    A1, A2, A3, A4 = ans1[0], ans2[0], ans3[0], ans4[0]
    B1 = np.log(density[boun1]) - A1 * alt_bc1
    C1, C2, C3, C4 = -1e5 / A1, -1e5 / A2, -1e5 / A3, -1e5 / A4
    b1 = C1 * np.exp(B1)
    coeff1, _ = curve_fit(Density, np.asarray(x_lay1), np.asarray(den_lay1), np.array([b1, C1]))
    b1_new, c1_new = coeff1[0], coeff1[1]
    den_bc1 = Density(alt_bc1, b1_new, c1_new)

    def fit_lay2(x, c): return fit_lay(x, den_bc1, alt_bc1, c)
    coeff2, _ = curve_fit(fit_lay2, np.asarray(x_lay2), np.asarray(den_lay2), C2)
    c2_new = coeff2[0]
    b2_new = find_par_b(den_bc1, c2_new, alt_bc1)
    den_bc2 = Density(alt_bc2, b2_new, c2_new)

    def fit_lay3(x, c): return fit_lay(x, den_bc2, alt_bc2, c)
    coeff3, _ = curve_fit(fit_lay3, np.asarray(x_lay3), np.asarray(den_lay3), C3)
    c3_new = coeff3[0]
    b3_new = find_par_b(den_bc2, c3_new, alt_bc2)
    den_bc3 = Density(alt_bc3, b3_new, c3_new)

    def fit_lay4(x, c): return fit_lay(x, den_bc3, alt_bc3, c)
    coeff4, _ = curve_fit(fit_lay4, np.asarray(x_lay4), np.asarray(den_lay4), C4)
    c4_new = coeff4[0]
    b4_new = find_par_b(den_bc3, c4_new, alt_bc3)
    
    atmdep_bc3 = fn_atm_depth(alt_bc3, a_from_atmdep(0.01128292 - 1e-09 * 100.*100000., 100, b4_new, c4_new), b4_new, c4_new)
    a4 = a_from_atmdep(0.01128292 - 1e-09 * 100.*100000., 100, b4_new, c4_new)
    a3 = a_from_atmdep(atmdep_bc3, alt_bc3, b3_new, c3_new)
    atmdep_bc2 = fn_atm_depth(alt_bc2, a3, b3_new, c3_new)
    a2 = a_from_atmdep(atmdep_bc2, alt_bc2, b2_new, c2_new)
    atmdep_bc1 = fn_atm_depth(alt_bc1, a2, b2_new, c2_new)
    a1 = a_from_atmdep(atmdep_bc1, alt_bc1, b1_new, c1_new)
    
    a, b, c = [a1, a2, a3, a4, 0.01128292], [b1_new, b2_new, b3_new, b4_new, 1], [c1_new, c2_new, c3_new, c4_new, 1e9]

    if createplot and matplotlibAvailable:
        alt_simu_gdas = np.arange(0, altitude[23], 0.2)
        rho_gdas = [Density(h, b[min(int(np.searchsorted([altitude[boun1], altitude[boun2], altitude[23]], h)), 3)], 
                             c[min(int(np.searchsorted([altitude[boun1], altitude[boun2], altitude[23]], h)), 3)]) for h in alt_simu_gdas]
        plt.plot(altitude*1000, density, 'bo', label='data')
        plt.plot(alt_simu_gdas*1000, rho_gdas, 'r-', label='fit')
        plt.xlabel("altitude (m)"); plt.ylabel("air density (g/cm$^3$)"); plt.legend(); plt.show()

    lowest_alt = alt_ground - interpolationSteps * np.round((alt_ground - minheight) / interpolationSteps)
    alt_ext = np.arange(lowest_alt, alt_ground, interpolationSteps)
    refIndex_total = (RI_ground * Density(alt_ext/1000, b1_new, c1_new) / density_ground * 1e-6 + 1).tolist() + refractiveIndex.tolist()
    alt_total = alt_ext.tolist() + alt_values.tolist()

    with open(output, 'w') as f:
        f.write("# atmospheric parameters ATMLAY, A, B, C respectively\n")
        f.write('{: .8E} {: .8E} {: .8E} {: .8E} {: .8E}\n'.format(0, alt_bc1*1e5, alt_bc2*1e5, alt_bc3*1e5, 1e7))
        f.write('{: .8E} {: .8E} {: .8E} {: .8E} {: .8E}\n'.format(*a))
        f.write('{: .8E} {: .8E} {: .8E} {: .8E} {: .8E}\n'.format(*b))
        f.write('{: .8E} {: .8E} {: .8E} {: .8E} {: .8E}\n'.format(*c))
        f.write("# atmospheric height [m] and refractive index columns \n")
        for h, ri in zip(alt_total, refIndex_total):
            f.write("{: .8E} {: .15E}\n".format(h, ri))
    print('Output written to: {}'.format(output))
    
    if cleanup:
        gdasfile = os.path.join(gdaspath_abs, gdasname)
        if os.path.exists(gdasfile):
            try:
                os.remove(gdasfile)
                print(f"Removed large GDAS file: {gdasfile}")
            except Exception as e:
                print(f"Warning: Could not remove GDAS file {gdasfile}: {e}")
                
    return True

def find_or_generate_atmosphere(event_id, atmosphere_dir, gdas_cache_dir=None):
    """
    Return the path to a GDAS atmosphere file for a LOFAR event.

    Searches for ``ATMOSPHERE_{event_id}.DAT`` in `atmosphere_dir`. If the
    file does not exist it is generated by downloading the appropriate GDAS
    reanalysis file and fitting a 5-layer atmosphere model to it. The raw
    GDAS binary is stored in `gdas_cache_dir` (defaults to `atmosphere_dir`)
    and removed after the fit to keep disk usage low.

    Parameters
    ----------
    event_id : int
        LOFAR event ID (seconds elapsed since 2010-01-01 00:00:00 UTC).
    atmosphere_dir : str
        Directory that contains, or will receive, ``ATMOSPHERE_*.DAT`` files.
    gdas_cache_dir : str, optional
        Directory for temporary GDAS binary downloads. Defaults to
        `atmosphere_dir`.

    Returns
    -------
    str
        Absolute path to the atmosphere file.

    Raises
    ------
    RuntimeError
        If the file cannot be found and generation fails.
    """
    _logger = logging.getLogger("NuRadioReco.LOFAR.gdas_tool")

    atm_filename = f"ATMOSPHERE_{int(event_id)}.DAT"

    # Search the shared read-only directory first.
    shared_path = os.path.join(os.path.abspath(atmosphere_dir), atm_filename)
    if os.path.isfile(shared_path):
        _logger.info("Found atmosphere file: %s", shared_path)
        return shared_path

    _logger.info("Atmosphere file not found in %s; generating from GDAS data.", atmosphere_dir)

    # Write the generated file to gdas_cache_dir if provided, otherwise try
    # atmosphere_dir. This avoids failing when atmosphere_dir is read-only shared storage.
    unix_timestamp = int(event_id) + 1262304000
    if gdas_cache_dir is not None:
        write_dir = os.path.abspath(gdas_cache_dir)
    else:
        write_dir = os.path.abspath(atmosphere_dir)
    os.makedirs(write_dir, exist_ok=True)
    atm_path = os.path.join(write_dir, atm_filename)

    ok = generate_atmosphere(
        unix_timestamp,
        output=atm_path,
        observatory="lofar",
        gdaspath=write_dir,
        cleanup=True,
    )
    if not ok or not os.path.isfile(atm_path):
        raise RuntimeError(
            "GDAS atmosphere generation failed for event %d (unix timestamp %d)."
            % (int(event_id), unix_timestamp)
        )

    _logger.info("Generated atmosphere file: %s", atm_path)
    return atm_path


if __name__ == "__main__":
    options = parser.parse_args()
    generate_atmosphere(options.utctimestamp, output=options.output, 
                        observatory=options.observatory, coordinates=options.coordinates,
                        minheight=options.minheight, maxheight=options.maxheight, 
                        interpolationSteps=options.interpolationSteps, 
                        gdaspath=options.gdaspath, verbose=options.verbose, 
                        createplot=getattr(options, 'createplot', False),
                        cleanup=options.cleanup)
