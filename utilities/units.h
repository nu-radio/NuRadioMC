#ifndef _utl_Units_h_
#define _utl_Units_h_

namespace utl {

  /**
  	 \brief default system of units

     You should use the units defined in this file whenever you
     have a dimensional quantity in your code.  For example,
     write:
     \code
      double s = 1.5*km;
     \endcode
     instead of:
     \code
       double s = 1.5;   // don't forget this is in km!
     \endcode
     The conversion factors defined in this file
     convert your data into NuRadioMC base units, so that
     all dimensional quantities in the code are in a
     single system of units!  You can also
     use the conversions defined here to, for example,
     display data with the unit of your choice.  For example:
     \code
       cout << "s = " << s/mm << " mm";
     \endcode

     The base units are :
        - meter                   (meter)
        - nanosecond              (nanosecond)
        - electron Volt           (eV)
        - positron charge         (eplus)
        - degree Kelvin           (kelvin)
        - the amount of substance (mole)
        - luminous intensity      (candela)
        - radian                  (radian)
        - steradian               (steradian)

     The SI numerical value of the positron charge is defined here,
     as it is needed for conversion factor : positron charge = eSI (coulomb)

     Adapted from Offline, the reconstruction Framework of the Pierre Auger Collaboration.
	 This is a slightly modified version of the units definitions written by the Geant4 collaboration
  */

  // Prefixes
  const double yocto = 1e-24;
  const double zepto = 1e-21;
  const double atto  = 1e-18;
  const double femto = 1e-15;
  const double pico  = 1e-12;
  const double nano  = 1e-9;
  const double micro = 1e-6;
  const double milli = 1e-3;
  const double centi = 1e-2;
  const double deci  = 1e-1;
  const double deka  = 1e+1;
  const double hecto = 1e+2;
  const double kilo  = 1e+3;
  const double mega  = 1e+6;
  const double giga  = 1e+9;
  const double tera  = 1e+12;
  const double peta  = 1e+15;
  const double exa   = 1e+18;
  const double zetta = 1e+21;
  const double yotta = 1e+24;

  // Length [L]
  const double meter  = 1;
  const double meter2 = meter*meter;
  const double meter3 = meter*meter*meter;

  const double millimeter  = milli*meter;
  const double millimeter2 = millimeter*millimeter;
  const double millimeter3 = millimeter*millimeter*millimeter;

  const double centimeter  = centi*meter;
  const double centimeter2 = centimeter*centimeter;
  const double centimeter3 = centimeter*centimeter*centimeter;

  const double kilometer  = kilo*meter;
  const double kilometer2 = kilometer*kilometer;
  const double kilometer3 = kilometer*kilometer*kilometer;

  const double parsec = 3.0856775807e+16 * meter;
  const double kiloParsec = kilo * parsec;
  const double megaParsec = mega * parsec;

  const double micrometer = micro*meter;
  const double nanometer  = nano*meter;
  const double angstrom   = 1e-10*meter;
  const double fermi      = femto*meter;

  const double barn      = 1e-28*meter2;
  const double millibarn = milli*barn;
  const double microbarn = micro*barn;
  const double nanobarn  = nano*barn;
  const double picobarn  = pico*barn;

  // symbols
  const double mm  = millimeter;
  const double mm2 = millimeter2;
  const double mm3 = millimeter3;

  const double cm  = centimeter;
  const double cm2 = centimeter2;
  const double cm3 = centimeter3;

  const double m  = meter;
  const double m2 = meter2;
  const double m3 = meter3;

  const double km  = kilometer;
  const double km2 = kilometer2;
  const double km3 = kilometer3;

  // Angle
  const double radian      = 1;
  const double milliradian = milli*radian;
  const double degree      = (3.14159265358979323846/180)*radian;

  const double steradian   = 1;

  // symbols
  const double rad  = radian;
  const double mrad = milliradian;
  const double sr   = steradian;
  const double deg  = degree;

  // Time [T]
  const double nanosecond  = 1;
  const double nanosecond2  = nanosecond*nanosecond;
  const double second      = giga*nanosecond;
  const double millisecond = milli*second;
  const double microsecond = micro*second;
  const double picosecond  = pico*second;
  const double minute      = 60*second;
  const double hour        = 60*minute;
  const double day         = 24*hour;

  const double hertz = 1/second;
  const double kilohertz = kilo*hertz;
  const double megahertz = mega*hertz;
  const double gigahertz = giga*hertz;

  const double Hz = hertz;
  const double kHz = kilohertz;
  const double MHz = megahertz;
  const double GHz = gigahertz;

  // symbols
  const double ns = nanosecond;
  const double s  = second;
  const double ms = millisecond;

  // Electric charge [Q]
  const double eplus   = 1;                // positron charge
  const double eSI     = 1.602176462e-19;  // positron charge in coulomb
  const double coulomb = eplus/eSI;        // coulomb = 6.24150 e+18*eplus

  // Energy [E]
  const double electronvolt      = 1;
  const double megaelectronvolt  = mega*electronvolt;
  const double kiloelectronvolt  = kilo*electronvolt;
  const double gigaelectronvolt  = giga*electronvolt;
  const double teraelectronvolt  = tera*electronvolt;
  const double petaelectronvolt  = peta*electronvolt;
  const double exaelectronvolt   = exa*electronvolt;
  const double zettaelectronvolt = zetta*electronvolt;

  const double joule = electronvolt/eSI;  // joule = 6.24150 e+12 * MeV

  // symbols
  const double MeV = megaelectronvolt;
  const double eV  = electronvolt;
  const double keV = kiloelectronvolt;
  const double GeV = gigaelectronvolt;
  const double TeV = teraelectronvolt;
  const double PeV = petaelectronvolt;
  const double EeV = exaelectronvolt;
  const double ZeV = zettaelectronvolt;

  // Mass [E][T^2][L^-2]
  const double kilogram  = joule*second*second/(meter*meter);
  const double gram      = milli*kilogram;
  const double milligram = milli*gram;

  // symbols
  const double kg = kilogram;
  const double g  = gram;
  const double mg = milligram;

  // Power [E][T^-1]
  const double watt = joule/second;  // watt = 6.24150 e+3 * MeV/ns

  // Force [E][L^-1]
  const double newton = joule/meter;  // newton = 6.24150 e+9 * MeV/mm

  // Pressure [E][L^-3]
  //const double hep_pascal = newton/m2;    // pascal = 6.24150 e+3 * MeV/mm3
  //const double pascal     = hep_pascal;   // pascal = 6.24150 e+3 * MeV/mm3
  const double pascal     = newton/m2;      // pascal = 6.24150 e+3 * MeV/mm3
  const double bar        = 100000*pascal;  // bar    = 6.24150 e+8 * MeV/mm3
  const double atmosphere = 101325*pascal;  // atm    = 6.32420 e+8 * MeV/mm3

  // symbols
  const double hPa = hecto*pascal;

  // Electric current [Q][T^-1]
  const double ampere      = coulomb/second;  // ampere = 6.24150 e+9 * eplus/ns
  const double milliampere = milli*ampere;
  const double microampere = micro*ampere;
  const double nanoampere  = nano*ampere;

  // Electric potential [E][Q^-1]
  const double megavolt = megaelectronvolt/eplus;
  const double kilovolt = milli*megavolt;
  const double volt     = micro*megavolt;
  const double millivolt = milli*volt;
  const double microvolt = micro*volt;

  const double V = volt;

  // Electric resistance [E][T][Q^-2]
  const double ohm = volt/ampere;  // ohm = 1.60217e-16*(MeV/eplus)/(eplus/ns)

  // Electric capacitance [Q^2][E^-1]
  const double farad      = coulomb/volt;  // farad = 6.24150e+24 * eplus/Megavolt
  const double millifarad = milli*farad;
  const double microfarad = micro*farad;
  const double nanofarad  = nano*farad;
  const double picofarad  = pico*farad;

  // Magnetic Flux [T][E][Q^-1]
  const double weber = volt*second;  // weber = 1000*megavolt*ns

  // Magnetic Field [T][E][Q^-1][L^-2]
  const double tesla      = volt*second/meter2;  // tesla =0.001*megavolt*ns/mm2
  const double microtesla = micro*tesla;

  const double gauss     = 1e-4*tesla;
  const double kilogauss = deci*tesla;

  // Inductance [T^2][E][Q^-2]
  const double henry = weber/ampere;  // henry = 1.60217e-7*MeV*(ns/eplus)**2

  // Temperature
  const double kelvin = 1;

  // Amount of substance
  const double mole = 1;

  // Activity [T^-1]
  const double becquerel = 1/second;
  const double curie = 3.7e+10 * becquerel;

  // Absorbed dose [L^2][T^-2]
  const double gray = joule/kilogram;

  // Luminous intensity [I]
  const double candela = 1;

  // Luminous flux [I]
  const double lumen = candela*steradian;

  // Illuminance [I][L^-2]
  const double lux = lumen/meter2;

  // Miscellaneous
  const double fraction    = 1;
  const double perCent     = 0.01;
  const double percent     = perCent;
  const double perThousand = 0.001;
  const double permil      = perThousand;
  const double perMillion  = 0.000001;


}


#endif
