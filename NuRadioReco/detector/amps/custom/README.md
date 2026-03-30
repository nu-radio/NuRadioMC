Custom amplifier responses can be implemented by including them as CSV files under this directory.

They should be included as {amp_type}.csv, with the columns frequency (in Hz), S21 (magnitude), S21 phase (in radians).
They may include one or several 'header' lines, but these should be prefaced with #. Example:

    # Custom amplifier response for Gen2 detector simulation
    # Frequency [Hz], S21 [mag], S21 [phase] [rad]
    1000000., 0.300000, 0.400000
    (...)
