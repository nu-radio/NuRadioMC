overview of times
===========
Time delays are introduced by several hardware components. These time delays are often corrected for by folding/unfolding the complex transfer function (for an amp e.g. the measurement of the S12 parameter). The unfolding is typically done in the frequency domain where a convolution becomes a simple multiplication. As a consequence of typically short trace length (<~1000 samples) and because a Fourier transform implies implicitly a periodic signal, a pulse being at the beginning of the trace can end up being at the end of the trace. To avoid this behavior we use the following procedure:

We smoothly filter the first 5% and last 5% of the trace using a Tukey window function. This is a function that goes smoothly from 0 to 1.
To avoid rollover of the pulse, we add 128ns of zeros to the beginning and end of the trace. Steps 2) and 3) are performed by the channelStopFilter module
Both electric fields and channels have a trace_start_time variable. The get_times() function will automatically add the trace_start_time to the time array.

ARIANNA specific details:
-------------------------
Our hardware produces an artifact (a glitch) at the STOP position (i.e. the physical beginning of the trace). Because of the way the hardware works, the STOP position is not at the beginning of the trace but can be anywhere. During read in of the snowshovel calibrated data files, the trace is rolled such that the physical beginning (the STOP position) is at sample zero of the trace. This glitch is removed by the procedure described above.


Station time
------------
The trace_start_times are all given relative to the station_time of the station the E-field or channel belongs to. The station_time is stored in an astopy.time.Time object for sub nanosecond precision on absolute times.
The trace_start_time itself is stored as a float. For simulations, the trace_start_time is relative to the vertex time, i.e., the time of the particle interaction. 
For data: TODO, describe how current RNO-G data is handled

Trace start times in channels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Specify when the trace starts relative to the station time. Effects that change the pulse time for all frequencies equally (for example cable delays) are most often taken into account by changing the trace_start_time.

Trace start times in E-fields
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Work similar to trace_start_time in channels, but with a caveat: Since EM-waves move, electric_field objects hold a position (relative to the station they are associated with). The E-field is therefore defined as the field an observer would measure at the given position. Note that this position does not necessarily have to coincide with the position of a channel the E-field is associated with. This is the case for (some) cosmic-ray simulations where the same E-field at the surface is used for all surface LPDAs.

overview of modules that alter time
===================================
We list all relevant modules that is used for a MC simulation and reconstruction. For a pure data reconstruction, the first few modules are just not used

* readCoREAS: CoREAS reader prepends n samples to the simulated trace. This is done so that the trace does not directly start with the pulse and to have a good frequency resolution.

* efieldToVoltageConverter: the traces are rolled (rotated) according to the time delay due to the geometric separation of the antennas and cable delays.

* hardwareResponseIncorporator (sim to data):
    * the channel traces are folded with the amplifier response which also includes some time delay
    * note that the hardwareResponseIncorporator does not take cable delays into account, as this is done by the efieldToVoltageConverter

* triggerTimeAdjuster 
    * 'sim_to_data' mode: This modules cuts the trace to the correct length (as specified in the detector description) around the trigger time with a pre-trigger time that is passed to the module. The default settings are 50ns pre trigger time. In the case of multiple triggers it either uses the trigger that was specified by the user as an argument to this module, or it uses the trigger with the earliest trigger time. The pre-trigger times are saved to the trigger object (only to the one that was used to determine the trigger time). In the end, the trace_start_time is set to the trigger time. This is done because this reflects how raw experimental data looks like. 
    * 'data_to_sim' mode: The module determines the trigger that was used to cut the trace to its current length (the 'sim_to_data' step above in case of simulations) and adjusts the trace_start_time according to the different readout delays. In case of multiple triggers, either the user specifies the trigger name or the trigger that has the field `pre_trigger_times` set will be used. For the latter case, if multiple triggers have the `pre_trigger_times` set, a warning is raised but the `trace_start_time` is adjusted for all pre-trigger times of all triggers. After applying this module in the "data_to_sim" direction, the position in the trace that caused the trigger can be found via `trigger_time` - `trace_start_time`.

* channelStopFilter: this module prepends and appends all channels with a fixed length (128ns by default). The 'prepend' time is subtracted from the station start time (because all channels get the same time delay)

* hardwareResponseIncorporator (data reconstruction):
    * unfolds amplifier -> also implies a time delay in the channel trace
    * cable delay is subtracted from the trace start time (due to the limited trace length, the trace is not rolled to account for cable delays)

* voltageToEfieldConverter:
    * the traces from all used channels are cut to the overlapping region (including delays due to geometry and differences in delays due to different hardware components, e.g. cables of different length's)
    * the E-field trace_start_time is set accordingly