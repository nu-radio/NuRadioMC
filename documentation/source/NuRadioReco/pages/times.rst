Overview of times
=================
This page provides an overview of the different times defined in different places in NuRadioMC/NuRadioReco.
Generally speaking, the global time of an `Event <NuRadioReco.framework.event.Event>` is stored as
a :ref:`station or event time <NuRadioReco/pages/times:Station time (Event time)>`. The times corresponding to the `voltage time trace <NuRadioReco.framework.channel.Channel>`,
`electric fields <NuRadioReco.framework.electric_field.ElectricField>` or `triggers <NuRadioReco.framework.trigger.Trigger>`
are then stored as floats relative to this global time inside each object. They can be obtained by the
`get_times() <NuRadioReco.framework.base_trace.BaseTrace.get_times>` method for trace-like objects (`ElectricField <NuRadioReco.framework.electric_field.ElectricField>` ,
`Channel <NuRadioReco.framework.channel.Channel>`), or by the `get_trigger_time <NuRadioReco.framework.trigger.Trigger.get_trigger_time>`
method for `Trigger <NuRadioReco.framework.trigger.Trigger>` objects.

Time delays are introduced by several hardware components. These time delays are often accounted for by folding/unfolding the complex transfer function (for an amplifier e.g. the measurement of the S12 parameter).
The unfolding is typically done in the frequency domain where a convolution becomes a simple multiplication.
As a consequence of typically short trace length (<~1000 samples) and because a Fourier transform implies implicitly a periodic signal,
a pulse being at the beginning of the trace can end up being at the end of the trace.
This can be avoided by using the `NuRadioReco.modules.channelStopFilter` module, which appends zeros at either end of the trace
and applies a Tukey window to taper the ends of the trace towards zero.

.. Note::
  For the **ARIANNA** experiment, the hardware produces an artifact (a glitch) at the STOP position (i.e. the physical beginning of the trace).
  Because of the way the hardware works, the STOP position is not at the beginning of the trace but can be anywhere.
  During read in of the snowshovel calibrated data files, the trace is rolled such that the physical beginning (the STOP position) is at sample zero of the trace.
  This glitch is removed by the `channelStopFilter <NuRadioReco.modules.channelStopFilter>` procedure described in the :ref:`module overview below <NuRadioReco/pages/times:Overview of modules that affect time>` .

Station time (Event time)
-------------------------
The global time at which the event takes place is stored as the `event time <NuRadioReco.framework.event.Event.get_event_time>`
in the `Event <NuRadioReco.framework.event.Event>` object.
This time usually corresponds to the "vertex time" of the first interaction for simulations,
and the time at which the data was recorded in the DAQ for data.
It is stored as an `astropy.time.Time` object to enable sub-ns precision on the absolute time.

In **simulated data**, the `event time <NuRadioReco.framework.event.Event.get_event_time>`
is generally the same as the `station_time <NuRadioReco.framework.station.Station.get_station_time>` stored
in the `Station <NuRadioReco.framework.station.Station>` object.
In **experimental data**, the `station_time <NuRadioReco.framework.station.Station.get_station_time>`
usually corresponds to the time the data was read out (recorded), and the `event time <NuRadioReco.framework.event.Event.get_event_time>` may not always be defined.
In this case, because different stations may operate and trigger independently,
the station_times of different stations are not guaranteed to agree, even if they were triggered by the same source.


Times in `Channel <NuRadioReco.framework.channel.Channel>`, `ElectricField <NuRadioReco.framework.electric_field.ElectricField>` and
`Trigger <NuRadioReco.framework.trigger.Trigger>` objects are all defined relative to the
`station_time <NuRadioReco.framework.station.Station.get_station_time>` of the `Station <NuRadioReco.framework.station.Station>`
they are stored in (see the description of the :doc:`NuRadio data structure </NuRadioReco/pages/event_structure>`).
These times are stored as an array of floats.
For trace-like objects (`Channels <NuRadioReco.framework.channel.Channel>` and `ElectricField <NuRadioReco.framework.electric_field.ElectricField>`),
the times can be obtained through the `get_times() <NuRadioReco.framework.base_trace.BaseTrace.get_times>` method of these classes.
Additionally, the trace start time (the first value of `get_times() <NuRadioReco.framework.base_trace.BaseTrace.get_times>`)
is accessible as the `trace_start_time <NuRadioReco.framework.base_trace.BaseTrace.get_trace_start_time>` .


Trace start times in channels
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Specify when the trace starts relative to the station time. Effects that change the pulse time for all frequencies equally (for example cable delays) are most often taken into account by changing the `trace_start_time <NuRadioReco.framework.base_trace.BaseTrace.get_trace_start_time>`.

Trace start times in E-fields
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Work similar to `trace_start_time <NuRadioReco.framework.base_trace.BaseTrace.get_trace_start_time>` in channels, but with a caveat: Since EM-waves move, electric_field objects hold a position (relative to the station they are associated with). The E-field is therefore defined as the field an observer would measure at the given position. Note that this position does not necessarily have to coincide with the position of a channel the E-field is associated with. This is the case for (some) cosmic-ray simulations where the same E-field at the surface is used for all surface LPDAs.

Trigger times
^^^^^^^^^^^^^
The `trigger_time <NuRadioReco.framework.trigger.Trigger.get_trigger_time>`,
which is the time at which the trigger fired, is stored in the `Trigger <NuRadioReco.framework.trigger.Trigger>`
object (which can be obtained using `station.get_trigger() <NuRadioReco.framework.station.Station.get_trigger>`).
This is the time at which the trigger condition was first fulfilled.
As for the trace_start_time, the trigger time is defined relative to the
`station_time <NuRadioReco.framework.station.Station.get_station_time>` .


Overview of modules that affect time
------------------------------------
We list all relevant modules that are used for a MC simulation and reconstruction. For a pure data reconstruction, the first two modules are not used.

* `NuRadioReco.modules.io.coreas`: CoREAS reader prepends n samples to the simulated trace. This is done so that the trace does not directly start with the pulse and to have a good frequency resolution.

* `NuRadioReco.modules.efieldToVoltageConverter`:
  the voltage traces are delayed compared to the electric field signal due to the geometric separation of the antennas and cable delays.
  This is accounted for by shifting the `trace_start_time <NuRadioReco.framework.base_trace.BaseTrace.get_trace_start_time>`.
  Note that the very similar `NuRadioReco.modules.efieldToVoltageConverterPerEfield` (which creates one
  `SimChannel <NuRadioReco.framework.sim_channel.SimChannel>` per electric field instead of combining the induced voltage traces in a single channel) does **not** include the cable delays!

* `NuRadioReco.modules.RNO_G.hardwareResponseIncorporator`, `NuRadioReco.modules.ARIANNA.hardwareResponseIncorporator`, `NuRadioReco.modules.ARA.hardwareResponseIncorporator`:

  If ``sim to data=True``:

    * the channel traces are folded with the amplifier response which also includes some time delay.
      This delay is applied to the trace in the frequency domain (i.e. the signal is shifted within the trace, rather than adjusting the trace_start_time)
    * note that the hardwareResponseIncorporator does not take cable delays into account, as this is done by the efieldToVoltageConverter

  If ``sim to data=False``:

    * unfolds amplifier -> also implies a time delay in the channel trace
    * cable delay is subtracted from the trace start time (due to the limited trace length, the trace is not rolled to account for cable delays)

* `NuRadioReco.modules.channelStopFilter`: this module prepends and appends all channels with a fixed length (128ns by default).
  The 'prepend' time is subtracted from the trace start time (because all channels get the same time delay).
  It additionally applies a tukey window to taper off the start and end (by default, the first and last 5%) of the trace.

* `NuRadioReco.modules.voltageToEfieldConverter`:
    * the traces from all used channels are cut to the overlapping region (including delays due to geometry and differences in delays due to different group delays in hardware, e.g. different antenna/amplifier responses)
    * the E-field `trace_start_time <NuRadioReco.framework.base_trace.BaseTrace.get_trace_start_time>` is set accordingly

* `NuRadioReco.modules.channelReadoutWindowCutter`:
    * Cuts out the readout windows from simulated traces according to the `trigger_time`, `pre_trigger_time`, and `number_of_samples` (i.e., length of trace) parameters.

* NuRadioReco.modules._deprecated.triggerTimeAdjuster (deprecated):
    * This module is now deprecated. It was replaced by the `NuRadioReco.modules.channelReadoutWindowCutter` module for simulation (doing essential what the mode ``sim_to_data`` did with the notable difference that it sets the trace start time to `trigger_time - pre_trigger_time`. The mode ``data_to_sim`` is not needed anymore, i.e., the experiment specific IO modules have to make sure that the trace start time is set correctly.
    * ``sim_to_data`` mode: This modules cuts the trace to the correct length (as specified in the detector description) around the trigger time with a pre-trigger time as defined by the respective trigger module. In the case of multiple triggers it used the primary trigger. If no primary trigger is defined, it uses the trigger with the earliest trigger time. In the end, the `trace_start_time <NuRadioReco.framework.base_trace.BaseTrace.get_trace_start_time>` is set to the trigger time. This is done because this reflects what raw experimental data looks like.
    * ``data_to_sim`` mode: The module determines the trigger that was used to cut the trace to its current length (the 'sim_to_data' step above in case of simulations) and adjusts the `trace_start_time <NuRadioReco.framework.base_trace.BaseTrace.get_trace_start_time>` according to the different readout delays. The "primary trigger" defines the readout delays. **After** applying this module in the "data_to_sim" direction, the position in the trace that caused the trigger can be found via the `trigger_time <NuRadioReco.framework.trigger.Trigger.get_trigger_time>`.
