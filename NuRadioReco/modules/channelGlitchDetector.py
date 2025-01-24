class GlitchFinder:

    def run(self, event, station):

        for ch in station.iter_channels():
            trace = ch.get_trace() * 4095/2.5
            times = ch.get_times()

            for i in range(len(trace) - 1):
                diff = trace[i+1] - trace[i]
                if (diff < -2000 or diff > 2000):
                    event.set_parameter(glitch, True)
                    ch.set_parameter(glitch_ch, True)
        
        event.set_parameter(glitch, False)

