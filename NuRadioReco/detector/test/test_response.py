from NuRadioReco.detector import response
from NuRadioReco.framework import electric_field
from NuRadioReco.utilities import analytic_pulse

import numpy as np


def equal_response(rep1, rep2):
    """ Returns equal if two response are equal """

    if rep1.get_names() != rep2.get_names():
        return False

    if rep1.get_time_delay() != rep2.get_time_delay():
        return False

    freq = np.arange(70, 1000)

    if not np.allclose(rep1(freq), rep2(freq)):
        return False

    return True


def test_response():

    n = 5000
    resp = 2 * np.ones(n) + 1j * np.ones(n)
    freq = np.linspace(0, 1.2, len(resp))
    resp_shifted = response.subtract_time_delay_from_response(freq, resp, time_delay=-200)

    fake_response = response.Response(freq, [np.abs(resp_shifted), np.angle(resp_shifted)], ["mag", "rad"], name="shifted")
    fake_response2 = response.Response(freq, [3 * np.ones(n), np.zeros(n)], ["mag", "rad"], name="normal")

    fake_response3 = fake_response2 * fake_response

    if equal_response(fake_response, fake_response2):
        raise ValueError("These two response should not be equal")

    if equal_response(fake_response2, fake_response3):
        raise ValueError("Error while multiplying two objects of `detector.response.Response`.")


def test_trace():

    ef = electric_field.ElectricField(channel_ids=[0])
    ef.set_frequency_spectrum(
        analytic_pulse.get_analytic_pulse_freq(1, -0.5, 1 * np.pi, 2048, 1, 2.4, bandpass=(0.1, 0.7)), sampling_rate=2.4)

    ef.set_trace(np.roll(ef.get_trace(), 480), "same")

    n = 5000
    resp = np.ones(n) + 1j * np.zeros(n)
    freq = np.linspace(0, 1.2, len(resp))
    resp_shifted = response.subtract_time_delay_from_response(freq, resp, time_delay=-200)
    fake_response = response.Response(freq, [np.abs(resp_shifted), np.angle(resp_shifted)], ["mag", "rad"], name="shifted")

    ef2 = ef * fake_response

    if np.allclose(ef2.get_trace(), ef.get_trace()):
        raise ValueError("Error while multiplying `framework.base_trace.BaseTrace` and `detector.response.Response`.")


if __name__ == "__main__":
    test_response()
    test_trace()