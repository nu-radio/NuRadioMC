#include <complex>
#include <vector>

/*! \file createAsk.h
    \brief A Documented file.

    Details.
*/


std::vector<std::complex<float> > getFrequencySpectrum(const double energy,
		const double theta, std::vector<float> &freqs, const bool isEMShower);
	/*! \fn std::vector<std::complex<float>> getFrequencySpectrum(const double energy,
		const double theta, std::vector<float> &freqs, const bool isEMShower)
	 * \brief get Askaryan pulse in frequency domain
	 * \param energy electromagnetic or hadronic energy of the shower
	 * \param theta Cherenkov angle
	 * \param freqs vector of frequencies
	 * \param isEMShower true for EM shower, false for had. shower
	 * \return complex frequency spectrum
	 */

std::pair<std::vector<float>, std::vector<std::vector<float> > > getTimeTrace(
		const double energy, const double theta, const double fmin,
		const double fmax, const double df, const bool isEMShower);
	/*! \fn std::pair<std::vector<float>, std::vector<std::vector<float> > > getTimeTrace(
		const double energy, const double theta, const double fmin,
		const double fmax, const double df, const bool isEMShower);
	 * \brief get Askaryan pulse in time domain
	 * \param energy electromagnetic or hadronic energy of the shower
	 * \param theta Cherenkov angle
	 * \param fmin lower frequency cutoff
	 * \param fmax uppper frequency cutoff
	 * \param df frequency resolution (determines length of time trace)
	 * \param isEMShower true for EM shower, false for had. shower
	 * \return times and three-dimensional electric field trace (eR, eTheta, ePhi)
	 */

void getTimeTrace2(double*& times, double*& ex, double*& ey, double*& ez,
		int& size, double energy, double theta, double fmin,
		double fmax, double df, bool isEMShower);

void getFrequencySpectrum2(double*& spectrumRealR, double*& spectrumImagR,
		double*& spectrumRealTheta, double*& spectrumImagTheta,
		double*& spectrumRealPhi, double*& spectrumImagPhi,
		int& size, const double energy, const double theta, double* freqs,
		int size_f, const bool isEMShower);

int main(int argc, char **argv);
