FROM python:3.6-slim
LABEL maintainer="The NuRadioReco Authors <physics-astro-nuradiomcdev@lists.uu.se>"

RUN apt-get update
RUN apt-get upgrade -y

# Install core dependencies
RUN pip install toml aenum astropy matplotlib numpy radiotools scipy tinydb tinydb-serialization

# Install optional dependencies
RUN pip install dash gunicorn uproot==4.1.1 h5py peakutils plotly pymongo sphinx sphinx_rtd_theme numpydoc pandas six DateTime importlib-metadata
# Install NuRadioReco
ADD NuRadioMC /usr/local/lib/python3.6/site-packages/NuRadioMC
ADD NuRadioReco /usr/local/lib/python3.6/site-packages/NuRadioReco

RUN useradd nuradio

USER   nuradio
EXPOSE 8050
WORKDIR /usr/local/lib/python3.6/site-packages/NuRadioReco/detector/webinterface
CMD [ "python", "./index.py" ]
