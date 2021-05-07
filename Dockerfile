FROM python:3.6-slim
LABEL maintainer="The NuRadioReco Authors <physics-astro-nuradiomcdev@lists.uu.se>"

RUN apt-get update
RUN apt-get upgrade -y

# Install core dependencies
RUN pip install aenum astropy matplotlib numpy radiotools scipy tinydb tinydb-serialization

# Install optional dependencies
RUN pip install dash gunicorn h5py peakutils plotly pymongo sphinx

# Install NuRadioReco
ADD NuRadioReco /usr/local/lib/python3.6/site-packages/NuRadioMC

RUN useradd nuradio

USER   nuradio
EXPOSE 8050
WORKDIR /usr/local/lib/python3.6/site-packages/NuRadioMC/NuRadioReco/detector/webinterface
CMD [ "python", "./index.py" ]
