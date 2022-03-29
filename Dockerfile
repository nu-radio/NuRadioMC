FROM python:3.9-slim
LABEL maintainer="The NuRadioReco Authors <physics-astro-nuradiomcdev@lists.uu.se>"

RUN apt-get update
RUN apt-get upgrade -y

# Install core dependencies
RUN pip install toml aenum astropy matplotlib numpy radiotools scipy tinydb tinydb-serialization

# Install optional dependencies
RUN pip install dash gunicorn h5py peakutils plotly pymongo sphinx pandas six DateTime importlib-metadata

#Uninstal werkzeug bug
pip uninstall Werkzeug

pip install -v https://github.com/pallets/werkzeug/archive/refs/tags/2.0.1.tar.gz

# Install NuRadioReco
ADD NuRadioMC /usr/local/lib/python3.9/site-packages/NuRadioMC
ADD NuRadioReco /usr/local/lib/python3.9/site-packages/NuRadioReco

RUN useradd nuradio

USER   nuradio
EXPOSE 8050
WORKDIR /usr/local/lib/python3.9/site-packages/NuRadioReco/detector/webinterface
CMD [ "python", "./index.py" ]
