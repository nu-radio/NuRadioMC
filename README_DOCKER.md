# How to build / run the docker container

## Install docker/ setup virtual machine
to get started with a container environment (docker in this case) on
your Mac systems you first would have to install docker and initialize
the setup

`brew install docker docker-machine virtualbox`
`docker-machine create default`

By running those commands you will get a virtual linux machine (running
in virtualbox) called ‘default’.
To be able to interact with this VM from the terminal of your Mac you
have to run the following command for setting the necessary environment
variables:

`eval $(docker-machine env default)`
## Build the container
`docker build --tag radio/detector .`

## Run the container
`docker run -it --rm -p 8050:8050 radio/detector`

(one may pass the url/user/password to the mongodb as environmental variables
`docker run -it --rm  -e mongo_user=<USER> -e mongo_password=<PW> -e mongo_server=<URL_TO_MONGODB> -p 8050:8050 radio/detector`)

## Open in browser
To connect to the application with your web browser you now have to
find out the IP address of the virtualbox VM, like

`docker-machine ip default`            

The URL for your browser should then look like

`http://<IP>:8050/`
