# Select the base image
#FROM nvcr.io/nvidia/pytorch:21.06-py3
FROM nvcr.io/nvidia/pytorch:23.04-py3
# Select the working directory
# Add cuda
RUN apt-get update
RUN echo ttf-mscorefonts-installer msttcorefonts/accepted-mscorefonts-eula select true | debconf-set-selections
RUN ln -snf /usr/share/zoneinfo/Etc/UTC /etc/localtime
RUN pip install opencv-python==4.8.0.74
# Add qt5
RUN apt-get install qt5-default -y
WORKDIR  /NUDGE/
RUN git clone https://github.com/k4ntz/OC_Atari
RUN pip install -e ./OC_Atari
ADD .ssh/ /root/.ssh/
RUN git clone git@github.com:akweury/NUDGE.git
# Install Python requirements
COPY ../NUDGE/requirements.txt ./requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt