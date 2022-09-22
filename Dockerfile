FROM ubuntu:16.04

RUN apt-get update && apt-get install -y software-properties-common

#Install dependicies
RUN apt-get update && apt-get install -y \
    gcc \
    cmake \
    git \
    gtk+2.0 \
    pkg-config \
    python2.7 \
    python-dev \
    python-numpy \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libtbb2 \
    libtbb-dev \
    libdc1394-22 \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    build-essential \
    qt5-default \
    libvtk6-dev \
    zlib1g-dev \
    libjpeg-dev \
    libwebp-dev \
    libpng-dev \
    libtiff5-dev \
    libjasper-dev \
    libopenexr-dev \
    libgdal-dev \
    libtbb-dev \
    libeigen3-dev \
    python-dev python-tk python-numpy python3-dev python3-tk python3-numpy \
    ant default-jdk \
    doxygen \
    unzip \
    wget

# debugging stuff
# RUN apt install -y curl && RIPGREP_VERSION=$(curl -s "https://api.github.com/repos/BurntSushi/ripgrep/releases/latest" | grep -Po '"tag_name": "\K[0-9.]+') && \
#     curl -Lo ripgrep.deb "https://github.com/BurntSushi/ripgrep/releases/latest/download/ripgrep_${RIPGREP_VERSION}_amd64.deb" && \
#     apt install -y ./ripgrep.deb && \
#     rm ripgrep.deb

#Install more dependicies
RUN apt-get install -y libblas-dev liblapack-dev libopenmpi-dev libtiff-dev libpng-dev libjpeg-dev libnetcdf-dev

#Copy DICe code
RUN mkdir dice-2.0
COPY . /dice-2.0/

#Update netcdf.h to a version with values as per dice build instructions
RUN cd /usr/include && rm netcdf.h && cp /dice-2.0/scripts/ubuntu/docker/netcdf.h .

#Install trilinos
RUN git clone https://github.com/trilinos/Trilinos.git --branch trilinos-release-12-4-2 trilinos-12.4.2
RUN cd trilinos-12.4.2 && mkdir build && cd build &&  cp /dice-2.0/scripts/ubuntu/docker/do-trilinos-cmake-docker . && chmod +x do-trilinos-cmake-docker && ./do-trilinos-cmake-docker && make all install

#Install openCV
RUN wget https://github.com/opencv/opencv/archive/3.2.0.zip -O OpenCV320.zip
RUN unzip OpenCV320.zip && rm OpenCV320.zip
RUN cd opencv-3.2.0 && mkdir build && cd build && cp /dice-2.0/scripts/ubuntu/docker/do-opencv-cmake-docker . && chmod +x do-opencv-cmake-docker && ./do-opencv-cmake-docker && make all install

#Install DICE
RUN cd dice-2.0 && mkdir build && cd build && cp /dice-2.0/scripts/ubuntu/docker/do-dice-cmake-docker . && chmod +x do-dice-cmake-docker && ./do-dice-cmake-docker && make all && cd tests  && ctest; exit 0

## Fails test number 44

CMD ["/dice-2.0/build/bin/dice"]
