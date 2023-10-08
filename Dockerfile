# pull official base image
#FROM python:3.8.5-slim-buster
FROM gcr.io/cloud-devrel-public-resources/gcloud-container-1.14.0:latest

RUN mkdir -p /tmp/mounted_model/0001

# set working directory
WORKDIR /usr/src/app

# Install curl
#RUN apt-get install curl

RUN apt-get update && apt-get -qq -y install \
    curl \
    wget \
    build-essential \ 
    cmake \ 
    git \
    unzip \ 
    pkg-config \
    python-dev \ 
    python-opencv \ 
    libopencv-dev \ 
    libjpeg-dev \ 
    libpng-dev \ 
    libtiff-dev \  
    libgtk2.0-dev \ 
    python-numpy \ 
    python-pycurl \ 
    libatlas-base-dev \
    gfortran \
    webp \ 
    python-opencv \ 
    qt5-default \
    libvtk6-dev \ 
    zlib1g-dev 


# Install Chrome for Selenium
RUN curl https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb -o /chrome.deb
RUN dpkg -i /chrome.deb || apt-get install -yf
RUN rm /chrome.deb

# Install chromedriver for Selenium
RUN curl https://chromedriver.storage.googleapis.com/2.31/chromedriver_linux64.zip -o /usr/local/bin/chromedriver
RUN chmod +x /usr/local/bin/chromedriver

# Install Open CV - Warning, this takes absolutely forever
RUN mkdir -p ~/opencv cd ~/opencv && \
    wget https://github.com/opencv/opencv/archive/4.5.0.zip && \
    unzip 4.5.0.zip && \
    rm 4.5.0.zip && \
    mv opencv-4.5.0 OpenCV && \
    cd OpenCV && \
    mkdir build && \ 
    cd build && \
    cmake \
    -DWITH_QT=ON \ 
    -DWITH_OPENGL=ON \ 
    -DFORCE_VTK=ON \
    -DWITH_TBB=ON \
    -DWITH_GDAL=ON \
    -DWITH_XINE=ON \
    -DBUILD_EXAMPLES=ON .. && \
    make -j4 && \
    make install && \ 
    ldconfig

# add and install requirements
COPY ./requirements.txt .
RUN pip install -r requirements.txt

# add app
COPY npsCaptchaSolverLinux.py .

COPY \captchaSolverModel /tmp/mounted_model/0001

# Run the script
CMD ["python", "npsCaptchaSolverLinux.py"]

# add and run as non-root user
#RUN adduser --disabled-password myuser
#USER myuser