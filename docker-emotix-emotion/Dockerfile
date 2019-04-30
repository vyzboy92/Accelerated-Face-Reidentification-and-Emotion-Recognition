FROM ubuntu:14.04
MAINTAINER vysakh@accubits.com

ENV \
  AFFDEX_DATA_DIR=/affdex-sdk/data \
  AFFECTIVA_SDK_VERSION=3.1.1-2802

RUN apt-get update && apt-get install --fix-missing -y \
    build-essential \
    cmake \
    git \
    gzip \
    libboost-system1.55-dev libboost-filesystem1.55-dev libboost-date-time1.55-dev libboost-regex1.55-dev libboost-thread1.55-dev libboost-timer1.55-dev libboost-chrono1.55-dev libboost-serialization1.55-dev libboost-log1.55-dev libboost-program-options1.55-dev \
    libopencv-dev \
    tar \
    wget \
	python-pip python-dev build-essential \
	python-numpy \
 && wget https://download.affectiva.com/linux/affdex-cpp-sdk-${AFFECTIVA_SDK_VERSION}-linux-64bit.tar.gz \
 && mkdir /affdex-sdk \
 && tar -xzvf /affdex-cpp-sdk-*-linux-64bit.tar.gz -C /affdex-sdk \
 && rm /affdex-cpp-sdk-*-linux-64bit.tar.gz \
 && git clone https://github.com/Affectiva/cpp-sdk-samples.git /sdk-samples \
 && mkdir build \
 && (cd build && cmake -DOpenCV_DIR=/usr/ -DBOOST_ROOT=/usr/ -DAFFDEX_DIR=/affdex-sdk /sdk-samples && make) \
 && export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HOME/affdex-sdk/lib \
 && rm -rf /sdk-samples \
 && ln /dev/null /dev/raw1394 \
 && apt-get remove --purge -y build-essential cmake git wget libboost-program-options1.55-dev libopencv-dev libopencv-highgui-dev libboost-system1.55-dev libboost-filesystem1.55-dev libboost-date-time1.55-dev libboost-regex1.55-dev libboost-thread1.55-dev libboost-timer1.55-dev libboost-chrono1.55-dev libboost-serialization1.55-dev libboost-log1.55-dev \
 && apt-get install -y libopencv-core2.4 libboost-program-options1.55.0 libopencv-highgui2.4 \
 && apt-get autoremove -y \
 && rm -rf /var/lib/apt/lists/*
 
RUN pip install Flask jsonpickle

COPY detect.sh /build/video-demo
COPY server.py /build/video-demo

WORKDIR "/build/video-demo"

RUN mkdir /tmp/image

RUN chmod +x /build/video-demo/detect.sh
RUN chmod +x /build/video-demo/server.py

ENTRYPOINT ["python"]
CMD ["server.py"]
