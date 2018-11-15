FROM 192.168.0.12:5000/conda_ubuntu:16.04_v1
ADD environment.yml /data/algo/environment.yml
ADD install.sh /data/algo/install.sh
ADD setup.py /data/algo/setup.py
ADD server.py /data/algo/server.py
COPY build /data/algo/build
COPY s4d /data/algo/s4d
COPY dist /data/algo/dist
ADD algo.py /data/algo/algo.py
ADD run.sh /data/algo/run.sh
ADD start.sh /data/algo/start.sh
ADD algo_pyproxy_agent /data/algo/algo_pyproxy_agent
ENV ETCD_ADDR 192.168.0.81:2379
ENV LOOKUPD_ADDR 192.168.0.81:4161
ENV ENV_NAME dev
WORKDIR /data/algo
EXPOSE 5000
RUN ["/bin/chmod", "777", "-R", "/data/algo"]
RUN ["/root/anaconda3/bin/conda", "env", "create","-n", "s4d" ,"-f","environment.yml"]
ENTRYPOINT /data/algo/start.sh


