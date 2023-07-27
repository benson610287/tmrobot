FROM nvidia/cuda:12.0.1-cudnn8-runtime-ubuntu20.04
############################## SYSTEM PARAMETERS ##############################
# * Arguments
ARG USER=initial
ARG GROUP=initial
ARG UID=1000
ARG GID=${UID}
ARG SHELL=/bin/bash
ARG HARDWARE=x86_64
ARG ENTRYPOINT_FILE=entrypint.sh

# * Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES all
# ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

# * Setup users and groups
RUN groupadd --gid ${GID} ${GROUP} \
    && useradd --gid ${GID} --uid ${UID} -ms ${SHELL} ${USER} \
    && mkdir -p /etc/sudoers.d \
    && echo "${USER}:x:${UID}:${UID}:${USER},,,:$HOME:${shell}" >> /etc/passwd \
    && echo "${USER}:x:${UID}:" >> /etc/group \
    && echo "${USER} ALL=(ALL) NOPASSWD: ALL" > "/etc/sudoers.d/${USER}" \
    && chmod 0440 "/etc/sudoers.d/${USER}"

# * Replace apt urls
# ? Change to tku
RUN sed -i 's@archive.ubuntu.com@ftp.tku.edu.tw@g' /etc/apt/sources.list
# ? Change to Taiwan
# RUN sed -i 's@archive.ubuntu.com@tw.archive.ubuntu.com@g' /etc/apt/sources.list

# T
ENV TZ=Asia/Taipei
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

############################### INSTALL #######################################
# * Install packages
RUN apt update && apt install -y --no-install-recommends \
    sudo \
    git \
    htop \
    wget \
    curl \
    psmisc \
    xterm \
    pciutils \
    # * Shell
    byobu \
    terminator \
    # * base tools
    python3-pip \
    python3-dev \
    python3-setuptools \
    # * Work tools
    #ros2
    software-properties-common \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*



# gnome-terminal libcanberra-gtk-module libcanberra-gtk3-module \
# dbus-x11 libglvnd0 libgl1 libglx0 libegl1 libxext6 libx11-6 \

# ROS2 Foxy install
RUN add-apt-repository universe -y \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main"  \
     | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt update \
    && apt install -y --no-install-recommends \
        ros-foxy-desktop python3-argcomplete    \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*
   
# ROS2 Foxy move2 install    
RUN pip install -U rosdep \
    && apt update \
    && rosdep init \
    && apt install -y --no-install-recommends \
    	build-essential \
    	cmake \
    	libbullet-dev \
    	python3-colcon-common-extensions \
    	python3-flake8 \
    	python3-pytest-cov \
    	python3-rosdep \
    	python3-vcstool \
    && python3 -m pip install -U \
    	argcomplete \
    	flake8-blind-except \
    	flake8-builtins \
    	flake8-class-newline \
    	flake8-comprehensions \
    	flake8-deprecated \
    	flake8-docstrings \
    	flake8-import-order \
    	flake8-quotes \
    	pytest-repeat \
    	pytest-rerunfailures \
    	pytest \
    && sh -c "su - ${USER} -c 'git clone https://github.com/ros-planning/moveit2.git \
    	-b foxy /home/$USER/ws_moveit2/src/moveit2'"  \
    && for repo in /home/$USER/ws_moveit2/src/moveit2/moveit2.repos \
    	$(f="moveit2/moveit2_foxy.repos"; test -r $f && echo $f); do vcs import < "$repo"; done \
    &&  sh -c "su - ${USER} -c 'rosdep update \
            && rosdep install -r \
    	    --from-paths /home/$USER/ws_moveit2/src \
    	    --ignore-src --rosdistro foxy -y'" \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*
RUN sh -c "su - ${USER} -c 'sudo apt update'" 
RUN apt install -y ros-foxy-eigen-stl-containers
RUN apt install -y ros-foxy-geometric-shapes
RUN apt install -y ros-foxy-moveit-msgs
RUN apt install -y ros-foxy-srdfdom
RUN apt install -y ros-foxy-control-msgs
RUN apt install -y ros-foxy-ros-testing
RUN apt install -y ros-foxy-warehouse-ros
RUN apt install -y ros-foxy-ompl
RUN apt install -y ros-foxy-moveit-resources-fanuc-description
RUN apt install -y ros-foxy-moveit-resources-fanuc-moveit-config
RUN apt install -y ros-foxy-moveit-resources-panda-description
RUN apt install -y ros-foxy-moveit-resources-panda-moveit-config
RUN apt install -y ros-foxy-control-toolbox
RUN sh -c "su - ${USER} -c 'source /opt/ros/foxy/setup.bash && cd /home/${USER}/ws_moveit2 && colcon build --event-handlers desktop_notification- status- --cmake-args -DCMAKE_BUILD_TYPE=Release'"
# RUN sh -c "su - ${USER} -c 'source /opt/ros/foxy/setup.bash'"
# RUN sh -c "su - ${USER} -c 'cd /home/${USER}/ws_moveit2'"
# RUN sh -c "su - ${USER} -c 'colcon build --event-handlers desktop_notification- status- --cmake-args -DCMAKE_BUILD_TYPE=Release'"
# RUN sh -c "su - ${USER} -c 'sudo apt update && source /opt/ros/foxy/setup.bash && cd /home/${USER}/ws_moveit2 && colcon build --event-handlers desktop_notification- status- --cmake-args -DCMAKE_BUILD_TYPE=Release'"
RUN apt-get install -y iputils-ping
RUN apt update \
    && sh -c "su - ${USER} -c 'pip3 install Flask'" \
    && sh -c "su - ${USER} -c 'pip3 install waitress'" \
    && sh -c "su - ${USER} -c 'pip3 install opencv-python==3.4.13.47 '" \
    && sh -c "su - ${USER} -c 'pip3 install datetime'" \
    && sh -c "su - ${USER} -c 'pip3 install mysql-connector-python'" \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*
############################### INSTALL #######################################
# * Switch workspace to /home/${USER}/.tmp
WORKDIR /home/${USER}/.tmp

# * Copy custom configuration
COPY config .

RUN bash ./script/shell_setup.sh "bash" \
    && bash ./script/pip_setup.sh \
    && rm -rf /home/${USER}/.tmp

# * Copy entrypoint
COPY --chmod=0775 ./${ENTRYPOINT_FILE} /entrypoint.sh

# * Switch workspace to /home/${USER}
WORKDIR /home/${USER}

# * Switch user to ${USER}
USER ${USER}
RUN sudo mkdir work

# * Make SSH available
EXPOSE 22

# * Switch workspace to ~/work
WORKDIR /home/${USER}/work 
RUN echo "source /home/${USER}/ws_moveit2/install/setup.bash" >> /home/"${USER}"/.bashrc
RUN echo "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp" >> ~/.bashrc
RUN echo "export ROS_DOMAIN_ID=78" >> ~/.bashrc



ENTRYPOINT [ "/entrypoint.sh", "terminator" ]
# ENTRYPOINT [ "/entrypoint.sh", "byobu" ]
# ENTRYPOINT [ "/entrypoint.sh", "bash" ]
# ENTRYPOINT [ "/entrypoint.sh" ]
