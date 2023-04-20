# Use a Rust base image
FROM nvidia/cuda:11.0.3-runtime-ubuntu20.04

# Update the package repository and install dependencies
# Get Ubuntu packages


# Update new packages
RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections


RUN apt-get update
RUN apt-get install -y -q

RUN apt-get install dialog apt-utils -y
RUN apt-get install -y \
    build-essential \
    curl \
    pkg-config \
    libssl-dev
RUN apt-get install -y wget unzip
RUN apt-get install tree
RUN apt-get install -y python3.8 python3.8-dev python3-pip

RUN pip3 install torch torchvision torchaudio
# Get Rust
#RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
#
## Add .cargo/bin to PATH
#ENV PATH="/root/.cargo/bin:${PATH}"
#ENV TORCH_CUDA_VERSION="cu117"

# Check cargo is visible
#RUN cargo --help


## Add the PyTorch repository
#RUN add-apt-repository ppa:ubuntu-toolchain-r/test

# Update the package repository and install PyTorch

# Set the working directory
WORKDIR /app
#RUN curl -LJO https://download.pytorch.org/libtorch/cu117/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcu117.zip && unzip libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcu117.zip
#RUN curl -LJO https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcpu.zip && unzip libtorch-cxx11-abi-shared-with-deps-1.13.1%2Bcpu.zip
# Copy the application code
COPY . .
RUN wget https://szdataset.s3.us-east-2.amazonaws.com/trained_models.zip && \
    unzip trained_models.zip && \
    rm trained_models.zip
RUN pip3 install -r requirements.txt
RUN tree --dirsfirst --charset=ascii .

#ENV LIBTORCH='/app/libtorch'
#ENV LD_LIBRARY_PATH='${LIBTORCH}/lib:$LD_LIBRARY_PATH'#COPY  /app/dist/ ./dist/

# Expose the application port
EXPOSE 8502

# Set the command to run when the container starts
#CMD ["./target/release/rust-new-project-template"]
#CMD ["./target/release/rust-new-project-template","text","-i", "The Chinese monarchy collapsed in 1912 with the Xinhai Revolution, when the Republic of China (ROC) replaced the Qing dynasty. In its early years as a republic, the country underwent a period of instability known as the \"Warlord Era\" before mostly reunifying in 1928 under a Nationalist government. A civil war between the nationalist Kuomintang (KMT) and the Chinese Communist Party (CCP) began in 1927. Japan invaded China in 1937, starting the Second Sino-Japanese War and temporarily halting the civil war. The surrender and expulsion of Japanese forces from China in 1945 left a power vacuum in the country, which led to renewed fighting between the CCP and the Kuomintang."]
#ENTRYPOINT ["cargo","run","--release", "text","-i"]
#CMD ["The Chinese monarchy collapsed in 1912 with the Xinhai Revolution, when the Republic of China (ROC) replaced the Qing dynasty. In its early years as a republic, the country underwent a period of instability known as the \"Warlord Era\" before mostly reunifying in 1928 under a Nationalist government. A civil war between the nationalist Kuomintang (KMT) and the Chinese Communist Party (CCP) began in 1927. Japan invaded China in 1937, starting the Second Sino-Japanese War and temporarily halting the civil war. The surrender and expulsion of Japanese forces from China in 1945 left a power vacuum in the country, which led to renewed fighting between the CCP and the Kuomintang."]
CMD ["streamlit", "run", "demo.py"]
