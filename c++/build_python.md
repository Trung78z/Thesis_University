sudo apt update
sudo apt install -y \
  build-essential \
  libffi-dev \
  libssl-dev \
  zlib1g-dev \
  libbz2-dev \
  libreadline-dev \
  libsqlite3-dev \
  libncursesw5-dev \
  libgdbm-dev \
  libnss3-dev \
  liblzma-dev \
  tk-dev \
  uuid-dev \
  libdb-dev
make clean
./configure --enable-optimizations
make -j$(nproc)
sudo make altinstall
