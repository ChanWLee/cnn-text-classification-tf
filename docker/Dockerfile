FROM ubuntu:16.04

ARG GIT_BRANCH=master

RUN echo building branch ${GIT_BRANCH}

RUN apt-get update && apt-get install -y git && apt-get install -y wget && apt-get install -y sudo && \
    apt-get install -y python3-pip python3-dev python3-venv

RUN cd && mkdir tf-cnn && \
    cd tf-cnn && \
    python3 -m venv env1 && \
    . env1/bin/activate && \
    pip3 install --upgrade pip && \
    pip3 install --upgrade tensorflow && \
    pip3 install --upgrade langdetect && \
    pip3 install --upgrade konlpy && \
    pip3 install --upgrade jpype1

RUN cd && wget https://bitbucket.org/eunjeon/mecab-ko/downloads/mecab-0.996-ko-0.9.2.tar.gz && \
#    wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.0.3-20170922.tar.gz && \
    wget https://bitbucket.org/eunjeon/mecab-ko-dic/downloads/mecab-ko-dic-2.1.1-20180720.tar.gz && \
    tar zxfv mecab-0.996-ko-0.9.2.tar.gz && cd mecab-0.996-ko-0.9.2 && ./configure && make && make install && cd .. && \
#    tar zxfv mecab-ko-dic-2.0.3-20170922.tar.gz && cd mecab-ko-dic-2.0.3-20170922 && sudo ldconfig && ./configure && make && make install && cd .. && \
    tar zxfv mecab-ko-dic-2.1.1-20180720.tar.gz && cd mecab-ko-dic-2.1.1-20180720 && sudo ldconfig && ./configure && make && make install && cd .. && \
    rm -f *.tar.gz

# for tagger error
RUN cd && git clone https://bitbucket.org/eunjeon/mecab-python-0.996.git && \
    cd mecab-python-0.996 && \
    python3 setup.py build && \
    python3 setup.py install --prefix=/root/tf-cnn/env1 && \
    echo 'export LC_ALL="C.UTF-8"' >> /etc/profile && \
    echo 'export LANG="C.UTF-8"' >> /etc/profile && \
    echo '. /etc/profile' >> ~/.bashrc

