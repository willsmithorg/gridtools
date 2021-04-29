FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN apt-get install -y vim
RUN pip install --upgrade pip==21.0.1
RUN pip install numpy pandas sklearn xgboost cached_property nltk regex tensorflow_hub
RUN git config --global user.email "will@willsmith.org"
RUN git config --global user.name "William Smith"

WORKDIR /ssd/ml/gridtools
ENV TFHUB_CACHE_DIR=/ssd/ml/tfhub_cache_dir
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
