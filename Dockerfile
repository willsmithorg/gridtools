FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN apt-get install -y vim
RUN pip install numpy pandas sklearn xgboost cached_property nltk regex tensorflow_hub
WORKDIR /ssd/ml/gridtools
ENV TFHUB_CACHE_DIR=/ssd/ml/tfhub_cache_dir
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
