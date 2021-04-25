FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN pip install numpy pandas sklearn xgboost cached_property nltk regex tensorflow_hub
WORKDIR /ssd/ml/gridtools
ENV TFHUB_CACHE_DIR=/ssd/ml/tfhub_cache_dir
CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
