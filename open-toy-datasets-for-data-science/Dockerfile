FROM python:3.9.2

WORKDIR /opt
RUN pip install --upgrade pip
RUN pip install numpy==1.20.1 \
				pandas==1.2.3 \
				matplotlib==3.3.4 \
				seaborn==0.11.1 \
				scikit-learn==0.24.1 \
				jupyterlab

WORKDIR /work

# "--ip=0.0.0.0": ip address for local host
CMD ["jupyter","lab","--ip=0.0.0.0","--allow-root","--LabApp.token=''"]
# docker run -p 8888:8888 -v ~/**:/work <Image ID>
