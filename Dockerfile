FROM dataloopai/dtlpy-agent:gpu.cuda.11.8.py3.10.opencv

USER 1000
COPY requirements.txt /tmp/
WORKDIR /tmp
RUN pip install --user -r /tmp/requirements.txt
RUN pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cu118



# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/gpu/video-tracker-sam:0.1.11 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/gpu/video-tracker-sam:0.1.11
