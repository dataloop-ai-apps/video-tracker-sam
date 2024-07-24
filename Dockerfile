FROM dataloopai/dtlpy-agent:gpu.cuda.11.8.py3.10.opencv
USER root
RUN apt update && apt install -y curl

USER 1000
ENV HOME=/tmp
COPY requirements.txt /tmp/
WORKDIR /tmp
RUN python3 -m pip install --upgrade pip
RUN pip install --user -r /tmp/requirements.txt
RUN pip install --user torch torchvision --index-url https://download.pytorch.org/whl/cu118



# docker build --no-cache -t gcr.io/viewo-g/piper/agent/runner/gpu/video-tracker-sam:0.1.12 -f Dockerfile .
# docker push gcr.io/viewo-g/piper/agent/runner/gpu/video-tracker-sam:0.1.11
# docker run -it gcr.io/viewo-g/piper/agent/runner/gpu/video-tracker-sam:0.1.11 bash
