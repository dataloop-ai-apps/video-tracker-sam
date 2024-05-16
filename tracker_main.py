import time
import dtlpy
import numpy as np
import logging
import cv2
import torch
import dtlpy as dl
import sys
import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4000"

sys.path.append('./FastSAM')
from sam_tracker.TrackedBox import TrackedBox
from SAM import FastSAM, MobileSAM
from sam_tracker.utils import Bbox
import subprocess as sp

logger = logging.getLogger(__name__)


def get_gpu_memory():
    output_to_list = lambda x: x.decode('ascii').split('\n')[:-1]
    COMMAND = "nvidia-smi --query-gpu=memory.used --format=csv"
    try:
        memory_use_info = output_to_list(sp.check_output(COMMAND.split(), stderr=sp.STDOUT))[1:]
    except sp.CalledProcessError as e:
        raise RuntimeError("command '{}' return with error (code {}): {}".format(e.cmd, e.returncode, e.output))
    memory_use_values = [int(x.split()[0]) for i, x in enumerate(memory_use_info)]
    # print(memory_use_values)
    return memory_use_values


class ServiceRunner(dtlpy.BaseServiceRunner):
    """
    Service runner class

    """

    def __init__(self):
        # ini params
        print('whaaaa', get_gpu_memory())
        self.MAX_AGE = 20
        self.THRESH = 0.4
        self.MIN_AREA = 20
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            print('[Tracker] [WARNING] cuda is NOT available.')
            self.device = 'cpu'
        self.sam = FastSAM(device=self.device, small=False)
        print('[Tracker] [INFO] Model loaded.')

    @staticmethod
    def _get_modality(mod: dict):
        if 'operation' in mod:
            if mod['operation'] == 'replace':
                return mod['itemId']
        elif 'type' in mod:
            if mod['type'] == 'replace':
                return mod['ref']
        else:
            return None

    def _get_item_stream_capture(self, item_stream_url):
        #############
        # replace to webm stream
        if dl.environment() in item_stream_url:
            # is dataloop stream - take webm
            item_id = item_stream_url[item_stream_url.find('items/') + len('items/'): -7]
            orig_item = dl.items.get(item_id=item_id)
            webm_id = None
            for mod in orig_item.metadata['system'].get('modalities', list()):
                ref = self._get_modality(mod)
                if ref is not None:
                    try:
                        _ = dl.items.get(item_id=ref)
                        webm_id = ref
                        break
                    except dl.exceptions.NotFound:
                        continue
            if webm_id is not None:
                # take webm if exists
                item_stream_url = item_stream_url.replace(item_id, webm_id)
        ############
        return cv2.VideoCapture('{}?jwt={}'.format(item_stream_url, dl.token()))

    def run(self, item_stream_url, bbs, start_frame, frame_duration=60, progress=None):
        """
        :param item_stream_url:  item.stream for Dataloop item, url for json video links
        :param bbs: dictionary of annotation.id : BB
        :param start_frame:
        :param frame_duration:
        :param progress:
        :return:
        """
        try:
            print('whaaaa', get_gpu_memory())

            if not isinstance(bbs, dict):
                raise ValueError('input "bbs" must be a dictionary of {id:bbox}')
            print('[Tracker] Started')

            print('[Tracker] video url: {}'.format(item_stream_url))
            d_size = 1024
            tic_get_cap = time.time()
            cap = self._get_item_stream_capture(item_stream_url)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            frame_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            frame_width = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            x_factor = frame_width / d_size
            y_factor = frame_height / d_size
            runtime_get_cap = time.time() - tic_get_cap
            print('[Tracker] starting from {} to {}'.format(start_frame, start_frame + frame_duration))

            print('[Tracker] received bbs(xyxy): {}'.format(bbs))
            runtime_load_frame = list()
            runtime_track = list()

            tic_total = time.time()
            output_dict = {bbox_id: dict() for bbox_id, _ in bbs.items()}
            states_dict = {bbox_id: TrackedBox(Bbox.from_xyxy(bb[0]['x'] / x_factor,
                                                              bb[0]['y'] / y_factor,
                                                              bb[1]['x'] / x_factor,
                                                              bb[1]['y'] / y_factor),
                                               max_age=self.MAX_AGE) for bbox_id, bb in bbs.items()}

            print('[Tracker] going to process {} frames'.format(frame_duration))
            for i_frame in range(1, frame_duration):
                print('whaaaa', get_gpu_memory())

                print('[Tracker] processing frame #{}'.format(start_frame + i_frame))
                tic = time.time()
                ret, frame = cap.read()
                states_dict_flag = all(bb.gone for bb in states_dict.values())
                if not ret or states_dict_flag:
                    print(f"[Tracker] stopped at frame {i_frame}: "
                          f"opencv frame read :{ret}, all bbs gone: {states_dict_flag}")
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                self.sam.set_image(image=cv2.resize(frame, (d_size, d_size)))
                runtime_load_frame.append(time.time() - tic)

                tic = time.time()
                for bbox_id, bb in bbs.items():
                    print('whaaaa', get_gpu_memory())

                    # track
                    bbox = states_dict[bbox_id].track(sam=self.sam,
                                                      thresh=self.THRESH,
                                                      min_area=self.MIN_AREA)
                    if bbox is None:
                        output_dict[bbox_id][start_frame + i_frame] = None
                    else:
                        output_dict[bbox_id][start_frame + i_frame] = dl.Box(top=bbox.y * y_factor,
                                                                             left=bbox.x * x_factor,
                                                                             bottom=bbox.y2 * y_factor,
                                                                             right=bbox.x2 * x_factor,
                                                                             label='dummy').to_coordinates(color=None)

                runtime_track.append(time.time() - tic)

            runtime_total = time.time() - tic_total
            fps = frame_duration / (runtime_total + 1e-6)
            print('[Tracker] Finished.')
            print('[Tracker] Runtime information: \n'
                  f'Total runtime: {runtime_total:.2f}[s]\n'
                  f'FPS: {fps:.2f}fps\n'
                  f'Get url capture object: {runtime_get_cap:.2f}[s]\n'
                  f'Total track time: {np.sum(runtime_load_frame) + np.sum(runtime_track):.2f}[s]\n'
                  f'Mean load per frame: {np.mean(runtime_load_frame):.2f}\n'
                  f'Mean track per frame: {np.mean(runtime_track):.2f}')
            print('[Tracker] DEVICE: {}'.format(self.device))
        except Exception:
            logger.exception('Failed during track:')
            raise
        return output_dict


if __name__ == "__main__":
    runner = ServiceRunner()
    inputs = {
        "item_stream_url": "",
        "bbs": {},
        "start_frame": 0,
        "frame_duration": 72
    }

    runner.run(**dl.executions.get('6646250c737a021a47459ba8').input)
