import numpy as np
import matplotlib
import pycocotools.mask as maskUtils
import torch
import asyncio

from mmdet.apis import init_detector, inference_detector, show_result_pyplot, async_inference_detector, show_result
from mmdet.apis.inference import inference_batch
from mmdet.utils.contextmanagers import concurrent
import mmcv

#config_file = '../configs/faster_rcnn_x101_32x4d_fpn_1x.py'
#checkpoint_file = '../checkpoints/faster_rcnn_x101_32x4d_fpn_1x_20181218-ad81c133.pth'
config_file = '../../helmet/configs/faster_rcnn_x101_32x4d_fpn_1x_fp16_hardhat.py'
checkpoint_file = '../../../model/uniform/0120/epoch_11.pth'

def sync_main_batch(loops, warm_loops=10, batch_size=2, io=True):
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    if batch_size % 2 != 0:
        raise "batch size should be a times of 2."

    if io:
        img = ['demo.jpg', 'demo1.jpg']
    else:
        img = [mmcv.imread('demo.jpg'), mmcv.imread('demo1.jpg')]
    imgs = []
    for _ in range(int(batch_size / 2)):
        imgs += img

    print('\nbatch size: {}'.format(batch_size))

    prog_bar = mmcv.ProgressBar(warm_loops*batch_size)
    for _ in range(warm_loops):
        result = inference_batch(model, imgs)
        for _ in range(batch_size):
            prog_bar.update()

    print()
    prog_bar = mmcv.ProgressBar(loops*batch_size)
    for _ in range(loops):
        result = inference_batch(model, imgs)
        for _ in range(batch_size):
            prog_bar.update()

    #for i in range(len(imgs)):
    #    show_result(imgs[i], result[i], model.CLASSES, out_file='result_sync_batch{}.jpg'.format(i))


def sync_main_single(loops, warm_loops=10, io=True):
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    if io:
        img = 'demo.jpg'
    else:
        img = mmcv.imread('demo.jpg')

    for _ in mmcv.track_iter_progress(range(warm_loops)):
        result = inference_detector(model, img)

    for _ in mmcv.track_iter_progress(range(loops)):
        result = inference_detector(model, img)

    show_result(img, result, model.CLASSES, out_file='result_sync_single.jpg')


async def async_main(loops, warm_loops=10):
    model = init_detector(config_file, checkpoint_file, device='cuda:0')

    # queue is used for concurrent inference of multiple images
    streamqueue = asyncio.Queue()
    # queue size defines concurrency level
    streamqueue_size = 10

    for _ in range(streamqueue_size):
        streamqueue.put_nowait(torch.cuda.Stream(device='cuda:0'))

    img = mmcv.imread('demo.jpg')

    '''for _ in mmcv.track_iter_progress(range(warm_loops)):
        async with concurrent(streamqueue):
            result = await async_inference_detector(model, img)'''

    for _ in mmcv.track_iter_progress(range(loops)):
        async with concurrent(streamqueue):
            result = await async_inference_detector(model, img)

    #show_result(img, result, model.CLASSES, out_file='result_async.jpg')


if __name__ == "__main__":
    '''print('async_main:')
    #asyncio.run(async_main(100))
 
    print('sync_main_single:')
    sync_main_single(50, 20)

    print('sync_main_batch:')
    sync_main_batch(50, 20, 2)
    sync_main_batch(50, 20, 4)
    sync_main_batch(50, 20, 6)
    sync_main_batch(50, 20, 8)'''
    sync_main_batch(100, 20, 4)
    #sync_main_batch(50, 20, 12)
    #sync_main_batch(50, 20, 16)
    print()
