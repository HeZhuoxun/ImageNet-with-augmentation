from nvidia.dali.plugin.pytorch import DALIClassificationIterator
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import math


class HybridTrainPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, augment=True, dali_cpu=False):
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=0, num_shards=1, random_shuffle=True)
        #let user decide which pipeline works him bets for RN version he runs
        dali_device = 'cpu' if dali_cpu else 'gpu'
        decoder_device = 'cpu' if dali_cpu else 'mixed'
        # This padding sets the size of the internal nvJPEG buffers to be able to handle all images from full-sized ImageNet
        # without additional reallocations
        device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
        host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
        if augment:
            self.decode = ops.ImageDecoderRandomCrop(device=decoder_device, output_type=types.RGB,
                                                     device_memory_padding=device_memory_padding,
                                                     host_memory_padding=host_memory_padding,
                                                     random_aspect_ratio=[0.8, 1.25],
                                                     random_area=[0.08, 1.0],
                                                     num_attempts=100)
            self.res = ops.Resize(device=dali_device, resize_x=crop, resize_y=crop, interp_type=types.INTERP_LINEAR)
            self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                                output_dtype=types.FLOAT,
                                                output_layout=types.NCHW,
                                                crop=(crop, crop),
                                                image_type=types.RGB,
                                                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                                std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
            self.coin = ops.CoinFlip(probability=0.5)
        else:
            self.decode = ops.ImageDecoder(device=decoder_device, output_type=types.RGB,
                                           device_memory_padding=device_memory_padding,
                                           host_memory_padding=host_memory_padding)
            self.res = ops.Resize(device=dali_device, resize_shorter=256, interp_type=types.INTERP_LINEAR)
            self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                                output_dtype=types.FLOAT,
                                                output_layout=types.NCHW,
                                                crop=(crop, crop),
                                                image_type=types.RGB,
                                                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                                std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
            self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return [output, self.labels]


class HybridValPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size):
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.input = ops.FileReader(file_root=data_dir, shard_id=0, num_shards=1, random_shuffle=False)
        self.decode = ops.ImageDecoder(device="mixed", output_type=types.RGB)
        self.res = ops.Resize(device="gpu", resize_shorter=size, interp_type=types.INTERP_LINEAR)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std=[0.229 * 255, 0.224 * 255, 0.225 * 255])

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)
        return [output, self.labels]


def dataloader(batch_size, data_dir, augment=True, cpu=True):
    traindir = data_dir + 'train'
    testdir = data_dir + 'val'
    pipe = HybridTrainPipe(batch_size=batch_size, num_threads=8, device_id=0,
                           data_dir=traindir, crop=224, augment=augment, dali_cpu=cpu)
    pipe.build()
    trainloader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader")))
    train_iter = math.ceil(pipe.epoch_size("Reader")/batch_size)

    pipe = HybridValPipe(batch_size=batch_size, num_threads=8, device_id=0,
                         data_dir=testdir, crop=224, size=256)
    pipe.build()
    testloader = DALIClassificationIterator(pipe, size=int(pipe.epoch_size("Reader")))
    test_iter = math.ceil(pipe.epoch_size("Reader") / batch_size)
    return trainloader, train_iter, testloader, test_iter

