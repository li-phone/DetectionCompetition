import numpy as np
import requests
import cv2
import time
import os.path as osp
from mmdet.third_party.parallel import Parallel


class TestParallel(object):

    @classmethod
    def setup_class(cls):
        cls.test_size = 20
        cls.test_range = 1
        cls.init_tasks = [cls.test_range for i in range(cls.test_size)]
        cls.img_file = osp.join(osp.dirname(__file__), '../color.jpg')

    @classmethod
    def process(cls, task):
        tasks = [cls.test_range for i in range(cls.test_size)]
        for i in range(len(tasks)):
            while tasks[i] > 0:
                tasks[i] -= 1
                # img = cv2.imread(cls.img_file)
                html = requests.get("https://baidu.com")
        return dict(count=tasks)

    def test_parallel_by_single_worker(self):
        settings = dict(tasks=self.init_tasks, process=self.process, collect=['count'], workers_num=1,
                        print_process=5)
        parallel = Parallel(**settings)
        start = time.time()
        results = parallel()
        end = time.time()
        # times: 86.52158761024475s
        print('times: {}s'.format(end - start))
        assert len(results['count']) == self.test_size * self.test_size
        for x in results['count']:
            assert x == 0

    def test_parallel_by_multi_worker(self):
        settings = dict(tasks=self.init_tasks, process=self.process, collect=['count'], workers_num=10,
                        print_process=5)
        parallel = Parallel(**settings)
        start = time.time()
        results = parallel()
        end = time.time()
        # times: 12.2323739528656s
        print('times: {}s'.format(end - start))
        assert len(results['count']) == self.test_size * self.test_size
        for x in results['count']:
            assert x == 0
