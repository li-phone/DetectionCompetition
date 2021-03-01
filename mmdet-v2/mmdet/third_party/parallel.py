import sys
import threading


class Parallel(object):
    """
        使用多线程去完成多任务资源的处理
        参数：
            tasks：任务资源
            process：处理过程函数
            collect：需要收集的数据
            workers_num：线程数量
            with_thread_lock：是否对线程加锁
        返回：
            返回results [dict]
    """

    def __init__(self, tasks, process, collect, workers_num=1, with_thread_lock=True,
                 process_params=None,
                 print_process=None):
        self.init_tasks = tasks
        self.task_size = len(tasks)
        self.process = process
        self.collect = collect
        self.workers_num = workers_num
        self.with_thread_lock = with_thread_lock
        self.process_params = {} if process_params is None else process_params
        self.print_process = print_process
        if self.with_thread_lock:
            self.thread_lock = threading.Lock()
        self.results = {k: [] for k in self.collect}

    def do_work(self):
        while len(self.init_tasks) > 0:
            # 取一个任务
            if self.with_thread_lock:
                with self.thread_lock:
                    if len(self.init_tasks) > 0:
                        task = self.init_tasks.pop()
                    else:
                        break
            else:
                if len(self.init_tasks) > 0:
                    task = self.init_tasks.pop()
                else:
                    break
            result = self.process(task, __results__=self.results, **self.process_params)
            if self.with_thread_lock:
                with self.thread_lock:
                    for k, v in result.items():
                        self.results[k].extend(v)
            else:
                for k, v in result.items():
                    self.results[k].extend(v)
            if self.print_process is not None and ((self.task_size - len(self.init_tasks)) % self.print_process == 0
                                                   or len(self.init_tasks) == 0):
                print('process {}/{}...'.format(self.task_size - len(self.init_tasks), self.task_size), flush=True)

    def __call__(self, **kwargs):
        threads = [threading.Thread(target=self.do_work)
                   for i in range(self.workers_num)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return self.results
