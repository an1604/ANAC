import threading


class Stream(object):
    def __init__(self):
        self.streams = list()
        self.task_list = list()
        self.action = None
        self.stopFlag = False
        self.myThread = threading.Thread(target=self.process_data)
        self.myThread.start()

    def process_data(self):
        while not self.stopFlag:
            if self.task_list and self.action is not None:
                x = self.task_list.pop()
                y = self.action(x)
                if isinstance(y, bool):
                    if y:
                        self.streams[-1].add(x)
                elif isinstance(y, (int, float)):
                    self.streams[-1].add(y)

    def add(self, item):
        self.task_list.append(item)

    def forEach(self, action):
        self.action = action

    def stop(self):
        self.stopFlag = True
        if self.streams:
            for stream in self.streams:
                stream.stop()

    def apply(self, action):
        self.action = action
        self.streams.append(Stream())
        return self.streams[-1]
