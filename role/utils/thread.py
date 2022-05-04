import threading


class custom_Thread(threading.Thread):
    def __init__(self, target, args=()):
        super(custom_Thread, self).__init__()
        self.target = target
        self.args = args

    def run(self):
        self.result = self.target(*self.args)

    def get_result(self):
        try:
            self.join()
            return self.result
        except Exception:
            return None
