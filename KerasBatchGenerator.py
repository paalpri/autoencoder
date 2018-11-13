import numpy as np
class KerasBatchGenerator(object):

    def __init__(self, data, num_steps, batch_size, skip_step):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size
        # this will track the progress of the batches sequentially through the
        # data set - once the data reaches the end of the data set it will reset
        # back to zero
        self.current_idx = 0
        # skip_step is the number of words which will be skipped before the next
        # batch is skimmed from the data set
        self.skip_step = skip_step

    def generate(self):
        x = np.zeros((self.batch_size, 1,self.num_steps))
        while True:
            for i in range(self.batch_size):
                if self.current_idx + self.num_steps >= len(self.data):
                    # reset the index back to the start of the data set
                    self.current_idx = 0
                x[i, :, :] = np.expand_dims(self.data[self.current_idx], axis=0)
                self.current_idx += self.skip_step
            yield x, x