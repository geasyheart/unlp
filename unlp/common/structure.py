# -*- coding: utf8 -*-

#


class History(object):
    """

    """

    def __init__(self):
        self.num_mini_batches = 0

    def step(self, gradient_accumulation):
        """

        :param gradient_accumulation:
        :return:
        """
        self.num_mini_batches += 1
        return self.num_mini_batches % gradient_accumulation == 0

    def num_training_steps(self, num_batches, gradient_accumulation):
        """

        :param num_batches:
        :param gradient_accumulation:
        :return:
        """

        return len([
            i for i in range(self.num_mini_batches + 1, self.num_mini_batches + num_batches + 1)
            if
            i % gradient_accumulation == 0
        ])
