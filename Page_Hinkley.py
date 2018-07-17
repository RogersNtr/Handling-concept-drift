class PageHinkley:
    def __init__(self, delta=0.005, lambda_=50, alpha=1 - 0.0001):
        """

        :param delta: Magnitude of changes
        :param lambda_: tth ethreshold to detect drift
        :param alpha: the weight of a given element in the datastream
        """
        self.delta = delta
        self.lambda_ = lambda_
        self.alpha_ = alpha
        self.sum = 0
        # incrementally calculated mean of input data
        self.x_mean = 0
        # number of considered values
        self.num = 0
        self.change_detected = False

    def reset_params(self):
        """
        Every time a change has been detected, all the collected statistics are reset.
        :return:
        """
        self.num = 0
        self.x_mean = 0
        self.sum = 0

    def set_input(self, x):
        """
        Main method for adding a new data value and automatically detect a possible concept drift.
        :param x: input data
        :return: boolean
        """
        self.detect_drift(x)
        return self.change_detected

    def detect_drift(self, x):
        """
        Concept drift detection following the formula from 'Knowledge Discovery from Data Streams' by João Gamma (p. 76)
        :param x: input data
        """
        # calculate the average and sum
        self.num += 1
        self.x_mean = (x + self.x_mean * (self.num - 1)) / self.num
        self.sum = self.sum * self.alpha_ + x - self.x_mean - self.delta

        self.change_detected = True if self.sum > self.lambda_ else False
        # print("Changed detected from Page_hinkley", self.change_detected)
        if self.change_detected:
            self.reset_params()
        return self.change_detected

    def point_of_drift(self):
        return self.num

