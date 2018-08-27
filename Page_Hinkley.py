class Hinkley_test:
    def __init__(self, delta=0.0000005, lambda_=2, alpha=1 - 0.0001):
        """

        :param delta: Magnitude of changes
        :param lambda_: the threshold to detect drift
        :param alpha: the weight of a given element in the datastream (a.k.a fading factor)
        """
        self.delta = delta
        self.lambda_ = lambda_
        self.alpha_ = alpha
        self.sum = 0
        # incrementally calculated mean of input data
        self.x_item_mean = 0
        # number of  values in the data stream
        self.num = 0
        self.is_change_detected = False

    def reset_parameters_(self):
        """
        Reset the parameters, each time a drift has been detected
        :return:
        """
        self.num = 0
        self.x_item_mean = 0
        self.sum = 0

    def set_input(self, x):
        """
        It helps to incrementally add a value to the PH-test and check for each value added, if there is a drift or not
        :param x: value from the data stream
        :return: boolean, isChangeDetected or not in the datastream
        """
        self.detect_drift_(x)
        return self.is_change_detected

    def detect_drift_(self, x):
        """
        Concept drift detection following the formula from 'Knowledge Discovery from Data Streams' by João Gamma (p. 76)
        :param x: input data
        """
        # calculate the average and sum
        self.num += 1
        self.x_item_mean = (x + self.x_item_mean * (self.num - 1)) / self.num
        # self.x_mean = self.x_mean + (x + self.x_mean * (self.num - 1)) / self.num
        self.sum = self.sum * self.alpha_ + x - self.x_item_mean - self.delta

        if self.sum > self.lambda_:
            self.is_change_detected = True
        else:
            self.is_change_detected = False
            # print("Changed detected from Page_hinkley", self.change_detected)
        if self.is_change_detected:
            self.reset_parameters_()
        return self.is_change_detected

    def point_of_drift(self):
        return self.num

