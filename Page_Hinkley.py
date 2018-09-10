class Hinkley_test:
    def __init__(self, delta=0.0000005, lambda_=2, alpha=1 - 0.0001):
        """

        :param delta: Magnitude of changes
        :param lambda_: a threshold related to the FAR that is allowed and hence to detect drift.
        :param alpha: the weight of a given element in the datastream (a.k.a fading factor)
        """
        self.delta = delta
        self.lambda_ = lambda_
        self.alpha_ = alpha
        self.average = 0
        self.x_item_mean = 0   # this represent, the mean at each iteration
        self.num_iter = 0  # Number of iteration done so far.
        self.is_change_detected = False

    def reset_parameters_(self):
        """
        Reset the parameters, each time a drift has been detected
        :return:
        """
        self.num_iter = 0
        self.x_item_mean = 0
        self.average = 0

    def set_data(self, x_item):
        """
        Incrementally add a value to the PH-test and directly check for drift as far as at item is added.
        :param x_item: value or instance add from the datastream
        :return: boolean, True if a change occurred in the data stream, False otherwise.
        """
        self.detect_drift_(x_item)
        return self.is_change_detected

    def detect_drift_(self, x_item):
        """
        Concept drift detection from 'Knowledge Discovery from Data Streams' by João Gamma (p. 76)
        :param x_item: input data
        """
        self.num_iter += 1
        self.x_item_mean = (x_item + self.x_item_mean * (self.num_iter - 1)) / self.num_iter
        # self.x_mean = self.x_mean + (x + self.x_mean * (self.num - 1)) / self.num
        self.average = self.average * self.alpha_ + x_item - self.x_item_mean - self.delta

        if self.average > self.lambda_:
            self.is_change_detected = True
        else:
            self.is_change_detected = False
            # print("Changed detected from Page_hinkley", self.change_detected)
        if self.is_change_detected:
            self.reset_parameters_()
        return self.is_change_detected

    def point_of_drift(self):
        """
        :return: position of the drift (where the drift occurs)
        """
        return self.num_iter

