class PH_test:
    """
    This compute the  page Hinkley test for detecting concept drift.
    Refer to the book  Knowledge discovery and dataScream
    """
    def __init__(self, datastream=None, delta_=0.005, lambda_=50, alpha_=1 - 0.0001):
        """
        :param delta_: the magnitude of change to be detected
        :param lambda_: a threshold related to the FAR that is allowed
        :param alpha_:The weight coefficient (or the fading factor)
        :param datastream the data to detect drift on.(should be a panda series or list type)
        """
        self.delta_ = delta_
        self.lambda_= lambda_
        self.alpha_ = alpha_
        self.data = datastream
        # print("data", type(self.data))
        self.x_item_mean = 0 # this represent, the mean at each iteration
        self.num_iter = 0 # Number of iteration done so far.
        self.average = 0
        self.isChangeDetected = False

    def set_data(self, datastream):
        self.data = datastream

    def reset_params(self):
        self.x_item_mean = 0
        self.num_iter = 0
        self.average = 0

    def detect_drift(self):
        """
        :return: if change has been detected or not.
        """
        list_average = []
        if self.data is not None:
            for x_t in self.data:
                print(x_t)
                self.num_iter = self.num_iter + 1
                self.x_item_mean = (x_t + self.x_item_mean*(self.num_iter - 1))/self.num_iter
                self.average = self.average*self.alpha_ + (x_t - self.x_item_mean - self.delta_)
                list_average.append(self.average)
            Mt = min(list_average)
            Pht = self.average - Mt # The PH test
            # if Pht > self.lambda_:
            if self.average > self.lambda_:
                self.isChangeDetected = True
                # self.reset_params()
                print("Drift detected at positon : " + str(self.num_iter))
            else:
                self.isChangeDetected = False
        else:
            print("you have not provided a datascream")

        return self.isChangeDetected

    def position_drift(self):
        """

        :return: position of the drift (where the drift occurs)
        """
        return self.num_iter