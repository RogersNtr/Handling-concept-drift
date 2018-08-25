class cumsumDM:
    def __init__(self, delta_=0.005, lambda_=20):
        self.minNumInstancesOption = 30 # 0
        self.delta_ = delta_
        self.lambda_ = lambda_
        self.data = 0
        # print("data", type(self.data))
        self.x_item_mean = 0.0  # this represent, the mean at each iteration
        self.num_iter = 0  # Number of iteration done so far.
        self.sum = 0
        self.estimation = 0.0
        self.isChangeDetected = False
        self.isInitialized = True

    def reset_params(self):
        self.num_iter = 0
        self.x_item_mean = 0.0
        self.sum = 0.0

    def input(self, x):
        # It monitors the error rate
        # if self.isChangeDetected or ~self.isInitialized:
        #     self.reset_params()
        #     self.isInitialized = True
        self.num_iter += 1
        self.x_item_mean = self.x_item_mean + (x - self.x_item_mean) / self.num_iter
        self.sum = max(0, self.sum + x -self.x_item_mean - self.delta_)


        if self.sum > self.lambda_:
            self.isChangeDetected = True
            self.reset_params()



        # print(prediction + " " + m_n + " " + (m_p + m_s) + " ");
        # self.estimation = self.x_item_mean
        # self.isChangeDetected = False

        # if self.num_iter < self.minNumInstancesOption:
        #     return
        return self.isChangeDetected


