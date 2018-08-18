
class ListNode:
    def __init__(self, M):
        self.M = M
        self.size = 0
        self.sum = []
        self.variance = []
        for i in range(0, self.M + 1):
            self.sum.append(i)
            self.variance.append(i)

        self.next = None
        self.prev = None

    def add_back(self, value, var):
        self.sum[self.size] = value
        self.variance[self.size] = var
        self.size += 1

    def drop_front(self, n=1):

        # To drop the first n elements
        for k in range(n, self.M+1):
            self.sum[k - n] = self.sum[k]
            self.variance[k - n] = self.variance[k]

        for k in range(0, n):
            self.sum[self.M - k] = 0
            self.variance[self.M - k] = 0

        self.size -= n

