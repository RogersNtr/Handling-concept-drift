import numpy as np
from math import log, sqrt
from ADWIN.List import *


class AdwinAlgo:
    def __init__(self, M):
        """

        :param M: Confidence on the result of the Algorithm
        """
        self.MINTCLOCK = 1
        self.MINLENGTHWINDOW = 16 # Capacity
        self.DELTA = 0.01
        self.MAXBUCKETS = M

        self.mintTime = 0
        self.mintClock = self.MINTCLOCK
        self.mdblError = 0.0
        self.mdblWidth = 0.0

        # Bucket
        self.bucketNumber = 0
        self.bucketList = AdwinList(self.MAXBUCKETS)
        self.lastBucketRow = 0

        self.W = 0 # Width of the window
        self.sum = 0
        self.var = 0

    def update(self, value):
        self.insert_element(value)
        self.compress_buckets()
        return self.check_drift()

    def insert_element(self, value_to_insert):
        # Insert new Bucket at the head of the current bucket
        self.W += 1
        self.bucketList.head.add_back(value_to_insert, 0.0)
        self.bucketNumber += 1

        # Updating statistics
        if self.W > 1:
            self.var += (self.W - 1) * (value_to_insert - self.sum/(self.W - 1)) * (value_to_insert - self.sum/(self.W - 1))/self.W

        self.sum += value_to_insert

    def compress_buckets(self):
        # Traverse the list of Buckets in Increasing order
        cursor = self.bucketList.head
        nextNode = None
        i = 0
        while True:
            # Number of buckets in a row
            k = cursor.size
            # merge buckets if row full (row > M +1)
            if k == self.MAXBUCKETS + 1:
                nextNode = cursor.next
                if nextNode is None:
                    self.bucketList.add_to_tail()
                    nextNode = cursor.next
                    self.lastBucketRow += 1

                n1 = self.bucketSize(i)
                n2 = self.bucketSize(i)
                u1 = cursor.sum[0]/n1
                u2 = cursor.sum[0]/n2
                incVariance = n1*n2*(u1 - u2)*(u1 - u2)/(n1 + n2)
                nextNode.add_back(cursor.sum[0] + cursor.sum[1], cursor.variance[0] + cursor.variance[1] + incVariance)
                self.bucketNumber -=1
                cursor.drop_front(2)
                if(nextNode.size <= self.MAXBUCKETS):
                    break
            else:
                break

            cursor = cursor.next
            i += 1
            if cursor is None:
                break

    def bucketSize(self, row):
        return pow(2, row)

    def check_drift(self):
        change = False
        quit_ = False
        it = None
        self.mintTime += 1

        if (self.mintTime % self.mintClock) and (self.W > self.MINLENGTHWINDOW):
            boolTalla = True
            while boolTalla:
                boolTalla = False
                quit_ = False
                n0 = 0
                n1 = self.W
                u0 = 0
                u1 = self.sum
                it = self.bucketList.tail
                i = self.lastBucketRow
                while True:
                    for k in range(0, it.size):
                        if i==0 and k==it.size - 1:
                            quit_ = True
                            break

                        n0 += self.bucketSize(i)
                        n1 -= self.bucketSize(i)
                        u0 += it.sum[k]
                        u1 -= it.sum[k]

                        mintMinWinLength = 5
                        if(n0 >= mintMinWinLength and n1 >= mintMinWinLength and self.cut_expression(n0, n1, u0, u1)):
                            boolTalla = True
                            change = True

                            if self.W > 0:
                                self.delete_element()
                                quit_ = True
                                break
                    it = it.prev
                    i -= 1
                    if ~quit_ and (it is not None):
                        break

        return change

    def delete_element(self):
        node = self.bucketList.tail
        n1 = self.bucketSize(self.lastBucketRow)
        self.W -=n1
        self.sum -= node.sum[0]
        u1 = node.sum[0]/n1
        incVariance = node.variance[0] + n1*self.W*(u1 - self.sum/self.W)*(u1 - self.sum/self.W)/(n1 + self.W)
        self.var -= incVariance

        # Delete Bucket
        node.drop_front()
        self.bucketNumber -= 1
        if node.size == 0:
            self.bucketList.remove_from_tail()
            self.lastBucketRow -= 1

    def cut_expression(self, N0, N1, u0, u1):
        n0 = N0
        n1 = N1
        n = self.W
        diff = u0/n0 - u1/n1

        v = self.var/self.W
        dd = log(2.0 * log(n)/self.DELTA)

        mintMinWinLength = 5
        m = (1.0 / (n0 - mintMinWinLength + 1)) + (1.0 / (n1 - mintMinWinLength + 1))
        epsilon = sqrt(2*m*v*dd) + (2.0/3.0)*dd*m

        if abs(diff) > epsilon:
            return True
        else:
            return False

    def get_estimation(self):
        if self.W > 0:
            return self.sum / self.W
        else:
            return 0

    def length(self):
        return self.W

    def print_res(self):
        it = self.bucketList.tail
        if it is None:
            print("It Null")

        i = self.lastBucketRow
        while True:
            for k in range(it.size - 1, -1, -1):
                print(str(i) + " [" + str(it.sum[k]) + " de " + str(self.bucketSize(i)) + "],")

            print()
            it = it.prev
            i -= 1
            if it is None:
                break

