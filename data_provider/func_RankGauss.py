import numpy as np
from collections import Counter, OrderedDict

class RankGauss_norm:
    '''Rank Gaussian Normalization 高斯标准化'''
    def __init__(self, data=None, precision=np.float32):
        #data: 1D array or list
        self._data = data
        self.precision = precision        
        self._output = None
        if self._data is None:
            self._trafo_map = None
        else:
            self.fit_transform(self._data)

    @property
    def data(self):
        return self._data

    @property
    def output(self):
        return self._output

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, p):
        if not isinstance(p, type):
            raise ValueError('precision must be a data type, e.g.: np.float64')
        self._precision = p

    def _RationalApproximation(self, t:float)->float: 
        c = [2.515517, 0.802853, 0.010328]
        d = [1.432788, 0.189269, 0.001308]
        return t - ((c[2]*t + c[1])*t + c[0]) / (((d[2]*t + d[1])*t + d[0])*t + 1.0)

    def _NormalCDFInverse(self, p:float) -> float:

        if (p <= 0.0 or p >= 1.0):
            raise Exception('0<p<1. The value of p was: {}'.format(p))
        if (p < 0.5):
            return -self._RationalApproximation(np.sqrt(-2.0*np.log(p)) )
        return self._RationalApproximation( np.sqrt(-2.0*np.log(1-p)) )

    def _vdErfInvSingle01(self, x:float) -> float:
        if x == 0:
            return 0
        elif x < 0:
            return -self._NormalCDFInverse(-x)*0.7
        else:
            return self._NormalCDFInverse(x)*0.7

    def fit_transform(self, dataIn:list) -> dict:
        self.fit(dataIn)
        return self.transform(dataIn)

    def fit(self, dataIn:list):
        self._data = dataIn
        trafoMap = OrderedDict()
        hist = Counter(dataIn)
        if len(hist) == 0:
            pass
        elif len(hist) == 1:
            key = list(hist.keys())[0]
            trafoMap[key] = 0.0
        elif len(hist) == 2:
            keys = sorted(list(hist.keys()))
            trafoMap[keys[0]] = 0.0
            trafoMap[keys[1]] = 1.0
        else:
            N = cnt = 0
            for it in hist:
                N += hist[it]
            assert (N == len(dataIn))
            mean = 0.0
            for it in sorted(list(hist.keys())):
                rankV = cnt / N
                rankV = rankV * 0.998 + 1e-3
                rankV = self._vdErfInvSingle01(rankV)
                assert(rankV >= -3.0 and rankV <= 3.0)
                mean += hist[it] * rankV
                trafoMap[it] = rankV
                cnt += hist[it]
            mean /= N
            for it in trafoMap:
                trafoMap[it] -= mean
        self._trafo_map = trafoMap
        return 

    def _binary_search(self, keys, val):
        start, end = 0, len(keys)-1
        while start+1 < end:
            mid = (start + end) // 2
            if val < keys[mid]:
                end = mid
            else:
                start = mid
        return keys[start], keys[end]

    def transform(self, dataIn:list) -> dict:
        dataOut = []
        trafoMap = self._trafo_map
        keys = list(trafoMap.keys())
        if len(keys) == 0:
            raise Exception('No transfermation map')
        for i in range(len(dataIn)):
            val = dataIn[i]
            trafoVal = 0.0
            if val <= keys[0]:
                trafoVal = trafoMap[keys[0]]
            elif val >= keys[-1]:
                trafoVal = trafoMap[keys[-1]]
            elif val in trafoMap:
                trafoVal = trafoMap[val]
            else:
                lower_key, upper_key = self._binary_search(keys, val)
                x1, y1 = lower_key, trafoMap[lower_key]
                x2, y2 = upper_key, trafoMap[upper_key]

                trafoVal = y1 + (val - x1) * (y2 - y1) / (x2 - x1)
            dataOut.append(trafoVal)
        dataOut = np.asarray(dataOut, dtype=self.precision)
        self._output = dataOut
        return self._output 