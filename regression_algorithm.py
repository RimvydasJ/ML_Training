#Linear regression:
#y=mx+b
# m(slope) = mean(x)*mean(y) - mean(x*y) / mean(x)^2 - mean(x^2)
# b(y-intercept) = mean(y) - m*mean(x)
# r^2 = 1 - (SE(y)/SE(mean(y))) - determines how good is the line. The closer to 1 the better
# SE(y) - squared error of regression line y - original y
# SE(mean(y)) - squared error of mean y - original y

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import random

style.use('fivethirtyeight')

#xs = np.array([1,2,3,4,5,6], dtype=np.float64)
#ys = np.array([5,4,6,5,6,7], dtype=np.float64)


def create_dataset(hm, variants, step=2, correlation=False):
    val,ys = 1,[]
    for i in range(hm):
        y = val + random.randrange(-variants,variants)
        ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step
    
    xs = [i for i in range(hm)]
    return np.array(xs, dtype=np.float64), np.array(ys, dtype=np.float64)


def best_fit_slope(xs,ys):
    m = ((mean(xs)*mean(ys)) - mean(xs*ys))/(mean(xs)**2 - mean(xs**2))
    return m

def y_intercept(xs,ys,m):
    return mean(ys) - (m*mean(xs))

def squared_error(ys_origin, ys_line):
    return sum((ys_line-ys_origin)**2)


def coeficient_of_determination(ys,regression_line):
    y_mean = [mean(ys) for y in ys]
    return 1-(squared_error(ys,regression_line)/squared_error(ys,y_mean))


xs,ys = create_dataset(40, 40, 2, correlation='pos')

m = best_fit_slope(xs,ys)
b = y_intercept(xs,ys,m)

regression_line = [m*xs[i]+b for i in range(len(xs))]

#predict_x = 8
#predict_y = (m*predict_x)+b


r_squared = coeficient_of_determination(ys,regression_line)
print(r_squared)

plt.scatter(xs,ys)
plt.plot(xs,regression_line)
plt.show()



