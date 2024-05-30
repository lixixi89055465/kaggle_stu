# -*- coding: utf-8 -*-
# @Time : 2024/5/30 10:51
# @Author : nanji
# @Site : 
# @File : testset_printoptions01.py
# @Software: PyCharm 
# @Comment :
import numpy as np

# np.set_printoptions(precision=4)
# print('0' * 100)
# print(np.array([1.2324234232]))


# import numpy as np
# np.set_printoptions(threshold=9)
# n1=np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
# print(n1)

# import numpy as np
# import math
# np.set_printoptions(suppress=True)
# n2=np.array([1e-10,-1e-10])
# print(n2)

# import numpy as np
# np.set_printoptions(linewidth=5)
# n3=np.array([20000,100])
# print(n3)

import numpy as np
np.set_printoptions(precision=4)
n4=np.array([1.223432423])
print(n4)
