#### Time series class important code
num =[[13/12,-7/24],[-7/24,7/48]]
num = np.array(num)
num_det = np.linalg.det(num)
num_det

den = [[13/12,-7/24],[-7/24,13/12]]
den = np.array(den)
den_det = np.linalg.det(den)

# 2 loops for j and k - While creating the GPAC table