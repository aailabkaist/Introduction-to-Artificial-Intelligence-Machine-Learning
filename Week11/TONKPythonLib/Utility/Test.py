from scipy.special import polygamma, gammaln, digamma
import time

startTime = time.time()
polygamma(0,10)
print("Elapsed Time : " + str((time.time()-startTime)))

startTime = time.time()
digamma(10)
print("Elapsed Time : " + str((time.time()-startTime)))
