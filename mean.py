import numpy as np
import matplotlib.pyplot as plt

x = np.array([4.6, 6.0, 2.0, 5.8])
sigma = np.array([2.0, 1.5, 5.0, 1.0])

Nmu = 1000
mu = np.linspace(0,10,Nmu)

p_xk_mu = np.zeros([np.shape(x)[0],Nmu])
for i in range(Nmu):
    p_xk_mu[:,i] = 1/np.sqrt(2*np.pi*sigma**2)*np.exp(-0.5*((x - mu[i])**2)/(sigma**2))

p_mu = 1/(np.max(mu)-np.min(mu))
p_x = np.trapz(x)
p_mu_kx = np.prod(p_xk_mu, axis = 0)*p_mu/p_x

L = np.sum(np.log(p_xk_mu), axis = 0)

arg_max = np.argmax(L)

d2L_dmu2 = (L[arg_max+1] - 2*L[arg_max] + L[arg_max-1]) / ((mu[1]-mu[0])**2)

plt.plot(mu,p_mu_kx)
plt.title('$\mu $= %f  $\pm$ %f'%(mu[arg_max],(-d2L_dmu2)**(-0.5)) )
plt.savefig('mean.pdf')