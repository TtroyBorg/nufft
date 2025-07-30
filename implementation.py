
import numpy as np
from scipy.fftpack import ifft2,fft2
import matplotlib.pyplot as plt
import time
dtype=np.complex128

def nufft_type2_2d(F_k, k1_values, k2_values, x_j, y_j, Mr, Msp, tau):
    M = len(x_j)

    G_k = np.exp(tau * (k1_values[:, None]**2 + k2_values**2))
  #deconvolve
   # F_k_tilde = F_k * G_k * np.sqrt(np.pi/tau)
    F_k_tilde=np.zeros((Mr,Mr), dtype=complex)

    for k1 in range(int(Mr/2)):
        for k2 in range(int(Mr/2 )):
          F_k_tilde[k1, k2] = (np.sqrt(np.pi / tau)) * G_k[k1,k2]  * F_k[k1, k2]
    #center = Mr // 2
   # half_M = Mr // 2
    #for k1 in range(-half_M, half_M):
    # for k2 in range(-half_M, half_M):
      # F_k_tilde[center + k1, center + k2] = np.sqrt(np.pi / tau) * np.exp(tau * (k1**2 + k2**2)) * F_k[half_M + k1, half_M + k2]


    #FFt- like sum
    f_tau=fft2d_r(F_k_tilde)
    #x_shifted = np.fft.ifftshift(F_k_tilde)
    #fft_rows = np.array([fft1d_r(row) for row in x_shifted])
    #fft_cols = np.array([fft1d_r(col) for col in fft_rows.T]).T
    #f_tau=fft2d_r(F_k_tilde)


  #precompute E3
    l_values=np.arange(Msp)
    E3 = np.exp(-((np.pi * l_values / Mr)**2) / tau)

    f_xy = np.zeros((M, M), dtype=complex)

    for j in range(M):
        for i in range(M):
            s1 = np.floor(x_j[i] * (Mr / (2*np.pi))) * (2*np.pi / Mr)
            s2 = np.floor(y_j[j] * (Mr / (2*np.pi))) * (2*np.pi / Mr)
            E1 = np.exp(-((x_j[i] - s1)**2 + (y_j[j] - s2)**2) / (4 * tau))
            E2x = np.exp((x_j[i] - s1) * np.pi / (Mr * tau))
            E2y = np.exp((y_j[j] - s2) * np.pi / (Mr * tau))
            # Compute V_0
            idx1 = min(int(np.floor(s1 * Mr / (2*np.pi))), Mr - 1)
            idx2 = min(int(np.floor(s2 * Mr / (2*np.pi))), Mr - 1)
            #V0 = f_tau[idx1, idx2] * E1
            # Apply spreading over Msp grid points
            for l2 in range(-Msp + 1, Msp):
                #Vy = V0 * (E2y**l2)*E3[abs(l2)]
                for l1 in range(-Msp + 1, Msp):
                  if 0<= idx1+l1<Mr and 0<= idx2+l2<Mr :
                    f_xy[i , j ] += f_tau[idx1 +l1, idx2+l2] * E1 * (E2y**l2)*E3[abs(l2)] * (E2x**l1/Mr) * E3[abs(l1)]/Mr

    return  f_xy

def compute_double_sum2(F, k1, k2, x, y):
    Mr = F.shape[0]
    M = len(x)
    #K1, K2 = np.meshgrid(k1, k2, indexing='ij')

    f_xy = np.zeros((M, M), dtype=complex)


    for i in range(M):
        for j in range(M):
            sUm = np.exp(1j * (k1[:, None] * x[i] + k2[None, :] * y[j]))
            f_xy[i, j] = np.sum(F * sUm)

    return f_xy

def fft1d_r(x):
    N = x.shape[0]
    if N == 1:
      return x
    even = fft1d_r(x[::2])
    odd = fft1d_r(x[1::2])
    twiddle = np.exp(2j * np.pi * np.arange(N // 2) / N)
    T = twiddle * odd
    return np.concatenate([even + T, even - T])

def fft2d_r(x):
    fft_rows = np.array([fft1d_r(row) for row in x])
    fft_cols = np.array([fft1d_r(col) for col in fft_rows.T]).T

    return fft_cols

def sinc_function(kx, ky):

    return np.sinc(kx / np.pi) * np.sinc(ky / np.pi)

M, Mr, Msp, tau = 8, 16, 8, 0.02
random_points = np.random.uniform(0, 2*np.pi, M)
x_j  = np.sort(random_points)
y_j = np.linspace(0, 2*np.pi, M)
k1_values=np.linspace(0, Mr, Mr)
k2_values=np.linspace(0, Mr, Mr)

#x=np.linspace(0, 2*np.pi, Mr)
#y=np.linspace(0, 2*np.pi, Mr)
f_k= 10*np.exp(-tau * (k1_values[:, None]**2 + k2_values**2))
#F_K=fft2(f_k)
#for i in range(Mr/)
#
# random Fourier coefficients
#F_k = (np.random.rand(Mr, Mr) + 1j * np.random.rand(Mr, Mr))
#F_k=fft2(f_k)
#f_k[int(Mr//2):,:]=0
#f_k[:,int(Mr//2):]=0
# accuracy
C=compute_double_sum2(f_k, k1_values, k2_values, x_j , y_j)
N=nufft_type2_2d(f_k, k1_values , k2_values , x_j, y_j, Mr, Msp, tau)
np.linalg.norm(C-N, ord='fro')/np.linalg.norm(C, ord='fro')

Mr = [8, 16,32,64, 128,256]
execution_times = []
Msp=4

# Benchmarking loop
for u in Mr:
    start_time = time.time()
    k1_values=np.linspace(0, u, u)
    k2_values=np.linspace(0, u, u)
    f_k= 10*np.exp(-tau * (k1_values[:, None]**2 + k2_values**2))
    N=nufft_type2_2d(f_k, k1_values , k2_values , x_j, y_j, u, Msp, tau)
    execution_times.append(time.time() - start_time)

# Plot results
plt.figure(figsize=(8,5))
plt.plot(Mr, execution_times, 'bo-', label="Benchmark results")  # Blue circles and solid line
for i in range(len(Mr)):  # Dashed vertical lines at each point
    plt.axvline(x=Mr[i], ymin=0, ymax=execution_times[i]/max(execution_times),
                linestyle='dashed', color='gray', alpha=0.6)

plt.xlabel("Mr")
plt.ylabel("Execution Time (seconds)")
plt.title("Algorithm Benchmark Performance")
plt.legend()
plt.grid(True)
plt.show()

Msp = [ 1,2,4, 8,12, 16]
execution_times = []

# Benchmarking loop
for u in Msp:
    start_time = time.time()
    N=nufft_type2_2d(f_k, k1_values , k2_values , x_j, y_j, Mr, u, tau)
    execution_times.append(time.time() - start_time)

# Plot results
plt.figure(figsize=(8,5))
plt.plot(Msp, execution_times, 'bo-', label="Benchmark results")  # Blue circles and solid line
for i in range(len(Msp)):  # Dashed vertical lines at each point
    plt.axvline(x=Msp[i], ymin=0, ymax=execution_times[i]/max(execution_times),
                linestyle='dashed', color='gray', alpha=0.6)

plt.xlabel("Msp")
plt.ylabel("Execution Time (seconds)")
plt.title("Algorithm Benchmark Performance")
plt.legend()
plt.grid(True)
plt.show()
