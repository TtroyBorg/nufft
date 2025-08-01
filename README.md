#  NUFFT (Nonuniform Fast Fourier Transform)

This repository contains a Python implementation of the ** NUFFT** (Nonuniform Fast Fourier Transform), based on the algorithm by Greengard and Lee (2004):  
**"Accelerating the Nonuniform Fast Fourier Transform"**, *SIAM Review*.

---

##  Mathematical Description

Given a set of **uniform Fourier coefficients** \( F(k_1, k_2) \), where \( k_1, k_2 \in [-M/2, M/2) \subset \mathbb{Z} \), the Type-2 NUFFT evaluates the function at **nonuniform spatial points** \( \{(x_j, y_j)\}_{j=1}^M \subset [0, 2\pi)^2 \):

\[
f(x_j, y_j) = \sum_{k_1, k_2} F(k_1, k_2) \, e^{i(k_1 x_j + k_2 y_j)}
\]

Direct evaluation of this sum is \( \mathcal{O}(M^2) \). The NUFFT algorithm accelerates this using the following steps:

---

### ⚙️ Algorithm Overview

Let:
- \( M \): number of Fourier modes per axis
- \( M_r = 2M \): oversampled FFT grid size
- \( \tau \): Gaussian width, typically \( \tau = \frac{12}{M^2} \)
- \( M_{\text{sp}} \): Gaussian spreading window size (typically 2 or 3)

1. **Gaussian Deconvolution in Fourier Space**  
   Multiply Fourier coefficients by \( G(k_1, k_2) = \exp(\tau(k_1^2 + k_2^2)) \) to undo Gaussian smoothing:

   \[
   \tilde{F}(k_1, k_2) = \sqrt{\frac{\pi}{\tau}} \, F(k_1, k_2) \, G(k_1, k_2)
   \]

2. **Zero Padding and FFT**  
   Embed \( \tilde{F} \) into a \( M_r \times M_r \) array (centered), and compute:

   \[
   f_{-\tau}(x, y) = \sum_{k_1, k_2} \tilde{F}(k_1, k_2) \, e^{i(k_1 x + k_2 y)} \approx \text{FFT}^{-1}[\tilde{F}]
   \]

3. **Gaussian Interpolation to Nonuniform Points**  
   Approximate the value at each nonuniform target \( (x_j, y_j) \) using a local Gaussian interpolation:

   \[
   f(x_j, y_j) \approx \sum_{\ell_1, \ell_2} f_{-\tau}(m_1 + \ell_1, m_2 + \ell_2) \cdot g_\tau(x_j - x_{m_1+\ell_1}, y_j - y_{m_2+\ell_2})
   \]

---

##  Accuracy Validation

The result of the NUFFT is compared with the **direct sum**:

```python
f(x_j, y_j) = sum(F[k1, k2] * exp(1j * (k1 * x_j + k2 * y_j)))
