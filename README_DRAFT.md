# Multi-Image Phasor Analysis for FLIM Data - A quick guide to setup and implementation

> **[PLACEHOLDER — TODO: Once I get repository access]**
> *`setup.bash` and `requirements.txt` will be added to the root directory to automate Conda environment creation and package installation.*

### Option A: Automated Conda Environment Setup (Recommended)
1. Clone the repository and navigate to the project root directory:
```bash
git clone https://github.com/SohailLab/R5-Analysis
cd R5-Analysis
```
2. Run the automated setup script to create a Conda environment and install all required dependencies:
```bash
bash setup.bash
```

(Placeholder: `setup.bash` will initialize the Conda environment and install `requirements.txt` packages including `tkinter`, `sv_ttk`, `scikit-learn`, `scipy`, and `scikit-image`.)
3. Activate the newly created Conda environment:
```bash
conda activate R5_analysis_env

```
### Option B: Manual Installation via `requirements.txt`

1. Set up and activate your preferred Python virtual environment or Conda environment.
2. Install the required packages from `requirements.txt`:
* **macOS/Linux:** `python3 -m pip install -r requirements.txt`
* **Windows:** `py -m pip install -r requirements.txt`
### Launching the Application

Run `GUI2.py` to launch the graphical user interface:
```bash
python GUI2.py
```
---

# Initial Overview & Workflow Methods
This application processes Fluorescence Lifetime Imaging Microscopy (FLIM) data through a graphical user interface built with the `tkinter` and `sv_ttk` libraries. The workflow converts raw time-domain photon arrival histograms into frequency-domain phasor coordinates, pools datasets for global statistical modeling, and segments spatial structures.
## 1. Preprocessing and Calibration
This stage converts raw time-domain photon counts into calibrated frequency-domain phasors $(G, S)$ while filtering out background noise.

### A. Spatial Binning, Normalization, and Thresholding
Fluorescence decay is a Poissonian process in which the Signal-to-Noise Ratio (SNR) scales with the square root of the photon counts ($\sqrt{N_{ph}}$). By summing photon counts from adjacent pixels, spatial binning increases $N_{ph}$ and reduces the variance of the resulting phasor coordinates.
1. **Spatial Integration:** The software applies $B \times B$ spatial binning to the raw intensity image (typically set to $3 \times 3$ in our workflow):

$$
I_{\text{binned}}(x,y,t) = \sum_{i=0}^{B-1} \sum_{j=0}^{B-1} I_{\text{raw}}(x+i, y+j, t)
$$

2. **Normalization:** The binned intensity data is max-normalized to standardize signal levels across different input files.
### B. Phasor Transformation
For each pixel passing the intensity threshold, the normalized and binned time-domain fluorescence decay $I(t)$ is transformed into frequency-domain coordinates $(G, S)$ via a discrete Fourier transform evaluated at a specific harmonic frequency:

$$
g(\omega) = \frac{\int_0^T I(t) \cos(\omega t) \, dt}{\int_0^T I(t) \, dt}, \quad s(\omega) = \frac{\int_0^T I(t) \sin(\omega t) \, dt}{\int_0^T I(t) \, dt}
$$

where $\omega = 2\pi f \cdot h$ ($f$ is the laser repetition rate and $h$ is the harmonic number).

### C. Calibration (Coordinate Rotation & Scaling)
Using `get_calibration_parameters` and `_preprocess_file`, the raw phasor coordinates are corrected for the Instrument Response Function (IRF).
The measured fluorescence decay curve, $f_m(t)$, is the convolution of the true sample fluorescence decay, $f_t(t)$, with the instrument response $IRF(t)$, plus noise $N(t)$:
$$
f_m(t) = [f_t(t) * IRF(t)] + N(t)
$$
In the frequency domain, time-domain convolution simplifies to multiplication, allowing deconvolution via complex division:
$$
F_m(i\omega) = F_t(i\omega) \times IRF(i\omega) \implies F_t(i\omega) = \frac{F_m(i\omega)}{IRF(i\omega)}
$$

Expressing the IRF in polar form ($M_{IRF}e^{-i\phi_{IRF}}$), this division corresponds to a scaling by the modulation factor $M_{IRF}$ and a rotation by the phase angle $\phi_{IRF}$:

$$\frac{G_m + iS_m}{M_{IRF}e^{-i\phi_{IRF}}} = G_t + iS_t$$
$$\vec{F}_t = \frac{1}{M_{IRF}} \begin{bmatrix} \cos(\phi_{IRF}) & \sin(\phi_{IRF}) \\ -\sin(\phi_{IRF}) & \cos(\phi_{IRF}) \end{bmatrix} \vec{F}_m$$

* $\vec{F}_m = (G_m, S_m)^T$ is the vector of measured phasor coordinates.
* $\vec{F}_t = (G_t, S_t)^T$ is the vector of calibrated (true) phasor coordinates.

We use , **Coumarin 6 in EtOH** ($\tau \approx 2.5\text{ ns}$)  as a known mono-exponential reference standard to determine the phase angle ($\phi_{IRF}$) and modulation ($M_{IRF}$) calibration factors at frequency $\omega$.

---

## 2. Global Phasor GMM Clustering

**Function:** `run_gmm_fitting_workflow`

Rather than fitting models to individual images in isolation, this workflow employs a **Global Gaussian Mixture Model (GMM)** approach. Processed $(G, S)$ phasor coordinates above the $15\%$ intensity threshold from all input files are pooled into a single global dataset.
GMM is a probabilistic (soft) clustering method that assumes the data arises from a mixture of underlying Gaussian distributions. This approach is particularly effective for phasor clustering because the spatial noise distribution of phasor coordinates in the $G\text{–}S$ plane naturally follows a mixture of Gaussians.

### A. The Gaussian Mixture Model (GMM)
We model the probability density function (PDF) of the pooled phasor coordinates $\mathbf{x} = (g, s)^T$ as a weighted sum of $K$ bivariate Gaussian components:
$$
p(\mathbf{x}) = \sum_{k=1}^{K} \phi_k \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$

* $\mathbf{x}$ is the 2D coordinate vector $(g, s)$.
* $\phi_k$ (or $\pi_k$) is the mixing weight for component $k$, satisfying $\sum_{k=1}^{K} \phi_k = 1$.
* $\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)$ is the bivariate Gaussian probability density function for component $k$:

$$
\mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k) = \frac{1}{2\pi \vert{}\boldsymbol{\Sigma}_k\vert{}^{1/2}} \exp \left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}_k^{-1} (\mathbf{x} - \boldsymbol{\mu}_k) \right)
$$

### B. Expectation-Maximization (EM) Algorithm
The pooled dataset is clustered using the Expectation-Maximization (EM) algorithm implemented in `sklearn.mixture.GaussianMixture` to find the maximum likelihood estimates of the parameters:
1. **Initialization:** Select initial component means ($\boldsymbol{\mu}_1, \dots, \boldsymbol{\mu}_K$), $2 \times 2$ covariance matrices ($\boldsymbol{\Sigma}_1, \dots, \boldsymbol{\Sigma}_K$), and mixing weights ($\phi_1, \dots, \phi_K$).
2. **Expectation (E-step):** Calculate the "responsibility" $\gamma_{ik}$ (posterior probability) that component $k$ generated data point $\mathbf{x}_i$:
$$
\gamma_{ik} = \frac{\phi_k \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)}{\sum_{j=1}^{K} \phi_j \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j, \boldsymbol{\Sigma}_j)}
$$

3. **Maximization (M-step):** Update the model parameters using the responsibilities from the E-step to maximize the log-likelihood across all $N$ pooled pixels:
* **Mixing Weights:** Updated to the average responsibility of component $k$ across all data points:
$$
\phi_k = \frac{1}{N} \sum_{i=1}^{N} \gamma_{ik}
$$

* **Means:** Updated to the responsibility-weighted average of all data points:
$$
\boldsymbol{\mu}_k = \frac{\sum_{i=1}^{N} \gamma_{ik} \mathbf{x}_i}{\sum_{i=1}^{N} \gamma_{ik}}
$$
* **Covariance Matrices:** Updated based on the new component means and normalized by total responsibility:
$$
\boldsymbol{\Sigma}_k = \frac{\sum_{i=1}^{N} \gamma_{ik} (\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^T}{\sum_{i=1}^{N} \gamma_{ik}}
$$


The algorithm alternates between the E-step and M-step until convergence. Because EM is an iterative optimization process, is is susceptibility to local minima and sensitivity to initial parameter values.

### C. Model Selection ($K$ Components)
[NOTE this section may change depending on the code attached]
The optimal number of components $K$ can be defined manually or selected quantitatively:
* **Auto-Find via BIC (`gmm_auto_find = True`):** The GUI can automatically iterate through component numbers and select the model that minimizes the Bayesian Information Criterion (BIC) score to prevent overfitting:
$$
\text{BIC} = k \ln(N) - 2 \ln(\hat{L})
$$

where $\hat{L}$ is the maximized likelihood, $N$ is the number of data points, and $k$ is the number of free parameters.
* **Deviation from the Standard "Elbow" Method & Manual Selection:** In high-dimensional or multi-cluster clustering, standard practice often relies on plotting the BIC curve and visually identifying an "elbow" where marginal gains level off. However, because our biological FLIM datasets typically contain a very small number of distinct fluorescent lifetime species (e.g., $K = 2$ or $3$), a pronounced elbow is rarely present. Consequently, our workflow either minimizes the BIC penalty directly or uses a manual component input informed by experimental controls. In practice, most datasets are processed using manual component selection.
### D. Post-BIC Component Validation: Ellipse Overlap Check (`check_ellipse_overlap`)

Relying solely on BIC minimization can still lead to over-splitting or redundant components, which is addressed in a subsequent validation step. After seeding initial cluster centers via $K$-means and fitting the GMM, the workflow calls `check_ellipse_overlap` to evaluate whether the $1\text{st}$ standard deviation ($1\sigma$) hyperellipses of any two Gaussian components overlap.
This geometric overlap test is formulated using the **squared Mahalanobis distance**:
$$
D^2 = (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})
$$

* $\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$ is a 2D coordinate vector in the $(G, S)$ phasor plane.

* $\boldsymbol{\mu} = \begin{bmatrix} \mu_1 \\ \mu_2 \end{bmatrix}$ is the component mean vector.

* $\boldsymbol{\Sigma}$ is the $2 \times 2$ covariance matrix (with determinant $\vert{}\boldsymbol{\Sigma}\vert{}$).
**Algorithm Workflow:**
1. **Input Parameters:** Pass the fitted Gaussian components ($\boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k$) and the number of boundary points $n$ to sample for comparison.
2. **Generate Pairwise Combinations:** Construct a list of all unique component pairs to compare (e.g., $C_1 \text{ vs } C_2$, $C_1 \text{ vs } C_3$, $C_2 \text{ vs } C_3$, $\dots$).
3. **Iterative Boundary Evaluation:** Loop through each component pair $(C_i, C_j)$:
* Define the $1\sigma$ ellipse boundary of component $C_i$ using the locus of points where $D^2 = 1$:
$$
1 = (\mathbf{x} - \boldsymbol{\mu}_i)^T \boldsymbol{\Sigma}_i^{-1} (\mathbf{x} - \boldsymbol{\mu}_i)
$$


* Generate $n$ discrete points along the $1\sigma$ boundary of $C_i$ and evaluate whether any of those points fall inside the $1\sigma$ boundary of component $C_j$ (i.e., satisfying $(\mathbf{x} - \boldsymbol{\mu}_j)^T \boldsymbol{\Sigma}_j^{-1} (\mathbf{x} - \boldsymbol{\mu}_j) \le 1$).
* If any boundary point of $C_i$ lies within $C_j$, immediately return `True` (indicating component overlap).
1. **Final Verdict:** If all pairwise comparisons complete without intersection, return `False` (confirming well-separated components).
---
## 3. Image Segmentation: Global GMM Reconstruction & Application

**Function:** `run_gmm_application_workflow`

Once the global GMM parameters ($\phi_k, \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k$) are learned from the pooled dataset, the software reconstructs the GMM for each individual image. To generate spatial overlays, the soft-clustered posterior probabilities must be converted into hard-clustered pixel assignments. Two distinct segmentation strategies are implemented in the GUI:

### A. Neighbor Pixel Segmentation ("Neighbor Assign" / Hard Cluster Overlay)
This pixel-wise refinement method uses spatial context to resolve ambiguous, low-confidence pixels:
1. **Posterior Probability Calculation:** For each pixel $\mathbf{x}_i$, the posterior probability $\gamma_{ik}$ of belonging to component $k$ is computed.
2. **High-Confidence Assignment:** Pixels with an assignment probability $\gamma_{ik} \ge 0.60$ ($60\%$ confidence) are directly assigned to their highest-likelihood component.
3. **Neighborhood Voting for Low-Confidence Pixels:** For pixels with a maximum confidence below $60\%$ ($\gamma_{ik} < 0.60$), the algorithm inspects the $8$ nearest spatial neighbors within a $3 \times 3$ window. The ambiguous pixel is reassigned to the majority component among its confident neighbors using `scipy.stats.mode`.



### B. Aggregate-Based Classification ("Aggregate Assign" Overlay)
This object-level method assumes that each contiguous biological aggregate exhibits a single distinct fluorescence lifetime and assigns the entire structure to its most probable GMM component:
1. **Binary Masking & Connected Components:** A binary mask (using the $15\%$ maximum intensity threshold) is applied to the intensity image, and Connected Components Analysis (`skimage.measure.label`) is used to segment and label individual contiguous aggregates.
2. **Aggregate Log-Likelihood Summation:** For each labeled aggregate $A$, the total log-likelihood of its phasor cloud belonging to component $k$ is calculated by summing the log-likelihood of every constituent pixel $\mathbf{x}_m \in A$ using `scipy.stats.multivariate_normal.logpdf`:
$$
\mathcal{L}_k(A) = \sum_{\mathbf{x}_m \in A} \ln \mathcal{N}(\mathbf{x}_m \mid \boldsymbol{\mu}_k, \boldsymbol{\Sigma}_k)
$$


3. **Maximum Likelihood Assignment:** The entire aggregate is assigned the color of the component $k$ that maximizes the total log-likelihood:
$$
\text{Cluster}(A) = \underset{k}{\mathrm{argmax}} \ \mathcal{L}_k(A)
$$


> **Comment on Methodological Trade-off:** While Aggregate-Based Classification produces a cleaner visual overlay than pixel-wise segmentation, users should carefully evaluate whether the single-lifetime-per-aggregate assumption holds for their specific biological system...



