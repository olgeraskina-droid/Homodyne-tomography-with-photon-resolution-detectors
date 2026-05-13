import numpy as np
from scipy import linalg
from config import dim_s  # used in create_ro_s

_ubs_cache = {}

def ubs(dimension, r):
    """Unitary beam-splitter transformation matrix."""
    key = (dimension, r)
    if key in _ubs_cache:
        return _ubs_cache[key]
    a = np.zeros((dimension, dimension), dtype=np.complex128)
    for i in range(0, dimension - 1):
        a[i][i + 1] = np.sqrt(i + 1)
    a_t = a.conj().T
    aa = np.kron(a_t, a) - np.kron(a, a_t)
    u = linalg.expm((np.pi / 4 * r) * aa)
    _ubs_cache[key] = u
    return u

def density_matrix_coherent_lo(n, alpha_lo, theta):
    """Coherent state density matrix for the local oscillator."""
    alpha_complex = alpha_lo * np.exp(1j * theta)
    norm_factor = np.exp(-alpha_lo ** 2)
    sqrt_arr = np.sqrt(np.arange(n))
    sqrt_arr[0] = 1.0
    c_m = np.ones(n, dtype=complex)
    c_n = np.ones(n, dtype=complex)
    for i in range(1, n):
        c_m[i] = c_m[i-1] * alpha_complex / sqrt_arr[i]
        c_n[i] = c_n[i-1] * np.conj(alpha_complex) / sqrt_arr[i]
    ro = norm_factor * np.outer(c_m, c_n)
    return ro

def create_random_density_matrix(dim):
    """Generate a random mixed density matrix."""
    g = np.random.randn(dim, dim) + 1j * np.random.randn(dim, dim)
    rho = g @ g.conj().T
    purity = np.trace(rho @ rho).real
    type_str = f'mixed with purity {purity:.6f}'
    return rho / np.trace(rho), type_str

def density_matrix_coherent(n, alpha_s, theta):
    """Coherent state density matrix (for signal)."""
    alpha_complex = alpha_s * np.exp(1j * theta)
    norm_factor = np.exp(-alpha_s ** 2)
    sqrt_arr = np.sqrt(np.arange(n))
    sqrt_arr[0] = 1.0
    c_m = np.ones(n, dtype=complex)
    c_n = np.ones(n, dtype=complex)
    for i in range(1, n):
        c_m[i] = c_m[i-1] * alpha_complex / sqrt_arr[i]
        c_n[i] = c_n[i-1] * np.conj(alpha_complex) / sqrt_arr[i]
    ro = norm_factor * np.outer(c_m, c_n)
    ro = ro / np.trace(ro)
    type_str = 'coherent; with alpha_s = ' + str(alpha_s)
    return ro, type_str

def fock_density_matrix(n: int, N: int) -> np.ndarray:
    """Fock state |n> density matrix."""
    if n >= N:
        raise ValueError(f"n = {n} must be less than dimension N = {N}")
    rho = np.zeros((N, N), dtype=complex)
    rho[n, n] = 1.0
    type_str = 'Fock; with n = ' + str(n)
    return rho, type_str

def create_fock_comb(dim: int):
    """Random superposition of all Fock states."""
    coefficients = np.random.uniform(0, 1, size=dim)
    norm = np.sqrt(np.sum(coefficients ** 2))
    if norm == 0:
        coefficients = np.zeros(dim)
        coefficients[0] = 1.0
    else:
        coefficients = coefficients / norm
    psi = np.zeros(dim, dtype=complex)
    for i in range(dim):
        psi[i] = coefficients[i]
    psi_col = psi.reshape(-1, 1)
    rho = psi_col @ psi_col.conj().T
    type_str = 'fock comb with ' + str(psi)
    return rho, type_str

def create_ro_s(str_type, n, alpha_s):
    """Create a signal density matrix according to a type string."""
    if str_type == 'c':
        ro_s_0, type_str = density_matrix_coherent(dim_s, alpha_s, 0)
    elif str_type == 'f':
        ro_s_0, type_str = fock_density_matrix(n, dim_s)
    elif str_type == 'fc':
        ro_s_0, type_str = create_fock_comb(dim_s)
    else:
        ro_s_0, type_str = create_random_density_matrix(dim_s)
    return ro_s_0, type_str

def is_density_matrix(matrix, tol=1e-10):
    """Check if a matrix is a valid density matrix."""
    if not np.allclose(matrix, matrix.conj().T, atol=tol):
        print("Not Hermitian")
        return False
    trace = np.trace(matrix)
    if not np.isclose(trace, 1.0, atol=tol):
        print(f"(trace = {trace:.10f})")
        return False
    eigenvalues = np.linalg.eigvalsh(matrix)
    if not np.all(eigenvalues >= -tol):
        print(f"Negative eigenvalues: {eigenvalues}")
        return False
    return True