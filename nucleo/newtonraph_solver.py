import numpy as np

A = 0.17
B = 0.45
C = 0.45
D = 0.55  # em metros [m]
theta_a = np.pi / 2


def f(x, ta=None):
    ang_a = theta_a if ta is None else ta
    theta_b = x[0][0]
    theta_d = x[1][0]
    return np.array([
        [A * np.cos(ang_a) + B * np.cos(theta_b) - C * np.cos(theta_d) - D],
        [A * np.sin(ang_a) + B * np.sin(theta_b) - C * np.sin(theta_d)]
    ])


def jacobian(x):
    theta_b = x[0][0]
    theta_d = x[1][0]
    return np.array([
        [-B * np.sin(theta_b), C * np.sin(theta_d)],
        [B * np.cos(theta_b), -C * np.cos(theta_d)]
    ])


def det_jacobian(x):
    """Determinante analítico de J(x): B * C * sin(theta_D - theta_B)."""
    theta_b = x[0][0]
    theta_d = x[1][0]
    return B * C * np.sin(theta_d - theta_b)


def MNR(f_func, j_func, x_ini, tol=1e-8, kmax=50):
    x_aprox = np.array(x_ini, dtype=float).reshape(2, 1)
    k_ini = 0
    convergiu = False

    for k in range(kmax):
        try:
            j_aprox = j_func(x_aprox)
            f_aprox = -f_func(x_aprox)
            Dx = np.linalg.solve(j_aprox, f_aprox)
        except np.linalg.LinAlgError:
            return x_aprox, k, False, np.inf

        x_aprox = x_aprox + Dx
        k_ini = k + 1
        residuo = np.linalg.norm(f_func(x_aprox), np.inf)

        if np.max(np.abs(Dx)) < tol and residuo < tol:
            convergiu = True
            return x_aprox, k_ini, convergiu, residuo

    return x_aprox, k_ini, convergiu, np.linalg.norm(f_func(x_aprox), np.inf)


if __name__ == "__main__":
    theta_a = np.pi / 2
    alpha, beta = np.random.rand(), np.random.rand()
    x_teste = np.array([[np.pi * alpha], [np.pi * beta]])
    convergiu = False

    for kmax in range(1, 51):
        x_sol, ite, conv, res = MNR(f, jacobian, x_teste, kmax=kmax)
        if conv:
            print(f"kmax={kmax}: x={x_sol.ravel()}, iter={ite}, convergiu={conv}, residuo={res:.3e}")
            convergiu = True
            break

    if not convergiu:
        print(
            f"O chute inicial {x_teste.ravel()},\n"
            "não gerou uma solução aproximada nos critérios de convergência\n"
            "do resíduo nem na verificação do erro de aproximações consecutivas."
        )