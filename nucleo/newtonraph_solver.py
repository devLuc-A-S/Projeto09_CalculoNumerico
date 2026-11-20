import numpy as np

A=0.17
B=0.45
C=0.45
D=0.55 # em metros [m]
theta_a = (np.pi)/2

def f(x):
    theta_b = x[0][0] # x_ij
    theta_d = x[1][0]
    return np.array([[A*np.cos(theta_a)+B*np.cos(theta_b)-C*np.cos(theta_d)-D],
                    [A*np.sin(theta_a)+B*np.sin(theta_b)-C*np.sin(theta_d)]]) # x = [theta_b, theta_d]^T
def jacobian(x):
    theta_b = x[0][0]
    theta_d = x[1][0]
    return np.array([
                     [(-B)*np.sin(theta_b), C*np.sin(theta_d)],
                     [B*np.cos(theta_b), -C*np.cos(theta_d)]
                     ])
def MNR(f, j, x_ini, tol=1e-8, kmax=50):
    x_aprox = x_ini.copy()
    k_ini = 0
    convergiu = False
    for k in range(kmax):
        j_aprox = j(x_aprox)
        f_aprox = -f(x_aprox)
        Dx = np.linalg.solve(j_aprox, f_aprox)
        x_aprox=x_aprox+Dx
        k_ini = k+1
        if np.max(np.abs(Dx)) < tol:
            convergiu = True
            break
    return x_aprox, k_ini, convergiu, np.linalg.norm(f(x_aprox), np.inf)

if __name__ == "__main__":
    alpha, beta = np.random.ranf(), np.random.ranf()
    x_teste = np.array([[np.pi*alpha], [np.pi*beta]])
    convergiu = False
    for kmax in range(0, 51):
        x_sol, ite, conv, res = MNR(f, jacobian, x_teste, kmax=kmax)
        print(f"kmax={kmax}: x={x_sol.ravel()}, iter={ite}, convergiu={conv}, residuo={res:.3e}")
        convergiu = conv
    if not convergiu:
        print(f"O chute inicial {x_teste},\n"
              "não gerou uma solução aproximada nos critérios de convergência\n"
              "do resíduo nem na verificação do erro de aproximações consecutivas")