import numpy as np
import matplotlib.pyplot as plt

# Tenta importar o arquivo base do grupo (se estiver na mesma pasta).
# Se o arquivo nao for encontrado, usa automaticamente uma copia identica da base.
try:
    import legado.newtonraph_solver as nr
except ModuleNotFoundError:
    print("Aviso: 'newtonraph_solver.py' nao encontrado na pasta. Usando copia local da base.")

    class _BaseLocal:
        A = 0.17
        B = 0.45
        C = 0.45
        D = 0.55
        theta_a = np.pi / 2

        @staticmethod
        def f(x):
            theta_b = x[0][0]
            theta_d = x[1][0]
            return np.array([
                [_BaseLocal.A * np.cos(_BaseLocal.theta_a) + _BaseLocal.B * np.cos(theta_b) - _BaseLocal.C * np.cos(theta_d) - _BaseLocal.D],
                [_BaseLocal.A * np.sin(_BaseLocal.theta_a) + _BaseLocal.B * np.sin(theta_b) - _BaseLocal.C * np.sin(theta_d)]
            ])

        @staticmethod
        def jacobian(x):
            theta_b = x[0][0]
            theta_d = x[1][0]
            return np.array([
                [(-_BaseLocal.B) * np.sin(theta_b), _BaseLocal.C * np.sin(theta_d)],
                [_BaseLocal.B * np.cos(theta_b), -_BaseLocal.C * np.cos(theta_d)]
            ])

        @staticmethod
        def MNR(f, j, x_ini, tol=1e-8, kmax=50):
            x_aprox = x_ini.copy()
            k_ini = 0
            convergiu = False
            for k in range(kmax):
                j_aprox = j(x_aprox)
                f_aprox = -f(x_aprox)
                Dx = np.linalg.solve(j_aprox, f_aprox)
                x_aprox = x_aprox + Dx
                k_ini = k + 1
                if np.max(np.abs(Dx)) < tol:
                    convergiu = True
                    break
            return x_aprox, k_ini, convergiu, np.linalg.norm(f(x_aprox), np.inf)

    nr = _BaseLocal


def executar_etapa4():
    # 1. Configuracao da Etapa 4: 360 valores de 0 ate 2*pi
    N = 360
    thetas_a = np.linspace(0, 2 * np.pi, N)

    thetas_b = np.zeros(N)
    thetas_d = np.zeros(N)
    iteracoes = np.zeros(N)
    residuos = np.zeros(N)

    # 2. Chute inicial no formato 2x1 do codigo base
    chute = np.array([[0.5], [2.0]])

    # 3. Loop principal usando o MNR da base
    for i, ta in enumerate(thetas_a):
        nr.theta_a = ta
        x_sol, ite, conv, res = nr.MNR(nr.f, nr.jacobian, chute)

        if not conv:
            print(f"Nao convergiu em theta_a = {ta:.2f} rad (residuo = {res:.3e})")

        thetas_b[i] = x_sol[0][0]
        thetas_d[i] = x_sol[1][0]
        iteracoes[i] = ite
        residuos[i] = res

        # A solucao atual vira o chute inicial da proxima posicao
        chute = x_sol

    # Resumo numerico para conferir com o relatorio
    print(f"Iteracoes na 1a posicao: {int(iteracoes[0])}")
    print(f"Media de iteracoes: {np.mean(iteracoes):.2f}")
    print(f"Residuo maximo: {np.max(residuos):.2e}")
    print(f"theta_b min/max (rad): {np.min(thetas_b):.3f} / {np.max(thetas_b):.3f}")
    print(f"theta_d min/max (rad): {np.min(thetas_d):.3f} / {np.max(thetas_d):.3f}")

    # --- GRAFICO 1: Angulos theta_b e theta_d ---
    plt.figure(figsize=(8, 4))
    plt.plot(thetas_a, thetas_b, label=r"$\theta_B$ (perna)")
    plt.plot(thetas_a, thetas_d, label=r"$\theta_D$ (coxa)")
    plt.xlabel(r"$\theta_A$ (rad)")
    plt.ylabel("Ângulos (rad)")
    plt.title(r"Evolução de $\theta_B$ e $\theta_D$ em uma volta completa")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("etapa4-angulos.png", dpi=300)
    plt.show()

    # --- GRAFICO 2: Movimento do mecanismo (A, B, C, D) ---
    xA, yA = 0.0, 0.0
    xD, yD = nr.D, 0.0
    xB = nr.A * np.cos(thetas_a)
    yB = nr.A * np.sin(thetas_a)
    xC = xB + nr.B * np.cos(thetas_b)
    yC = yB + nr.B * np.sin(thetas_b)

    plt.figure(figsize=(6, 6))
    plt.plot(xB, yB, '--', color='gray', label='Trajetória do pedal (B)')
    plt.plot(xC, yC, '--', color='orange', label='Trajetória do joelho (C)')

    for i in range(0, N, 30):
        plt.plot([xA, xB[i], xC[i], xD], [yA, yB[i], yC[i], yD], 'o-', alpha=0.5)

    plt.axis('equal')
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.title("Movimento do mecanismo de quatro barras")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("etapa4-movimento.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    executar_etapa4()