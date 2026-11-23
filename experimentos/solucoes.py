import numpy as np
import matplotlib.pyplot as plt
import nucleo.newtonraph_solver as nr
from experimentos.aproximacoes_iniciais import investigar_aproximacoes


def executar_etapa4(N=361, plotar=True):
    _, chute = investigar_aproximacoes(plotar=False)
    if chute is None:
        chute = np.array([[0.5], [2.0]])

    thetas_a = np.linspace(0, 2 * np.pi, N)
    thetas_b = np.zeros(N)
    thetas_d = np.zeros(N)
    iteracoes = np.zeros(N, dtype=int)
    residuos = np.zeros(N)

    for i, ta in enumerate(thetas_a):
        nr.theta_a = ta
        x_sol, ite, conv, res = nr.MNR(nr.f, nr.jacobian, chute)

        if not conv:
            print(f"Não convergiu em theta_a = {ta:.2f} rad (resíduo = {res:.3e})")

        thetas_b[i] = x_sol[0][0]
        thetas_d[i] = x_sol[1][0]
        iteracoes[i] = ite
        residuos[i] = res

        chute = x_sol

    if plotar:
        print(f"Iterações na 1ª posição: {iteracoes[0]}")
        print(f"Média de iterações: {np.mean(iteracoes):.2f}")
        _plotar_etapa4(thetas_a, thetas_b, thetas_d, N)

    return thetas_a, thetas_b, thetas_d, iteracoes, residuos


def _plotar_etapa4(thetas_a, thetas_b, thetas_d, N):
    plt.figure(figsize=(8, 4))
    plt.plot(thetas_a, thetas_b, label=r"$\theta_B$ (perna)")
    plt.plot(thetas_a, thetas_d, label=r"$\theta_D$ (coxa)")
    plt.xlabel(r"$\theta_A$ (rad)")
    plt.ylabel("Ângulos (rad)")
    plt.title(r"Evolução de $\theta_B$ e $\theta_D$ em uma volta completa")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("saidas/etapa4-angulos.png", dpi=300)
    plt.show()

    xA, yA = 0.0, 0.0
    xD, yD = nr.D, 0.0
    xB = nr.A * np.cos(thetas_a)
    yB = nr.A * np.sin(thetas_a)
    xC = xB + nr.B * np.cos(thetas_b)
    yC = yB + nr.B * np.sin(thetas_b)
    print(f"theta_B min / max (rad): {np.min(thetas_b):.3f} / {np.max(thetas_b):.3f}")
    print(f"theta_D min / max (rad): {np.min(thetas_d):.3f} / {np.max(thetas_d):.3f}")
    print(f"Altura do joelho yC min / max (m): {np.min(yC):.3f} / {np.max(yC):.3f}")
    print(f"Diferenca de periodicidade (theta_B): {abs(thetas_b[0] - thetas_b[-1]):.2e}")
    print(f"Diferenca de periodicidade (theta_D): {abs(thetas_d[0] - thetas_d[-1]):.2e}")
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
    plt.savefig("saidas/etapa4-movimento.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    executar_etapa4(plotar=True)