import numpy as np
import matplotlib.pyplot as plt
import nucleo.newtonraph_solver as nr


def investigar_aproximacoes(n_grade=15, plotar=True):
    nr.theta_a = np.pi / 2
    solucoes = []
    chutes = np.linspace(0, 2 * np.pi, n_grade, endpoint=False)
    total_convergencias = 0

    for theta_b_ini in chutes:
        for theta_d_ini in chutes:
            x_ini = np.array([[theta_b_ini], [theta_d_ini]])
            x_sol, ite, conv, res = nr.MNR(nr.f, nr.jacobian, x_ini)

            if conv:
                total_convergencias += 1
                x_norm = np.mod(x_sol.ravel(), 2 * np.pi)

                repetida = any(
                    np.max(np.abs(np.angle(np.exp(1j * (x_norm - s))))) < 1e-5
                    for s in solucoes
                )
                if not repetida:
                    solucoes.append(x_norm)

    solucoes.sort(key=lambda s: s[0])

    chute_fisico = None
    for s in solucoes:
        tb, td = s
        if (0 < tb < np.pi / 2) and (np.pi / 2 < td < np.pi):
            chute_fisico = np.array([[tb], [td]])
            break

    if plotar:
        print("RESULTADOS DA INVESTIGAÇÃO (ETAPAS 2 E 3)")
        print("-----------------------------------------")
        print(f"Chutes iniciais testados: {len(chutes)**2}")
        print(f"Convergências obtidas: {total_convergencias}")
        print(f"Soluções distintas encontradas: {len(solucoes)}")

        for i, s in enumerate(solucoes):
            tb, td = s
            x_col = np.array([[tb], [td]])
            fisica = (0 < tb < np.pi / 2) and (np.pi / 2 < td < np.pi)
            print(f"\nCONFIGURAÇÃO {i+1}")
            print(f"Theta B = {np.degrees(tb):.4f} graus ({tb:.4f} rad)")
            print(f"Theta D = {np.degrees(td):.4f} graus ({td:.4f} rad)")
            print(f"Resíduo = {np.linalg.norm(nr.f(x_col), np.inf):.3e}")
            print(f"det(J)  = {nr.det_jacobian(x_col):.4f}")
            print("Corresponde à configuração física." if fisica else "Configuração alternativa (joelho invertido).")

        _plotar_configuracoes(solucoes)

    return solucoes, chute_fisico


def _plotar_configuracoes(solucoes):
    fig, axes = plt.subplots(1, len(solucoes), figsize=(6 * len(solucoes), 5))
    if len(solucoes) == 1:
        axes = [axes]

    for i, (s, ax) in enumerate(zip(solucoes, axes)):
        tb, td = s
        ponto_A = np.array([0.0, 0.0])
        ponto_B = ponto_A + nr.A * np.array([np.cos(nr.theta_a), np.sin(nr.theta_a)])
        ponto_C = ponto_B + nr.B * np.array([np.cos(tb), np.sin(tb)])
        ponto_D = np.array([nr.D, 0.0])

        pontos = np.array([ponto_A, ponto_B, ponto_C, ponto_D, ponto_A])
        ax.plot(pontos[:, 0], pontos[:, 1], 'o-', linewidth=2, markersize=7)

        for nome, p in zip(['A', 'B', 'C', 'D'], [ponto_A, ponto_B, ponto_C, ponto_D]):
            ax.annotate(nome, p, xytext=(5, 5), textcoords='offset points', fontsize=12)

        fisica = (0 < tb < np.pi / 2) and (np.pi / 2 < td < np.pi)
        titulo = "Configuração física" if fisica else "Configuração alternativa"
        ax.set_title(
            f"Configuração {i+1}: {titulo}\n"
            f"θB = {np.degrees(tb):.2f}° | θD = {np.degrees(td):.2f}°"
        )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.grid(True)
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    investigar_aproximacoes(plotar=True)