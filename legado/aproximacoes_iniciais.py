import numpy as np
import matplotlib.pyplot as plt

# PARÂMETROS DO MECANISMO

A = 0.17
B = 0.45
C = 0.45
D = 0.55

theta_a = np.pi/2


# SISTEMA NÃO LINEAR

def f(x):

    theta_b = x[0]
    theta_d = x[1]

    return np.array([
        A*np.cos(theta_a) + B*np.cos(theta_b)
        - C*np.cos(theta_d) - D,

        A*np.sin(theta_a) + B*np.sin(theta_b)
        - C*np.sin(theta_d)
    ])


# MATRIZ JACOBIANA

def jacobian(x):

    theta_b = x[0]
    theta_d = x[1]

    return np.array([
        [-B*np.sin(theta_b), C*np.sin(theta_d)],
        [ B*np.cos(theta_b), -C*np.cos(theta_d)]
    ])


# MÉTODO DE NEWTON-RAPHSON

def MNR(f, j, x_ini, tol=1e-8, kmax=50):

    x = np.array(x_ini, dtype=float)

    for k in range(kmax):

        try:
            Dx = np.linalg.solve(j(x), -f(x))
        except np.linalg.LinAlgError:
            return x, k, False

        x = x + Dx

        erro = np.max(np.abs(Dx))
        residuo = np.linalg.norm(f(x), np.inf)

        if erro < tol and residuo < tol:
            return x, k+1, True

    return x, kmax, False


# INVESTIGAÇÃO DAS APROXIMAÇÕES INICIAIS

solucoes = []

chutes = np.linspace(0, 2*np.pi, 15, endpoint=False)

total_convergencias = 0

for theta_b_ini in chutes:
    for theta_d_ini in chutes:

        x_ini = [theta_b_ini, theta_d_ini]

        x_sol, ite, conv = MNR(f, jacobian, x_ini)

        if conv:

            total_convergencias += 1

            # Normalização dos ângulos entre 0 e 2pi
            x_sol = np.mod(x_sol, 2*np.pi)

            # Verificação de soluções repetidas
            repetida = False

            for s in solucoes:

                diferenca = np.angle(np.exp(1j*(x_sol-s)))

                if np.max(np.abs(diferenca)) < 1e-5:
                    repetida = True
                    break

            if not repetida:
                solucoes.append(x_sol)


# ORGANIZAÇÃO DOS RESULTADOS

solucoes.sort(key=lambda s: s[0])

print("RESULTADOS DA INVESTIGAÇÃO")
print("--------------------------------")

print(f"Chutes iniciais testados: {len(chutes)**2}")
print(f"Convergências obtidas: {total_convergencias}")
print(f"Soluções distintas encontradas: {len(solucoes)}")

for i, s in enumerate(solucoes):

    tb, td = s

    print(f"\nCONFIGURAÇÃO {i+1}")
    print(f"Theta B = {np.degrees(tb):.4f} graus")
    print(f"Theta D = {np.degrees(td):.4f} graus")
    print(f"Resíduo = {np.linalg.norm(f(s), np.inf):.3e}")

    # Verificação da configuração física do desenho
    fisica = (0 < tb < np.pi/2) and (np.pi/2 < td < np.pi)

    if fisica:
        print("Corresponde à configuração física do desenho.")
    else:
        print("Não corresponde à configuração física do desenho.")


# REPRESENTAÇÃO GEOMÉTRICA DAS SOLUÇÕES

fig, axes = plt.subplots(
    1, len(solucoes),
    figsize=(6*len(solucoes), 5)
)

if len(solucoes) == 1:
    axes = [axes]

for i, (s, ax) in enumerate(zip(solucoes, axes)):

    tb, td = s

    # Coordenadas dos pontos A, B, C e D

    ponto_A = np.array([0, 0])

    ponto_B = ponto_A + A*np.array([
        np.cos(theta_a), np.sin(theta_a)
    ])

    ponto_C = ponto_B + B*np.array([
        np.cos(tb), np.sin(tb)
    ])

    ponto_D = np.array([D, 0])

    # Fechamento geométrico do mecanismo

    pontos = np.array([
        ponto_A, ponto_B, ponto_C,
        ponto_D, ponto_A
    ])

    ax.plot(
        pontos[:,0], pontos[:,1],
        'o-', linewidth=2, markersize=7
    )

    # Identificação das articulações

    for nome, p in zip(
        ['A', 'B', 'C', 'D'],
        [ponto_A, ponto_B, ponto_C, ponto_D]
    ):
        ax.annotate(nome, p, xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=12)

    # Verificação da configuração física

    fisica = (0 < tb < np.pi/2) and (np.pi/2 < td < np.pi)

    if fisica:
        titulo = "Configuração física"
    else:
        titulo = "Configuração alternativa"

    ax.set_title(
        f"Configuração {i+1}: {titulo}\n"
        f"θB = {np.degrees(tb):.2f}° | "
        f"θD = {np.degrees(td):.2f}°"
    )

    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.grid(True)
    ax.set_aspect('equal')

plt.tight_layout()
plt.show()