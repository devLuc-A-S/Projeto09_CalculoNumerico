import numpy as np
import matplotlib.pyplot as plt
import nucleo.newtonraph_solver as nr
from experimentos.solucoes import executar_etapa4


def executar_etapa6():
    a, b, c, d = nr.A, nr.B, nr.C, nr.D
    thetaA_values, thetaB_values, thetaD_values, _, _ = executar_etapa4(plotar=False)

    gamma_values = np.arccos(
        np.clip(np.cos(thetaD_values - thetaB_values), -1.0, 1.0)
    )
    gamma_graus = np.degrees(gamma_values)

    Bx = a * np.cos(thetaA_values)
    By = a * np.sin(thetaA_values)
    Cx = Bx + b * np.cos(thetaB_values)
    Cy = By + b * np.sin(thetaB_values)
    Dx = d
    Dy = 0.0

    P = (Dx - Bx) * (Cy - By) - (Dy - By) * (Cx - Bx)
    det_J = b * c * np.sin(thetaD_values - thetaB_values)

    tol_geom = 1e-10
    admissivel = P > tol_geom

    print("RESULTADOS DO EXERCÍCIO 6 (E VERIFICAÇÃO DE SINGULARIDADE)")
    print("----------------------------------------------------------")
    print(f"Ângulo mínimo do joelho (gamma): {np.min(gamma_graus):.2f} graus")
    print(f"Ângulo máximo do joelho (gamma): {np.max(gamma_graus):.2f} graus")
    print("----------------------------------------------------------")
    print(f"Posições geometricamente admissíveis: {np.sum(admissivel)} / {len(thetaA_values)}")
    print(f"Posições fora do ramo escolhido:      {np.sum(~admissivel)}")
    print("----------------------------------------------------------")
    print(f"Produto vetorial P (min / max):       {np.min(P):.4f} / {np.max(P):.4f} m²")
    print(f"det(J) ao longo do ciclo (min / max): {np.min(det_J):.4f} / {np.max(det_J):.4f}")

    plt.figure(figsize=(10, 5))
    plt.plot(np.degrees(thetaA_values), gamma_graus, label="Ângulo interno do joelho (γ)")
    plt.xlabel("Ângulo do pedivela θA (graus)")
    plt.ylabel("Ângulo γ (graus)")
    plt.title("Variação do ângulo interno do joelho ao longo do ciclo")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("saidas/etapa6-angulo-joelho.png", dpi=300)
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(np.degrees(thetaA_values), P, label="Produto vetorial P (m²)")
    plt.plot(np.degrees(thetaA_values), det_J, '--', label="det(J) = b·c·sin(θD - θB)")
    plt.axhline(y=0, color="red", linestyle=":", label="Limite de inversão / Singularidade")
    plt.xlabel("Ângulo do pedivela θA (graus)")
    plt.ylabel("Amplitude")
    plt.title("Verificação do ramo físico (Etapa 6) e distância da singularidade (Etapa 5)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("saidas/etapa6-verificacao-geometrica.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    executar_etapa6()