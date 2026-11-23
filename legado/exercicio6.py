import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# EXERCICIO 6 - MOVIMENTO DO JOELHO
# ==========================================

# 1. Converter os resultados anteriores em arrays
thetaA_values = np.array(thetaA_values)
thetaB_values = np.array(thetaB_values)
thetaD_values = np.array(thetaD_values)

# ==========================================
# 2. CALCULO DO ANGULO INTERNO GAMMA
# ==========================================

gamma_values = np.arccos(
    np.clip(
        np.cos(thetaD_values - thetaB_values),
        -1, 1
    )
)

# Converter de radianos para graus
gamma_graus = np.degrees(gamma_values)

# ==========================================
# 3. RECONSTRUCAO DAS COORDENADAS
# ==========================================

# Coordenadas do ponto B
Bx = a * np.cos(thetaA_values)
By = a * np.sin(thetaA_values)

# Coordenadas do ponto C
Cx = Bx + b * np.cos(thetaB_values)
Cy = By + b * np.sin(thetaB_values)

# Coordenadas do ponto D
Dx = d
Dy = 0

# ==========================================
# 4. VERIFICACAO GEOMETRICA
# ==========================================

# Produto vetorial entre BD e BC
P = (
    (Dx - Bx)*(Cy - By)
    - (Dy - By)*(Cx - Bx)
)

# Tolerancia numerica
tol_geom = 1e-10

# Configuracao escolhida no exercicio 3:
# P positivo representa o ramo geometrico admissivel
admissivel = P > tol_geom

# ==========================================
# 5. APRESENTACAO DOS RESULTADOS
# ==========================================

print("RESULTADOS DO EXERCICIO 6")
print("--------------------------------")

print("Angulo minimo do joelho:",
      np.min(gamma_graus), "graus")

print("Angulo maximo do joelho:",
      np.max(gamma_graus), "graus")

print("--------------------------------")

print("Posicoes geometricamente admissiveis:",
      np.sum(admissivel))

print("Posicoes fora do ramo geometrico escolhido:",
      np.sum(~admissivel))

print("--------------------------------")

print("Menor valor do produto vetorial:",
      np.min(P))

print("Maior valor do produto vetorial:",
      np.max(P))

# ==========================================
# 6. GRAFICO DO ANGULO DO JOELHO
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    np.degrees(thetaA_values),
    gamma_graus,
    label="Angulo interno do joelho"
)

plt.xlabel("Angulo do pedivela (graus)")
plt.ylabel("Angulo gamma (graus)")
plt.title("Variacao do angulo interno do joelho")

plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# ==========================================
# 7. GRAFICO DA VERIFICACAO GEOMETRICA
# ==========================================

plt.figure(figsize=(10, 5))

plt.plot(
    np.degrees(thetaA_values),
    P,
    label="Produto vetorial P"
)

plt.axhline(
    y=0,
    color="red",
    linestyle="--",
    label="Limite entre configuracoes"
)

plt.xlabel("Angulo do pedivela (graus)")
plt.ylabel("Produto vetorial (m²)")
plt.title("Verificacao da configuracao geometrica")

plt.grid()
plt.legend()
plt.tight_layout()
plt.show()