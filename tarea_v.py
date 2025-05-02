# ejercicios_no_lineal.py
# Métodos Cuantitativos: Programación No Lineal
# Víctor Manrique

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import sympy as sp

# ----------------------------------------------------
# 1. INVERSOR: maximizar rendimientos con restricciones
# ----------------------------------------------------
def problema_inversor():
    """
    Maximizamos f(x,y) = 0.1 x + 0.08 y sujeto a:
      1) igualdad:   x + y = 1
      2) desigualdad: 0.02 x^2 + 0.03 y^2 <= 0.05
      3) x,y >= 0
    Usamos scipy.optimize.minimize cambiando signo para convertir en minimización.
    """
    # Función objetivo (negativa para maximizar)
    def f(vars):
        x, y = vars
        return -(0.1*x + 0.08*y)

    # Restricciones
    cons = [
        {'type': 'eq',   'fun': lambda v: v[0] + v[1] - 1},  # x + y - 1 = 0
        {'type': 'ineq', 'fun': lambda v: 0.05 - (0.02*v[0]**2 + 0.03*v[1]**2)}
    ]
    # Límites de las variables: 0 <= x,y <= 1
    bnds = [(0,1), (0,1)]
    # Punto inicial (factible)
    x0 = np.array([0.5, 0.5])

    # Llamada al solver
    res = minimize(f, x0, bounds=bnds, constraints=cons)
    if not res.success:
        raise RuntimeError(f"No convergió: {res.message}")

    x_opt, y_opt = res.x
    rendimiento = 0.1*x_opt + 0.08*y_opt
    print("1) INVERSOR:")
    print(f"   x* = {x_opt:.4f}, y* = {y_opt:.4f}, rendimiento ≈ {rendimiento:.4f}\n")


# ----------------------------------------------------
# 2. COMPAÑÍA: minimizar costos con tres productos
# ----------------------------------------------------
def problema_compania():
    """
    Minimizar C(x,y,z) = 5 x^2 + 3 y^2 + 1·z^2
    sujeto a x + y + z = 100.
    Hacemos solución analítica vía sistema de ecuaciones.
    """
    # Planteamos λ tal que:
    #   ∂C/∂x = 10x = λ
    #   ∂C/∂y = 6y  = λ
    #   ∂C/∂z = 2z  = λ
    # y x + y + z = 100.
    # De ahí x = λ/10, y = λ/6, z = λ/2.
    # Sustituimos en la suma:
    #   λ*(1/10 + 1/6 + 1/2) = 100 → λ = 100 / (1/10 + 1/6 + 1/2)

    denom = 1/10 + 1/6 + 1/2
    lam  = 100 / denom

    x_opt = lam/10
    y_opt = lam/6
    z_opt = lam/2

    print("2) COMPAÑÍA:")
    print(f"   x* ≈ {x_opt:.4f}, y* ≈ {y_opt:.4f}, z* ≈ {z_opt:.4f}\n")

    # (Opcional) Gráfica de las cantidades
    productos = ['A (x)', 'B (y)', 'C (z)']
    valores    = [x_opt, y_opt, z_opt]
    plt.figure()
    plt.bar(productos, valores)
    plt.ylabel('Cantidad óptima')
    plt.title('Producción óptima para minimizar costos')
    plt.show()


# ----------------------------------------------------
# 3. DESCENSO DEL GRADIENTE en f(x,y,z)
# ----------------------------------------------------
def problema_gradiente(alpha=0.1, iters=15):
    """
    Aplicamos descenso del gradiente a
      f(x,y,z) = x^2 + y^2 + z^2 - 2xy + 3z
    con paso alpha y número de iteraciones dado.
    """
    # Función f y su gradiente
    def f(v):
        x,y,z = v
        return x**2 + y**2 + z**2 - 2*x*y + 3*z

    def grad(v):
        x,y,z = v
        return np.array([2*x - 2*y,
                         2*y - 2*x,
                         2*z + 3])

    v = np.array([1.0, 1.0, 1.0])  # punto inicial
    hist = []

    for k in range(iters):
        hist.append(f(v))
        v = v - alpha * grad(v)

    print("3) DESCENSO DEL GRADIENTE:")
    print(f"   Valor final f ≈ {hist[-1]:.4f}")
    print(f"   Punto final ≈ {v}\n")

    # Gráfica de convergencia
    plt.figure()
    plt.plot(range(1, iters+1), hist, marker='o')
    plt.xlabel('Iteración')
    plt.ylabel('f(x,y,z)')
    plt.title(f'Descenso del gradiente (α={alpha})')
    plt.grid(True)
    plt.show()


# ----------------------------------------------------
# 4. MINIMIZACIÓN UNIVARIABLE con restricciones
# ----------------------------------------------------
def problema_univariable():
    """
    Minimizar f(x) = x^2 + 4x + 5 en el intervalo [2,5].
    Calculamos derivada y comparamos extremos.
    """
    f  = lambda x: x**2 + 4*x + 5
    df = lambda x: 2*x + 4

    # Candidato crítico: df(x)=0 → x=-2 (fuera de [2,5])
    x_cand = -2

    # Evaluar en límites
    xs = [2,5]
    vals = [f(x) for x in xs]
    x_opt = xs[np.argmin(vals)]
    f_opt = min(vals)

    print("4) MINIMIZACIÓN UNIVARIABLE:")
    print(f"   Mínimo en x* = {x_opt}, f(x*) = {f_opt}\n")


# ----------------------------------------------------
# 5. PUNTOS ESTACIONARIOS y clasificación
# ----------------------------------------------------
def problema_estacionarios():
    """
    Para f(x)=x^2, f(x)=-x^2, f(x)=x^3
    encontramos puntos críticos y clasificamos con la segunda derivada.
    Usamos sympy para ilustrar.
    """
    x = sp.symbols('x')
    funciones = {
        'x^2':   x**2,
        '-x^2': -x**2,
        'x^3':   x**3
    }

    print("5) PUNTOS ESTACIONARIOS:")
    for name, f in funciones.items():
        df  = sp.diff(f, x)
        d2f = sp.diff(df, x)
        crits = sp.solve(df, x)
        for c in crits:
            tipo = ("mínimo" if d2f.subs(x, c) > 0 
                    else "máximo" if d2f.subs(x, c) < 0 
                    else "punto de inflexión")
            print(f"   f(x)={name}: x={c} → {tipo}")
    print()

# ----------------------------------------------------
# EJECUCIÓN de todos los ejercicios
# ----------------------------------------------------
if __name__ == "__main__":
    problema_inversor()
    problema_compania()
    problema_gradiente(alpha=0.1, iters=15)
    problema_univariable()
    problema_estacionarios()
