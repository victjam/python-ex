# parcial_no_lineal.py
# Calculadora de optimización no lineal con POO y menú interactivo

import json
import sys
import numpy as np
import sympy as sp
from scipy.optimize import minimize

# Nombre de archivo JSON para guardar/cargar problemas
PROBLEMS_DATA_FILE = 'problems.json'


class ProblemDefinition:
    """
    Representa un problema de optimización:
      - objective_function: función objetivo como sympy.Expr
      - variables: lista de sympy.Symbol
      - equality_constraints: lista de sympy.Expr (==0)
      - inequality_constraints: lista de sympy.Expr (<=0)
    """
    def __init__(self, objective_function, variables, equality_constraints=None, inequality_constraints=None, name=""):
        self.objective_function = objective_function
        self.variables = variables
        self.equality_constraints = equality_constraints or []
        self.inequality_constraints = inequality_constraints or []
        self.name = name

    def to_json(self):
        # Serializa expresiones a texto
        return {
            'name': self.name,
            'variables': [str(var) for var in self.variables],
            'objective_function': str(self.objective_function),
            'equality_constraints': [str(constraint) for constraint in self.equality_constraints],
            'inequality_constraints': [str(constraint) for constraint in self.inequality_constraints],
        }

    @staticmethod
    def from_json(data):
        # Reconstruye desde texto usando sympy.sympify
        variables = sp.symbols(data['variables'])
        objective_function = sp.sympify(data['objective_function'])
        equality_constraints = [sp.sympify(constraint) for constraint in data['equality_constraints']]
        inequality_constraints = [sp.sympify(constraint) for constraint in data['inequality_constraints']]
        return ProblemDefinition(
            objective_function, 
            list(variables), 
            equality_constraints, 
            inequality_constraints, 
            name=data.get('name', '')
        )


class NonLinearCalculator:
    def __init__(self):
        self.problems = self.load_data()

    def load_data(self):
        """Carga definiciones desde JSON o crea ejemplos por defecto."""
        try:
            with open(PROBLEMS_DATA_FILE, 'r') as file:
                data = json.load(file)
            return [ProblemDefinition.from_json(problem_data) for problem_data in data]
        except FileNotFoundError:
            # Ejemplo por defecto: inversión
            investment_x, investment_y = sp.symbols('x y', real=True)
            default_problem = ProblemDefinition(
                objective_function=0.1*investment_x + 0.08*investment_y,
                variables=[investment_x, investment_y],
                equality_constraints=[investment_x + investment_y - 1],
                inequality_constraints=[0.02*investment_x**2 + 0.03*investment_y**2 - 0.05],
                name="Ejemplo: inversión"
            )
            return [default_problem]

    def save_data(self):
        """Guarda la lista de problemas en JSON."""
        with open(PROBLEMS_DATA_FILE, 'w') as file:
            json.dump([problem.to_json() for problem in self.problems], file, indent=2)

    def menu(self):
        """Menú principal."""
        while True:
            print("\n=== Calculadora No Lineal ===")
            print("1) Agregar problema")
            print("2) Listar problemas")
            print("3) Optimización con restricciones")
            print("4) Método del gradiente")
            print("5) Método de Lagrange (igualdad)")
            print("6) Optimización sin restricciones")
            print("7) Multivariables y clasificación (Hessiano)")
            print("0) Salir")
            option = input("Elige una opción: ").strip()
            
            if option == '1':
                self.add_problem()
            elif option == '2':
                self.list_problems()
            elif option == '3':
                self.solve_with_constraints()
            elif option == '4':
                self.gradient_methods()
            elif option == '5':
                self.lagrange_equality()
            elif option == '6':
                self.unconstrained_opt()
            elif option == '7':
                self.multivar_and_hessian()
            elif option == '0':
                self.save_data()
                print("¡Hasta luego!")
                sys.exit(0)
            else:
                print("Opción no válida.")

    def add_problem(self):
        """Permite al usuario definir un nuevo problema."""
        problem_name = input("Nombre descriptivo: ").strip()
        variables_input = input("Variables (separadas por coma): ").strip()
        variables = sp.symbols([var.strip() for var in variables_input.split(',')])
        objective_function_input = input("Función objetivo (en variables): ").strip()
        objective_function = sp.sympify(objective_function_input)
        
        equality_constraints = []
        inequality_constraints = []
        
        print("Introduce restricciones de igualdad (expr=0). En blanco para terminar.")
        while True:
            constraint = input("  Igualdad: ").strip()
            if not constraint: 
                break
            equality_constraints.append(sp.sympify(constraint.replace('=', '-(') + ')'))
            
        print("Introduce restricciones de desigualdad (expr<=0). En blanco para terminar.")
        while True:
            constraint = input("  Desigualdad: ").strip()
            if not constraint: 
                break
            inequality_constraints.append(sp.sympify(constraint.replace('<=', '-(') + ')'))
            
        new_problem = ProblemDefinition(
            objective_function, 
            list(variables), 
            equality_constraints, 
            inequality_constraints, 
            name=problem_name
        )
        self.problems.append(new_problem)
        print("Problema agregado.")

    def list_problems(self):
        """Muestra los problemas disponibles."""
        for index, problem in enumerate(self.problems):
            print(f"{index}) {problem.name}")

    def choose_problem(self):
        """Pide índice y devuelve problema."""
        self.list_problems()
        problem_index = int(input("Elige índice: ").strip())
        return self.problems[problem_index]

    def solve_with_constraints(self):
        """Resuelve problemas de optimización con restricciones."""
        problem = self.choose_problem()
        print(f"\n-- Optimización restringida: {problem.name} --")

        # a) Lagrange (solo igualdades)
        if problem.equality_constraints:
            print("a) Multiplicadores de Lagrange (igualdad)")
            lagrange_multipliers = sp.symbols(f"lambda0:{len(problem.equality_constraints)}")
            lagrangian = problem.objective_function
            
            # Construye Lagrangiana
            for multiplier, constraint in zip(lagrange_multipliers, problem.equality_constraints):
                lagrangian = lagrangian + multiplier * constraint
                
            # Sistema de ecuaciones: derivadas cero + restricciones
            equations = [sp.diff(lagrangian, var) for var in problem.variables] + problem.equality_constraints
            solutions = sp.solve(equations, list(problem.variables) + list(lagrange_multipliers), dict=True)
            print("  Soluciones candidatas:", solutions)

        # b) KKT (igualdad + desigualdad)
        if problem.inequality_constraints:
            print("\nb) Condiciones KKT (desigualdad)")
            kkt_multipliers = sp.symbols(f"mu0:{len(problem.inequality_constraints)}", nonnegative=True)
            kkt_lagrangian = problem.objective_function
            
            for multiplier, constraint in zip(lagrange_multipliers, problem.equality_constraints):
                kkt_lagrangian += multiplier * constraint
            for multiplier, constraint in zip(kkt_multipliers, problem.inequality_constraints):
                kkt_lagrangian += multiplier * constraint
                
            # Gradiente estacionario
            kkt_equations = [sp.diff(kkt_lagrangian, var) for var in problem.variables]
            # Complementariedad y factibilidad
            complementarity = [multiplier*constraint for multiplier, constraint in zip(kkt_multipliers, problem.inequality_constraints)]
            all_equations = kkt_equations + problem.equality_constraints + complementarity
            kkt_solutions = sp.solve(all_equations, list(problem.variables)+list(lagrange_multipliers)+list(kkt_multipliers), dict=True)
            print("  Soluciones KKT:", kkt_solutions)

        # c) Método numérico
        print("\nc) Método numérico (SciPy)")
        # Convertir a función de numpy
        numpy_objective = sp.lambdify(problem.variables, problem.objective_function, 'numpy')
        
        # Restricciones para scipy
        scipy_constraints = []
        for constraint in problem.equality_constraints:
            constraint_function = sp.lambdify(problem.variables, constraint, 'numpy')
            scipy_constraints.append({
                'type': 'eq', 
                'fun': lambda v, f=constraint_function: f(*v)
            })
            
        for constraint in problem.inequality_constraints:
            constraint_function = sp.lambdify(problem.variables, constraint, 'numpy')
            scipy_constraints.append({
                'type': 'ineq', 
                'fun': lambda v, f=constraint_function: -f(*v)
            })
            
        # Bounds genéricos (-inf,inf)
        num_variables = len(problem.variables)
        bounds = [(-np.inf, np.inf)] * num_variables
        initial_point = np.zeros(num_variables)
        
        result = minimize(
            lambda v: numpy_objective(*v), 
            initial_point, 
            bounds=bounds, 
            constraints=scipy_constraints
        )
        print("  Resultado numérico:", result)

    def gradient_methods(self):
        """Aplica el método del gradiente al problema seleccionado."""
        problem = self.choose_problem()
        step_size = float(input("Paso α (ej. 0.1): ").strip())
        num_iterations = int(input("Número de iteraciones: ").strip())
        print(f"\n-- Gradiente sobre: {problem.name} --")

        # Prepara funciones
        numpy_objective = sp.lambdify(problem.variables, problem.objective_function, 'numpy')
        gradient_symbols = [sp.diff(problem.objective_function, var) for var in problem.variables]
        numpy_gradient = sp.lambdify(problem.variables, gradient_symbols, 'numpy')

        current_point = np.zeros(len(problem.variables))  # inicializar en cero
        history = []
        
        for iteration in range(num_iterations):
            gradient = np.array(numpy_gradient(*current_point), dtype=float).flatten()
            current_point = current_point - step_size * gradient
            history.append(numpy_objective(*current_point))

        print("Descenso:", current_point, "f≈", history[-1])

    def lagrange_equality(self):
        """Aplica el método de Lagrange para restricciones de igualdad."""
        problem = self.choose_problem()
        if not problem.equality_constraints:
            print("No hay restricciones de igualdad.")
            return
        print(f"\n-- Lagrange (igualdad): {problem.name} --")
        # Reusa la sección a) de solve_with_constraints
        self.solve_with_constraints()

    def unconstrained_opt(self):
        """Resuelve problemas de optimización sin restricciones."""
        problem = self.choose_problem()
        print(f"\n-- Optimización sin restricciones: {problem.name} --")
        # Derivadas parciales y sistema grad f = 0
        gradients = [sp.diff(problem.objective_function, var) for var in problem.variables]
        solutions = sp.solve(gradients, problem.variables, dict=True)
        print("Soluciones candidatas:", solutions)

    def multivar_and_hessian(self):
        """Analiza puntos críticos usando el Hessiano."""
        problem = self.choose_problem()
        print(f"\n-- Multivariables y Hessiano: {problem.name} --")
        gradients = [sp.diff(problem.objective_function, var) for var in problem.variables]
        hessian_matrix = sp.hessian(problem.objective_function, problem.variables)
        critical_points = sp.solve(gradients, problem.variables, dict=True)
        
        for point in critical_points:
            hessian_at_point = hessian_matrix.subs(point)
            eigenvalues = hessian_at_point.eigenvals()
            print(f"Punto {point}: valores propios Hessiano =", eigenvalues)
            
            # Clasificación simple
            if all(eigenvalue > 0 for eigenvalue in eigenvalues):
                point_type = "mínimo local"
            elif all(eigenvalue < 0 for eigenvalue in eigenvalues):
                point_type = "máximo local"
            else:
                point_type = "punto de silla"
            print("  Clasificación:", point_type)


if __name__ == '__main__':
    calculator = NonLinearCalculator()
    calculator.menu()
