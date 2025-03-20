"""
Tarea IV. Metodos Cuantitativos. Programacion Entera
Problema de la Mochila
Autor: Victor Manrique
"""

class Objeto:
    """
    Clase que representa un objeto con peso y valor.
    """
    def __init__(self, peso, valor):
        self.peso = peso
        self.valor = valor

    def __str__(self):
        # Para imprimir de forma legible
        return f"Objeto(peso={self.peso}, valor={self.valor})"


class Mochila:
    """
    Clase que representa la mochila y contiene los métodos
    para resolver el problema de la mochila.
    """
    def __init__(self, capacidad_maxima, objetos):
        """
        capacidad_maxima: capacidad entera de la mochila
        objetos: lista de instancias de Objeto
        """
        self.capacidad_maxima = capacidad_maxima
        self.objetos = objetos
        self.objetos_seleccionados = []  # aquí guardaremos la solución

    def resolver_mochila(self):
        """
        Resuelve el problema de la mochila usando Programación Dinámica.
        Retorna la lista de objetos que conforman la solución óptima.
        """
        n = len(self.objetos)
        capacidad = self.capacidad_maxima

        # tabla[i][peso] almacenará el valor máximo alcanzable usando
        # los primeros i objetos con capacidad peso
        # (i y peso son índices 1-based en la lógica; tabla se define con i+1, peso+1)
        tabla = [[0]*(capacidad+1) for _ in range(n+1)]

        # Llenamos la tabla de programación dinámica
        for i in range(1, n+1):
            peso_objeto = self.objetos[i-1].peso
            valor_objeto = self.objetos[i-1].valor
            for peso_actual in range(1, capacidad+1):
                if peso_objeto <= peso_actual:
                    # Podemos escoger no tomar el objeto i, o tomarlo si cabe
                    # Tomamos el máximo
                    tabla[i][peso_actual] = max(
                        tabla[i-1][peso_actual],  # no tomar el objeto i
                        tabla[i-1][peso_actual - peso_objeto] + valor_objeto  # tomar el objeto i
                    )
                else:
                    # No cabe este objeto
                    tabla[i][peso_actual] = tabla[i-1][peso_actual]

        # El valor óptimo se encuentra en tabla[n][capacidad]
        valor_optimo = tabla[n][capacidad]

        # Reconstruimos qué objetos se seleccionaron
        objetos_seleccionados = []
        peso_restante = capacidad
        for i in range(n, 0, -1):
            if tabla[i][peso_restante] != tabla[i-1][peso_restante]:
                # Significa que el objeto i sí se tomó
                objetos_seleccionados.append(self.objetos[i-1])
                peso_restante -= self.objetos[i-1].peso

        # Revertimos la lista para que aparezcan en orden original
        objetos_seleccionados.reverse()

        # Guardamos en un atributo de la clase (por si se quiere acceder después)
        self.objetos_seleccionados = objetos_seleccionados
        return objetos_seleccionados, valor_optimo

    def imprimir_resultados(self):
        """
        Imprime en pantalla los objetos seleccionados, su valor total y el peso total.
        """
        objetos_sel, valor_total = self.resolver_mochila()

        # Calculamos el peso total
        peso_total = sum(obj.peso for obj in objetos_sel)

        print("OBJETOS SELECCIONADOS:")
        for obj in objetos_sel:
            print(f"  {obj}")
        print(f"Valor total: {valor_total}")
        print(f"Peso total: {peso_total}")


def main():
    
    print("PROBLEMA DE LA MOCHILA - EJEMPLO")
    # Se podrían pedir los datos al usuario, aquí directamente usamos el ejemplo
    capacidad = 10
    objetos_ejemplo = [
        Objeto(peso=5, valor=10),
        Objeto(peso=4, valor=40),
        Objeto(peso=6, valor=30),
        Objeto(peso=3, valor=50)
    ]
    
    mochila = Mochila(capacidad, objetos_ejemplo)
    mochila.imprimir_resultados()


if __name__ == "__main__":
    main()
