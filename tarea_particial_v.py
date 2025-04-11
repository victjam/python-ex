class WorkforceOptimizer:
    def __init__(self, demanda, costo_mantener, costo_fijo_contratacion, costo_por_contrato):
        self.demanda = demanda
        self.n = len(demanda)
        self.costo_mantener = costo_mantener
        self.costo_fijo_contratacion = costo_fijo_contratacion
        self.costo_por_contrato = costo_por_contrato
        self.dp = {}
        self.ruta = {}

    def calcular_costos(self):
        max_trab = max(self.demanda) + 5  # Para permitir trabajadores extra

        for trabajadores in range(max_trab):
            contratados = trabajadores
            costo = self.costo_fijo_contratacion + self.costo_por_contrato * contratados
            self.dp[(0, trabajadores)] = costo
            print(f"Semana 1, trabajadores: {trabajadores}, contratados: {contratados}, costo: {costo}")

        for semana in range(1, self.n):
            for trabajadores_actuales in range(max_trab):
                min_costo = float('inf')
                mejor_anterior = -1

                for trabajadores_previos in range(max_trab):
                    if (semana - 1, trabajadores_previos) in self.dp:
                        delta = trabajadores_actuales - trabajadores_previos
                        costo = self.dp[(semana - 1, trabajadores_previos)]

                        if delta > 0:
                            costo += self.costo_fijo_contratacion + self.costo_por_contrato * delta
                        elif delta < 0:
                            costo += self.costo_mantener * abs(delta)

                        if costo < min_costo:
                            min_costo = costo
                            mejor_anterior = trabajadores_previos

                self.dp[(semana, trabajadores_actuales)] = min_costo
                self.ruta[(semana, trabajadores_actuales)] = mejor_anterior
                print(f"Semana {semana+1}, trabajadores: {trabajadores_actuales}, mejor anterior: {mejor_anterior}, costo acumulado: {min_costo}")

    def obtener_solucion(self):
        self.calcular_costos()
        ultima_semana = self.n - 1
        min_costo = float('inf')
        mejor_final = -1

        for trabajadores in range(max(self.demanda) + 5):
            if (ultima_semana, trabajadores) in self.dp and trabajadores >= self.demanda[-1]:
                if self.dp[(ultima_semana, trabajadores)] < min_costo:
                    min_costo = self.dp[(ultima_semana, trabajadores)]
                    mejor_final = trabajadores

        plan = []
        semana = ultima_semana
        while semana >= 0:
            plan.append((semana + 1, mejor_final))
            mejor_final = self.ruta.get((semana, mejor_final), 0)
            semana -= 1

        plan.reverse()
        print("\n🔄 Plan óptimo de trabajadores por semana:")
        for semana, trabajadores in plan:
            print(f"Semana {semana}: {trabajadores} trabajadores")

        print(f"\n💰 Costo total mínimo: ${min_costo}")
        return plan, min_costo


if __name__ == "__main__":
    demanda = [5, 7, 8, 4, 6]
    optimizador = WorkforceOptimizer(demanda, 300, 400, 200)
    plan, costo_total = optimizador.obtener_solucion()
