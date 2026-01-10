class ejercicioRuta(Problem):
    def __init__(self, N):
        self.N = N
        estado_inicial = ['Ciudad'] * N
        estado_final = ['Ciudad'] * N


    def actions(self, state):
        acciones_validas = []
        ciudades_impares_visitadas = 0

        while(ciudades_impares_visitadas < 3):
            acciones_validas.append()