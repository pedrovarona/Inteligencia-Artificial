class ranasProblem(Problem):
    def __init__(self, N):
        self.N = N
        estado_inicial = ['L'] * N + ['_'] + ['R'] * N
        estado_final   = ['R'] * N + ['_'] + ['L'] * N
        super().__init__(tuple(estado_inicial), tuple(estado_final))

    def actions(self, state):
        n = len(state)
        acciones_validas = []
        v = state.index('_')

        if v-1 >= 0 and state[v-1] == 'L':
            acciones_validas.append((v-1, v))
        if v-2 >= 0 and state[v-2] == 'L' and state[v-1] == 'R':
            acciones_validas.append((v-2, v))
        if v+1 < n and state[v+1] == 'R':
            acciones_validas.append((v+1, v))
        if v+2 < n and state[v+2] == 'R' and state[v+1] == 'L':
            acciones_validas.append((v+2, v))

        return acciones_validas

    def result(self, state, action):
        v = state.index('_')
        copia_estado = list(state)

        if action == (v-1, v):
            copia_estado[v-1] = '_'
            copia_estado[v] = 'L'

        elif action == (v-2, v):
            copia_estado[v-2] = '_'
            copia_estado[v] = 'L'

        elif action == (v+1, v):
            copia_estado[v+1] = '_'
            copia_estado[v] = 'R'

        elif action == (v+2, v):
            copia_estado[v+2] = '_'
            copia_estado[v] = 'R'

        return tuple(copia_estado)
