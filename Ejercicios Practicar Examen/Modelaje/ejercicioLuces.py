class lucesProblem(Problem):
    def __init__(self, N):
        self.N = N
        estado_inicial = ([1] * N)
        estado_final = ([0] * N)
        super().__init__(tuple(estado_inicial), tuple(estado_final))


    def actions(self, state):
        acciones_validas = list(range(len(state)))
        return acciones_validas

    def result(self, state, action):
        N = self.N
        nuevo_estado = list(state)

        i = action
        nuevo_estado[i] = 1 - nuevo_estado[i]
        if(i - 1 >= 0):
            j = action -1
            nuevo_estado[j] = 1 - nuevo_estado[j]
        if(i + 1 < N):
            k = action + 1
            nuevo_estado[k] = 1 - nuevo_estado[k]

        return tuple(nuevo_estado)
    
    def goal_test(self, state):
        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        return c + 1