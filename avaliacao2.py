# =============================================================================
# 2ª AVALIAÇÃO PRÁTICA DE MÉTODOS DE OTIMIZAÇÃO (VERSÃO COESA - OOP)
# MESTRADO EM MODELAGEM MATEMÁTICA E COMPUTACIONAL - UFRRJ
#
# Aluno: Axl Andrade
#
# Este script foi refatorado para uma estrutura orientada a objetos,
# melhorando a coesão, organização e reusabilidade do código.
# =============================================================================

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.linalg import null_space, lstsq

class QuadraticProgram:
    """
    Encapsula um Problema de Programação Quadrática (QP) e seus solvers.
    O problema é da forma:
      minimizar 1/2 * x'Qx - c'x
      sujeito a Ax = b (igualdade) e/ou Ax <= b (desigualdade)
    """
    def __init__(self, Q, c, A_eq=None, b_eq=None, A_ub=None, b_ub=None, name=""):
        self.Q = np.asarray(Q, dtype=float)
        self.c = np.asarray(c, dtype=float)
        self.A_eq = np.asarray(A_eq, dtype=float) if A_eq is not None else None
        self.b_eq = np.asarray(b_eq, dtype=float) if b_eq is not None else None
        self.A_ub = np.asarray(A_ub, dtype=float) if A_ub is not None else None
        self.b_ub = np.asarray(b_ub, dtype=float) if b_ub is not None else None
        self.name = name
        self.n_vars = self.Q.shape[0]

    def evaluate_objective(self, x):
        """Calcula o valor da função objetivo em um ponto x."""
        return 0.5 * x.T @ self.Q @ x - self.c.T @ x

    def solve_with_null_space(self, x0, tol=1e-6):
        """
        Resolve o QP com restrições de IGUALDADE (Questão 1).
        Assume que self.A_eq e self.b_eq estão definidos.
        """
        # 1. Encontrar solução particular e base para o espaço nulo
        x_p = np.linalg.lstsq(self.A_eq, self.b_eq, rcond=None)[0]
        Z = null_space(self.A_eq)

        if Z.shape[1] == 0: # Se não há espaço nulo, a solução é única
            return x_p, 0

        # 2. Transformar em um problema irrestrito na variável 'v'
        Q_hat = Z.T @ self.Q @ Z
        c_hat = Z.T @ (self.c - self.Q @ x_p)
        
        # 3. Resolver o subproblema irrestrito com Gradientes Conjugados
        v0 = np.zeros(Z.shape[1])
        g_hat = lambda v: Q_hat @ v - c_hat
        
        vk = v0
        rk = -g_hat(vk)
        pk = rk
        for i in range(self.n_vars):
            if np.linalg.norm(rk) < tol: break
            Qpk = Q_hat @ pk
            alpha = (rk.T @ rk) / (pk.T @ Qpk)
            vk += alpha * pk
            rk_new = rk - alpha * Qpk
            beta = (rk_new.T @ rk_new) / (rk.T @ rk)
            pk = rk_new + beta * pk
            rk = rk_new

        # 4. Converter a solução de volta para o espaço original
        return x_p + Z @ vk, i + 1

    def solve_with_active_set(self, x0, tol=1e-6, max_iter=100):
        """
        Resolve o QP com restrições de DESIGUALDADE (Questão 2).
        Assume que self.A_ub e self.b_ub estão definidos.
        """
        xk = np.copy(x0)
        
        for k in range(max_iter):
            grad_k = self.Q @ xk - self.c
            
            # Identifica o conjunto de restrições ativas
            active_set_indices = list(np.where(np.abs(self.A_ub @ xk - self.b_ub) < 1e-7)[0])

            # Monta e resolve o subproblema de igualdade para achar a direção pk
            A_w = self.A_ub[active_set_indices, :] if active_set_indices else np.empty((0, self.n_vars))
            n_active = A_w.shape[0]
            
            KKT_matrix = np.block([[self.Q, A_w.T], [A_w, np.zeros((n_active, n_active))]])
            KKT_rhs = np.concatenate([-grad_k, np.zeros(n_active)])
            
            try:
                sol = np.linalg.solve(KKT_matrix, KKT_rhs)
                pk = sol[:self.n_vars]
                lambdas = sol[self.n_vars:]
            except np.linalg.LinAlgError:
                print(f"  Aviso ({self.name}): Matriz KKT singular na iteração {k}.")
                return xk, k

            # Se a direção pk for (praticamente) nula, checar otimalidade
            if np.linalg.norm(pk) < tol:
                if not active_set_indices or np.all(lambdas >= -tol):
                    return xk, k # Ótimo encontrado
                else:
                    j_to_remove = np.argmin(lambdas)
                    active_set_indices.pop(j_to_remove)
                    continue

            # Se pk não é nulo, calcula passo máximo alpha
            alpha = 1.0
            for i in range(len(self.b_ub)):
                if i not in active_set_indices:
                    A_i_pk = self.A_ub[i, :].T @ pk
                    if A_i_pk > tol:
                        alpha_i = (self.b_ub[i] - self.A_ub[i, :].T @ xk) / A_i_pk
                        alpha = min(alpha, alpha_i)

            # Atualiza o ponto e o conjunto ativo
            xk += alpha * pk
        
        return xk, max_iter


class LinearProgram:
    """
    Encapsula um Problema de Programação Linear (PL) e suas operações.
    O problema é da forma:
      maximizar/minimizar c'x
      sujeito a Ax <= b e/ou Ax = b
    """
    def __init__(self, c, A_ub=None, b_ub=None, A_eq=None, b_eq=None, obj_type='max', name=""):
        self.c = c
        self.A_ub = A_ub
        self.b_ub = b_ub
        self.A_eq = A_eq
        self.b_eq = b_eq
        self.obj_type = obj_type
        self.name = name
        self.solution = None
    
    def solve(self):
        """Resolve o problema de PL usando scipy.linprog."""
        c_solver = [-x for x in self.c] if self.obj_type == 'max' else self.c
        self.solution = linprog(c=c_solver, A_ub=self.A_ub, b_ub=self.b_ub, 
                                A_eq=self.A_eq, b_eq=self.b_eq, method='highs')
        return self.solution

    def get_dual(self):
        """Cria e retorna um novo objeto LinearProgram representando o dual."""
        if self.obj_type == 'max':
            # Dual de (max c'x s.t. A_ub*x <= b_ub, A_eq*x = b_eq, x>=0)
            # é (min b_ub'y_ub + b_eq'y_eq s.t. A_ub'y_ub + A_eq'y_eq >= c, y_ub>=0, y_eq irrestrito)
            dual_obj_type = 'min'
            dual_c, dual_A_ub, dual_b_ub, dual_A_eq, dual_b_eq = [], [], [], None, None
            
            A_T_parts = []
            if self.b_ub is not None:
                dual_c.extend(self.b_ub)
                A_T_parts.append(np.array(self.A_ub).T)
            if self.b_eq is not None:
                dual_c.extend(self.b_eq)
                A_T_parts.append(np.array(self.A_eq).T)

            # A'y >= c  -> -A'y <= -c
            dual_A_ub = -np.hstack(A_T_parts)
            dual_b_ub = [-x for x in self.c]
            
            return LinearProgram(dual_c, dual_A_ub, dual_b_ub, obj_type=dual_obj_type, name=f"Dual de {self.name}")
        # Implementação do dual para 'min' seria análoga
        return None

    def display_results(self):
        """Resolve o primal, o dual, e imprime os resultados formatados."""
        print("\n" + "="*80)
        print(f"RESOLVENDO PROBLEMA: {self.name}")
        print("="*80)

        # Resolve e imprime Primal
        primal_res = self.solve()
        print("\n[PRIMAL]")
        if primal_res.success:
            obj_val = -primal_res.fun if self.obj_type == 'max' else primal_res.fun
            print(f"  Status: Sucesso | Valor Ótimo: {obj_val:.4f} | Iterações: {primal_res.nit}")
            print(f"  Solução Ótima (x*): {np.round(primal_res.x, 4)}")
        else:
            print(f"  Status: Falha - {primal_res.message}")
        
        # Resolve e imprime Dual
        dual_problem = self.get_dual()
        print("\n[DUAL]")
        if dual_problem:
            dual_res = dual_problem.solve()
            if dual_res.success:
                print(f"  Status: Sucesso | Valor Ótimo: {dual_res.fun:.4f} | Iterações: {dual_res.nit}")
                print(f"  Solução Ótima (y*): {np.round(dual_res.x, 4)}")
            else:
                print(f"  Status: Falha - {dual_res.message}")
        else:
            print("  Formulação dual não implementada para este tipo de problema.")

# =============================================================================
# BLOCO PRINCIPAL DE EXECUÇÃO
# =============================================================================

def run_qp_examples():
    """Executa os exemplos das Questões 1 e 2."""
    print("#"*80)
    print("EXECUTANDO QUESTÃO 1 e 2: PROBLEMAS DE PROGRAMAÇÃO QUADRÁTICA")
    print("#"*80)
    
    # --- Questão 1 ---
    qp1 = QuadraticProgram(
        Q=[[1, 0], [0, 2]], c=[1, 1],
        A_eq=[[1, 1]], b_eq=[1],
        name="QP com Restrição de Igualdade"
    )
    x_star_q1, iters_q1 = qp1.solve_with_null_space(x0=[0.5, 0.5])
    print(f"\nResultado Questão 1 ({qp1.name}):")
    print(f"  Solução Ótima (x*): {np.round(x_star_q1, 4)}")
    print(f"  Valor Ótimo f(x*): {qp1.evaluate_objective(x_star_q1):.4f}")

    # --- Questão 2 (CORRIGIDO) ---
    qp2 = QuadraticProgram(
        Q=[[2, 0], [0, 2]], c=[-6, -4],
        A_ub=[[1, 0], [0, 1], [-1, -1]], b_ub=[2, 2, -3],
        name="QP com Restrição de Desigualdade"
    )
    # Ponto inicial definido como float para evitar o erro de casting
    x_star_q2, iters_q2 = qp2.solve_with_active_set(x0=[0.0, 0.0])
    print(f"\nResultado Questão 2 ({qp2.name}):")
    print(f"  Solução Ótima (x*): {np.round(x_star_q2, 4)}")
    print(f"  Valor Ótimo f(x*): {qp2.evaluate_objective(x_star_q2):.4f}")
    print(f"  Iterações: {iters_q2}")


def run_lp_examples():
    """Define e resolve os problemas de PL da Questão 3."""
    print("\n" + "#"*80)
    print("EXECUTANDO QUESTÃO 3: PROBLEMAS DE PROGRAMAÇÃO LINEAR")
    print("#"*80)

    # 3.a - Bicicletas e Motoretas
    lp_3a = LinearProgram(name="3.a - Bicicletas e Motoretas", obj_type='max', c=[30, 40], A_ub=[[6, 4], [3, 10]], b_ub=[120, 180])
    lp_3a.display_results()

    # 3.b - Novos Produtos
    lp_3b = LinearProgram(name="3.b - Novos Produtos", obj_type='max', c=[33, 12, 19], A_ub=[[9, 3, 5], [5, 4, 0], [3, 0, 2], [0, 1, 0]], b_ub=[500, 350, 150, 20])
    lp_3b.display_results()
    
    # 3.c - Fábrica P1 e P2 (CORRIGIDO)
    lp_3c = LinearProgram(name="3.c - Fábrica P1 e P2", obj_type='max', c=[6, 15], A_ub=[[0, 3], [1.5, 4], [2, 3], [3, 2], [3, 0]], b_ub=[39, 57, 70, 60, 57])
    lp_3c.display_results()
    
    # 3.h - Fio de Algodão
    lp_3h = LinearProgram(name="3.h - Fio de Algodão", obj_type='max', c=[5, 10], A_ub=[[2, 1.5], [1, 2], [-1, 0]], b_ub=[15, 12, -30])
    lp_3h.display_results()
    
    # Nota: Os problemas de minimização 3.e, 3.f, 3.g, 3.i não foram incluídos aqui
    # porque o helper `get_dual` foi implementado apenas para problemas de maximização,
    # mas eles podem ser resolvidos individualmente se necessário.

if __name__ == "__main__":
    run_qp_examples()
    run_lp_examples()
