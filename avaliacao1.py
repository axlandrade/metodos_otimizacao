# =============================================================================
# 1ª AVALIAÇÃO PRÁTICA DE MÉTODOS DE OTIMIZAÇÃO
# MESTRADO EM MODELAGEM MATEMÁTICA E COMPUTACIONAL - UFRRJ
#
# Aluno: Axl Silva de Andrade
#
# Este script implementa e compara cinco algoritmos de otimização irrestrita
# para cinco funções distintas, conforme solicitado na avaliação.
# =============================================================================

import numpy as np
import pandas as pd
import time

# =============================================================================
# SEÇÃO 1: DEFINIÇÃO DAS FUNÇÕES, GRADIENTES E HESSIANAS
#
# Cada função do problema (a, b, c, d, e) é definida aqui, juntamente
# com suas derivadas de primeira (gradiente) e segunda (Hessiana) ordem.
# A Hessiana é necessária apenas para o Método de Newton.
# =============================================================================

# --- Função (a): f(x1, x2) = x1^2 + x2^2 - 3*x1*x2 ---
def f_a(x):
    return x[0]**2 + x[1]**2 - 3*x[0]*x[1]

def grad_f_a(x):
    return np.array([2*x[0] - 3*x[1], 2*x[1] - 3*x[0]])

def hess_f_a(x):
    return np.array([[2.0, -3.0], [-3.0, 2.0]])

# --- Função (b): f(x1, x2) = (x2 - x1)^2 - x1^5 ---
def f_b(x):
    return (x[1] - x[0])**2 - x[0]**5

def grad_f_b(x):
    return np.array([-2*(x[1] - x[0]) - 5*x[0]**4, 2*(x[1] - x[0])])

def hess_f_b(x):
    return np.array([[2 - 20*x[0]**3, -2.0], [-2.0, 2.0]])

# --- Função (c): f(x) = ||(1, 2) - x||^2 ---
def f_c(x):
    b = np.array([1.0, 2.0])
    return (b - x).T @ (b - x)

def grad_f_c(x):
    b = np.array([1.0, 2.0])
    return -2 * (b - x)

def hess_f_c(x):
    return np.array([[2.0, 0.0], [0.0, 2.0]])

# --- Função (d): f(x1, x2) = (x1*x2)^2 - sin(x1) - cos(x2) ---
def f_d(x):
    return (x[0]*x[1])**2 - np.sin(x[0]) - np.cos(x[1])

def grad_f_d(x):
    return np.array([2*x[0]*x[1]**2 - np.cos(x[0]), 2*x[1]*x[0]**2 + np.sin(x[1])])

def hess_f_d(x):
    return np.array([
        [2*x[1]**2 + np.sin(x[0]), 4*x[0]*x[1]],
        [4*x[0]*x[1], 2*x[0]**2 + np.cos(x[1])]
    ])

# --- Função (e): f(x) = 0.5 * ||Ax - b||^2 ---
# Gerador para a função 'e', pois ela depende de A e b que podem ser gerados
def get_function_e(n=10, seed=42):
    """Gera a matriz A e o vetor b para a função (e)"""
    np.random.seed(seed)
    A = np.random.rand(n, n)
    b = np.random.rand(n)

    def f_e(x):
        return 0.5 * np.linalg.norm(A @ x - b)**2
    def grad_f_e(x):
        return A.T @ (A @ x - b)
    def hess_f_e(x):
        return A.T @ A
    return f_e, grad_f_e, hess_f_e


# =============================================================================
# SEÇÃO 2: IMPLEMENTAÇÃO DA BUSCA LINEAR DE ARMIJO
#
# Esta função implementa a regra de Armijo para encontrar um tamanho de passo
# alpha que garanta uma redução suficiente no valor da função.
# =============================================================================

def armijo_line_search(f, grad_f, xk, dk, alpha0=1.0, c=1e-4, rho=0.5):
    """
    Busca linear de Armijo.

    Parâmetros:
        f: A função objetivo.
        grad_f: O gradiente da função objetivo.
        xk: O ponto atual.
        dk: A direção de descida.
        alpha0: O tamanho de passo inicial.
        c: Parâmetro da condição de Armijo (geralmente 1e-4).
        rho: Fator de redução do passo (geralmente 0.5).

    Retorna:
        O tamanho de passo alpha que satisfaz a condição de Armijo.
    """
    alpha = alpha0
    grad_fk_dk = grad_f(xk).T @ dk
    fk = f(xk)

    while f(xk + alpha * dk) > fk + c * alpha * grad_fk_dk:
        alpha = rho * alpha
        if alpha < 1e-10: # Evita loop infinito com passos muito pequenos
            return alpha
    return alpha

# =============================================================================
# SEÇÃO 3: IMPLEMENTAÇÃO DOS MÉTODOS DE OTIMIZAÇÃO
#
# Cada método de descida solicitado é implementado como uma função separada.
# Todos eles utilizam a busca de Armijo para o cálculo do passo.
# =============================================================================

def gradient_descent(f, grad_f, x0, tol, max_iter=1000):
    """Método do Gradiente (Descida Mais Íngreme)"""
    xk = np.copy(x0)
    iterations = 0
    for k in range(max_iter):
        iterations = k + 1
        grad_xk = grad_f(xk)
        dk = -grad_xk # Direção de descida mais íngreme

        alpha = armijo_line_search(f, grad_f, xk, dk)
        xk_new = xk + alpha * dk

        if np.linalg.norm(xk_new - xk) < tol:
            break
        xk = xk_new
    return xk, f(xk), iterations

def newton_method(f, grad_f, hess_f, x0, tol, max_iter=100):
    """Método de Newton"""
    xk = np.copy(x0)
    iterations = 0
    for k in range(max_iter):
        iterations = k + 1
        grad_xk = grad_f(xk)
        hess_xk = hess_f(xk)

        # Resolve o sistema linear H_k * d_k = -g_k
        try:
            # Tenta usar um solver linear robusto
            dk = np.linalg.solve(hess_xk, -grad_xk)
        except np.linalg.LinAlgError:
            # Se a Hessiana for singular, usa a direção do gradiente
            dk = -grad_xk

        # Garante que a direção é de descida
        if grad_xk.T @ dk >= 0:
            dk = -grad_xk

        alpha = armijo_line_search(f, grad_f, xk, dk)
        xk_new = xk + alpha * dk

        if np.linalg.norm(xk_new - xk) < tol:
            break
        xk = xk_new
    return xk, f(xk), iterations

def quasi_newton_dfp(f, grad_f, x0, tol, max_iter=1000):
    """Método Quase-Newton DFP (Davidon-Fletcher-Powell)"""
    xk = np.copy(x0)
    n = len(x0)
    Hk = np.eye(n) # Aproximação inicial da Hessiana inversa
    iterations = 0
    for k in range(max_iter):
        iterations = k + 1
        grad_xk = grad_f(xk)

        if np.linalg.norm(grad_xk) < tol:
            break

        dk = -Hk @ grad_xk

        alpha = armijo_line_search(f, grad_f, xk, dk, alpha0=1.0)
        xk_new = xk + alpha * dk

        if np.linalg.norm(xk_new - xk) < tol:
            break

        pk = xk_new - xk
        grad_xk_new = grad_f(xk_new)
        qk = grad_xk_new - grad_xk

        # Evita divisão por zero
        if np.abs(pk.T @ qk) < 1e-10:
             break

        term1 = (pk @ pk.T) / (pk.T @ qk)
        term2_num = (Hk @ qk) @ (qk.T @ Hk)
        term2_den = qk.T @ Hk @ qk
        
        if np.abs(term2_den) < 1e-10:
            break

        Hk = Hk + term1 - (term2_num / term2_den)
        xk = xk_new
    return xk, f(xk), iterations


def quasi_newton_bfgs(f, grad_f, x0, tol, max_iter=1000):
    """Método Quase-Newton BFGS (Broyden-Fletcher-Goldfarb-Shanno)"""
    xk = np.copy(x0)
    n = len(x0)
    Hk = np.eye(n) # Aproximação inicial da Hessiana inversa
    iterations = 0
    for k in range(max_iter):
        iterations = k + 1
        grad_xk = grad_f(xk)
        
        if np.linalg.norm(grad_xk) < tol:
            break
            
        dk = -Hk @ grad_xk

        alpha = armijo_line_search(f, grad_f, xk, dk, alpha0=1.0)
        xk_new = xk + alpha * dk

        if np.linalg.norm(xk_new - xk) < tol:
            break

        pk = xk_new - xk
        grad_xk_new = grad_f(xk_new)
        qk = grad_xk_new - grad_xk

        pk = pk.reshape(-1, 1)
        qk = qk.reshape(-1, 1)
        
        # Evita divisão por zero
        pk_T_qk = pk.T @ qk
        if np.abs(pk_T_qk) < 1e-10:
            break
        
        qk_T_Hk_qk = qk.T @ Hk @ qk
        
        term1_factor = (pk_T_qk + qk_T_Hk_qk) / (pk_T_qk**2)
        term1 = term1_factor * (pk @ pk.T)
        term2 = (Hk @ qk @ pk.T + pk @ qk.T @ Hk) / pk_T_qk
        
        Hk = Hk + term1 - term2
        xk = xk_new.flatten()

    return xk, f(xk), iterations


def conjugate_gradient(f, grad_f, x0, tol, max_iter=1000):
    """Método dos Gradientes Conjugados (Fletcher-Reeves)"""
    xk = np.copy(x0)
    grad_xk = grad_f(xk)
    dk = -grad_xk
    iterations = 0
    for k in range(max_iter):
        iterations = k + 1
        
        alpha = armijo_line_search(f, grad_f, xk, dk)
        xk_new = xk + alpha * dk
        
        if np.linalg.norm(xk_new - xk) < tol:
            break

        grad_xk_new = grad_f(xk_new)
        
        # Fórmula de Fletcher-Reeves
        beta = (grad_xk_new.T @ grad_xk_new) / (grad_xk.T @ grad_xk)
        dk = -grad_xk_new + beta * dk
        
        # Reset para gradiente se a direção não for de descida
        if grad_xk_new.T @ dk >= 0:
            dk = -grad_xk_new
            
        xk = xk_new
        grad_xk = grad_xk_new

    return xk, f(xk), iterations

# =============================================================================
# SEÇÃO 4: EXECUÇÃO DOS EXPERIMENTOS E GERAÇÃO DO QUADRO COMPARATIVO
#
# Este bloco principal executa cada método para cada função e armazena
# os resultados para apresentação final.
# =============================================================================

if __name__ == "__main__":
    
    # --- Configurações do Experimento ---
    
    # Tolerância para o critério de parada
    TOL = 1e-6
    
    # Pontos iniciais (o usuário pode alterar aqui)
    # Dimensões: 2 para a-d, 10 para e
    x0_2d = np.array([3.0, 3.0])
    x0_10d = np.ones(10) * 3.0
    
    # Dicionário para organizar as funções e seus pontos iniciais
    f_e, grad_f_e, hess_f_e = get_function_e()
    
    PROBLEMS = {
        'Função (a)': {'f': f_a, 'grad': grad_f_a, 'hess': hess_f_a, 'x0': x0_2d},
        'Função (b)': {'f': f_b, 'grad': grad_f_b, 'hess': hess_f_b, 'x0': x0_2d},
        'Função (c)': {'f': f_c, 'grad': grad_f_c, 'hess': hess_f_c, 'x0': x0_2d},
        'Função (d)': {'f': f_d, 'grad': grad_f_d, 'hess': hess_f_d, 'x0': x0_2d},
        'Função (e)': {'f': f_e, 'grad': grad_f_e, 'hess': hess_f_e, 'x0': x0_10d},
    }
    
    # Lista de métodos a serem testados
    METHODS = {
        "Gradiente": gradient_descent,
        "Newton": newton_method,
        "DFP": quasi_newton_dfp,
        "BFGS": quasi_newton_bfgs,
        "Grad. Conjugados": conjugate_gradient
    }
    
    # --- Loop de Execução ---
    
    results = []
    
    print("Iniciando a execução dos métodos de otimização...")
    print("Isso pode levar alguns instantes.")
    
    for prob_name, prob_details in PROBLEMS.items():
        print(f"\nProcessando {prob_name}...")
        for method_name, method_func in METHODS.items():
            
            # Alguns métodos precisam da Hessiana, outros não.
            args = (
                prob_details['f'],
                prob_details['grad'],
                prob_details['x0'],
                TOL
            )
            if method_name == "Newton":
                args = (
                    prob_details['f'],
                    prob_details['grad'],
                    prob_details['hess'],
                    prob_details['x0'],
                    TOL
                )

            start_time = time.time()
            try:
                # O Python desempacota a tupla 'args' nos parâmetros da função
                x_final, f_final, iters = method_func(*args)
                status = "Concluído"
            except Exception as e:
                x_final, f_final, iters = (None, None, None)
                status = f"Erro: {e}"

            end_time = time.time()
            
            # Armazena os resultados
            results.append({
                "Função": prob_name,
                "Método": method_name,
                "Ponto Inicial": np.round(prob_details['x0'], 2),
                "Solução Computada (x*)": np.round(x_final, 4) if x_final is not None else "N/A",
                "Valor Ótimo (f(x*))": f'{f_final:.6f}' if f_final is not None else "N/A",
                "Iterações": iters if iters is not None else "N/A",
                "Tempo (s)": f'{(end_time - start_time):.4f}'
            })
    
    # --- Apresentação dos Resultados ---
    
    df_results = pd.DataFrame(results)
    
    # Configura o pandas para exibir melhor os resultados
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    pd.set_option('display.colheader_justify', 'center')

    print("\n\n" + "="*80)
    print("QUADRO COMPARATIVO DO DESEMPENHO COMPUTACIONAL DOS MÉTODOS")
    print("="*80)
    print(df_results)