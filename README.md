# Trabalhos Práticos - IM7403 Métodos de Otimização

Este repositório contém as soluções em Python para as avaliações práticas da disciplina IM7403 - Métodos de Otimização, ministrada pelo Prof. Ronaldo Malheiros Gregório no Programa de Pós-graduação em Modelagem Matemática e Computacional (PPGMMC) da UFRRJ, referente ao período de 2025.1.

## 📝 Conteúdo

O projeto está dividido em duas partes principais, correspondentes às duas avaliações práticas.

### 🚀 Avaliação Prática 1: Otimização Irrestrita

Implementação e análise comparativa de algoritmos clássicos de descida para problemas de otimização sem restrições.

* **Algoritmos Implementados:**
    * Método do Gradiente
    * Método de Newton
    * Métodos Quase-Newton (DFP e BFGS)
    * Método dos Gradientes Conjugados
* **Técnicas Auxiliares:**
    * Busca Linear de Armijo para determinação do tamanho do passo.
* **Análise:** O estudo demonstrou o comportamento dos algoritmos em funções convexas, não-convexas (com ponto de sela) e ilimitadas, validando a teoria de convergência e divergência.

### 🎯 Avaliação Prática 2: Otimização com Restrições

Implementação de algoritmos para Programação Quadrática (QP) e modelagem de problemas de Programação Linear (PL).

* **Algoritmos de Programação Quadrática (QP):**
    * **Método do Espaço Nulo:** Para problemas com restrições de igualdade (`Ax = b`).
    * **Método de Restrições Ativas:** Para problemas com restrições de desigualdade (`Ax <= b`).
* **Modelagem em Programação Linear (PL):**
    * Formulação e resolução de 9 problemas práticos de alocação de recursos.
    * Análise da relação Primal-Dual para cada problema, verificando na prática os teoremas da Dualidade Forte e a relação de inviabilidade/ilimitabilidade.

## 🛠️ Como Executar

Cada avaliação está em seu próprio script auto-contido.

### Pré-requisitos

É necessário ter Python 3 instalado, juntamente com as seguintes bibliotecas:

```bash
pip install numpy scipy pandas tabulate
