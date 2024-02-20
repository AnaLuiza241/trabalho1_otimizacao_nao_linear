from optimize import gradient_descent_adaptive_step
from visualize import function_contour
import numpy as np
import matplotlib.pyplot as plt

f1 = lambda x: x[0] - x[1] + 2*x[0]**2 + 2*x[0]*x[1] + x[1]**2
g1 = lambda x: np.array([1 + 4*x[0] + 2*x[1], -1 + 2*x[0] + 2*x[1]])

f2 = lambda x: 10*x[0]**2 + 2*x[1]**2
g2 = lambda x: np.array([20*x[0], 4*x[1]])

f3 = lambda x: 2*x[0]**2 + x[1]**2 + 2*x[0]*x[1]
g3 = lambda x: np.array([4*x[0] + 2*x[1], 2*x[1] + 2*x[0]])

x0 = np.array([-3, -4])

functions = [(f1, g1), (f2, g2), (f3, g3)]

tolerancias = [1e-6, 1e-4, 1e-2, 1e-1]
histPadrao = []
histAurea = []
histArmijo = []

for i, (f, g) in enumerate(functions):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title(f'Função {i+1}')

    gd_gs_points = []
    gd_ar_points = []
    gda_points = []
    
    for tol in tolerancias:
        _, gd_gs, _, _ = gradient_descent_adaptive_step(x0, f, g, step_size=0.09, niter=1000, tol=tol, metod='golden_section')
        _, gd_ar, _, _ = gradient_descent_adaptive_step(x0, f, g, step_size=0.09, niter=1000, tol=tol, metod='armijo')
        _, gda, _, _ = gradient_descent_adaptive_step(x0, f, g, step_size=0.09, niter=1000, tol=tol)
       
        gd_gs_points.append((tol, len(gd_gs)))
        gd_ar_points.append((tol, len(gd_ar)))
        gda_points.append((tol, len(gda)))

        if tol == 1e-6:
            histAurea.append(len(gd_gs))
            histArmijo.append(len(gd_ar))
            histPadrao.append(len(gda))


    gd_gs_points = np.array(gd_gs_points)
    gd_ar_points = np.array(gd_ar_points)
    gda_points = np.array(gda_points)
    
    ax.plot(gd_gs_points[:, 0], gd_gs_points[:, 1], 'o-', label='GS', linewidth=3)
    ax.plot(gd_ar_points[:, 0], gd_ar_points[:, 1], 'x-', label='Armijo', linewidth=2)
    ax.plot(gda_points[:, 0], gda_points[:, 1], '+-', label='Adaptativo Padrão', linewidth=1)


    # Adicionar rótulos aos pontos
    for point in gd_gs_points:
        ax.annotate(f'{point[1]}', (point[0], point[1]), textcoords="offset points", xytext=(0, 5), ha='center')

    for point in gd_ar_points:
        ax.annotate(f'{point[1]}', (point[0], point[1]), textcoords="offset points", xytext=(0, 5), ha='center')

    for point in gda_points:
        ax.annotate(f'{point[1]}', (point[0], point[1]), textcoords="offset points", xytext=(0, 5), ha='center')


    ax.set_xlabel('Tolerância')
    ax.set_ylabel('Número de Iterações')
    ax.legend()
    
    plt.savefig(f'funcao_{i+1}_otimizacao.png')
    plt.close()

# Faz o histograma com os dados em histArmijo histAurea, histArmijo, histPadrao. cada um dos vetores será um grupo no histograma. e pra cada grupo vão ter 3 valores que serão as barras. O x é a função 1,2,3 que corresponde a posição do vetor, o y vai ser o valor que ta no vetor e a cor será do grupo. 
    
# Plotar histograma

import numpy as np
import matplotlib.pyplot as plt

# ... (código anterior)

# Faz o histograma com os dados em histArmijo, histAurea, histPadrao.
# Cada um dos vetores será um grupo no histograma.
# Para cada grupo, haverá 3 valores que serão as barras.
# O x é a função 1, 2, 3 que corresponde à posição do vetor,
# o y será o valor que está no vetor e a cor será do grupo.

fig_hist, ax_hist = plt.subplots(figsize=(10, 6))

barWidth = 0.25

r1 = np.arange(len(histArmijo))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]

ax_hist.bar(r1, histArmijo, color='green', width=barWidth, edgecolor='grey', label='Armijo')
ax_hist.bar(r2, histAurea, color='orange', width=barWidth, edgecolor='grey', label='Seção Áurea')
ax_hist.bar(r3, histPadrao, color='blue', width=barWidth, edgecolor='grey', label='Adaptativo Padrão')

ax_hist.set_xlabel('Funções')
ax_hist.set_ylabel('Número de Iterações')
ax_hist.set_xticks([r + barWidth for r in range(len(histArmijo))])
ax_hist.set_xticklabels(['Função 1', 'Função 2', 'Função 3'])
ax_hist.legend()

plt.savefig('histogram.png')
plt.show()
