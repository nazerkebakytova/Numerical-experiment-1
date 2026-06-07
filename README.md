"""
=============================================================================
Численный эксперимент: класс SW_2^r(0,1)^s, s=2, r=2
Тестовая функция: f(x1,x2) = x1^2 * x2^2 * (1-x1)^2 * (1-x2)^2

Предельная погрешность (теорема С(N)D-2):
    eps_N = (ln N)^{r(s-1)} / N^{r+1/2} = (ln N)^2 / N^{5/2}  при s=2, r=2


 numpy, matplotlib
=============================================================================
"""

import numpy as np
import matplotlib
try:
    matplotlib.use('TkAgg')       # работает в большинстве окружений
    import matplotlib.pyplot as plt
    plt.figure()                  # тест
    plt.close()
    INTERACTIVE = True
    print("Бэкенд: TkAgg (интерактивный)")
except Exception:
    try:
        matplotlib.use('Qt5Agg')
        import matplotlib.pyplot as plt
        plt.figure()
        plt.close()
        INTERACTIVE = True
        print("Бэкенд: Qt5Agg (интерактивный)")
    except Exception:
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        INTERACTIVE = False
        print("Бэкенд: Agg — графики будут сохранены в PNG, но не показаны на экране")

from mpl_toolkits.mplot3d import Axes3D   # noqa: F401

# ── Параметры ─────────────────────────────────────────────────────────────────
R_VALUES = [5, 7, 10, 15, 20, 30, 40, 50, 70, 100, 150, 200, 300, 500]
GRID_N   = 40
SEED     = 42

BLUE  = '#1a4f8a'
RED   = '#c0392b'
GRAY  = '#666666'


# ── Коэффициенты Фурье ────────────────────────────────────────────────────────
def g_hat(m):
    """
    g_hat(m) = integral_0^1 x^2*(1-x)^2 * e^{-2*pi*i*m*x} dx
    Вычисляется рекуррентно:
        I_0 = (1 - e^{-2pi*i*m}) / (2*pi*i*m)
        I_n = -e^{-2pi*i*m}/(2*pi*i*m) + n/(2*pi*i*m) * I_{n-1}
        g_hat(m) = I_2 - 2*I_3 + I_4
    """
    if m == 0:
        return 1.0 / 30.0
    w = 2 * np.pi * m
    def I(n):
        if n == 0:
            return (1.0 - np.exp(-1j * w)) / (1j * w)
        return -np.exp(-1j * w) / (1j * w) + n / (1j * w) * I(n - 1)
    return I(2) - 2*I(3) + I(4)


print("Предвычисление коэффициентов Фурье...")
G_CACHE = {m: g_hat(m) for m in range(-max(R_VALUES)-1, max(R_VALUES)+2)}
print("Готово.")


# ── Гиперболический крест ─────────────────────────────────────────────────────
def hyperbolic_cross(R):
    return [(m1, m2)
            for m1 in range(-R, R+1)
            for m2 in range(-R, R+1)
            if max(1, abs(m1)) * max(1, abs(m2)) <= R]


# ── Агрегат восстановления ────────────────────────────────────────────────────
def reconstruct(gamma, coeffs, x1, x2, chunk=200):
    m1a = np.array([m for m, _ in gamma])
    m2a = np.array([m for _, m in gamma])
    out = np.zeros(len(x1))
    for s in range(0, len(x1), chunk):
        e = min(s + chunk, len(x1))
        ph = 2*np.pi * (np.outer(m1a, x1[s:e]) + np.outer(m2a, x2[s:e]))
        out[s:e] = (coeffs @ np.exp(1j * ph)).real
    return out


# ── Основной цикл ────────────────────────────────────────────────────────────
def run_experiment():
    xs = np.linspace(0.02, 0.98, GRID_N)
    X1, X2 = np.meshgrid(xs, xs)
    x1f, x2f = X1.flatten(), X2.flatten()
    f_true = x1f**2 * x2f**2 * (1-x1f)**2 * (1-x2f)**2

    rng = np.random.default_rng(SEED)
    results = []

    for R in R_VALUES:
        gamma  = hyperbolic_cross(R)
        N      = len(gamma)
        lN     = np.log(N)
        coeffs = np.array([G_CACHE[m1] * G_CACHE[m2] for m1, m2 in gamma])

        Phi = reconstruct(gamma, coeffs, x1f, x2f)
        EN  = float(np.max(np.abs(f_true - Phi)))

        # eta=1: eps = EN/N
        eps1 = EN / N
        z1   = coeffs + eps1 * rng.uniform(-1, 1, N)
        EN1  = float(np.max(np.abs(f_true - reconstruct(gamma, z1, x1f, x2f))))

        # eta=logN: eps = logN * EN/N
        eps2 = lN * EN / N
        z2   = coeffs + eps2 * rng.uniform(-1, 1, N)
        EN2  = float(np.max(np.abs(f_true - reconstruct(gamma, z2, x1f, x2f))))

        # Теоретическая предельная погрешность: (ln N)^2 / N^{5/2}
        eps_th = lN**2 / N**2.5

        results.append(dict(R=R, N=N, EN=EN, eps_N=eps1,
                            EN1=EN1, r1=EN1/EN,
                            EN2=EN2, r2=EN2/EN,
                            eps_th=eps_th))
        print(f"R={R:4d}  N={N:6d}  EN={EN:.3e}  "
              f"eps_N={eps1:.3e}  ENeps={EN1:.3e}  "
              f"ratio={EN1/EN:.4f}  eps_th={eps_th:.3e}")
    return results


def print_table(results):
    print(f"\n{'R':>5} {'N':>7} {'EN':>12} {'eps_N':>13} "
          f"{'ENeps':>12} {'ENeps/EN':>10} {'eps_th':>14}")
    print("-" * 75)
    for r in results:
        print(f"{r['R']:5d} {r['N']:7d} {r['EN']:12.3e} {r['eps_N']:13.3e} "
              f"{r['EN1']:12.3e} {r['r1']:10.4f} {r['eps_th']:14.4e}")


def _savefig(fig, path):
    """Сохранить и показать фигуру."""
    fig.savefig(path, dpi=180, bbox_inches='tight')
    print(f"Сохранён {path}")
    if INTERACTIVE:
        plt.show(block=False)
        plt.pause(0.5)
    else:
        plt.close(fig)


# ── Рисунок 1: Гиперболический крест ─────────────────────────────────────────
def fig_cross(savepath='fig1_cross.png'):
    g50 = hyperbolic_cross(50)
    m1s = [m for m, _ in g50]
    m2s = [m for _, m in g50]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(m1s, m2s, s=2, c=BLUE, marker='s', linewidths=0)
    ax.set_xlabel('$m_1$', fontsize=12)
    ax.set_ylabel('$m_2$', fontsize=12)
    ax.set_title(r'$m=(m_1,m_2)\in\Gamma_R,\ R=50,\ N=1029$', fontsize=11)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25, lw=0.5)
    plt.tight_layout()
    _savefig(fig, savepath)


# ── Рисунок 2: 3D поверхность f ──────────────────────────────────────────────
def fig_func3d(savepath='fig2_func3d.png'):
    n  = 60
    xs = np.linspace(0, 1, n)
    X1, X2 = np.meshgrid(xs, xs)
    F  = X1**2 * X2**2 * (1-X1)**2 * (1-X2)**2

    fig = plt.figure(figsize=(6, 5))
    ax  = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X1, X2, F, cmap='viridis', alpha=0.92, linewidth=0)
    ax.set_xlabel('$x_1$', fontsize=11, labelpad=4)
    ax.set_ylabel('$x_2$', fontsize=11, labelpad=4)
    ax.set_zlabel('$f$',   fontsize=11, labelpad=2)
    ax.set_title(r'$f(x_1,x_2)=x_1^2x_2^2(1{-}x_1)^2(1{-}x_2)^2$',
                 fontsize=10, pad=8)
    ax.view_init(elev=28, azim=225)
    ax.tick_params(labelsize=8)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=12, pad=0.08)
    plt.tight_layout()
    _savefig(fig, savepath)


# ── Рисунок 3: 3D поверхность ошибки ─────────────────────────────────────────
def fig_err3d(savepath='fig3_err3d.png'):
    R      = 50
    gamma  = hyperbolic_cross(R)
    N      = len(gamma)
    coeffs = np.array([G_CACHE[m1]*G_CACHE[m2] for m1, m2 in gamma])

    ne  = 35
    xse = np.linspace(0.02, 0.98, ne)
    X1e, X2e = np.meshgrid(xse, xse)
    x1e, x2e = X1e.flatten(), X2e.flatten()
    fte  = x1e**2 * x2e**2 * (1-x1e)**2 * (1-x2e)**2
    Phi  = reconstruct(gamma, coeffs, x1e, x2e)
    Err  = np.abs(fte - Phi).reshape(ne, ne)

    fig  = plt.figure(figsize=(6, 5))
    ax   = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X1e, X2e, Err, cmap='hot_r', alpha=0.92, linewidth=0)
    ax.set_xlabel('$x_1$', fontsize=11, labelpad=4)
    ax.set_ylabel('$x_2$', fontsize=11, labelpad=4)
    ax.set_zlabel(r'$|f-\Phi_N(f)|$', fontsize=10, labelpad=2)
    ax.set_title(rf'$|f-\Phi_N(f)|$, $R={R}$, $N={N}$', fontsize=11, pad=8)
    ax.view_init(elev=28, azim=225)
    ax.tick_params(labelsize=8)
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=12, pad=0.08)
    plt.tight_layout()
    _savefig(fig, savepath)


# ── Рисунок 4(a,b): Убывание погрешности ─────────────────────────────────────
def fig_errors_ab(results, savepath='fig4_ab.png'):
    N_a   = np.array([r['N']   for r in results], dtype=float)
    EN_a  = np.array([r['EN']  for r in results])
    EN1_a = np.array([r['EN1'] for r in results])
    EN2_a = np.array([r['EN2'] for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.loglog(N_a, EN_a,  'o-',  color=BLUE, ms=5, lw=1.5,
              label=r'$E_N$ (точная информация)')
    ax.loglog(N_a, EN1_a, 's--', color=RED,  ms=4, lw=1.2,
              label=r'$E_N^\varepsilon$ при $\tilde\varepsilon_N$')
    C = EN_a[6] * N_a[6]**1.5 / np.log(N_a[6])**2
    ax.loglog(N_a, C * np.log(N_a)**2 / N_a**1.5, ':', color=GRAY, lw=1.3,
              label=r'$C(\ln N)^2/N^{3/2}$')
    ax.set_xlabel('$N$', fontsize=12)
    ax.set_ylabel(r'$E_N$', fontsize=12)
    ax.set_title(r'Убывание погрешности $E_N$, $\eta_N\equiv 1$', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both', lw=0.5)

    ax = axes[1]
    ax.loglog(N_a, EN_a,  'o-',  color=BLUE, ms=5, lw=1.5,
              label=r'$E_N$ (точная информация)')
    ax.loglog(N_a, EN2_a, 's--', color=RED,  ms=4, lw=1.2,
              label=r'$E_N^\varepsilon$ при $\eta_N\tilde\varepsilon_N$')
    ax.set_xlabel('$N$', fontsize=12)
    ax.set_ylabel(r'$E_N^\varepsilon$', fontsize=12)
    ax.set_title(r'Погрешность $E_N^\varepsilon$, $\eta_N\equiv\log N$', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, which='both', lw=0.5)

    plt.tight_layout()
    _savefig(fig, savepath)


# ── Рисунок 5(c,d): Нормированная погрешность ────────────────────────────────
def fig_ratio_cd(results, savepath='fig5_cd.png'):
    N_a  = np.array([r['N']  for r in results], dtype=float)
    r1_a = np.array([r['r1'] for r in results])
    r2_a = np.array([r['r2'] for r in results])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    ax = axes[0]
    ax.semilogx(N_a, r1_a, 'o-', color=BLUE, ms=5, lw=1.5)
    ax.axhline(1.0, color='k', ls='--', lw=1.2, alpha=0.6, label='$= 1$')
    ax.set_ylim([0.7, 1.5])
    ax.set_xlabel('$N$', fontsize=12)
    ax.set_ylabel(r'$E_N^\varepsilon / E_N$', fontsize=12)
    ax.set_title(r'Нормированная погрешность, $\eta_N\equiv 1$', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, lw=0.5)

    ax = axes[1]
    ax.semilogx(N_a, r2_a, 's-', color=RED, ms=5, lw=1.5,
                label=r'$E_N^\varepsilon/E_N$')
    ax.semilogx(N_a, np.log(N_a), '--', color=GRAY, lw=1.2, label=r'$\ln N$')
    ax.set_xlabel('$N$', fontsize=12)
    ax.set_ylabel(r'$E_N^\varepsilon / E_N$', fontsize=12)
    ax.set_title(r'Нормированная погрешность, $\eta_N\equiv\log N$', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, lw=0.5)

    plt.tight_layout()
    _savefig(fig, savepath)


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('=' * 60)
    print('SW_2^2(0,1)^2, s=2, r=2')
    print('=' * 60)

    fig_cross()
    fig_func3d()
    fig_err3d()

    print('\nВычисление...')
    results = run_experiment()
    print_table(results)

    fig_errors_ab(results)
    fig_ratio_cd(results)

    if INTERACTIVE:
        print('\nВсе графики показаны. Нажмите Enter для завершения...')
        input()
    else:
        print('\nВсе PNG-файлы сохранены в текущую папку.')
