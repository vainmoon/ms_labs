import numpy as np
from scipy.stats import f
import matplotlib.pyplot as plt


# Функция для проведения теста Фишера
def fisher_test(sample1, sample2, alpha):
    # Вычисляем несмещенные оценки дисперсий
    s1_squared = np.var(sample1, ddof=1)
    s2_squared = np.var(sample2, ddof=1)

    # Вычисляем статистику критерия Фишера
    F = s1_squared / s2_squared if s1_squared >= s2_squared else s2_squared / s1_squared

    # Определяем степени свободы
    k1 = len(sample1) - 1
    k2 = len(sample2) - 1

    # Вычисляем критическое значение
    crit_value = f.ppf(1 - alpha / 2, k1, k2)

    # Определяем результат
    if F < crit_value:
        result = "Принять H0"
    else:
        result = "Отвергнуть H0"

    return s1_squared, s2_squared, F, crit_value, result


# Генерация выборок
def generate_normal_samples(mean1, mean2, std_dev1, std_dev2, size1, size2):
    sample1 = np.random.normal(mean1, std_dev1, size1)
    sample2 = np.random.normal(mean2, std_dev2, size2)
    return sample1, sample2


# Заданные параметры
mean1 = 0
mean2 = 0
std_dev1 = 1
std_dev2 = 1
alpha = 0.05

# Создание нормального распределения мощностью 100
population = np.random.normal(mean1, std_dev1, 100)

# Случай 1: выборки мощностью 20 и 40
sample1_case1, sample2_case1 = generate_normal_samples(mean1, mean2, std_dev1, std_dev2, 20, 40)
s1_squared_case1, s2_squared_case1, F_case1, crit_value_case1, result_case1 = fisher_test(sample1_case1, sample2_case1,
                                                                                          alpha)

# Случай 2: выборки мощностью 20 и 100
sample1_case2, sample2_case2 = generate_normal_samples(mean1, mean2, std_dev1, std_dev2, 20, 100)
s1_squared_case2, s2_squared_case2, F_case2, crit_value_case2, result_case2 = fisher_test(sample1_case2, sample2_case2,
                                                                                          alpha)

# Отображение результатов
print("Результаты исследования:")
print("\nСлучай 1: выборки мощностью 20 и 40")
print("Оценка дисперсии для первой выборки:", s1_squared_case1)
print("Оценка дисперсии для второй выборки:", s2_squared_case1)
print("Статистика критерия Фишера:", F_case1)
print("Критическое значение:", crit_value_case1)
print("Результат теста:", result_case1)

print("\nСлучай 2: выборки мощностью 20 и 100")
print("Оценка дисперсии для первой выборки:", s1_squared_case2)
print("Оценка дисперсии для второй выборки:", s2_squared_case2)
print("Статистика критерия Фишера:", F_case2)
print("Критическое значение:", crit_value_case2)
print("Результат теста:", result_case2)

# Визуализация выборок и нормального распределения
plt.figure(figsize=(10, 6))
plt.hist(population, bins=20, alpha=0.7, label='Нормальное распределение N=100', edgecolor='black')
plt.hist(sample1_case1, bins=10, alpha=0.5, label='Выборка 1 мощностью 20 (случай 1)', edgecolor='black')
plt.hist(sample2_case1, bins=10, alpha=0.5, label='Выборка 2 мощностью 40 (случай 1)', edgecolor='black')
plt.title('Распределение выборок для случая 1')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.hist(population, bins=20, alpha=0.7, label='Нормальное распределение N=100', edgecolor='black')
plt.hist(sample1_case2, bins=10, alpha=0.5, label='Выборка 1 мощностью 20 (случай 2)', edgecolor='black')
plt.hist(sample2_case2, bins=10, alpha=0.5, label='Выборка 2 мощностью 100 (случай 2)', edgecolor='black')
plt.title('Распределение выборок для случая 2')
plt.legend()
plt.show()