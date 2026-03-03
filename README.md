# 🏦 Credit Default Prediction — ML Course Project

> Прогнозирование невыполнения кредитных обязательств на основе данных о клиентах банка.  
> Курс: AI in Finance | ВШЭ

---

## 📋 Задача

Построить модель бинарной классификации для предсказания факта кредитного дефолта (`Credit Default`: 1 — дефолт, 0 — нет) на основе данных о заёмщиках.

**Метрика качества:** F1-score для класса 1 (дефолт)  
**Целевой порог:** F1 > 0.5

---

## 📁 Структура данных

| Файл | Описание |
|---|---|
| `course_project_train.csv` | Обучающая выборка, 7500 строк, 17 признаков |
| `course_project_test.csv` | Тестовая выборка, 2500 строк, 16 признаков |
| `predictions_final.csv` | Финальные прогнозы на тестовой выборке |
| `HW_regression_Ilia_Zabegaev_final.ipynb` | Ноутбук с полным решением |

### Признаки датасета

| Признак | Описание |
|---|---|
| `Home Ownership` | Тип жилья (аренда / ипотека / собственность) |
| `Annual Income` | Годовой доход заёмщика |
| `Years in current job` | Стаж на текущем месте работы |
| `Tax Liens` | Налоговые обременения |
| `Number of Open Accounts` | Количество открытых счетов |
| `Years of Credit History` | Длина кредитной истории (лет) |
| `Maximum Open Credit` | Максимальная сумма открытой кредитной линии |
| `Number of Credit Problems` | Количество кредитных проблем |
| `Months since last delinquent` | Месяцев с последней просрочки |
| `Bankruptcies` | Количество банкротств |
| `Purpose` | Цель кредита |
| `Term` | Срок кредита (Short / Long Term) |
| `Current Loan Amount` | Текущая сумма кредита |
| `Current Credit Balance` | Текущий кредитный баланс |
| `Monthly Debt` | Ежемесячный долг |
| `Credit Score` | Кредитный рейтинг |
| `Credit Default` | **Целевая переменная** (1 — дефолт, 0 — нет) |

---

## 🔬 Пайплайн разработки модели

### Шаг 1 — EDA (Разведочный анализ данных)

**Ключевые находки:**

- Дисбаланс классов: 71.8% — нет дефолта, 28.2% — дефолт (~1:2.5)
- Пропуски в критичных признаках: `Months since last delinquent` — 54.4%, `Annual Income` и `Credit Score` — по 20.8%
- `Annual Income` и `Credit Score` пропущены у одних и тех же 1557 строк
- **Аномалии:**
  - `Current Loan Amount` = 99,999,999 — явная заглушка (870 строк, 11.6%)
  - `Credit Score` max = 7510 — часть значений ошибочно умножена на 10
  - `Home Ownership`: категория `Have Mortgage` дублирует `Home Mortgage` (12 строк)

### Шаг 2 — Preprocessing (Предобработка данных)

| Проблема | Решение |
|---|---|
| `Credit Score` > 850 | Делим на 10 → диапазон 585–751 |
| `Current Loan Amount` = 99999999 | Заменяем на NaN → медиана (265,826) |
| `Annual Income`, `Credit Score` — 20% пропусков | Заполнение медианой по train |
| `Bankruptcies`, `Years in current job` | Заполнение медианой |
| `Months since last delinquent` — 54% пропусков | Два признака: `Has_Delinquent` (флаг) + `Months` = 0 |
| `Have Mortgage` | Объединяем с `Home Mortgage` |
| `Years in current job` — строки | Парсинг в числа (0–10) |
| `Term` | Бинаризация: Long Term = 1 |
| `Home Ownership`, `Purpose` | Label Encoding (fit только на train) |
| Все числовые признаки | StandardScaler (fit только на train) |

> **Важное решение по `Months since last delinquent`:** пропуск означает отсутствие просрочек, а не неизвестное значение. Поэтому создан отдельный бинарный признак `Has_Delinquent`, а пропуски заполнены 0, а не медианой или -1.

### Шаг 3 — Feature Engineering и отбор признаков

**Топ признаков по корреляции с целевой переменной:**

| Признак | Корреляция | Интерпретация |
|---|---|---|
| `Term` | +0.181 | Долгосрочные кредиты → чаще дефолт |
| `Credit Score` | -0.170 | Ниже рейтинг → выше риск |
| `Annual Income` | -0.095 | Ниже доход → выше риск |
| `Current Loan Amount` | +0.082 | Больше сумма → выше риск |
| `Home Ownership` | +0.065 | Тип жилья влияет на риск |

**Отбор признаков (ANOVA F-test):** статистически значимых признаков — 7 (p-value < 0.05). При моделировании использовались все 17 признаков.

**Созданные FE-признаки:**

| Признак | Формула | Смысл |
|---|---|---|
| `Debt to Income` | Monthly Debt / (Annual Income / 12) | Долговая нагрузка |
| `Credit Utilization` | Current Credit Balance / Maximum Open Credit | Использование лимита |
| `Loan to Income` | Current Loan Amount / Annual Income | Кредит к доходу |
| `Has Delinquent` | notna(Months since last delinquent) | Был ли факт просрочки |
| `Has Credit Problems` | Number of Credit Problems > 0 | Есть ли кредитные проблемы |

### Шаг 4 — Моделирование

**Балансировка классов:** `class_weight='balanced'` во всех моделях  
**Валидация:** Stratified K-Fold (5 фолдов) + hold-out выборка 20%

---

## 📊 Результаты всех экспериментов

| № | Модель | F1-score |
|---|---|---|
| 1 | Baseline LogReg — 7 признаков (sklearn) | 0.4699 |
| 2 | **Tuned LogReg — все признаки (sklearn)** | **0.4875 ◄ ЛУЧШАЯ** |
| 3 | Custom LogReg — самописная | 0.4828 |
| 4 | LogReg + Feature Engineering | 0.4786 |
| 5 | Polynomial LogReg degree=2 | 0.4838 |
| 6 | Random Forest | 0.4864 |

**Лучшие гиперпараметры:** `C=10, solver='liblinear', class_weight='balanced'`  
**Финальный прогноз:** Tuned LogReg на 17 признаках

---

## ⭐ Самописная логистическая регрессия

Реализован алгоритм градиентного спуска с нуля:

```python
class LogisticRegressionCustom:
    def __init__(self, lr=0.1, n_iter=1000):
        ...
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    def fit(self, X, y):
        # Градиентный спуск + Binary Cross-Entropy Loss
        ...
    def predict(self, X, threshold=0.5):
        ...
```

**Сравнение Custom vs sklearn:**

| | sklearn LogReg | Custom LogReg |
|---|---|---|
| F1-score | **0.4875** | 0.4828 |
| Реализация | Из коробки | Градиентный спуск вручную |
| Порог классификации | 0.5 | 0.22 (подобран) |

Разница в F1 между реализациями составила **0.0047** — модели дают практически идентичное качество.

---

## 💡 Ключевые выводы

1. **Логистическая регрессия достигла потолка ~0.487.** Данные имеют нелинейную структуру зависимостей, которую линейная модель не улавливает. Все варианты (Feature Engineering, Polynomial Features, разные наборы признаков) дали схожий результат.

2. **Самые важные признаки** — `Term` и `Credit Score`. Долгосрочные кредиты с низким кредитным рейтингом — главный индикатор риска дефолта.

3. **Пропуски несут смысл.** 54% пропусков в `Months since last delinquent` — это не отсутствие данных, а отсутствие просрочек. Правильная обработка этого признака важна для качества модели.

4. **Дисбаланс классов (1:2.5)** требует явной балансировки — без `class_weight='balanced'` модель игнорирует класс 1.

5. **Для достижения F1 > 0.5** на этих данных необходимы нелинейные модели: Random Forest, Gradient Boosting (XGBoost, LightGBM, CatBoost).

---

## 🛠️ Технологии

- Python 3.12
- pandas, numpy
- scikit-learn
- matplotlib, seaborn
- Google Colab

---

## 🚀 Воспроизведение результатов

```bash
# 1. Клонировать репозиторий
git clone https://github.com/dominator666/test-repo/

# 2. Открыть ноутбук в Google Colab
# HW_regression_Ilia_Zabegaev_final.ipynb

# 3. Загрузить данные
# course_project_train.csv
# course_project_test.csv

# 4. Runtime → Restart and run all
```

---

*Автор: Илья Забегаев | ВШЭ, курс AI in Finance*
