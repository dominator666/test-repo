# 🏦 Credit Default Prediction — ML Course Project

> Прогнозирование невыполнения кредитных обязательств на основе данных о клиентах банка.  
> Курс: AI in Finance | ВШЭ

---

## 📋 Задача

Построить модель бинарной классификации для предсказания факта кредитного дефолта (`Credit Default`: 1 — дефолт, 0 — нет) на основе данных о заёмщиках.

**Метрика качества:** F1-score для класса 1 (дефолт)  
**Целевой порог:** F1 > 0.5

---

## 📁 Структура репозитория

```
├── HW_regression_Ilia_Zabegaev_final.ipynb  		  # Ноутбук с решением
├── course_project_train.csv                              # Обучающая выборка
├── course_project_test.csv                               # Тестовая выборка
├── predictions_final.csv                                 # Финальные прогнозы
└── README.md
```

---

## 📊 Описание данных

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

**Размер данных:** train — 7500 строк, test — 2500 строк  
**Баланс классов:** 0 (71.8%) / 1 (28.2%) — умеренный дисбаланс

---

## 🔬 ML-пайплайн

### Шаг 1 — EDA (Разведочный анализ данных)

Ключевые находки:
- Пропуски: `Months since last delinquent` — 54%, `Annual Income` и `Credit Score` — по 20.8%
- `Current Loan Amount` = 99,999,999 — заглушка в 870 строках (11.6%)
- `Credit Score` max = 7510 — часть значений ошибочно умножена на 10
- `Have Mortgage` дублирует `Home Mortgage` (12 строк)
- Топ-2 признака по корреляции с дефолтом: `Term` (+0.181), `Credit Score` (-0.170)

### Шаг 2 — Preprocessing (Предобработка данных)

| Проблема | Решение |
|---|---|
| `Credit Score` > 850 | Делим на 10 → диапазон 585–751 |
| `Current Loan Amount` = 99999999 | Заменяем на NaN → медиана по train |
| `Annual Income`, `Credit Score` — 20% пропусков | Медиана по train |
| `Bankruptcies`, `Years in current job` | Медиана по train |
| `Months since last delinquent` — 54% пропусков | **Два признака:** `Has_Delinquent` (флаг) + `Months` = 0 |
| `Have Mortgage` | Объединяем с `Home Mortgage` |
| `Years in current job` — строки ("10+ years") | Парсинг в числа (0–10) |
| `Term` | Бинаризация: Long Term = 1 |
| `Home Ownership`, `Purpose` | Label Encoding (fit только на train) |
| Все числовые признаки | StandardScaler (fit только на train) |

> **Ключевое решение:** 54% пропусков в `Months since last delinquent` — это не отсутствие данных, а отсутствие просрочек. Создан отдельный бинарный признак `Has_Delinquent`, пропуски заполнены 0.

### Шаг 3 — Feature Engineering

Создано 5 новых признаков на основе финансовой логики:

| Признак | Формула | Смысл |
|---|---|---|
| `Debt to Income` | Monthly Debt / (Annual Income / 12) | Долговая нагрузка |
| `Credit Utilization` | Current Credit Balance / Maximum Open Credit | Использование кредитного лимита |
| `Loan to Income` | Current Loan Amount / Annual Income | Кредит относительно дохода |
| `Has Delinquent` | notna(Months since last delinquent) | Был ли факт просрочки |
| `Has Credit Problems` | Number of Credit Problems > 0 | Есть ли кредитные проблемы |

Итого признаков после FE: **22** (17 исходных + 5 новых)

### Шаг 4 — Моделирование

**Стратегия:**
- Балансировка классов: `class_weight='balanced'` во всех моделях
- Валидация: Stratified K-Fold (5 фолдов) + hold-out 20%
- Подбор гиперпараметров: GridSearchCV
- Подбор порога классификации через `predict_proba()` для максимизации F1

---

## 📈 Результаты всех экспериментов

| № | Модель | Признаки | Гиперпараметры | F1-score |
|---|---|---|---|---|
| 1 | Baseline LogReg (sklearn) | 7 (F-test) | default | 0.4699 |
| **2** | **Tuned LogReg (sklearn)** | **17 (все)** | **C=100, liblinear** | **0.4903 ◄** |
| 3 | Custom LogReg (самописная) | 17 (все) | lr=0.1, n_iter=2000, порог=0.20 | 0.4828 |
| 4 | LogReg + Feature Engineering | 22 | C=1, lbfgs | 0.4786 |
| 5 | Polynomial LogReg degree=2 | 17 | C=1, liblinear | 0.4838 |
| 6 | Random Forest | 10 (FE subset) | n_est=300, depth=8 | 0.4804 |

**Лучшая модель:** Tuned LogReg — все признаки (sklearn)  
**Лучший F1:** 0.4903  
**Оптимальный порог:** 0.50

**Финальный прогноз на тесте:**
```
0 (нет дефолта): 1464 (58.6%)
1 (дефолт):      1036 (41.4%)
```

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
        # Binary Cross-Entropy Loss + градиентный спуск
        ...
    def predict_proba(self, X):
        ...
    def predict(self, X, threshold=0.5):
        ...
```

**Сравнение Custom vs sklearn на всех 17 признаках:**

| | sklearn LogReg | Custom LogReg |
|---|---|---|
| F1-score | **0.4903** | 0.4828 |
| Оптимальный порог | 0.50 | 0.20 |
| Параметры | C=100, liblinear | lr=0.1, n_iter=2000 |
| Реализация | Из коробки | Градиентный спуск вручную |

Разница в F1: **0.0075** — результаты практически идентичны, что подтверждает корректность самописного алгоритма.

---

## 💡 Ключевые выводы

1. **Логистическая регрессия достигла потолка ~0.49.** Все эксперименты — 7 признаков, 17, 22, Polynomial Features, Feature Engineering — дали схожий результат. Данные имеют нелинейную структуру зависимостей.

2. **Самые важные признаки** — `Term` (+0.181) и `Credit Score` (-0.170). Долгосрочные кредиты с низким кредитным рейтингом — главный индикатор риска дефолта.

3. **Пропуски несут смысл.** 54% пропусков в `Months since last delinquent` означают отсутствие просрочек, а не неизвестное значение. Правильная обработка через `Has_Delinquent` важна для качества модели.

4. **Дисбаланс классов (1:2.5)** требует явной балансировки через `class_weight='balanced'` — без неё модель практически игнорирует класс дефолта.

5. **Подбор порога** через `predict_proba()` даёт гибкость — можно управлять балансом precision/recall. Для данной модели оптимальный порог совпал с 0.5, что говорит о хорошей калибровке модели.

6. **Для достижения F1 > 0.5** на этих данных нужны нелинейные модели: XGBoost, LightGBM, CatBoost.

---

## 🛠️ Технологии

- Python 3.12
- pandas, numpy
- scikit-learn (LogisticRegression, GridSearchCV, StratifiedKFold, StandardScaler, SelectKBest, Pipeline, PolynomialFeatures, RandomForestClassifier)
- matplotlib, seaborn
- Google Colab

---

## 🚀 Воспроизведение результатов

```bash
# 1. Клонировать репозиторий
git clone https://https://github.com/dominator666/test-repo

# 2. Открыть ноутбук в Google Colab:
# HW_regression_Ilia_Zabegaev_final.ipynb

# 3. Загрузить файлы данных через Files → Upload:
# course_project_train.csv
# course_project_test.csv

# 4. Runtime → Restart and run all
```

Прогнозы сохранятся в `predictions_final.csv` (2500 строк, значения 0/1).

---

*Автор: Илья Забегаев | ВШЭ, курс AI in Finance*
