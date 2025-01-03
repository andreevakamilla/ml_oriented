# Titanic Survival Prediction: Анализ и Классификация

В этом проекте мы анализируем данные пассажиров Титаника, чтобы предсказать вероятность выживания. Для этого используется набор данных Titanic с платформы Kaggle. Проект сочетает исследовательский анализ данных и машинное обучение.

## Что мы делаем?

1. **Анализ данных:** Исследуем, как различные признаки (пол, возраст, класс билета и т.д.) влияют на выживаемость, с помощью визуализации (`matplotlib`, `seaborn`).
2. **Классификация:** Реализуем простой классификатор методом ближайших соседей (kNN) и логические правила в виде деревьев решений для предсказания выживаемости.
3. **Оценка:** Сравниваем модели по точности, чтобы выявить наиболее эффективный подход.

## Что мы используем?

- **`numpy` и `pandas`:** Для работы с данными.
- **`matplotlib` и `seaborn`:** Для визуализации зависимостей.
- **Метод kNN:** Для предсказаний на основе ближайших соседей.
- **Простые правила:** Деревья решений для интерпретируемых предсказаний.

## Почему это работает?

- **kNN:** Позволяет находить похожих пассажиров и предсказывать их выживаемость на основе данных обучающей выборки.
- **Деревья решений:** Простые и понятные логические правила эффективно учитывают ключевые признаки, такие как пол и возраст. 

Проект демонстрирует, как базовые алгоритмы машинного обучения могут применяться для реальных задач анализа данных.
