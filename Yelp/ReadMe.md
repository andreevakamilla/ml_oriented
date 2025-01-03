# Анализ данных Yelp

## Описание проекта
Данный проект посвящен анализу данных Yelp для исследования компаний в различных городах. Основные задачи:
- Определение города с наибольшим количеством компаний.
- Выявление районов с наилучшей средней оценкой компаний.
- Нахождение компаний с наивысшими отзывами.

## Используемые инструменты
- **Python**: основной язык программирования для обработки и анализа данных.
- **Pandas**: для чтения данных из CSV-файлов и обработки табличных данных.
- **Matplotlib** и **Seaborn**: для построения визуализаций, включая графики и тепловые карты.
- **Plotly**: для интерактивной визуализации, такой как отображение данных на карте.
- **Numpy**: для математических операций, включая округление координат.

## Логика работы
1. **Загрузка и предварительная обработка данных**:
   - Загружаем данные о компаниях из файла `yelp_business.csv`.
   - Считаем количество компаний в каждом городе и находим город с наибольшим числом компаний.

2. **Фильтрация данных**:
   - Убираем компании, находящиеся за пределами центра города, используя широту и долготу.

3. **Анализ оценок**:
   - Загружаем данные отзывов из файла `yelp_review.csv`.
   - Соединяем таблицу отзывов с таблицей компаний для анализа средней оценки и количества отзывов.

4. **Разбиение города на районы**:
   - Округляем координаты для создания районов ("клеток").
   - Рассчитываем средние оценки для компаний в каждом районе.

5. **Визуализация**:
   - Строим тепловую карту районов с помощью Seaborn.
   - Используем Plotly для отображения компаний и районов на интерактивной карте.

## Почему это работает
- **Проверенные библиотеки**: Pandas, Matplotlib, Seaborn и Plotly широко применяются для анализа и визуализации данных, обеспечивая надежность и гибкость.
- **Четкая структура данных**: CSV-файлы легко читаются и обрабатываются с использованием Pandas.
- **Интерактивность**: Plotly позволяет глубже исследовать данные благодаря интерактивным графикам.
- **Географический анализ**: Округление координат помогает визуализировать данные по районам и выявлять закономерности.

## Как использовать
1. Подготовьте файлы данных (`yelp_business.csv`, `yelp_review.csv`).
2. Запустите скрипт для анализа данных.
3. Изучите визуализации и выводы для дальнейшего анализа или принятия решений.

