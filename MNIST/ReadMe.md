
# Сравнение сверточных нейросетей на датасете MNIST

## Описание

Цель данного проекта — сравнить различные архитектуры сверточных нейросетей на датасете рукописных цифр MNIST. В данном датасете представлены черно-белые изображения цифр от 0 до 9. Мы будем экспериментировать с различными параметрами архитектур нейросетей, такими как количество сверточных и линейных слоев, размер ядер свертки, а также производить обучение моделей с использованием оптимизатора Adam и функции потерь CrossEntropyLoss.

## Зачем это нужно?

Сравнение различных конфигураций сверточных нейросетей позволяет выбрать оптимальную архитектуру для классификации изображений. Мы пытаемся достичь точности на валидационном наборе не менее 98%. MNIST — это простой и хорошо изученный набор данных, который идеально подходит для тестирования различных архитектур нейросетей.

## Используемые технологии

- **PyTorch**: для реализации нейросетей, обучения и оценки.
- **Torchvision**: для загрузки и обработки данных MNIST.
- **Matplotlib**: для визуализации результатов обучения.
- **NumPy**: для обработки данных и массивов.

## Шаги эксперимента

1. **Загрузка и подготовка данных**
   Мы используем датасет MNIST из `torchvision.datasets` и преобразуем его в тензоры для последующей подачи в нейросети.



2. **Визуализация данных**
   Для лучшего понимания, что содержится в датасете, выводим несколько примеров изображений.



3. **Модели нейросетей**
   Для первого эксперимента мы создаем 5 различных моделей с разным количеством сверточных и линейных слоев. Модели включают как минимум один сверточный слой и один линейный слой.



4. **Обучение моделей**
   Для каждой модели мы используем функцию потерь `nn.CrossEntropyLoss` и оптимизатор `torch.optim.Adam`.


5. **Оценка результатов**
   Мы строим графики потерь и точности для всех моделей, чтобы сравнить их производительность.


6. **Выводы по результатам**
   Мы анализируем, как количество слоев и размер ядер свертки влияют на качество модели и время обучения. Например, более сложные модели с большим количеством слоев могут показывать лучшие результаты на тренировочных данных, но не всегда улучшают точность на тестовых данных.