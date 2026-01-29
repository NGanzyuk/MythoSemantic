#!/usr/bin/env python3
"""
Модуль для анализа эмбеддингов из .npy и .json файлов.
Поддерживает любые модели: labse, e5, bge, mxbai и т.д.
"""

import os
import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import sys
from typing import Dict, List, Optional


class EmbeddingAnalyzer:
    """
    Класс для анализа эмбеддингов, сохранённых в формате:
    - sentence_embeddings.npy  (numpy array: [n_samples, embedding_dim])
    - sentence_metadata.json   (list of dicts, each with 'text' key)
    """

    def __init__(self, embeddings_dir: str):
        """
        Инициализация анализатора.

        :param embeddings_dir: Путь к папке с эмбеддингами (например: "embeddings/labse")
        """
        self.embeddings_dir = os.path.abspath(embeddings_dir)
        self.emb_path = os.path.join(self.embeddings_dir, "sentence_embeddings.npy")
        self.meta_path = os.path.join(self.embeddings_dir, "sentence_metadata.json")
        self.embeddings = None
        self.metadata = None
        self._load_data()

    def _load_data(self):
        """Загружает эмбеддинги и метаданные."""
        if not os.path.exists(self.embeddings_dir):
            raise FileNotFoundError(f"Папка с эмбеддингами не найдена: {self.embeddings_dir}")

        for path in [self.emb_path, self.meta_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Файл не найден: {path}")

        print(f"🔍 Загрузка эмбеддингов и метаданных из: {self.embeddings_dir}")
        try:
            self.embeddings = np.load(self.emb_path)
            with open(self.meta_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Ошибка при загрузке данных: {e}")

        if len(self.embeddings) != len(self.metadata):
            print(f"⚠️ Предупреждение: количество эмбеддингов ({len(self.embeddings)}) ≠ количеству метаданных ({len(self.metadata)})")

        print(f"✅ Успешно загружено: {len(self.embeddings)} эмбеддингов размерностью {self.embeddings.shape[1]}")

    def get_statistics(self) -> Dict:
        """Возвращает статистику по эмбеддингам."""
        if self.embeddings is None:
            raise RuntimeError("Данные не загружены. Вызовите _load_data()")

        return {
            "shape": self.embeddings.shape,
            "dtype": str(self.embeddings.dtype),
            "min": float(self.embeddings.min()),
            "max": float(self.embeddings.max()),
            "mean": float(self.embeddings.mean()),
            "std": float(self.embeddings.std()),
            "n_samples": len(self.embeddings),
        }

    def print_statistics(self):
        """Выводит статистику в консоль."""
        stats = self.get_statistics()
        print(f"\n📊 Статистика эмбеддингов ({os.path.basename(self.embeddings_dir)}):")
        print(f"   • Размер: {stats['shape']}")
        print(f"   • Тип: {stats['dtype']}")
        print(f"   • Минимум: {stats['min']:.6f}")
        print(f"   • Максимум: {stats['max']:.6f}")
        print(f"   • Среднее: {stats['mean']:.6f}")
        print(f"   • Стандартное отклонение: {stats['std']:.6f}")

    def print_first_examples(self, n: int = 5):
        """Выводит первые n примеров эмбеддингов и текстов."""
        print(f"\n📄 Первые {min(n, len(self.embeddings))} примеров:")
        for i in range(min(n, len(self.embeddings))):
            emb = self.embeddings[i]
            text = self.metadata[i].get('text', '[нет текста]')
            print(f"\n{i+1}. Текст: {text[:120]}{'...' if len(text) > 120 else ''}")
            print(f"   Эмбеддинг (первые 5 значений): {emb[:5]}...")

    def plot_tsne(self, sample_size: int = 1000, save_plot: bool = True, show_plot: bool = True):
        """
        Строит t-SNE визуализацию первых sample_size эмбеддингов.

        :param sample_size: Количество эмбеддингов для визуализации (макс. 1000 рекомендуется)
        :param save_plot: Сохранить ли изображение
        :param show_plot: Показать ли график в интерактивном режиме
        """
        sample_size = min(sample_size, len(self.embeddings))
        print(f"\n🎨 Генерация t-SNE визуализации (первые {sample_size} эмбеддингов)...")

        sample = self.embeddings[:sample_size]

        try:
            tsne = TSNE(n_components=2, perplexity=30, n_iter=300, random_state=42, n_jobs=-1)
            embedding_2d = tsne.fit_transform(sample)

            plt.figure(figsize=(12, 8))
            plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], s=3, alpha=0.7, c='blue', edgecolors='none')
            model_name = os.path.basename(self.embeddings_dir)
            plt.title(f"t-SNE визуализация эмбеддингов ({model_name})\n(первые {sample_size} предложений)", fontsize=14)
            plt.xlabel("t-SNE компонента 1")
            plt.ylabel("t-SNE компонента 2")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

            if save_plot:
                plot_path = os.path.join(self.embeddings_dir, "tsne_visualization.png")
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                print(f"✅ Визуализация сохранена в: {plot_path}")

            if show_plot:
                plt.show()

        except Exception as e:
            print(f"⚠️ Ошибка при построении t-SNE: {e}")

    def save_summary(self):
        """Сохраняет сводку анализа в текстовый файл."""
        stats = self.get_statistics()
        summary_path = os.path.join(self.embeddings_dir, "analysis_summary.txt")

        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(f"=== СВОДКА АНАЛИЗА ЭМБЕДДИНГОВ ===\n\n")
            f.write(f"Путь: {self.embeddings_dir}\n")
            f.write(f"Модель: {os.path.basename(self.embeddings_dir)}\n")
            f.write(f"Количество эмбеддингов: {stats['n_samples']}\n")
            f.write(f"Размерность: {stats['shape'][1]}\n")
            f.write(f"Тип данных: {stats['dtype']}\n")
            f.write(f"Минимум: {stats['min']:.8f}\n")
            f.write(f"Максимум: {stats['max']:.8f}\n")
            f.write(f"Среднее: {stats['mean']:.8f}\n")
            f.write(f"Стандартное отклонение: {stats['std']:.8f}\n")
            f.write(f"\nФайлы: \n")
            f.write(f"  - Эмбеддинги: {os.path.basename(self.emb_path)}\n")
            f.write(f"  - Метаданные: {os.path.basename(self.meta_path)}\n")

        print(f"📝 Сводка сохранена в: {summary_path}")

    def analyze(self, n_examples: int = 5, tsne_sample_size: int = 1000):
        """
        Полный анализ: статистика + примеры + t-SNE + сохранение отчёта.

        :param n_examples: Количество примеров для вывода
        :param tsne_sample_size: Количество эмбеддингов для t-SNE
        """
        print(f"🚀 Начало анализа эмбеддингов: {self.embeddings_dir}")
        self.print_statistics()
        self.print_first_examples(n_examples)
        self.plot_tsne(sample_size=tsne_sample_size)
        self.save_summary()
        print(f"\n🎉 Анализ завершён для {self.embeddings_dir}!")


# === Пример использования ===
if __name__ == "__main__":
    # Поддержка нескольких моделей
    base_dir = "embeddings"
    if os.path.exists(base_dir):
        models = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    else:
        models = []

    for model_dir in models:
        try:
            analyzer = EmbeddingAnalyzer(model_dir)
            analyzer.analyze(n_examples=3, tsne_sample_size=500)  # Уменьшаем размер для скорости
        except Exception as e:
            print(f"❌ Ошибка при анализе {model_dir}: {e}")
            continue