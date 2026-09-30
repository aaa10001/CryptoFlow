"""
CryptoFlow · 加密流量分类器
=============================

使用随机森林对加密流量进行分类，识别应用类型。
支持模型训练、评估、预测和特征重要性分析。

应用类型：
- browsing (浏览)
- video (视频)
- chat (聊天)
- file_transfer (文件传输)
- email (邮件)
- streaming (流媒体)
- voip
- p2p
- dns
"""

import numpy as np
import pickle
import os
from typing import List, Dict, Tuple, Optional
from collections import Counter

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler, LabelEncoder


class TrafficClassifier:
    """
    加密流量分类器
    使用随机森林模型对加密流量进行应用类型识别
    """

    # 应用类型名称
    APP_NAMES = [
        'browsing',      # 0: 网页浏览
        'video',         # 1: 视频
        'chat',          # 2: 即时通讯
        'file_transfer', # 3: 文件传输
        'email',         # 4: 电子邮件
        'streaming',     # 5: 流媒体
        'voip',          # 6: 网络电话
        'p2p',           # 7: 点对点
        'dns',           # 8: DNS查询
    ]

    # 异常类型名称
    ANOMALY_NAMES = [
        'anomaly_data_exfiltration',  # 数据泄露
        'anomaly_c2_tunnel',          # C2隧道
        'anomaly_dns_tunnel',         # DNS隧道
    ]

    def __init__(self, model_path: str = None):
        """
        初始化分类器

        Args:
            model_path: 预训练模型路径（可选）
        """
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_importance = None
        self.feature_names = None
        self.is_trained = False

        # 所有可能的标签
        self.all_labels = self.APP_NAMES + self.ANOMALY_NAMES
        self.label_encoder.fit(self.all_labels)

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def train(self, X: np.ndarray, y: np.ndarray,
              feature_names: List[str] = None,
              test_size: float = 0.2,
              optimize: bool = False) -> Dict:
        """
        训练分类模型

        Args:
            X: 特征矩阵 (n_samples, n_features)
            y: 标签 (n_samples,)
            feature_names: 特征名称列表
            test_size: 测试集比例
            optimize: 是否进行超参数优化

        Returns:
            训练结果字典
        """
        if feature_names:
            self.feature_names = feature_names

        # 编码标签
        y_encoded = self.label_encoder.transform(y)

        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=42, stratify=y_encoded
        )

        # 标准化
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # 超参数优化（可选）
        if optimize:
            print("🔍 正在进行超参数优化...")
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
            }
            grid_search = GridSearchCV(
                RandomForestClassifier(random_state=42, n_jobs=-1),
                param_grid,
                cv=5,
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train_scaled, y_train)
            self.model = grid_search.best_estimator_
            print(f"✅ 最佳参数: {grid_search.best_params_}")
        else:
            # 使用默认参数
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'
            )
            self.model.fit(X_train_scaled, y_train)

        # 评估
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)

        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        # 特征重要性
        self.feature_importance = self.model.feature_importances_

        # 交叉验证
        cv_scores = cross_val_score(
            self.model, X_train_scaled, y_train, cv=5, scoring='f1_weighted'
        )

        self.is_trained = True

        # 生成分类报告
        target_names = self.label_encoder.classes_
        report = classification_report(
            y_test, y_pred,
            target_names=target_names,
            zero_division=0,
            output_dict=True
        )

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)

        results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'cv_mean': float(np.mean(cv_scores)),
            'cv_std': float(np.std(cv_scores)),
            'classification_report': report,
            'confusion_matrix': cm,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'n_features': X.shape[1],
            'n_classes': len(np.unique(y_encoded)),
        }

        return results

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测流量类型

        Args:
            X: 特征矩阵

        Returns:
            (predictions, probabilities)
        """
        if not self.is_trained or self.model is None:
            raise ValueError("模型尚未训练，请先调用 train()")

        X_scaled = self.scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        y_proba = self.model.predict_proba(X_scaled)

        # 解码标签
        predictions = self.label_encoder.inverse_transform(y_pred)

        return predictions, y_proba

    def predict_single(self, features: np.ndarray) -> Tuple[str, float]:
        """
        预测单个样本

        Args:
            features: 特征向量

        Returns:
            (prediction, confidence)
        """
        X = features.reshape(1, -1)
        pred, proba = self.predict(X)
        confidence = float(np.max(proba[0]))
        return pred[0], confidence

    def get_feature_importance(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """
        获取特征重要性排名

        Args:
            top_n: 返回前N个重要特征

        Returns:
            [(feature_name, importance), ...]
        """
        if self.feature_importance is None:
            return []

        if self.feature_names is None:
            self.feature_names = [f'feature_{i}' for i in range(len(self.feature_importance))]

        # 排序
        indices = np.argsort(self.feature_importance)[::-1]
        top_features = []
        for i in indices[:top_n]:
            top_features.append((self.feature_names[i], float(self.feature_importance[i])))

        return top_features

    def save_model(self, path: str):
        """
        保存模型到文件

        Args:
            path: 保存路径
        """
        if not self.is_trained:
            raise ValueError("模型尚未训练，无法保存")

        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_importance': self.feature_importance,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
        }

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"💾 模型已保存到: {path}")

    def load_model(self, path: str):
        """
        从文件加载模型

        Args:
            path: 模型文件路径
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {path}")

        with open(path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.label_encoder = model_data['label_encoder']
        self.feature_importance = model_data['feature_importance']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']

        print(f"📂 模型已加载: {path}")

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        if not self.is_trained:
            return {'status': '未训练'}

        info = {
            'status': '已训练',
            'model_type': type(self.model).__name__,
            'n_estimators': self.model.n_estimators,
            'max_depth': self.model.max_depth,
            'n_features': self.model.n_features_in_,
            'n_classes': len(self.model.classes_),
            'classes': list(self.label_encoder.classes_),
        }

        if self.feature_importance is not None:
            info['top_features'] = self.get_feature_importance(10)

        return info


def train_classifier(X: np.ndarray, y: np.ndarray,
                     feature_names: List[str] = None,
                     optimize: bool = False,
                     save_path: str = None) -> Tuple[TrafficClassifier, Dict]:
    """
    便捷函数：训练分类器

    Args:
        X: 特征矩阵
        y: 标签
        feature_names: 特征名称
        optimize: 是否超参数优化
        save_path: 模型保存路径

    Returns:
        (classifier, results)
    """
    classifier = TrafficClassifier()
    results = classifier.train(X, y, feature_names, optimize=optimize)

    if save_path:
        classifier.save_model(save_path)

    return classifier, results


if __name__ == '__main__':
    # 测试分类器
    from data_generator import generate_training_data
    from feature_extractor import FlowFeatureExtractor

    print("=" * 60)
    print("CryptoFlow · 分类器测试")
    print("=" * 60)

    # 生成训练数据
    print("\n📊 生成训练数据...")
    X, y_class, y_anomaly = generate_training_data(n_samples=300, anomaly_ratio=0.15)

    # 获取特征名称
    extractor = FlowFeatureExtractor()
    feature_names = extractor.get_feature_names()

    # 将数值标签转换为名称
    from data_generator import TrafficSimulator
    simulator = TrafficSimulator()
    app_types = list(simulator.APP_PROFILES.keys())
    anomaly_types = list(simulator.ANOMALY_PROFILES.keys())
    all_types = app_types + [f'anomaly_{t}' for t in anomaly_types]
    y_labels = np.array([all_types[i] for i in y_class])

    print(f"  样本数: {X.shape[0]}")
    print(f"  特征数: {X.shape[1]}")
    print(f"  类别数: {len(np.unique(y_labels))}")
    print(f"  类别分布: {dict(Counter(y_labels))}")

    # 训练分类器
    print("\n🔧 训练分类器...")
    classifier, results = train_classifier(X, y_labels, feature_names)

    # 输出结果
    print(f"\n📈 训练结果:")
    print(f"  准确率: {results['accuracy'] * 100:.2f}%")
    print(f"  精确率: {results['precision'] * 100:.2f}%")
    print(f"  召回率: {results['recall'] * 100:.2f}%")
    print(f"  F1分数: {results['f1_score'] * 100:.2f}%")
    print(f"  交叉验证: {results['cv_mean'] * 100:.2f}% ± {results['cv_std'] * 100:.2f}%")

    # 特征重要性
    print("\n🔝 特征重要性 Top 10:")
    for name, importance in classifier.get_feature_importance(10):
        print(f"  {name}: {importance * 100:.2f}%")

    # 测试预测
    print("\n🔮 测试预测:")
    test_flow, test_label = simulator.generate_flow(app_type='video')
    test_features = extractor.extract_from_raw(test_flow)
    pred, conf = classifier.predict_single(test_features)
    print(f"  真实: {test_label}")
    print(f"  预测: {pred} (置信度: {conf * 100:.1f}%)")

    print("\n✅ 分类器测试完成")