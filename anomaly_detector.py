"""
CryptoFlow · 加密流量异常检测器
================================

使用孤立森林 (IsolationForest) 检测加密流量中的异常行为。
支持多种异常检测策略：
1. 全局异常检测：基于所有特征的异常评分
2. 局部异常检测：基于特征子集的异常检测
3. 实时异常检测：对新流量的实时判断
"""

import numpy as np
import pickle
import os
from typing import List, Dict, Tuple, Optional

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor


class AnomalyDetector:
    """
    加密流量异常检测器
    使用孤立森林算法检测异常流量
    """

    # 异常类型定义
    ANOMALY_TYPES = {
        'data_exfiltration': '数据泄露 - 大量上传加密数据',
        'c2_tunnel': 'C2通信 - 规律性心跳与长连接',
        'dns_tunnel': 'DNS隧道 - 异常的DNS查询流量',
        'unusual_tls': '异常TLS握手 - 不常见的TLS参数组合',
        'high_entropy': '高熵异常 - 非典型加密流量分布',
        'timing_anomaly': '时序异常 - 异常的包间隔模式',
    }

    def __init__(self, contamination: float = 0.1, model_path: str = None):
        """
        初始化异常检测器

        Args:
            contamination: 预期异常比例
            model_path: 预训练模型路径（可选）
        """
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()
        self.threshold = None
        self.is_trained = False
        self.feature_names = None

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def train(self, X: np.ndarray, feature_names: List[str] = None) -> Dict:
        """
        训练异常检测模型

        Args:
            X: 特征矩阵 (n_samples, n_features)
            feature_names: 特征名称列表

        Returns:
            训练结果字典
        """
        if feature_names:
            self.feature_names = feature_names

        # 标准化
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        # 训练孤立森林
        self.model = IsolationForest(
            n_estimators=200,
            max_samples='auto',
            contamination=self.contamination,
            max_features=1.0,
            bootstrap=False,
            n_jobs=-1,
            random_state=42,
            verbose=0,
        )
        self.model.fit(X_scaled)

        # 计算异常分数
        anomaly_scores = self.model.score_samples(X_scaled)
        # 决策函数（分数越低越异常）
        decision_scores = self.model.decision_function(X_scaled)

        # 设置阈值（基于训练数据）
        self.threshold = np.percentile(anomaly_scores, self.contamination * 100)

        # 预测标签
        y_pred = self.model.predict(X_scaled)
        n_anomalies = np.sum(y_pred == -1)

        self.is_trained = True

        results = {
            'n_samples': len(X),
            'n_features': X.shape[1],
            'contamination': self.contamination,
            'n_anomalies': n_anomalies,
            'anomaly_ratio': n_anomalies / len(X),
            'threshold': float(self.threshold),
            'mean_score': float(np.mean(anomaly_scores)),
            'std_score': float(np.std(anomaly_scores)),
            'min_score': float(np.min(anomaly_scores)),
            'max_score': float(np.max(anomaly_scores)),
        }

        return results

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        检测异常

        Args:
            X: 特征矩阵

        Returns:
            (predictions, scores)
            predictions: 1为正常, -1为异常
            scores: 异常分数（越低越异常）
        """
        if not self.is_trained or self.model is None:
            raise ValueError("模型尚未训练，请先调用 train()")

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        scores = self.model.decision_function(X_scaled)

        return predictions, scores

    def predict_single(self, features: np.ndarray) -> Dict:
        """
        检测单个样本

        Args:
            features: 特征向量

        Returns:
            {
                'is_anomaly': bool,
                'score': float,
                'confidence': float,
                'risk_level': str
            }
        """
        X = features.reshape(1, -1)
        pred, scores = self.predict(X)

        # 归一化异常分数到 [0, 1]，越高越异常
        anomaly_score = 1.0 / (1.0 + np.exp(-scores[0] * 5))  # sigmoid
        anomaly_score = 1.0 - anomaly_score  # 反转：高值表示异常

        # 置信度
        confidence = abs(scores[0]) / (abs(scores[0]) + 1.0)

        # 风险等级
        if anomaly_score > 0.8:
            risk_level = 'high'
        elif anomaly_score > 0.5:
            risk_level = 'medium'
        elif anomaly_score > 0.3:
            risk_level = 'low'
        else:
            risk_level = 'normal'

        return {
            'is_anomaly': bool(pred[0] == -1),
            'score': float(anomaly_score),
            'raw_score': float(scores[0]),
            'confidence': float(confidence),
            'risk_level': risk_level,
        }

    def analyze_anomaly_features(self, features: np.ndarray,
                                 top_n: int = 5) -> List[Dict]:
        """
        分析导致异常的关键特征

        Args:
            features: 特征向量
            top_n: 返回前N个异常特征

        Returns:
            [{'name': str, 'value': float, 'contribution': float}, ...]
        """
        if self.feature_names is None:
            self.feature_names = [f'feature_{i}' for i in range(len(features))]

        # 使用特征值与均值的偏差作为异常贡献度
        feature_contributions = []
        for i, (name, value) in enumerate(zip(self.feature_names, features)):
            # 假设特征值在 [0, 1] 范围内
            deviation = abs(value - 0.5) * 2  # 归一化偏差
            feature_contributions.append({
                'name': name,
                'value': float(value),
                'contribution': float(deviation),
            })

        # 按贡献度排序
        feature_contributions.sort(key=lambda x: x['contribution'], reverse=True)

        return feature_contributions[:top_n]

    def get_anomaly_explanation(self, features: np.ndarray) -> str:
        """
        生成异常解释

        Args:
            features: 特征向量

        Returns:
            可读的异常解释文本
        """
        result = self.predict_single(features)
        if not result['is_anomaly']:
            return "流量正常，未检测到异常行为。"

        # 分析异常特征
        top_features = self.analyze_anomaly_features(features, top_n=5)

        explanation_parts = [
            f"⚠️ 检测到异常流量 (风险等级: {result['risk_level']})",
            f"异常分数: {result['score']:.4f}",
            f"置信度: {result['confidence']:.4f}",
            "",
            "关键异常特征:",
        ]

        for i, feat in enumerate(top_features[:5], 1):
            explanation_parts.append(
                f"  {i}. {feat['name']}: {feat['value']:.4f} "
                f"(异常贡献: {feat['contribution']:.2%})"
            )

        # 匹配已知异常模式
        matched_patterns = self._match_anomaly_patterns(features)
        if matched_patterns:
            explanation_parts.append("")
            explanation_parts.append("可能的异常类型:")
            for pattern, desc in matched_patterns:
                explanation_parts.append(f"  • {desc}")

        return '\n'.join(explanation_parts)

    def _match_anomaly_patterns(self, features: np.ndarray) -> List[Tuple[str, str]]:
        """匹配已知异常模式"""
        if self.feature_names is None:
            return []

        patterns = []

        # 创建特征名到值的映射
        feat_dict = {}
        for name, value in zip(self.feature_names, features):
            # 提取特征类别的短名称用于匹配
            parts = name.split('_', 1)
            if len(parts) == 2:
                feat_dict[name] = value

        # 检测数据泄露特征
        if ('tls_up_ratio' in feat_dict and feat_dict['tls_up_ratio'] > 0.7):
            patterns.append(('data_exfiltration',
                             self.ANOMALY_TYPES['data_exfiltration']))

        # 检测C2隧道特征
        if ('flow_fwd_bwd_ratio' in feat_dict and
            abs(feat_dict['flow_fwd_bwd_ratio'] - 1.0) < 0.1):
            patterns.append(('c2_tunnel', self.ANOMALY_TYPES['c2_tunnel']))

        # 检测DNS隧道
        if ('proto_dns_query_present' in feat_dict and
            feat_dict['proto_dns_query_present'] > 0.5 and
            'len_payload_len_mean' in feat_dict and
            feat_dict['len_payload_len_mean'] > 0.3):
            patterns.append(('dns_tunnel', self.ANOMALY_TYPES['dns_tunnel']))

        # 检测异常TLS握手
        if ('tls_has_tls' in feat_dict and feat_dict['tls_has_tls'] > 0.5 and
            'tls_cipher_count' in feat_dict and feat_dict['tls_cipher_count'] < 0.2):
            patterns.append(('unusual_tls', self.ANOMALY_TYPES['unusual_tls']))

        # 检测高熵异常
        if ('entropy_high_entropy_ratio' in feat_dict and
            feat_dict['entropy_high_entropy_ratio'] > 0.8):
            patterns.append(('high_entropy', self.ANOMALY_TYPES['high_entropy']))

        # 检测时序异常
        if ('iat_pkt_rate' in feat_dict and feat_dict['iat_pkt_rate'] < 0.1):
            patterns.append(('timing_anomaly', self.ANOMALY_TYPES['timing_anomaly']))

        return patterns

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
            'threshold': self.threshold,
            'contamination': self.contamination,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained,
        }

        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"💾 异常检测模型已保存到: {path}")

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
        self.threshold = model_data['threshold']
        self.contamination = model_data['contamination']
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']

        print(f"📂 异常检测模型已加载: {path}")

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        if not self.is_trained:
            return {'status': '未训练'}

        return {
            'status': '已训练',
            'model_type': 'IsolationForest',
            'contamination': self.contamination,
            'threshold': float(self.threshold) if self.threshold else None,
            'n_estimators': self.model.n_estimators,
            'n_features': self.model.n_features_in_,
        }


class LocalAnomalyDetector:
    """
    局部异常因子检测器
    使用LOF算法检测局部异常模式
    """

    def __init__(self, n_neighbors: int = 20, contamination: float = 0.1):
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False

    def train(self, X: np.ndarray):
        """训练LOF模型"""
        X_scaled = self.scaler.fit_transform(X)

        self.model = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=True,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """检测异常"""
        if not self.is_trained:
            raise ValueError("模型尚未训练")

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        scores = self.model.decision_function(X_scaled)

        return predictions, scores


def train_anomaly_detector(X: np.ndarray,
                           feature_names: List[str] = None,
                           contamination: float = 0.1,
                           save_path: str = None) -> Tuple[AnomalyDetector, Dict]:
    """
    便捷函数：训练异常检测器

    Args:
        X: 特征矩阵
        feature_names: 特征名称
        contamination: 预期异常比例
        save_path: 模型保存路径

    Returns:
        (detector, results)
    """
    detector = AnomalyDetector(contamination=contamination)
    results = detector.train(X, feature_names)

    if save_path:
        detector.save_model(save_path)

    return detector, results


if __name__ == '__main__':
    # 测试异常检测器
    from data_generator import generate_training_data, TrafficSimulator
    from feature_extractor import FlowFeatureExtractor

    print("=" * 60)
    print("CryptoFlow · 异常检测器测试")
    print("=" * 60)

    # 生成训练数据（包含异常）
    print("\n📊 生成训练数据...")
    X, y_class, y_anomaly = generate_training_data(n_samples=500, anomaly_ratio=0.15)

    # 获取特征名称
    extractor = FlowFeatureExtractor()
    feature_names = extractor.get_feature_names()

    print(f"  样本数: {X.shape[0]}")
    print(f"  特征数: {X.shape[1]}")
    print(f"  真实异常比例: {np.mean(y_anomaly) * 100:.1f}%")

    # 训练异常检测器
    print("\n🔧 训练异常检测器...")
    detector, results = train_anomaly_detector(
        X, feature_names, contamination=0.15
    )

    # 输出结果
    print(f"\n📈 训练结果:")
    print(f"  检测到的异常数: {results['n_anomalies']}")
    print(f"  异常比例: {results['anomaly_ratio'] * 100:.2f}%")
    print(f"  阈值: {results['threshold']:.4f}")

    # 测试异常检测
    simulator = TrafficSimulator()
    print("\n🔮 测试正常流量检测:")
    normal_flow, _ = simulator.generate_flow(app_type='browsing')
    normal_features = extractor.extract_from_raw(normal_flow)
    normal_result = detector.predict_single(normal_features)
    print(f"  是否异常: {normal_result['is_anomaly']}")
    print(f"  异常分数: {normal_result['score']:.4f}")
    print(f"  风险等级: {normal_result['risk_level']}")

    print("\n🔮 测试异常流量检测:")
    anomaly_flow, _ = simulator.generate_flow(is_anomaly=True, anomaly_type='data_exfiltration')
    anomaly_features = extractor.extract_from_raw(anomaly_flow)
    anomaly_result = detector.predict_single(anomaly_features)
    print(f"  是否异常: {anomaly_result['is_anomaly']}")
    print(f"  异常分数: {anomaly_result['score']:.4f}")
    print(f"  风险等级: {anomaly_result['risk_level']}")

    # 异常解释
    print("\n📝 异常解释:")
    explanation = detector.get_anomaly_explanation(anomaly_features)
    print(explanation)

    print("\n✅ 异常检测器测试完成")