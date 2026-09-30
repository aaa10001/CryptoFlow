"""
CryptoFlow · 加密流量模拟数据生成器
=====================================

生成模拟的加密流量数据用于训练和测试。
模拟真实网络流量的统计分布特征，包括：
- 浏览流量 (browsing)
- 视频流量 (video)
- 聊天流量 (chat)
- 文件传输 (file_transfer)
- 邮件流量 (email)
- 流媒体 (streaming)
- VoIP
- P2P
- DNS
"""

import numpy as np
import hashlib
import time
from typing import List, Dict, Tuple, Optional
from collections import Counter


class TrafficSimulator:
    """
    加密流量模拟器
    生成符合真实流量统计分布的模拟数据
    """

    # 各应用类型的流量特征参数
    APP_PROFILES = {
        'browsing': {
            'packet_count': (10, 50),
            'packet_size_mean': (400, 1200),
            'packet_size_std': (200, 500),
            'iat_mean': (0.01, 0.5),
            'iat_std': (0.01, 0.3),
            'tls_prob': 0.95,
            'tls_version_prob': {0x0303: 0.4, 0x0304: 0.6},
            'up_ratio': (0.3, 0.5),
            'small_pkt_ratio': (0.2, 0.4),
            'large_pkt_ratio': (0.1, 0.3),
            'entropy_mean': (6.0, 7.5),
            'payload_len_mean': (100, 500),
            'duration': (0.5, 5.0),
        },
        'video': {
            'packet_count': (100, 500),
            'packet_size_mean': (1000, 1450),
            'packet_size_std': (100, 300),
            'iat_mean': (0.001, 0.02),
            'iat_std': (0.001, 0.01),
            'tls_prob': 0.98,
            'tls_version_prob': {0x0303: 0.3, 0x0304: 0.7},
            'up_ratio': (0.1, 0.2),
            'small_pkt_ratio': (0.05, 0.15),
            'large_pkt_ratio': (0.6, 0.85),
            'entropy_mean': (7.0, 7.8),
            'payload_len_mean': (800, 1400),
            'duration': (10.0, 120.0),
        },
        'chat': {
            'packet_count': (5, 30),
            'packet_size_mean': (100, 400),
            'packet_size_std': (50, 200),
            'iat_mean': (0.5, 5.0),
            'iat_std': (0.3, 3.0),
            'tls_prob': 0.90,
            'tls_version_prob': {0x0303: 0.5, 0x0304: 0.5},
            'up_ratio': (0.4, 0.6),
            'small_pkt_ratio': (0.4, 0.7),
            'large_pkt_ratio': (0.0, 0.1),
            'entropy_mean': (5.0, 6.5),
            'payload_len_mean': (50, 300),
            'duration': (1.0, 30.0),
        },
        'file_transfer': {
            'packet_count': (50, 300),
            'packet_size_mean': (800, 1400),
            'packet_size_std': (200, 400),
            'iat_mean': (0.005, 0.05),
            'iat_std': (0.005, 0.03),
            'tls_prob': 0.85,
            'tls_version_prob': {0x0303: 0.7, 0x0304: 0.3},
            'up_ratio': (0.1, 0.3),
            'small_pkt_ratio': (0.1, 0.2),
            'large_pkt_ratio': (0.4, 0.7),
            'entropy_mean': (6.5, 7.5),
            'payload_len_mean': (600, 1200),
            'duration': (5.0, 60.0),
        },
        'email': {
            'packet_count': (10, 40),
            'packet_size_mean': (300, 800),
            'packet_size_std': (100, 300),
            'iat_mean': (0.1, 2.0),
            'iat_std': (0.05, 1.0),
            'tls_prob': 0.92,
            'tls_version_prob': {0x0303: 0.8, 0x0304: 0.2},
            'up_ratio': (0.3, 0.5),
            'small_pkt_ratio': (0.2, 0.4),
            'large_pkt_ratio': (0.1, 0.3),
            'entropy_mean': (5.5, 7.0),
            'payload_len_mean': (200, 600),
            'duration': (1.0, 10.0),
        },
        'streaming': {
            'packet_count': (80, 400),
            'packet_size_mean': (900, 1400),
            'packet_size_std': (150, 350),
            'iat_mean': (0.002, 0.03),
            'iat_std': (0.001, 0.02),
            'tls_prob': 0.97,
            'tls_version_prob': {0x0303: 0.2, 0x0304: 0.8},
            'up_ratio': (0.1, 0.25),
            'small_pkt_ratio': (0.05, 0.15),
            'large_pkt_ratio': (0.5, 0.8),
            'entropy_mean': (7.0, 7.8),
            'payload_len_mean': (700, 1300),
            'duration': (30.0, 300.0),
        },
        'voip': {
            'packet_count': (50, 200),
            'packet_size_mean': (100, 300),
            'packet_size_std': (20, 80),
            'iat_mean': (0.01, 0.05),
            'iat_std': (0.005, 0.02),
            'tls_prob': 0.70,
            'tls_version_prob': {0x0303: 0.6, 0x0304: 0.4},
            'up_ratio': (0.4, 0.6),
            'small_pkt_ratio': (0.5, 0.8),
            'large_pkt_ratio': (0.0, 0.05),
            'entropy_mean': (4.0, 5.5),
            'payload_len_mean': (50, 200),
            'duration': (5.0, 60.0),
        },
        'p2p': {
            'packet_count': (30, 150),
            'packet_size_mean': (500, 1200),
            'packet_size_std': (300, 600),
            'iat_mean': (0.01, 0.1),
            'iat_std': (0.01, 0.08),
            'tls_prob': 0.40,
            'tls_version_prob': {0x0303: 0.8, 0x0304: 0.2},
            'up_ratio': (0.3, 0.6),
            'small_pkt_ratio': (0.2, 0.4),
            'large_pkt_ratio': (0.2, 0.5),
            'entropy_mean': (6.0, 7.5),
            'payload_len_mean': (400, 1000),
            'duration': (10.0, 120.0),
        },
        'dns': {
            'packet_count': (2, 10),
            'packet_size_mean': (50, 200),
            'packet_size_std': (20, 100),
            'iat_mean': (0.01, 0.5),
            'iat_std': (0.01, 0.3),
            'tls_prob': 0.0,
            'tls_version_prob': {},
            'up_ratio': (0.4, 0.6),
            'small_pkt_ratio': (0.7, 0.95),
            'large_pkt_ratio': (0.0, 0.0),
            'entropy_mean': (3.0, 5.0),
            'payload_len_mean': (30, 150),
            'duration': (0.01, 0.5),
        },
    }

    # 异常流量特征（偏离正常模式）
    ANOMALY_PROFILES = {
        'data_exfiltration': {
            'packet_count': (20, 80),
            'packet_size_mean': (1300, 1500),
            'packet_size_std': (50, 150),
            'iat_mean': (0.001, 0.01),
            'iat_std': (0.001, 0.005),
            'tls_prob': 0.99,
            'tls_version_prob': {0x0303: 0.9, 0x0304: 0.1},
            'up_ratio': (0.6, 0.9),
            'small_pkt_ratio': (0.0, 0.05),
            'large_pkt_ratio': (0.8, 0.95),
            'entropy_mean': (7.5, 7.9),
            'payload_len_mean': (1200, 1500),
            'duration': (0.5, 5.0),
        },
        'c2_tunnel': {
            'packet_count': (5, 20),
            'packet_size_mean': (50, 200),
            'packet_size_std': (20, 100),
            'iat_mean': (1.0, 10.0),
            'iat_std': (0.5, 5.0),
            'tls_prob': 0.95,
            'tls_version_prob': {0x0303: 0.5, 0x0304: 0.5},
            'up_ratio': (0.4, 0.6),
            'small_pkt_ratio': (0.6, 0.9),
            'large_pkt_ratio': (0.0, 0.1),
            'entropy_mean': (6.0, 7.0),
            'payload_len_mean': (50, 150),
            'duration': (30.0, 300.0),
        },
        'dns_tunnel': {
            'packet_count': (20, 100),
            'packet_size_mean': (200, 500),
            'packet_size_std': (50, 150),
            'iat_mean': (0.01, 0.1),
            'iat_std': (0.01, 0.05),
            'tls_prob': 0.0,
            'tls_version_prob': {},
            'up_ratio': (0.4, 0.6),
            'small_pkt_ratio': (0.1, 0.3),
            'large_pkt_ratio': (0.0, 0.1),
            'entropy_mean': (6.5, 7.5),
            'payload_len_mean': (150, 400),
            'duration': (1.0, 10.0),
        },
    }

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.app_types = list(self.APP_PROFILES.keys())
        self.anomaly_types = list(self.ANOMALY_PROFILES.keys())

    def generate_flow(self, app_type: str = None, is_anomaly: bool = False,
                      anomaly_type: str = None) -> Tuple[List[Dict], str]:
        """
        生成一个流量流

        Args:
            app_type: 应用类型，None则随机选择
            is_anomaly: 是否为异常流量
            anomaly_type: 异常类型

        Returns:
            (packets, label) 包列表和标签
        """
        if is_anomaly:
            if anomaly_type:
                profile = self.ANOMALY_PROFILES[anomaly_type]
            else:
                anomaly_type = self.rng.choice(self.anomaly_types)
                profile = self.ANOMALY_PROFILES[anomaly_type]
            label = f'anomaly_{anomaly_type}'
        else:
            if app_type is None:
                app_type = self.rng.choice(self.app_types)
            profile = self.APP_PROFILES[app_type]
            label = app_type

        # 生成包
        packets = self._generate_packets(profile, label)

        return packets, label

    def _generate_packets(self, profile: Dict, label: str) -> List[Dict]:
        """根据profile生成包序列"""
        # 包数量
        n_packets = int(self.rng.integers(profile['packet_count'][0],
                                          profile['packet_count'][1] + 1))

        # 包长度
        mean_size = self.rng.uniform(*profile['packet_size_mean'])
        std_size = self.rng.uniform(*profile['packet_size_std'])
        lengths = np.abs(self.rng.normal(mean_size, std_size, n_packets))
        lengths = np.clip(lengths, 40, 1500).astype(int)

        # 时间戳
        iat_mean = self.rng.uniform(*profile['iat_mean'])
        iat_std = self.rng.uniform(*profile['iat_std'])
        iats = np.abs(self.rng.normal(iat_mean, iat_std, n_packets))
        iats = np.clip(iats, 0.0001, None)
        timestamps = np.cumsum(iats)

        # 方向
        up_ratio = self.rng.uniform(*profile['up_ratio'])
        directions = self.rng.choice(['up', 'down'], n_packets,
                                     p=[up_ratio, 1 - up_ratio])

        # 端口
        if 'dns' in label:
            src_port = int(self.rng.integers(49152, 65535))
            dst_port = 53
            protocol = 17  # UDP
        elif 'anomaly_dns' in label:
            src_port = int(self.rng.integers(49152, 65535))
            dst_port = 53
            protocol = 17
        else:
            src_port = int(self.rng.integers(49152, 65535))
            dst_port = 443
            protocol = 6  # TCP

        # TLS信息
        tls_info = None
        if self.rng.random() < profile['tls_prob']:
            tls_info = self._generate_tls_info(profile['tls_version_prob'], label)

        # 载荷
        payloads = []
        for length in lengths:
            payload = self._generate_payload(int(length * 0.8), profile, label)
            payloads.append(payload)

        packets = []
        for i in range(n_packets):
            pkt = {
                'length': int(lengths[i]),
                'timestamp': float(timestamps[i]),
                'src_port': src_port,
                'dst_port': dst_port,
                'protocol': protocol,
                'payload': payloads[i],
                'direction': str(directions[i]),
                'tls_info': tls_info if i == 0 else None,
            }
            packets.append(pkt)

        return packets

    def _generate_tls_info(self, version_prob: Dict, label: str) -> Dict:
        """生成TLS握手信息"""
        if not version_prob:
            return None

        versions = list(version_prob.keys())
        probs = list(version_prob.values())
        version = int(self.rng.choice(versions, p=probs))

        # 密码套件
        cipher_suites = [
            0x1301, 0x1302, 0x1303,  # TLS 1.3
            0xC02B, 0xC02C, 0xC02F, 0xC030,  # ECDHE
            0x009C, 0x009D,  # RSA AES-GCM
            0xC013, 0xC014,  # ECDHE RSA
        ]
        n_ciphers = int(self.rng.integers(3, min(8, len(cipher_suites) + 1)))
        selected_ciphers = list(self.rng.choice(cipher_suites, n_ciphers, replace=False))

        # 扩展
        extensions = ['sni', 'alpn', 'supported_groups', 'ec_point_formats',
                      'session_ticket', 'renegotiation_info']
        if version == 0x0304:
            extensions.extend(['key_share', 'psk_key_exchange_modes'])
        n_exts = int(self.rng.integers(3, len(extensions) + 1))
        selected_exts = list(self.rng.choice(extensions, n_exts, replace=False))

        # ALPN
        alpn = []
        if 'alpn' in selected_exts:
            alpn_protos = ['http/1.1', 'h2', 'h3']
            alpn = list(self.rng.choice(alpn_protos, int(self.rng.integers(1, 4)), replace=False))

        # 证书
        cert_chain_len = int(self.rng.integers(1, 4))
        certs = []
        cert_lengths = []
        for _ in range(cert_chain_len):
            cert_len = int(self.rng.integers(500, 2000))
            cert_lengths.append(cert_len)
            certs.append({
                'length': cert_len,
                'self_signed': bool(self.rng.random() < 0.1),
                'validity_days': int(self.rng.integers(30, 825)),
            })

        # 密码强度
        cipher_strengths = [128, 256] if version == 0x0304 else [128, 192, 256]
        selected_strengths = list(self.rng.choice(cipher_strengths, n_ciphers, replace=True))

        tls_info = {
            'version': version,
            'cipher_suites': selected_ciphers,
            'extensions': selected_exts,
            'alpn': alpn,
            'sni': f"www.{self.rng.choice(['example', 'test', 'sample', 'demo'])}.com",
            'certificate_lengths': cert_lengths,
            'certificates': certs,
            'key_exchange_length': int(self.rng.integers(32, 256)),
            'sig_hash_algs': list(self.rng.integers(0, 10, int(self.rng.integers(2, 6)))),
            'supported_groups': list(self.rng.integers(0, 30, int(self.rng.integers(2, 8)))),
            'ec_point_formats': list(self.rng.integers(0, 3, int(self.rng.integers(1, 4)))),
            'cipher_strengths': selected_strengths,
        }

        return tls_info

    def _generate_payload(self, length: int, profile: Dict, label: str) -> bytes:
        """生成载荷数据"""
        if length <= 0:
            return b''

        entropy_mean = profile['entropy_mean']
        target_entropy = self.rng.uniform(*entropy_mean)

        if 'dns' in label:
            # DNS查询：低熵
            domains = ['google.com', 'facebook.com', 'youtube.com', 'baidu.com',
                       'example.com', 'test.org', 'api.service.com']
            domain = self.rng.choice(domains)
            return domain.encode()[:length]

        if target_entropy > 6.5:
            # 高熵：加密数据
            return self.rng.bytes(length)
        elif target_entropy > 5.0:
            # 中熵：混合数据
            ratio = self.rng.random()
            n_enc = int(length * ratio)
            n_text = length - n_enc
            enc_part = self.rng.bytes(n_enc)
            text_part = self.rng.bytes(n_text)  # 模拟文本
            return enc_part + text_part
        else:
            # 低熵：文本数据
            text = "HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n\r\n" + \
                   "<html><body>" + "x" * length
            return text.encode()[:length]

    def generate_dataset(self, n_samples: int = 500,
                         anomaly_ratio: float = 0.1) -> Tuple[List, List, List]:
        """
        生成完整数据集

        Args:
            n_samples: 样本总数
            anomaly_ratio: 异常样本比例

        Returns:
            (flows, labels, is_anomaly)
        """
        flows = []
        labels = []
        is_anomaly = []

        n_anomaly = int(n_samples * anomaly_ratio)
        n_normal = n_samples - n_anomaly

        # 生成正常流量
        for _ in range(n_normal):
            app_type = self.rng.choice(self.app_types)
            flow, label = self.generate_flow(app_type=app_type, is_anomaly=False)
            flows.append(flow)
            labels.append(label)
            is_anomaly.append(False)

        # 生成异常流量
        for _ in range(n_anomaly):
            anomaly_type = self.rng.choice(self.anomaly_types)
            flow, label = self.generate_flow(is_anomaly=True, anomaly_type=anomaly_type)
            flows.append(flow)
            labels.append(label)
            is_anomaly.append(True)

        # 打乱
        indices = list(range(len(flows)))
        self.rng.shuffle(indices)
        flows = [flows[i] for i in indices]
        labels = [labels[i] for i in indices]
        is_anomaly = [is_anomaly[i] for i in indices]

        return flows, labels, is_anomaly

    def get_app_type_distribution(self) -> Dict[str, float]:
        """返回应用类型分布（用于生成平衡数据集）"""
        return {app: 1.0 / len(self.app_types) for app in self.app_types}


def generate_training_data(n_samples: int = 500, anomaly_ratio: float = 0.1,
                           seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    生成训练数据（特征矩阵 + 标签）

    Args:
        n_samples: 样本数
        anomaly_ratio: 异常比例
        seed: 随机种子

    Returns:
        (X, y_class, y_anomaly)
        X: 特征矩阵
        y_class: 分类标签（数值）
        y_anomaly: 异常标签（0正常/1异常）
    """
    from feature_extractor import FlowFeatureExtractor

    simulator = TrafficSimulator(seed=seed)
    extractor = FlowFeatureExtractor()

    flows, labels, is_anomaly = simulator.generate_dataset(n_samples, anomaly_ratio)

    # 提取特征
    X = []
    for flow in flows:
        features = extractor.extract_from_raw(flow)
        X.append(features)
    X = np.array(X)

    # 分类标签编码
    app_types = list(simulator.APP_PROFILES.keys())
    anomaly_types = list(simulator.ANOMALY_PROFILES.keys())
    all_types = app_types + [f'anomaly_{t}' for t in anomaly_types]
    label_to_id = {t: i for i, t in enumerate(all_types)}
    y_class = np.array([label_to_id.get(l, len(all_types) - 1) for l in labels])

    # 异常标签
    y_anomaly = np.array(is_anomaly, dtype=int)

    return X, y_class, y_anomaly


if __name__ == '__main__':
    # 测试数据生成
    simulator = TrafficSimulator(seed=42)

    print("=" * 60)
    print("CryptoFlow · 数据生成器测试")
    print("=" * 60)

    # 测试各应用类型
    for app_type in simulator.app_types:
        flow, label = simulator.generate_flow(app_type=app_type)
        print(f"\n📦 {label}: {len(flow)} 个包, "
              f"平均长度 {np.mean([p['length'] for p in flow]):.0f} 字节, "
              f"TLS: {'有' if flow[0].get('tls_info') else '无'}")

    # 测试异常流量
    print("\n⚠️  异常流量:")
    for anomaly_type in simulator.anomaly_types:
        flow, label = simulator.generate_flow(is_anomaly=True, anomaly_type=anomaly_type)
        print(f"  {label}: {len(flow)} 个包, "
              f"平均长度 {np.mean([p['length'] for p in flow]):.0f} 字节")

    # 生成完整数据集
    print("\n📊 生成完整数据集...")
    X, y_class, y_anomaly = generate_training_data(n_samples=200, anomaly_ratio=0.15)
    print(f"  特征矩阵: {X.shape}")
    print(f"  分类标签: {len(np.unique(y_class))} 类")
    print(f"  异常比例: {np.mean(y_anomaly) * 100:.1f}%")
    print(f"  特征维度: {X.shape[1]}")

    print("\n✅ 数据生成器测试完成")