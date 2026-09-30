"""
CryptoFlow · 加密流量特征提取模块
====================================

从网络流量中提取 80+ 维特征用于机器学习分类和异常检测。

特征类别：
1. 包长度统计特征 (20维)
2. 包间隔时间特征 (15维)
3. TLS握手特征 (20维)
4. 流统计特征 (15维)
5. 熵特征 (10维)
6. 协议特征 (10维)
"""

import numpy as np
import hashlib
import math
from collections import Counter
from typing import List, Dict, Tuple, Optional


class FlowFeatureExtractor:
    """
    流量特征提取器
    从原始流量数据中提取多维特征向量
    """

    # TLS 密码套件特征库（常见套件）
    TLS_CIPHER_SUITES = {
        0x0001: 'TLS_NULL_WITH_NULL_NULL',
        0x0002: 'TLS_RSA_WITH_NULL_SHA',
        0x0004: 'TLS_RSA_WITH_AES_128_CBC_SHA',
        0x0005: 'TLS_RSA_WITH_AES_256_CBC_SHA',
        0x002F: 'TLS_RSA_WITH_AES_128_CBC_SHA',
        0x0035: 'TLS_RSA_WITH_AES_256_CBC_SHA',
        0x009C: 'TLS_RSA_WITH_AES_128_GCM_SHA256',
        0x009D: 'TLS_RSA_WITH_AES_256_GCM_SHA384',
        0x1301: 'TLS_AES_128_GCM_SHA256',
        0x1302: 'TLS_AES_256_GCM_SHA384',
        0x1303: 'TLS_CHACHA20_POLY1305_SHA256',
        0xC013: 'TLS_ECDHE_RSA_WITH_AES_128_CBC_SHA',
        0xC014: 'TLS_ECDHE_RSA_WITH_AES_256_CBC_SHA',
        0xC02B: 'TLS_ECDHE_ECDSA_WITH_AES_128_GCM_SHA256',
        0xC02C: 'TLS_ECDHE_ECDSA_WITH_AES_256_GCM_SHA384',
        0xC02F: 'TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256',
        0xC030: 'TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384',
        0xCCA8: 'TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305_SHA256',
        0xCCA9: 'TLS_ECDHE_ECDSA_WITH_CHACHA20_POLY1305_SHA256',
    }

    # TLS 版本映射
    TLS_VERSIONS = {
        0x0300: 'SSLv3',
        0x0301: 'TLSv1.0',
        0x0302: 'TLSv1.1',
        0x0303: 'TLSv1.2',
        0x0304: 'TLSv1.3',
    }

    # 应用类型标签
    APP_TYPES = {
        'browsing': 0,
        'video': 1,
        'chat': 2,
        'file_transfer': 3,
        'email': 4,
        'streaming': 5,
        'voip': 6,
        'p2p': 7,
        'dns': 8,
        'other': 9,
    }

    def __init__(self):
        self.feature_names = []
        self._init_feature_names()

    def _init_feature_names(self):
        """初始化特征名称列表"""
        # 包长度统计
        for stat in ['min', 'max', 'mean', 'std', 'median', 'q25', 'q75', 'iqr',
                     'skew', 'kurtosis', 'total_bytes', 'total_packets',
                     'mean_up', 'mean_down', 'ratio_up_down',
                     'small_pkt_ratio', 'large_pkt_ratio',
                     'pkt_len_entropy', 'payload_len_mean', 'payload_len_std']:
            self.feature_names.append(f'len_{stat}')

        # 包间隔时间
        for stat in ['min', 'max', 'mean', 'std', 'median', 'q25', 'q75', 'iqr',
                     'skew', 'kurtosis', 'mean_up', 'mean_down',
                     'ratio_up_down', 'total_duration', 'pkt_rate']:
            self.feature_names.append(f'iat_{stat}')

        # TLS 握手特征
        for feat in ['has_tls', 'tls_version', 'cipher_count', 'ext_count',
                     'cert_len_mean', 'cert_len_std', 'sni_present',
                     'alpn_count', 'key_exchange_len', 'sig_hash_alg_count',
                     'supported_groups_count', 'ec_point_format_count',
                     'session_ticket_present', 'renegotiation_present',
                     'heartbeat_present', 'encrypt_then_mac_present',
                     'extended_master_secret_present', 'tls13_support',
                     'cipher_strength_mean', 'tls_handshake_entropy']:
            self.feature_names.append(f'tls_{feat}')

        # 流统计
        for stat in ['src_port', 'dst_port', 'protocol', 'flow_duration',
                     'fwd_packets', 'bwd_packets', 'fwd_bytes', 'bwd_bytes',
                     'fwd_pkt_rate', 'bwd_pkt_rate', 'fwd_byte_rate',
                     'bwd_byte_rate', 'fwd_bwd_ratio', 'pkt_size_var',
                     'flow_active_mean']:
            self.feature_names.append(f'flow_{stat}')

        # 熵特征
        for feat in ['payload_entropy', 'header_entropy', 'byte_dist_entropy',
                     'entropy_std', 'entropy_max', 'entropy_min',
                     'entropy_mean', 'high_entropy_ratio',
                     'low_entropy_ratio', 'entropy_change_rate']:
            self.feature_names.append(f'entropy_{feat}')

        # 协议特征
        for feat in ['is_encrypted', 'encryption_confidence',
                     'has_certificate', 'cert_chain_len',
                     'self_signed_cert', 'cert_validity_days',
                     'dns_query_present', 'http_over_tls',
                     'quic_detected', 'protocol_entropy']:
            self.feature_names.append(f'proto_{feat}')

    def get_feature_count(self) -> int:
        """返回特征总数"""
        return len(self.feature_names)

    def get_feature_names(self) -> List[str]:
        """返回所有特征名称"""
        return self.feature_names.copy()

    def extract_from_raw(self, packets: List[Dict]) -> np.ndarray:
        """
        从原始包数据中提取特征

        Args:
            packets: 包列表，每个包为字典格式：
                {
                    'length': int,           # 包长度
                    'timestamp': float,      # 时间戳
                    'src_port': int,         # 源端口
                    'dst_port': int,         # 目的端口
                    'protocol': int,         # IP协议号
                    'payload': bytes,        # 载荷数据
                    'direction': str,        # 'up' 或 'down'
                    'tls_info': dict or None # TLS信息（可选）
                }

        Returns:
            特征向量 (numpy array)
        """
        if not packets:
            return np.zeros(self.get_feature_count())

        features = []

        # 1. 包长度统计特征 (20维)
        features.extend(self._extract_length_features(packets))

        # 2. 包间隔时间特征 (15维)
        features.extend(self._extract_iat_features(packets))

        # 3. TLS握手特征 (20维)
        features.extend(self._extract_tls_features(packets))

        # 4. 流统计特征 (15维)
        features.extend(self._extract_flow_stats(packets))

        # 5. 熵特征 (10维)
        features.extend(self._extract_entropy_features(packets))

        # 6. 协议特征 (10维)
        features.extend(self._extract_protocol_features(packets))

        return np.array(features, dtype=np.float64)

    def _extract_length_features(self, packets: List[Dict]) -> List[float]:
        """提取包长度统计特征"""
        lengths = [p['length'] for p in packets]
        up_lengths = [p['length'] for p in packets if p.get('direction') == 'up']
        down_lengths = [p['length'] for p in packets if p.get('direction') == 'down']

        if not lengths:
            return [0.0] * 20

        arr = np.array(lengths)
        up_arr = np.array(up_lengths) if up_lengths else np.array([0])
        down_arr = np.array(down_lengths) if down_lengths else np.array([0])

        # 基本统计
        pkt_min = float(np.min(arr))
        pkt_max = float(np.max(arr))
        pkt_mean = float(np.mean(arr))
        pkt_std = float(np.std(arr))
        pkt_median = float(np.median(arr))
        pkt_q25 = float(np.percentile(arr, 25))
        pkt_q75 = float(np.percentile(arr, 75))
        pkt_iqr = pkt_q75 - pkt_q25

        # 高阶统计
        skew = float(np.mean(((arr - pkt_mean) / (pkt_std + 1e-10)) ** 3)) if pkt_std > 0 else 0
        kurtosis = float(np.mean(((arr - pkt_mean) / (pkt_std + 1e-10)) ** 4)) - 3 if pkt_std > 0 else 0

        total_bytes = float(np.sum(arr))
        total_packets = float(len(arr))

        mean_up = float(np.mean(up_arr))
        mean_down = float(np.mean(down_arr))
        ratio_up_down = mean_up / (mean_down + 1e-10)

        # 小包和大包比例
        small_pkt_ratio = float(np.sum(arr < 100) / len(arr))
        large_pkt_ratio = float(np.sum(arr > 1400) / len(arr))

        # 包长度熵
        pkt_len_entropy = self._compute_entropy(arr)

        # 载荷长度
        payload_lens = [len(p.get('payload', b'')) for p in packets]
        payload_arr = np.array(payload_lens)
        payload_len_mean = float(np.mean(payload_arr))
        payload_len_std = float(np.std(payload_arr))

        return [
            pkt_min, pkt_max, pkt_mean, pkt_std, pkt_median,
            pkt_q25, pkt_q75, pkt_iqr, skew, kurtosis,
            total_bytes, total_packets, mean_up, mean_down, ratio_up_down,
            small_pkt_ratio, large_pkt_ratio, pkt_len_entropy,
            payload_len_mean, payload_len_std
        ]

    def _extract_iat_features(self, packets: List[Dict]) -> List[float]:
        """提取包间隔时间特征"""
        if len(packets) < 2:
            return [0.0] * 15

        timestamps = [p['timestamp'] for p in packets]
        iats = np.diff(timestamps)
        iats = iats[iats >= 0]  # 过滤负值

        if len(iats) == 0:
            return [0.0] * 15

        # 上下行间隔
        up_iats = []
        down_iats = []
        for i in range(1, len(packets)):
            if packets[i].get('direction') == 'up' and packets[i-1].get('direction') == 'up':
                up_iats.append(timestamps[i] - timestamps[i-1])
            elif packets[i].get('direction') == 'down' and packets[i-1].get('direction') == 'down':
                down_iats.append(timestamps[i] - timestamps[i-1])

        up_arr = np.array(up_iats) if up_iats else np.array([0])
        down_arr = np.array(down_iats) if down_iats else np.array([0])

        iat_min = float(np.min(iats))
        iat_max = float(np.max(iats))
        iat_mean = float(np.mean(iats))
        iat_std = float(np.std(iats))
        iat_median = float(np.median(iats))
        iat_q25 = float(np.percentile(iats, 25))
        iat_q75 = float(np.percentile(iats, 75))
        iat_iqr = iat_q75 - iat_q25

        skew = float(np.mean(((iats - iat_mean) / (iat_std + 1e-10)) ** 3)) if iat_std > 0 else 0
        kurtosis = float(np.mean(((iats - iat_mean) / (iat_std + 1e-10)) ** 4)) - 3 if iat_std > 0 else 0

        mean_up = float(np.mean(up_arr))
        mean_down = float(np.mean(down_arr))
        ratio_up_down = mean_up / (mean_down + 1e-10)

        total_duration = timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0
        pkt_rate = len(packets) / (total_duration + 1e-10)

        return [
            iat_min, iat_max, iat_mean, iat_std, iat_median,
            iat_q25, iat_q75, iat_iqr, skew, kurtosis,
            mean_up, mean_down, ratio_up_down, total_duration, pkt_rate
        ]

    def _extract_tls_features(self, packets: List[Dict]) -> List[float]:
        """提取TLS握手特征"""
        has_tls = 0
        tls_version = 0
        cipher_count = 0
        ext_count = 0
        cert_len_mean = 0
        cert_len_std = 0
        sni_present = 0
        alpn_count = 0
        key_exchange_len = 0
        sig_hash_alg_count = 0
        supported_groups_count = 0
        ec_point_format_count = 0
        session_ticket_present = 0
        renegotiation_present = 0
        heartbeat_present = 0
        encrypt_then_mac_present = 0
        extended_master_secret_present = 0
        tls13_support = 0
        cipher_strength_mean = 0
        tls_handshake_entropy = 0

        for pkt in packets:
            tls_info = pkt.get('tls_info')
            if tls_info:
                has_tls = 1
                tls_version = tls_info.get('version', 0) / 0x0304  # 归一化到 [0,1]
                cipher_count = min(len(tls_info.get('cipher_suites', [])), 50) / 50
                ext_count = min(len(tls_info.get('extensions', [])), 30) / 30

                cert_lens = tls_info.get('certificate_lengths', [])
                if cert_lens:
                    cert_arr = np.array(cert_lens)
                    cert_len_mean = float(np.mean(cert_arr)) / 2000  # 归一化
                    cert_len_std = float(np.std(cert_arr)) / 1000

                sni_present = 1 if tls_info.get('sni') else 0
                alpn_count = min(len(tls_info.get('alpn', [])), 10) / 10
                key_exchange_len = tls_info.get('key_exchange_length', 0) / 256
                sig_hash_alg_count = min(len(tls_info.get('sig_hash_algs', [])), 10) / 10
                supported_groups_count = min(len(tls_info.get('supported_groups', [])), 10) / 10
                ec_point_format_count = min(len(tls_info.get('ec_point_formats', [])), 5) / 5

                session_ticket_present = 1 if 'session_ticket' in tls_info.get('extensions', []) else 0
                renegotiation_present = 1 if 'renegotiation_info' in tls_info.get('extensions', []) else 0
                heartbeat_present = 1 if 'heartbeat' in tls_info.get('extensions', []) else 0
                encrypt_then_mac_present = 1 if 'encrypt_then_mac' in tls_info.get('extensions', []) else 0
                extended_master_secret_present = 1 if 'extended_master_secret' in tls_info.get('extensions', []) else 0
                tls13_support = 1 if tls_info.get('version') == 0x0304 else 0

                # 密码强度
                cipher_strengths = tls_info.get('cipher_strengths', [128])
                cipher_strength_mean = float(np.mean(cipher_strengths)) / 256

                # TLS握手熵
                handshake_data = str(tls_info).encode()
                tls_handshake_entropy = self._compute_entropy(
                    np.frombuffer(handshake_data, dtype=np.uint8)
                ) / 8

                break  # 只取第一个TLS包的信息

        return [
            has_tls, tls_version, cipher_count, ext_count,
            cert_len_mean, cert_len_std, sni_present, alpn_count,
            key_exchange_len, sig_hash_alg_count, supported_groups_count,
            ec_point_format_count, session_ticket_present,
            renegotiation_present, heartbeat_present,
            encrypt_then_mac_present, extended_master_secret_present,
            tls13_support, cipher_strength_mean, tls_handshake_entropy
        ]

    def _extract_flow_stats(self, packets: List[Dict]) -> List[float]:
        """提取流统计特征"""
        if not packets:
            return [0.0] * 15

        src_ports = [p.get('src_port', 0) for p in packets]
        dst_ports = [p.get('dst_port', 0) for p in packets]
        protocols = [p.get('protocol', 6) for p in packets]

        src_port = float(np.mean(src_ports)) / 65535
        dst_port = float(np.mean(dst_ports)) / 65535
        protocol = float(Counter(protocols).most_common(1)[0][0]) / 255

        timestamps = [p['timestamp'] for p in packets]
        flow_duration = (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0

        fwd_packets = sum(1 for p in packets if p.get('direction') == 'up')
        bwd_packets = sum(1 for p in packets if p.get('direction') == 'down')
        fwd_bytes = sum(p['length'] for p in packets if p.get('direction') == 'up')
        bwd_bytes = sum(p['length'] for p in packets if p.get('direction') == 'down')

        fwd_pkt_rate = fwd_packets / (flow_duration + 1e-10)
        bwd_pkt_rate = bwd_packets / (flow_duration + 1e-10)
        fwd_byte_rate = fwd_bytes / (flow_duration + 1e-10)
        bwd_byte_rate = bwd_bytes / (flow_duration + 1e-10)

        fwd_bwd_ratio = fwd_packets / (bwd_packets + 1e-10)

        lengths = [p['length'] for p in packets]
        pkt_size_var = float(np.var(lengths)) / (1500 ** 2)  # 归一化

        # 流活跃度
        if len(timestamps) > 1:
            iats = np.diff(timestamps)
            flow_active_mean = float(np.mean(iats[iats < 1.0])) if np.any(iats < 1.0) else 0
        else:
            flow_active_mean = 0

        return [
            src_port, dst_port, protocol, flow_duration,
            float(fwd_packets), float(bwd_packets),
            float(fwd_bytes), float(bwd_bytes),
            fwd_pkt_rate, bwd_pkt_rate, fwd_byte_rate, bwd_byte_rate,
            fwd_bwd_ratio, pkt_size_var, flow_active_mean
        ]

    def _extract_entropy_features(self, packets: List[Dict]) -> List[float]:
        """提取熵特征"""
        if not packets:
            return [0.0] * 10

        # 计算每个包的载荷熵
        payload_entropies = []
        for pkt in packets:
            payload = pkt.get('payload', b'')
            if len(payload) > 0:
                entropy = self._compute_entropy(np.frombuffer(payload, dtype=np.uint8))
                payload_entropies.append(entropy)

        if not payload_entropies:
            return [0.0] * 10

        entropy_arr = np.array(payload_entropies)

        payload_entropy = float(np.mean(entropy_arr)) / 8  # 归一化到 [0,1]
        header_entropy = self._compute_entropy(
            np.frombuffer(str(packets[0]).encode(), dtype=np.uint8)
        ) / 8

        # 字节分布熵
        all_bytes = b''.join([p.get('payload', b'') for p in packets if p.get('payload')])
        if all_bytes:
            byte_dist = np.zeros(256)
            for b in all_bytes:
                byte_dist[b] += 1
            byte_dist = byte_dist / len(all_bytes)
            byte_dist_entropy = -np.sum(byte_dist[byte_dist > 0] * np.log2(byte_dist[byte_dist > 0])) / 8
        else:
            byte_dist_entropy = 0

        entropy_std = float(np.std(entropy_arr)) / 8
        entropy_max = float(np.max(entropy_arr)) / 8
        entropy_min = float(np.min(entropy_arr)) / 8
        entropy_mean = float(np.mean(entropy_arr)) / 8

        high_entropy_ratio = float(np.sum(entropy_arr > 6)) / len(entropy_arr)
        low_entropy_ratio = float(np.sum(entropy_arr < 2)) / len(entropy_arr)

        # 熵变化率
        if len(entropy_arr) > 1:
            entropy_change_rate = float(np.mean(np.abs(np.diff(entropy_arr)))) / 8
        else:
            entropy_change_rate = 0

        return [
            payload_entropy, header_entropy, byte_dist_entropy,
            entropy_std, entropy_max, entropy_min, entropy_mean,
            high_entropy_ratio, low_entropy_ratio, entropy_change_rate
        ]

    def _extract_protocol_features(self, packets: List[Dict]) -> List[float]:
        """提取协议特征"""
        is_encrypted = 0
        encryption_confidence = 0
        has_certificate = 0
        cert_chain_len = 0
        self_signed_cert = 0
        cert_validity_days = 0
        dns_query_present = 0
        http_over_tls = 0
        quic_detected = 0
        protocol_entropy = 0

        for pkt in packets:
            tls_info = pkt.get('tls_info')
            if tls_info:
                is_encrypted = 1
                encryption_confidence = 0.9 + 0.1 * (tls_info.get('version', 0) / 0x0304)

                certs = tls_info.get('certificates', [])
                if certs:
                    has_certificate = 1
                    cert_chain_len = min(len(certs), 5) / 5
                    self_signed_cert = 1 if any(c.get('self_signed', False) for c in certs) else 0
                    validities = [c.get('validity_days', 365) for c in certs]
                    cert_validity_days = min(validities) / 365 if validities else 0

                # HTTP over TLS检测
                alpn = tls_info.get('alpn', [])
                if 'http/1.1' in alpn or 'h2' in alpn:
                    http_over_tls = 1

                break

        # DNS查询检测
        for pkt in packets:
            if pkt.get('dst_port') == 53 or pkt.get('src_port') == 53:
                dns_query_present = 1
                break

        # QUIC检测
        for pkt in packets:
            if pkt.get('dst_port') == 443 or pkt.get('src_port') == 443:
                if pkt.get('protocol') == 17:  # UDP
                    quic_detected = 1
                    break

        # 协议熵
        proto_data = f"{is_encrypted}{has_certificate}{dns_query_present}{http_over_tls}{quic_detected}".encode()
        if proto_data:
            protocol_entropy = self._compute_entropy(
                np.frombuffer(proto_data, dtype=np.uint8)
            ) / 8

        return [
            float(is_encrypted), float(encryption_confidence),
            float(has_certificate), float(cert_chain_len),
            float(self_signed_cert), float(cert_validity_days),
            float(dns_query_present), float(http_over_tls),
            float(quic_detected), float(protocol_entropy)
        ]

    @staticmethod
    def _compute_entropy(data: np.ndarray) -> float:
        """计算数据熵"""
        if len(data) == 0:
            return 0.0
        _, counts = np.unique(data, return_counts=True)
        probs = counts / len(data)
        return float(-np.sum(probs * np.log2(probs + 1e-10)))


def extract_features_from_flows(flows: List[List[Dict]]) -> np.ndarray:
    """
    从多个流中提取特征矩阵

    Args:
        flows: 流列表，每个流是一个包列表

    Returns:
        特征矩阵 (n_flows, n_features)
    """
    extractor = FlowFeatureExtractor()
    features = []
    for flow in flows:
        feat = extractor.extract_from_raw(flow)
        features.append(feat)
    return np.array(features)


if __name__ == '__main__':
    # 测试特征提取
    extractor = FlowFeatureExtractor()
    print(f"特征总数: {extractor.get_feature_count()}")
    print(f"特征名称: {extractor.get_feature_names()[:5]}...")
    print(f"特征名称: {extractor.get_feature_names()[-5:]}...")