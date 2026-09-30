#!/usr/bin/env python3
"""
CryptoFlow · 加密流量分析与异常检测工具
========================================

主程序入口 - 命令行接口

用法:
    python main.py --generate          # 生成模拟数据
    python main.py --train             # 训练模型
    python main.py --analyze           # 分析流量（使用模拟数据）
    python main.py --all               # 完整流程
    python main.py --pcap <file.pcap>  # 分析PCAP文件
    python main.py --live              # 实时捕获分析
    python main.py --info              # 查看模型信息
"""

import numpy as np
import argparse
import os
import sys
import json
import time
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from collections import Counter

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from feature_extractor import FlowFeatureExtractor, extract_features_from_flows
from data_generator import TrafficSimulator, generate_training_data
from classifier import TrafficClassifier, train_classifier
from anomaly_detector import AnomalyDetector, train_anomaly_detector


# 模型保存路径
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'classifier.pkl')
ANOMALY_PATH = os.path.join(MODEL_DIR, 'anomaly_detector.pkl')


def print_banner():
    """打印启动横幅"""
    banner = """
╔══════════════════════════════════════════════════╗
║              CryptoFlow · v1.0.0                 ║
║         加密流量分析与异常检测工具                ║
╚══════════════════════════════════════════════════╝
    """
    print(banner)


def print_report_header():
    """打印报告头部"""
    print("\n" + "=" * 60)
    print(f"  CryptoFlow · 加密流量分析报告")
    print(f"  生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)


def cmd_generate(args):
    """生成模拟数据"""
    print_banner()
    print("📊 生成模拟加密流量数据...\n")

    simulator = TrafficSimulator(seed=args.seed)

    # 生成数据集
    X, y_class, y_anomaly = generate_training_data(
        n_samples=args.samples,
        anomaly_ratio=args.anomaly_ratio,
        seed=args.seed
    )

    # 统计信息
    n_normal = np.sum(y_anomaly == 0)
    n_anomaly = np.sum(y_anomaly == 1)

    print(f"✅ 数据生成完成!")
    print(f"  • 总样本数: {X.shape[0]}")
    print(f"  • 特征维度: {X.shape[1]}")
    print(f"  • 正常流量: {n_normal} ({n_normal / len(y_anomaly) * 100:.1f}%)")
    print(f"  • 异常流量: {n_anomaly} ({n_anomaly / len(y_anomaly) * 100:.1f}%)")

    # 保存数据
    if args.save:
        data_path = os.path.join(MODEL_DIR, 'training_data.npz')
        os.makedirs(MODEL_DIR, exist_ok=True)
        np.savez(data_path, X=X, y_class=y_class, y_anomaly=y_anomaly)
        print(f"\n💾 数据已保存到: {data_path}")

    return X, y_class, y_anomaly


def cmd_train(args):
    """训练模型"""
    print_banner()
    print("🔧 训练机器学习模型...\n")

    # 生成或加载数据
    if args.load_data and os.path.exists(args.load_data):
        print(f"📂 加载训练数据: {args.load_data}")
        data = np.load(args.load_data)
        X = data['X']
        y_class = data['y_class']
        y_anomaly = data['y_anomaly']
    else:
        print("📊 生成训练数据...")
        X, y_class, y_anomaly = generate_training_data(
            n_samples=args.samples,
            anomaly_ratio=args.anomaly_ratio,
            seed=args.seed
        )

    # 获取特征名称
    extractor = FlowFeatureExtractor()
    feature_names = extractor.get_feature_names()

    # 将数值标签转换为名称
    simulator = TrafficSimulator()
    app_types = list(simulator.APP_PROFILES.keys())
    anomaly_types = list(simulator.ANOMALY_PROFILES.keys())
    all_types = app_types + [f'anomaly_{t}' for t in anomaly_types]
    y_labels = np.array([all_types[i] for i in y_class])

    print(f"\n📊 数据集信息:")
    print(f"  • 样本数: {X.shape[0]}")
    print(f"  • 特征数: {X.shape[1]}")
    print(f"  • 类别数: {len(np.unique(y_labels))}")
    print(f"  • 异常比例: {np.mean(y_anomaly) * 100:.1f}%")

    # 训练分类器
    print("\n🔧 训练流量分类器 (RandomForest)...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    classifier, class_results = train_classifier(
        X, y_labels, feature_names,
        optimize=args.optimize,
        save_path=CLASSIFIER_PATH
    )

    print(f"\n📈 分类器性能:")
    print(f"  • 准确率: {class_results['accuracy'] * 100:.2f}%")
    print(f"  • 精确率: {class_results['precision'] * 100:.2f}%")
    print(f"  • 召回率: {class_results['recall'] * 100:.2f}%")
    print(f"  • F1分数: {class_results['f1_score'] * 100:.2f}%")
    print(f"  • 交叉验证: {class_results['cv_mean'] * 100:.2f}% ± {class_results['cv_std'] * 100:.2f}%")

    # 训练异常检测器
    print("\n🔧 训练异常检测器 (IsolationForest)...")
    detector, anomaly_results = train_anomaly_detector(
        X, feature_names,
        contamination=args.anomaly_ratio,
        save_path=ANOMALY_PATH
    )

    print(f"\n📈 异常检测器性能:")
    print(f"  • 检测到的异常: {anomaly_results['n_anomalies']} "
          f"({anomaly_results['anomaly_ratio'] * 100:.2f}%)")
    print(f"  • 异常阈值: {anomaly_results['threshold']:.4f}")

    # 特征重要性
    print("\n🔝 特征重要性 Top 10:")
    for name, importance in classifier.get_feature_importance(10):
        print(f"  {name}: {importance * 100:.2f}%")

    print(f"\n✅ 模型训练完成!")
    print(f"  • 分类器: {CLASSIFIER_PATH}")
    print(f"  • 异常检测器: {ANOMALY_PATH}")

    return classifier, detector


def cmd_analyze(args):
    """分析流量"""
    print_banner()
    print_report_header()

    # 加载模型
    if not os.path.exists(CLASSIFIER_PATH) or not os.path.exists(ANOMALY_PATH):
        print("❌ 模型文件不存在，请先运行 --train 训练模型")
        return

    print("\n📂 加载模型...")
    classifier = TrafficClassifier(model_path=CLASSIFIER_PATH)
    detector = AnomalyDetector(model_path=ANOMALY_PATH)

    # 生成测试数据
    print("📊 生成测试流量...")
    simulator = TrafficSimulator(seed=args.seed)
    extractor = FlowFeatureExtractor()

    n_samples = args.samples
    test_flows = []
    test_labels = []
    test_is_anomaly = []

    # 生成正常流量
    for _ in range(n_samples - int(n_samples * args.anomaly_ratio)):
        app_type = simulator.rng.choice(simulator.app_types)
        flow, label = simulator.generate_flow(app_type=app_type)
        test_flows.append(flow)
        test_labels.append(label)
        test_is_anomaly.append(False)

    # 生成异常流量
    for _ in range(int(n_samples * args.anomaly_ratio)):
        anomaly_type = simulator.rng.choice(simulator.anomaly_types)
        flow, label = simulator.generate_flow(is_anomaly=True, anomaly_type=anomaly_type)
        test_flows.append(flow)
        test_labels.append(label)
        test_is_anomaly.append(True)

    # 提取特征
    print("🔍 提取特征...")
    X = []
    for flow in test_flows:
        features = extractor.extract_from_raw(flow)
        X.append(features)
    X = np.array(X)

    # 分类预测
    print("🔮 执行分类...")
    predictions, probabilities = classifier.predict(X)

    # 异常检测
    print("⚠️  执行异常检测...")
    anomaly_preds, anomaly_scores = detector.predict(X)

    # 生成报告
    print("\n" + "=" * 60)
    print("📊 流量分析报告")
    print("=" * 60)

    # 1. 流量统计
    n_encrypted = sum(1 for flow in test_flows
                      if any(p.get('tls_info') for p in flow))
    print(f"\n📊 流量统计")
    print(f"  ├─ 总流数: {len(test_flows)}")
    print(f"  ├─ 加密流: {n_encrypted} ({n_encrypted / len(test_flows) * 100:.1f}%)")
    print(f"  └─ 非加密流: {len(test_flows) - n_encrypted} "
          f"({(1 - n_encrypted / len(test_flows)) * 100:.1f}%)")

    # 2. 分类结果
    print(f"\n🔍 分类结果")
    pred_counts = Counter(predictions)
    for label, count in pred_counts.most_common():
        print(f"  ├─ {label}: {count} ({count / len(predictions) * 100:.1f}%)")

    # 3. 异常检测结果
    n_detected_anomalies = np.sum(anomaly_preds == -1)
    print(f"\n⚠️  异常检测")
    print(f"  ├─ 正常: {len(anomaly_preds) - n_detected_anomalies} "
          f"({(1 - n_detected_anomalies / len(anomaly_preds)) * 100:.1f}%)")
    print(f"  └─ 异常: {n_detected_anomalies} "
          f"({n_detected_anomalies / len(anomaly_preds) * 100:.1f}%)")

    # 4. 详细异常信息
    if n_detected_anomalies > 0:
        print(f"\n📝 异常详情:")
        anomaly_indices = np.where(anomaly_preds == -1)[0]
        for idx in anomaly_indices[:10]:  # 最多显示10个
            features = X[idx]
            result = detector.predict_single(features)
            explanation = detector.get_anomaly_explanation(features)
            print(f"\n  ┌─ 流 #{idx + 1}")
            print(f"  ├─ 真实标签: {test_labels[idx]}")
            print(f"  ├─ 预测分类: {predictions[idx]}")
            print(f"  ├─ 异常分数: {result['score']:.4f}")
            print(f"  ├─ 风险等级: {result['risk_level']}")
            print(f"  └─ 异常解释: {explanation.split(chr(10))[0] if explanation else 'N/A'}")

    # 5. 性能统计
    if args.show_stats:
        print(f"\n📈 性能统计")
        # 分类准确率
        correct = sum(1 for true, pred in zip(test_labels, predictions) if true == pred)
        print(f"  ├─ 分类准确率: {correct / len(predictions) * 100:.2f}%")
        # 异常检测准确率
        anomaly_correct = sum(1 for true, pred in zip(test_is_anomaly, anomaly_preds)
                              if (true and pred == -1) or (not true and pred == 1))
        print(f"  └─ 异常检测准确率: {anomaly_correct / len(anomaly_preds) * 100:.2f}%")

    print("\n" + "=" * 60)
    print("✅ 分析完成")
    print("=" * 60)


def cmd_all(args):
    """完整流程：生成数据 → 训练模型 → 分析"""
    print_banner()
    print("🚀 启动完整流程...\n")

    # Step 1: 生成数据
    print("=" * 50)
    print("Step 1/3: 生成训练数据")
    print("=" * 50)
    X, y_class, y_anomaly = generate_training_data(
        n_samples=args.samples,
        anomaly_ratio=args.anomaly_ratio,
        seed=args.seed
    )
    print(f"  ✅ 生成 {X.shape[0]} 个样本, {X.shape[1]} 维特征\n")

    # Step 2: 训练模型
    print("=" * 50)
    print("Step 2/3: 训练模型")
    print("=" * 50)
    extractor = FlowFeatureExtractor()
    feature_names = extractor.get_feature_names()

    simulator = TrafficSimulator()
    app_types = list(simulator.APP_PROFILES.keys())
    anomaly_types = list(simulator.ANOMALY_PROFILES.keys())
    all_types = app_types + [f'anomaly_{t}' for t in anomaly_types]
    y_labels = np.array([all_types[i] for i in y_class])

    os.makedirs(MODEL_DIR, exist_ok=True)

    classifier, class_results = train_classifier(
        X, y_labels, feature_names,
        save_path=CLASSIFIER_PATH
    )
    print(f"  ✅ 分类器准确率: {class_results['accuracy'] * 100:.2f}%\n")

    detector, anomaly_results = train_anomaly_detector(
        X, feature_names,
        contamination=args.anomaly_ratio,
        save_path=ANOMALY_PATH
    )
    print(f"  ✅ 异常检测器训练完成\n")

    # Step 3: 分析
    print("=" * 50)
    print("Step 3/3: 流量分析")
    print("=" * 50)
    cmd_analyze(args)

    print("\n" + "=" * 50)
    print("🎉 完整流程完成!")
    print("=" * 50)


def cmd_info(args):
    """查看模型信息"""
    print_banner()

    print("📂 模型信息\n")

    # 分类器信息
    if os.path.exists(CLASSIFIER_PATH):
        classifier = TrafficClassifier(model_path=CLASSIFIER_PATH)
        info = classifier.get_model_info()
        print("🔧 流量分类器:")
        print(f"  • 状态: {info['status']}")
        print(f"  • 模型类型: {info.get('model_type', 'N/A')}")
        print(f"  • 决策树数量: {info.get('n_estimators', 'N/A')}")
        print(f"  • 最大深度: {info.get('max_depth', 'N/A')}")
        print(f"  • 特征数: {info.get('n_features', 'N/A')}")
        print(f"  • 类别数: {info.get('n_classes', 'N/A')}")
        if 'top_features' in info:
            print(f"  • 重要特征:")
            for name, importance in info['top_features'][:5]:
                print(f"    - {name}: {importance * 100:.2f}%")
    else:
        print("❌ 分类器模型未找到，请先运行 --train")

    print()

    # 异常检测器信息
    if os.path.exists(ANOMALY_PATH):
        detector = AnomalyDetector(model_path=ANOMALY_PATH)
        info = detector.get_model_info()
        print("⚠️  异常检测器:")
        print(f"  • 状态: {info['status']}")
        print(f"  • 模型类型: {info.get('model_type', 'N/A')}")
        print(f"  • 异常比例: {info.get('contamination', 'N/A')}")
        print(f"  • 阈值: {info.get('threshold', 'N/A')}")
        print(f"  • 决策树数量: {info.get('n_estimators', 'N/A')}")
        print(f"  • 特征数: {info.get('n_features', 'N/A')}")
    else:
        print("❌ 异常检测模型未找到，请先运行 --train")


def cmd_pcap(args):
    """分析PCAP文件（需要scapy）"""
    print_banner()
    print(f"📁 分析PCAP文件: {args.pcap}\n")

    if not os.path.exists(args.pcap):
        print(f"❌ 文件不存在: {args.pcap}")
        return

    try:
        from scapy.all import rdpcap, IP, TCP, UDP, TLS
    except ImportError:
        print("❌ 需要安装 scapy: pip install scapy")
        print("   或者使用 --generate 生成模拟数据进行分析")
        return

    # 加载模型
    if not os.path.exists(CLASSIFIER_PATH) or not os.path.exists(ANOMALY_PATH):
        print("❌ 模型文件不存在，请先运行 --train 训练模型")
        return

    classifier = TrafficClassifier(model_path=CLASSIFIER_PATH)
    detector = AnomalyDetector(model_path=ANOMALY_PATH)
    extractor = FlowFeatureExtractor()

    # 读取PCAP
    print("📖 读取PCAP文件...")
    packets = rdpcap(args.pcap)
    print(f"  • 总包数: {len(packets)}")

    # 提取流
    print("🔍 提取流量流...")
    flows = extract_flows_from_pcap(packets)
    print(f"  • 流数: {len(flows)}")

    # 分析每个流
    print("🔮 分析流量...")
    for i, flow in enumerate(flows[:50]):  # 最多分析50个流
        features = extractor.extract_from_raw(flow)
        pred, conf = classifier.predict_single(features)
        anomaly_result = detector.predict_single(features)

        print(f"\n  流 #{i + 1}:")
        print(f"    包数: {len(flow)}")
        print(f"    分类: {pred} (置信度: {conf * 100:.1f}%)")
        print(f"    异常: {'⚠️ ' if anomaly_result['is_anomaly'] else '✅ '}"
              f"分数: {anomaly_result['score']:.4f}")

    print(f"\n✅ PCAP分析完成")


def extract_flows_from_pcap(packets) -> List[List[Dict]]:
    """
    从scapy包列表中提取流

    Args:
        packets: scapy包列表

    Returns:
        流列表
    """
    flows_dict = {}

    for pkt in packets:
        if IP not in pkt:
            continue

        ip_layer = pkt[IP]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        protocol = ip_layer.proto

        if TCP in pkt:
            src_port = pkt[TCP].sport
            dst_port = pkt[TCP].dport
        elif UDP in pkt:
            src_port = pkt[UDP].sport
            dst_port = pkt[UDP].dport
        else:
            continue

        # 流标识（双向）
        flow_key = tuple(sorted([(src_ip, src_port), (dst_ip, dst_port)]))

        # 提取TLS信息
        tls_info = None
        if TLS in pkt:
            tls_info = extract_tls_info(pkt[TLS])

        flow_pkt = {
            'length': len(pkt),
            'timestamp': float(pkt.time),
            'src_port': src_port,
            'dst_port': dst_port,
            'protocol': protocol,
            'payload': bytes(pkt[IP].payload) if IP in pkt else b'',
            'direction': 'up' if (src_ip, src_port) == flow_key[0] else 'down',
            'tls_info': tls_info,
        }

        if flow_key not in flows_dict:
            flows_dict[flow_key] = []
        flows_dict[flow_key].append(flow_pkt)

    return list(flows_dict.values())


def extract_tls_info(tls_layer) -> Optional[Dict]:
    """从scapy TLS层提取信息"""
    try:
        info = {
            'version': getattr(tls_layer, 'version', 0),
            'cipher_suites': [],
            'extensions': [],
            'alpn': [],
            'sni': None,
            'certificate_lengths': [],
            'certificates': [],
        }
        return info
    except Exception:
        return None


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='CryptoFlow - 加密流量分析与异常检测工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python main.py --generate          # 生成模拟数据
    python main.py --train             # 训练模型
    python main.py --analyze           # 分析流量
    python main.py --all               # 完整流程
    python main.py --info              # 查看模型信息
    python main.py --pcap traffic.pcap # 分析PCAP文件
        """
    )

    # 命令
    parser.add_argument('--generate', action='store_true',
                        help='生成模拟加密流量数据')
    parser.add_argument('--train', action='store_true',
                        help='训练分类和异常检测模型')
    parser.add_argument('--analyze', action='store_true',
                        help='分析加密流量')
    parser.add_argument('--all', action='store_true',
                        help='完整流程：生成数据 → 训练 → 分析')
    parser.add_argument('--info', action='store_true',
                        help='查看已训练模型的信息')
    parser.add_argument('--pcap', type=str, metavar='FILE',
                        help='分析PCAP文件')
    parser.add_argument('--live', action='store_true',
                        help='实时捕获分析（需要root权限）')

    # 参数
    parser.add_argument('--samples', type=int, default=500,
                        help='样本数量 (默认: 500)')
    parser.add_argument('--anomaly-ratio', type=float, default=0.15,
                        help='异常流量比例 (默认: 0.15)')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    parser.add_argument('--optimize', action='store_true',
                        help='进行超参数优化（较慢）')
    parser.add_argument('--save', action='store_true',
                        help='保存生成的数据')
    parser.add_argument('--load-data', type=str, metavar='FILE',
                        help='加载已有的训练数据')
    parser.add_argument('--show-stats', action='store_true',
                        help='显示详细统计信息')
    parser.add_argument('--interface', type=str, default='eth0',
                        help='网络接口 (默认: eth0)')
    parser.add_argument('--count', type=int, default=100,
                        help='捕获包数量 (默认: 100)')

    args = parser.parse_args()

    # 如果没有参数，显示帮助
    if len(sys.argv) == 1:
        parser.print_help()
        return

    # 创建模型目录
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 执行命令
    try:
        if args.generate:
            cmd_generate(args)
        elif args.train:
            cmd_train(args)
        elif args.analyze:
            cmd_analyze(args)
        elif args.all:
            cmd_all(args)
        elif args.info:
            cmd_info(args)
        elif args.pcap:
            cmd_pcap(args)
        elif args.live:
            print("⚠️  实时捕获功能需要安装 scapy 和 root 权限")
            print("   请使用: sudo python main.py --live --interface eth0")
        else:
            parser.print_help()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()