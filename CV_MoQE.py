import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, datasets, models
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
import multiprocessing
import datetime
from torchvision.transforms import RandAugment, AutoAugment, AutoAugmentPolicy
from collections import defaultdict


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


# ====== PyTorch专家包装器 ======
class PyTorchExpert(nn.Module):
    def __init__(self, model_path, num_classes=1000):
        super().__init__()
        self.model = models.resnet50(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state_dict, strict=False)
    def forward(self, x):
        return self.model(x)


# ====== ONNX专家包装器 ======
class ONNXExpert(nn.Module):
    def __init__(self, model_path, add_classifier=False, embedding_dim=2048, num_classes=1000, input_is_uint8=False):
        super().__init__()
        import onnxruntime as ort
        session_options = ort.SessionOptions()
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.enable_mem_pattern = False
        session_options.enable_mem_reuse = False
        session_options.log_severity_level = 3

        providers = [
            ('CUDAExecutionProvider', {'device_id': 0}),
            'CPUExecutionProvider'
        ]
        self.model = ort.InferenceSession(model_path, session_options=session_options, providers=providers)
        self.input_name = self.model.get_inputs()[0].name
        self.output_name = self.model.get_outputs()[0].name
        self.input_is_uint8 = input_is_uint8
        self.dummy_param = nn.Parameter(torch.zeros(1))
        self.add_classifier = add_classifier
        if add_classifier:
            self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        device = x.device
        x_np = x.detach().cpu().numpy()
        if self.input_is_uint8:
            mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
            std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)
            x_np = x_np * std + mean
            x_np = (x_np * 255).clip(0, 255).astype(np.uint8)
        else:
            x_np = x_np.astype(np.float32)
        outputs = self.model.run([self.output_name], {self.input_name: x_np})
        feats = torch.from_numpy(outputs[0]).to(torch.float32).to(device)
        if self.add_classifier:
            feats = feats.view(feats.size(0), -1)  # 自动展平
            logits = self.classifier(feats)
            return logits
        else:
            return feats


class DualGateMoE(nn.Module):
    def __init__(self, expert_paths, feature_dim=2048, num_experts=4, classifier_path=None, 
                 num_classes=1000, onnx_feat_path=None, num_heads=8):
        super().__init__()

        #根据情况自己配置专家数量
        self.experts = nn.ModuleList([

        ])
        
        # 使用多头ResNet14+SE门控网络
        self.gate = MultiHeadResNet8Gate(
            num_experts=num_experts, 
            feature_dim=feature_dim, 
            temperature=1.0,
            num_heads=8  # 8个注意力头
        )

    def forward(self, x_img, x_feat):
        # 获取门控权重（多头门控系统返回详细信息）
        gate_result = self.gate(x_feat, return_logits=True)
        if isinstance(gate_result, tuple) and len(gate_result) == 2:
            gate_weights, gate_info = gate_result
            if isinstance(gate_info, dict):
                gate_logits = gate_info.get('final_logits', gate_info)
            else:
                gate_logits = gate_info
        else:
            gate_weights = gate_result
            gate_logits = gate_weights
        #根据自己的配置自行配置
        # 计算专家输出
        expert_outputs = [

        ]
        expert_outputs = torch.stack(expert_outputs, dim=1)  # [B, num_experts, num_classes]
        gate = gate_weights.unsqueeze(-1)  # [B, num_experts, 1]
        combined = (expert_outputs * gate).sum(dim=1)
        
        # 返回结果（包含多头门控信息用于损失计算）
        gate_metadata = {'gate_logits': gate_logits}
        if isinstance(gate_result, tuple) and len(gate_result) == 2 and isinstance(gate_result[1], dict):
            gate_metadata.update(gate_result[1])
        return combined, gate_weights, gate_metadata


# ====== 联合模型（五层全连接特征提取器 + MoE） ======
class JointModel(nn.Module):
    def __init__(self, prenet, moe):
        super().__init__()
        self.prenet = prenet
        self.moe = moe
    
    def forward(self, x):
        x_feat = self.prenet(x)
        return self.moe(x, x_feat)

class EnhancedMoELoss(nn.Module):
    def __init__(self, num_experts, balance_weight=0.0, diversity_weight=0.0, 
                 entropy_weight=0.0, temperature=1.0, gini_weight=0.0, 
                 concentration_weight=0.0, variance_weight=0.0):
        super().__init__()
        self.num_experts = num_experts
        self.balance_weight = balance_weight
        self.diversity_weight = diversity_weight
        self.entropy_weight = entropy_weight
        self.gini_weight = gini_weight
        self.concentration_weight = concentration_weight
        self.variance_weight = variance_weight
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.0)  # 完全取消标签平滑，最大化准确度重要性
        
        # 动态权重调整参数
        self.min_entropy_threshold = np.log(num_experts) * 0.7  # 最小熵阈值
        self.max_concentration_ratio = 0.6  # 最大专家集中度
        
    def gini_coefficient(self, gate_weights):
        batch_size = gate_weights.size(0)
        gini_coeffs = []
        
        for i in range(batch_size):
            weights = gate_weights[i].sort()[0]  # 排序
            n = len(weights)
            index = torch.arange(1, n + 1, dtype=torch.float32, device=weights.device)
            gini = (2 * (index * weights).sum()) / (n * weights.sum()) - (n + 1) / n
            gini_coeffs.append(gini)
        
        return torch.stack(gini_coeffs).mean()
    
    def concentration_penalty(self, gate_weights):
        max_weights = gate_weights.max(dim=1)[0]  # [batch_size]
        # 如果最大权重超过阈值，施加惩罚
        concentration_penalty = torch.relu(max_weights - self.max_concentration_ratio).mean()
        return concentration_penalty
    
    def variance_regularization(self, gate_weights):
        expert_variance = gate_weights.var(dim=0)  
        min_variance = 0.01  # 最小方差阈值
        variance_penalty = torch.relu(min_variance - expert_variance).sum()
        return variance_penalty
    
    def expert_consistency_loss(self, gate_weights):
        if gate_weights.size(0) > 1:
            diff = gate_weights[1:] - gate_weights[:-1]  # [batch_size-1, num_experts]
            consistency_loss = (diff ** 2).mean()
            return consistency_loss
        return torch.tensor(0.0, device=gate_weights.device)
    
    def adaptive_entropy_loss(self, gate_weights, gate_logits=None):
        """自适应熵损失 - 根据当前熵水平动态调整"""
        if gate_logits is not None:
            gate_probs = F.softmax(gate_logits / self.temperature, dim=1)
        else:
            gate_probs = gate_weights
            
        # 计算当前平均熵
        current_entropy = -(gate_probs * torch.log(gate_probs + 1e-10)).sum(dim=1).mean()
        
        # 如果当前熵低于阈值，增强熵正则化
        entropy_scale = torch.relu(self.min_entropy_threshold - current_entropy) + 1.0
        entropy_loss = -(gate_probs * torch.log(gate_probs + 1e-10)).sum(dim=1).mean()
        
        return entropy_loss * entropy_scale, current_entropy
    
    def forward(self, logits, targets, gate_weights, gate_logits=None, epoch=None):
        device = gate_weights.device
        
        # 主任务损失
        ce_loss = self.ce_loss(logits, targets)
        
        # 1. 负载均衡损失 - 确保专家使用均衡（增强版）
        expert_usage = gate_weights.mean(dim=0)  # [num_experts]
        target_usage = torch.ones_like(expert_usage) / self.num_experts
        
        # 使用更严格的均衡损失
        balance_loss = F.mse_loss(expert_usage, target_usage)
        # 添加L1损失增强均衡性
        balance_l1_loss = F.l1_loss(expert_usage, target_usage)
        total_balance_loss = balance_loss + 0.5 * balance_l1_loss
        
        # 2. 基尼系数损失 - 衡量专家使用的公平性
        gini_loss = self.gini_coefficient(gate_weights)
        
        # 3. 专家集中度惩罚
        concentration_loss = self.concentration_penalty(gate_weights)
        
        # 4. 方差正则化
        variance_loss = self.variance_regularization(gate_weights)
        
        # 5. 自适应熵正则化
        entropy_loss, current_entropy = self.adaptive_entropy_loss(gate_weights, gate_logits)
        
        # 6. 专家一致性损失
        consistency_loss = self.expert_consistency_loss(gate_weights)
        
        # 7. 多样性损失 - 鼓励不同样本使用不同专家组合（增强版）
        diversity_loss = -(gate_weights * torch.log(gate_weights + 1e-10)).sum(dim=1).mean()
        
        # 8. 专家利用率惩罚 - 防止某些专家完全不被使用（增强版）
        min_usage = expert_usage.min()
        max_usage = expert_usage.max()
        usage_imbalance = max_usage - min_usage
        
        # 放宽专家使用限制，确保每个专家至少被使用5%，最多不超过50%
        min_usage_penalty = torch.relu(0.05 - min_usage)
        max_usage_penalty = torch.relu(max_usage - 0.5)
        usage_penalty = min_usage_penalty + max_usage_penalty + 0.05 * usage_imbalance
        
        # 9. 温度自适应调整和激进的权重衰减
        if epoch is not None:
            # 激进的正则化衰减，让准确度快速成为主导
            decay_factor = max(0.1, 1.0 - epoch * 0.15)  # 更激进的正则化衰减
            self.balance_weight *= decay_factor
            self.entropy_weight *= decay_factor
            self.diversity_weight *= decay_factor
            self.gini_weight *= decay_factor
            self.concentration_weight *= decay_factor
            self.variance_weight *= decay_factor
        
        # 总损失计算（准确度主导版：CE损失占90%，正则化占10%）
        # 动态计算权重以确保正则化占比为10%
        accuracy_weight = 1.0  # 准确度权重
        
        # 计算当前CE损失值用于动态调整
        ce_loss_value = ce_loss.item() if ce_loss.item() > 0 else 1.0
        
        # 完全移除正则化，只使用交叉熵损失
        target_regularization_ratio = 0.0
        
        # 不计算正则化损失，直接设为0
        raw_regularization = torch.tensor(0.0, device=ce_loss.device)
        actual_reg_weight = 0.0

        # 当前实现中未使用的正则化项设置为零，防止未定义变量错误
        gate_l2_reg = torch.tensor(0.0, device=ce_loss.device)
        accuracy_bonus = torch.tensor(0.0, device=ce_loss.device)
        
        # 主损失：只使用交叉熵损失，完全不使用正则化
        total_loss = ce_loss
        
        
        return {
            'total_loss': total_loss,
            'ce_loss': ce_loss.item(),
            'balance_loss': total_balance_loss.item(),
            'diversity_loss': diversity_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'gini_loss': gini_loss.item(),
            'concentration_loss': concentration_loss.item(),
            'variance_loss': variance_loss.item(),
            'usage_penalty': usage_penalty.item(),
            'consistency_loss': consistency_loss.item(),
            'current_entropy': current_entropy.item(),
            'gate_l2_reg': gate_l2_reg.item(),
            'accuracy_bonus': accuracy_bonus.item(),
            'expert_usage': expert_usage.detach().cpu().numpy().tolist(),
            'regularization_ratio': (actual_reg_weight * raw_regularization.item()) / total_loss.item() if total_loss.item() > 0 else 0.0,
            'actual_reg_weight': actual_reg_weight,
            'target_reg_ratio': target_regularization_ratio,
            'ce_percentage': (ce_loss.item() / total_loss.item() * 100) if total_loss.item() > 0 else 0.0,
            'reg_percentage': (actual_reg_weight * raw_regularization.item() / total_loss.item() * 100) if total_loss.item() > 0 else 0.0
        }

# ====== 训练与验证函数 ======
def train_one_epoch_joint(model, loader, optimizer, device, scheduler=None, epoch=None):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0
    
    # 直接使用标准交叉熵损失，完全移除所有正则化
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    
    progress_bar = tqdm(loader, desc="训练")
    
    # 梯度累积步数
    accumulation_steps = 2
    optimizer.zero_grad(set_to_none=True)
    
    for batch_idx, (imgs, labels) in enumerate(progress_bar):
        imgs = imgs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        labels = labels.to(device, non_blocking=True)
        
        with torch.amp.autocast('cuda'):
            x_feat = model.prenet(imgs)
            
            # 获取ResNet14门控的输出
            logits, gate_weights, gate_info = model.moe(imgs, x_feat)
            
            # 提取用于损失计算的gate_logits
            gate_logits = gate_info.get('gate_logits', None)
            
            # 直接使用交叉熵损失，不使用任何正则化
            main_loss = criterion(logits, labels)
            
        scaler.scale(main_loss).backward()
        
        # 只在累积完成后更新参数
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(loader):
            scaler.unscale_(optimizer)
            # 分别对不同组件进行梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.moe.gate.parameters(), max_norm=0.5)  # 门控网络更小的梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.prenet.parameters(), max_norm=1.0)   # 特征提取器标准裁剪
            scaler.step(optimizer)
            if scheduler is not None:
                scheduler.step()
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
                
        loss_sum += main_loss.item() * imgs.size(0)
        pred = logits.argmax(1)
        correct += (pred == labels).sum().item()
        total += imgs.size(0)
        
        if batch_idx % 50 == 0:  
            # 只显示交叉熵损失和准确率
            progress_bar.set_description(
                f"训练 [{epoch or 0}] 准确率: {(pred == labels).float().mean().item():.4f} "
                f"交叉熵损失: {main_loss.item():.4f} (无正则化)"
            )
        
    return loss_sum / total, correct / total

def evaluate_joint(model, loader, device):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="验证"):
            imgs = imgs.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            labels = labels.to(device, non_blocking=True)
            outputs = model(imgs)
            # 如果输出是元组，取第一个元素（logits）
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            pred = logits.argmax(1)
            correct += (pred == labels).sum().item()
            total += imgs.size(0)
    return correct / total


# ====== 五层全连接特征提取器 ======
class FiveLayerFCFeatureExtractor(nn.Module):
    def __init__(self, out_dim=2048, pretrained=True):
        super().__init__()
        
        resnet = models.resnet34(pretrained=pretrained)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # 五层全连接特征提取器
        self.fc_layers = nn.Sequential(
            # 第一层：512 -> 1024
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),  # 完全移除dropout
            
            # 第二层：1024 -> 2048
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),  # 完全移除dropout
            
            # 第三层：2048 -> 1536
            nn.Linear(2048, 1536),
            nn.BatchNorm1d(1536),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),  # 完全移除dropout
            
            # 第四层：1536 -> 1024
            nn.Linear(1536, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),  # 完全移除dropout
            
            # 第五层：1024 -> out_dim
            nn.Linear(1024, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0)  # 完全移除dropout
        )
        
        # 初始化全连接层权重
        for m in self.fc_layers.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.pool(x).flatten(1)  # [B, 512]
        
        # 再通过五层全连接进一步提取特征
        x = self.fc_layers(x)  # [B, out_dim]
        return x


# ====== ResNet8+SE门控网络======
class SEModule(nn.Module):
    """Squeeze-and-Excitation模块，用于通道注意力"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class MultiHeadResNet8Gate(nn.Module):
    """Multi-head ResNet-8 gate network, adjusted for lighter backbone."""

    def __init__(self, num_experts=4, feature_dim=2048, temperature=1.0, num_heads=8):
        super().__init__()
        self.num_experts = num_experts
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.tensor(temperature))

        # Five-layer FC (same as before)
        self.fc_layers = nn.Sequential(
            nn.Linear(feature_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),
            nn.Linear(128, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0)
        )

        # Reshape: [B, 16, 8, 8] (16*8*8=1024)
        self.reshape_dim = 16
        self.reshape_hw = 8

        # ResNet-8 backbone: 3 stages, 1 BasicBlock each, channels 16-32-64
        self.inplanes = self.reshape_dim
        self.layer1 = self._make_layer(BasicBlock, 16, 1, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 32, 1, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 64, 1, stride=2)

        # Global pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Multi-head: adjusted for 64-dim output
        self.head_dim = 64 // num_heads  # 8
        assert 64 % num_heads == 0, "Dimension must be divisible by heads"

        # QKV projections
        self.query_proj = nn.Linear(64, 64)
        self.key_proj = nn.Linear(64, 64)
        self.value_proj = nn.Linear(64, 64)

        # Rest of init remains similar
        self.attention_bias = nn.Parameter(torch.randn(num_heads, num_heads) * 0.1)
        initial_attention_weights = torch.tensor([
            0.15,  
            0.10,  
            0.20,  
            0.05,  
            0.18,  
            0.12,  
            0.08,  
            0.12   
        ])
        # 确保初始权重和为1
        initial_attention_weights = initial_attention_weights / initial_attention_weights.sum()
        
        # 使用logits形式，这样可以自动保证softmax后和为1
        self.learnable_attention_logits = nn.Parameter(
            torch.log(initial_attention_weights + 1e-8), requires_grad=True
        )
        
        # 头权重偏好向量，让不同头有不同的专家偏好
        head_preferences = torch.tensor([
            [0.4, 0.3, 0.2, 0.1],  # 头1偏好专家1
            [0.1, 0.4, 0.3, 0.2],  # 头2偏好专家2
            [0.2, 0.1, 0.4, 0.3],  # 头3偏好专家3
            [0.3, 0.2, 0.1, 0.4],  # 头4偏好专家4
            [0.25, 0.25, 0.3, 0.2], # 头5平衡偏好
            [0.2, 0.3, 0.25, 0.25], # 头6平衡偏好
            [0.3, 0.2, 0.25, 0.25], # 头7平衡偏好
            [0.25, 0.3, 0.2, 0.25]  # 头8平衡偏好
        ])
        self.head_preferences = nn.Parameter(head_preferences, requires_grad=True)
        
        # 多头输出融合
        self.multi_head_fusion = nn.Linear(256, 128)
        
        # 每个头独立的门控决策
        self.head_gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.head_dim, 32),
                nn.ReLU(inplace=True),
                nn.Dropout(0.0),  # 完全移除dropout
                nn.Linear(32, num_experts)
            ) for _ in range(num_heads)
        ])
        
        # 多头融合门控（最终决策层）
        self.final_fusion_gate = nn.Sequential(
            nn.Dropout(0.0),  # 完全移除dropout
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.0),  # 完全移除dropout
            nn.Linear(64, num_experts)
        )
        
        # 初始化（更好的初始化策略）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # 对门控网络使用更小的初始化方差
                if 'gate_fc' in str(m):
                    nn.init.normal_(m.weight, 0, 0.005)  # 更小的初始化
                else:
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
        # 特殊初始化注意力投影层，增加多样性
        nn.init.normal_(self.query_proj.weight, 0, 0.02)
        nn.init.normal_(self.key_proj.weight, 0, 0.01)  # Key权重更小，增加稳定性
        nn.init.normal_(self.value_proj.weight, 0, 0.02)
        
        # 偏置初始化为小的随机值
        if self.query_proj.bias is not None:
            nn.init.uniform_(self.query_proj.bias, -0.01, 0.01)
        if self.key_proj.bias is not None:
            nn.init.uniform_(self.key_proj.bias, -0.01, 0.01)
        if self.value_proj.bias is not None:
            nn.init.uniform_(self.value_proj.bias, -0.01, 0.01)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        # 在整个layer后加SE模块
        layer_seq = nn.Sequential(*layers)
        se_module = SEModule(planes * block.expansion)
        return nn.Sequential(layer_seq, se_module)
    def multi_head_attention(self, x):
        """多头注意力计算"""
        batch_size = x.size(0)
        
        # 计算Q, K, V
        Q = self.query_proj(x).view(batch_size, self.num_heads, self.head_dim)  # [B, H, D/H]
        K = self.key_proj(x).view(batch_size, self.num_heads, self.head_dim)
        V = self.value_proj(x).view(batch_size, self.num_heads, self.head_dim)
        
        # 计算注意力分数并添加可学习偏置
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, H]
        scores = scores + self.attention_bias.unsqueeze(0)  # 添加偏置打破对称性
        attention_weights = F.softmax(scores, dim=-1)
        
        # 应用注意力到V
        attended_values = torch.matmul(attention_weights, V)  # [B, H, D/H]
        
        # 重塑并融合多头输出
        multi_head_output = attended_values.view(batch_size, -1)  # [B, 256]
        fused_output = self.multi_head_fusion(multi_head_output)  # [B, 128]
        
        return attended_values, fused_output, attention_weights

    def forward(self, extracted_features, return_logits=False):
        x = self.fc_layers(extracted_features)  # [B, 1024]
        x = x.view(x.size(0), self.reshape_dim, self.reshape_hw, self.reshape_hw)  # [B, 16, 8, 8]
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.flatten(1)  # [B, 256]
        
        # 多头注意力处理
        attended_values, fused_features, attention_weights = self.multi_head_attention(x)
        batch_size = x.size(0)
        head_gate_logits = []
        for i, head_gate in enumerate(self.head_gates):
            head_features = attended_values[:, i, :]  # [B, head_dim]
            head_logits = head_gate(head_features)    # [B, num_experts]
            head_gate_logits.append(head_logits)
        
        head_gate_logits = torch.stack(head_gate_logits, dim=1)  # [B, num_heads, num_experts]
        learnable_attention_weights = F.softmax(self.learnable_attention_logits, dim=0)  # [num_heads]
        
        # 将可学习权重扩展到batch维度
        batch_attention_weights = learnable_attention_weights.unsqueeze(0).expand(batch_size, -1)  # [B, num_heads]
        
        # 使用可学习的注意力权重进行加权融合
        weighted_head_logits = (head_gate_logits * batch_attention_weights.unsqueeze(-1)).sum(dim=1)  # [B, num_experts]
        
        # 路径2：学习型融合门控（神经网络学习权重）
        fusion_gate_logits = self.final_fusion_gate(fused_features)  # [B, num_experts]
        
        # 3. 双路径融合：注意力驱动 + 学习驱动
        alpha, beta = 0.7, 0.3  # 注意力加权为主导，学习融合为辅助
        final_gate_logits = alpha * weighted_head_logits + beta * fusion_gate_logits
        
        # 使用温度调节的softmax
        gate_weights = F.softmax(final_gate_logits / torch.clamp(self.temperature, min=0.1, max=5.0), dim=1)
        
        if return_logits:
            # 返回详细的多头门控信息用于分析
            gate_info = {
                'final_logits': final_gate_logits,
                'weighted_head_logits': weighted_head_logits,
                'fusion_logits': fusion_gate_logits,
                'head_logits': head_gate_logits,
                'attention_weights': attention_weights,
                'attention_weights_norm': batch_attention_weights,  # 使用可学习的注意力权重
                'learnable_attention_weights': learnable_attention_weights,  # 可学习权重
                'head_preferences': self.head_preferences,
                'individual_head_weights': [F.softmax(head_logits, dim=-1).mean(dim=0) for head_logits in head_gate_logits.unbind(dim=1)]
            }
            return gate_weights, gate_info
        return gate_weights

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(" 启动训练 - 设备:", device)
    output_dir = "path/to/output"
    os.makedirs(output_dir, exist_ok=True)
    
    # ====== 加载五层全连接特征提取 ======
    print(" 加载五层全连接特征提取器...")
    feature_dim = 2048
    prenet = FiveLayerFCFeatureExtractor(out_dim=feature_dim, pretrained=True).to(device)
    
    # 加载预训练的五层全连接特征提取器参数
    pretrained_feature_path = os.path.join(output_dir, "five_layer_fc_feature_extractor_best.pth")
    if os.path.exists(pretrained_feature_path):
        print(f" 发现预训练的五层全连接特征提取器参数: {pretrained_feature_path}")
        try:
            # 加载预训练参数
            pretrained_state_dict = torch.load(pretrained_feature_path, map_location='cpu')
            
            # 创建映射字典，将保存的参数名映射到当前模型的参数名
            model_state_dict = prenet.state_dict()
            mapped_state_dict = {}
            
            for key, value in pretrained_state_dict.items():
                # 处理参数名映射
                if key.startswith(('backbone.', 'fc_layers.')):
                    if key in model_state_dict:
                        mapped_state_dict[key] = value
            
            # 加载映射后的参数
            missing_keys, unexpected_keys = prenet.load_state_dict(mapped_state_dict, strict=False)
            
            print(f" 成功加载预训练特征提取器参数: {len(mapped_state_dict)} 个参数")
            
        except Exception as e:
            print(f" 加载预训练参数失败: {e}")
    else:
        print(f" 未找到预训练参数，使用ResNet8的ImageNet预训练权重初始化")
    
    # 让门控网络和特征提取器参数都可训练
    for param in prenet.parameters():
        param.requires_grad = True
        
    # ====== 加载MoE门控网络（ResNet8+SE） ======
    print(" 加载ResNet8+SE门控网络...")
    expert_paths = [
        "path/to/expert1.pth",
        "path/to/expert2.pth",
        "path/to/expert3.pth",
        "path/to/expert4.pth"
    ]
    
    
    # 冻结所有专家参数
    total_expert_params = 0
    frozen_expert_params = 0
    for i, expert in enumerate(moe.experts):
        for param in expert.parameters():
            param.requires_grad = False
            total_expert_params += 1
            frozen_expert_params += 1

    print(f" 专家参数冻结完成: {frozen_expert_params} 个参数已冻结")

    # 让门控网络和特征提取器参数都可训练
    gate_params = sum(p.numel() for p in moe.gate.parameters() if p.requires_grad)
    prenet_params = sum(p.numel() for p in prenet.parameters() if p.requires_grad)
    print(f" 可训练参数: ResNet14+SE门控网络 {gate_params:,} + 特征提取器 {prenet_params:,} = {gate_params + prenet_params:,}")

    for param in moe.gate.parameters():
        param.requires_grad = True
    for param in prenet.parameters():
        param.requires_grad = True
    
    joint_model = JointModel(prenet, moe).to(device, memory_format=torch.channels_last)
    print(" 联合模型构建完成")
    
    # 模型参数统计
    joint_total = sum(p.numel() for p in joint_model.parameters())
    joint_trainable = sum(p.numel() for p in joint_model.parameters() if p.requires_grad)
    joint_frozen = joint_total - joint_trainable
    
    print(f" 模型参数: 总计 {joint_total:,}, 可训练 {joint_trainable:,} ({joint_trainable/joint_total*100:.1f}%), 冻结 {joint_frozen:,}")
    
    # ====== 数据加载 ======
    data_dir = input("请输入ImageNet数据集根目录: ") or "path/to/imagenet"
    # 增强的数据增强策略，提升模型泛化能力
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # 更保守的裁剪
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # 颜色增强
        transforms.RandomRotation(degrees=15),  # 轻微旋转
        transforms.RandomGrayscale(p=0.1),  # 随机灰度
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.33), ratio=(0.3, 3.3))  # 随机擦除
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_set = datasets.ImageFolder(os.path.join(data_dir, "train"), transform=train_transform)
    val_set = datasets.ImageFolder(os.path.join(data_dir, "val"), transform=val_transform)

    print(f" 数据集: 训练 {len(train_set):,} 样本, 验证 {len(val_set):,} 样本")
    cpu_count = max(4, min(multiprocessing.cpu_count() - 2, 32))
    num_workers = cpu_count
    prefetch_factor = max(2, min(8, 32 // num_workers))
    train_loader = torch.utils.data.DataLoader(
        train_set,  
        batch_size=48,  
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=64, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor
    )
    patience = 7  
    best_acc = 0.0
    epochs_no_improve = 0
    
    # 记录每轮的验证准确率和训练损失
    last_val_acc = 0.0
    avg_loss_history = []
    avg_train_acc_history = []
    
    print(" 损失函数配置：准确度主导模式（10%正则化）")
    print("   - 准确度损失：90% 权重（无标签平滑）") 
    print("   - 正则化损失：10% 权重（防止门控崩溃）")
    print("   - 动态权重调整：确保正则化占比恒定为10%")
    print("   - 正则化权重将随训练逐渐衰减")
    print()
    
    for epoch in range(1, 21):  
        best_model_path = os.path.join(output_dir, "joint_model_v4_best.pth")
        print(f"训练第{epoch}/20轮...")
        
        # 动态学习率策略，优化不同组件
        if epoch <= 5:  # 前5轮使用较小学习率预热
            prenet_lr = 5e-6  
            gate_lr = 1e-5   
        elif epoch <= 10:  # 中期使用标准学习率
            prenet_lr = 1e-5  
            gate_lr = 2e-5   
        else:  # 后期进一步微调
            prenet_lr = 5e-6  
            gate_lr = 1e-5
            
        # 使用AdamW优化器，更好的权重衰减
        from torch.optim import AdamW
        optimizer = AdamW([
            {'params': moe.gate.parameters(), 'lr': gate_lr, 'weight_decay': 0.0},
            {'params': prenet.parameters(), 'lr': prenet_lr, 'weight_decay': 0.0}  # 完全移除权重衰减
        ], eps=1e-8, betas=(0.9, 0.999))
        
        steps_per_epoch = len(train_loader) 
        # 使用余弦退火学习率调度器，更平滑的学习率变化
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=steps_per_epoch,  # 每个epoch内完成一个余弦周期
            eta_min=gate_lr * 0.01  # 最小学习率为初始值的1%
        )
        
        train_loss, train_acc = train_one_epoch_joint(joint_model, train_loader, optimizer, device, scheduler, epoch)
        val_acc = evaluate_joint(joint_model, val_loader, device)
        print(f'[Epoch {epoch}] 准确率优先 训练准确率{train_acc*100:.2f}% 验证准确率{val_acc*100:.2f}% 训练损失{train_loss:.4f}')

        # ====== 检查门控系统分配的四个专家概率 ======
        joint_model.eval()
        gate_probs = []
        attention_weights_list = []
        
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(device)
                x_feat = joint_model.prenet(imgs)
                
                # 获取多头门控的输出
                gate_result = joint_model.moe.gate(x_feat, return_logits=True)
                if isinstance(gate_result, tuple) and len(gate_result) == 2:
                    gate_weights, gate_info = gate_result
                    if isinstance(gate_info, dict) and 'attention_weights' in gate_info:
                        attention_weights_list.append(gate_info['attention_weights'].cpu())
                else:
                    gate_weights = gate_result
                gate_probs.append(gate_weights.cpu())
                
        gate_probs = torch.cat(gate_probs, dim=0)  # [N, 4]
        avg_probs = gate_probs.mean(dim=0)
        print(f'[Epoch {epoch}] 门控分配四个专家的平均概率: {avg_probs.tolist()}')
        
        # 显示多头注意力和门控统计信息
        if attention_weights_list:
            attention_weights = torch.cat(attention_weights_list, dim=0)  # [N, num_heads, num_heads]
            avg_attention = attention_weights.mean(dim=0)  # [num_heads, num_heads]
            print(f'[Epoch {epoch}] 多头注意力对角线权重: {avg_attention.diag().tolist()[:4]}')
            
            # 显示各个头的门控偏好和注意力权重
            gate_result = joint_model.moe.gate(x_feat[:8], return_logits=True)  # 只取前8个样本
            if isinstance(gate_result, tuple) and len(gate_result) == 2:
                _, gate_info = gate_result
                if 'individual_head_weights' in gate_info:
                    print(f'[Epoch {epoch}] 多头门控详细信息:')
                    for i, head_weights in enumerate(gate_info['individual_head_weights'][:4]):  # 只显示前4个头
                        print(f'  头{i+1}: {[f"{w:.3f}" for w in head_weights.tolist()]}')
                    
                    # 显示可学习的注意力权重分布
                    if 'learnable_attention_weights' in gate_info:
                        learnable_weights = gate_info['learnable_attention_weights'][:4]  # 只显示前4个头
                        print(f'  可学习注意力权重: {[f"{w:.3f}" for w in learnable_weights.tolist()]}')

        if val_acc > best_acc:
            print(f"发现新的最佳准确率: {val_acc*100:.4f}% (之前: {best_acc*100:.4f}%)")
            best_acc = val_acc
            epochs_no_improve = 0
            
            try:
                # 创建备份目录
                backup_dir = os.path.join(output_dir, "best_model_backup")
                os.makedirs(backup_dir, exist_ok=True)
                
                # 保存模型状态字典（只保存可训练的参数）
                moe_save_path = os.path.join(output_dir, "moe_v4_enhanced_best.pth")
                joint_save_path = os.path.join(output_dir, "joint_model_v4_best.pth")
                
                # 保存MoE模型（包含门控网络和专家网络的完整状态）
                # 虽然专家网络被冻结，但保存完整状态便于后续加载
                torch.save(moe.state_dict(), moe_save_path)
                
                # 保存完整的联合模型
                torch.save(joint_model.state_dict(), joint_save_path)
                
                # 保存特征提取器参数
                prenet_save_path = os.path.join(output_dir, "feature_extractor_v4_best.pth")
                torch.save(prenet.state_dict(), prenet_save_path)
                
                # 保存门控网络参数
                gate_save_path = os.path.join(output_dir, "gate_network_v4_best.pth")
                torch.save(moe.gate.state_dict(), gate_save_path)
                
                # 保存详细的训练状态信息
                training_state = {
                    'epoch': epoch,
                    'best_acc': best_acc,
                    'train_acc': train_acc,
                    'train_loss': train_loss,
                    'gate_probs': avg_probs.tolist(),
                    'model_state_dict': joint_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'model_config': {
                        'feature_dim': feature_dim,
                        'num_experts': 4,
                        'temperature': float(moe.gate.temperature.item()) if hasattr(moe.gate, 'temperature') else 1.0,
                        'gate_type': 'ResNet14_SE'
                    },
                    'training_config': {
                        'prenet_lr': prenet_lr,
                        'gate_lr': gate_lr,
                        'batch_size': 48,
                        'weight_decay': 0.0
                    }
                }
                training_state_path = os.path.join(output_dir, "training_state_v4_best.pth")
                torch.save(training_state, training_state_path)
                
                # 保存备份副本
                import shutil
                backup_files = [
                    (joint_save_path, f"joint_model_epoch_{epoch}_acc_{val_acc*100:.2f}.pth"),
                    (training_state_path, f"training_state_epoch_{epoch}_acc_{val_acc*100:.2f}.pth")
                ]
                
                for src, backup_name in backup_files:
                    if os.path.exists(src):
                        backup_path = os.path.join(backup_dir, backup_name)
                        shutil.copy2(src, backup_path)

                print(f" 最佳模型已保存 (准确率: {val_acc*100:.4f}%)")
                
            except Exception as e:
                print(f" 保存模型时出错: {e}")
                print("继续训练...")
            

        else:
            epochs_no_improve += 1
            print(f"[EarlyStopping] 验证集准确率连续{epochs_no_improve}轮未提升")
            
            # 动态调整策略：准确率停滞时进一步降低正则化，强化准确度重要性
            if epochs_no_improve == 2:
                print(f" 准确率停滞，进一步降低正则化权重，强化准确度重要性")
                # 这里可以在下一轮训练中使用更低的正则化权重
                
            # 学习率调整策略：如果连续2轮没有提升，减少学习率
            if epochs_no_improve == 2:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.7  # 更温和的学习率衰减
                print(f" 学习率已调整为原来的70%，继续优化准确率")
            elif epochs_no_improve == 4:
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 0.5  # 更大的衰减
                print(f" 学习率已减半，最后冲刺阶段")
                
            if epochs_no_improve >= patience:
                print(f"[EarlyStopping] 验证集准确率连续{patience}轮未提升，提前终止训练！")
                break
        
        # 记录每轮的验证准确率和训练损失
        avg_loss_history.append(train_loss)
        avg_train_acc_history.append(train_acc)
    
    print(f"训练完成，最佳验证准确率: {best_acc*100:.2f}%")
    
    print(f" 训练完成! 最佳准确率: {best_acc*100:.4f}%")
    print(f" 模型文件保存在: {output_dir}")
    
    # 创建模型加载脚本
    load_script_path = os.path.join(output_dir, "load_best_model.py")

    
    print(f" 模型加载脚本: {load_script_path}")


if __name__ == "__main__":
    main() 
