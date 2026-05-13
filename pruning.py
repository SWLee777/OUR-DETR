import torch
import torch.nn as nn
import torch.optim as optim
import random
import copy
import time
import numpy as np
import math
import os
import yaml
from pathlib import Path
import datetime
from ultralytics import RTDETR
import re
from collections import OrderedDict
from contextlib import redirect_stdout
import csv

from typing import Optional


class StructuralPruner:
  

    def __init__(self, model_cfg_path: str):
        self.model_cfg_path = model_cfg_path
        self.original_cfg = self.load_yaml_config()
        self.pruned_cfg = copy.deepcopy(self.original_cfg)

    def load_yaml_config(self) -> dict:
        """加载YAML配置文件"""
        with open(self.model_cfg_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)

    def save_yaml_config(self, save_path: str):
        """保存修改后的YAML配置"""
        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.pruned_cfg, f, default_flow_style=False, allow_unicode=True)

    def find_and_update_channels(self, layer_name: str, original_channels: int, pruned_channels: int):
        """在配置文件中查找并更新通道数"""
        print(f"[Config-Prune] Updating {layer_name}: {original_channels} -> {pruned_channels}")
        self._recursive_update_channels(self.pruned_cfg, layer_name, original_channels, pruned_channels)

    def _recursive_update_channels(self, config, layer_name: str, original_channels: int, pruned_channels: int):
        """递归更新配置中的通道数"""
        if isinstance(config, dict):
            for key, value in config.items():
                if isinstance(value, (dict, list)):
                    self._recursive_update_channels(value, layer_name, original_channels, pruned_channels)
                elif isinstance(value, (int, float)) and value == original_channels:
                    if key in ['out_channels', 'in_channels', 'channels', 'c1', 'c2', 'width']:
                        config[key] = pruned_channels
                        print(f"  Updated {key}: {original_channels} -> {pruned_channels}")

        elif isinstance(config, list):
            for i, item in enumerate(config):
                if isinstance(item, (dict, list)):
                    self._recursive_update_channels(item, layer_name, original_channels, pruned_channels)
                elif isinstance(item, (int, float)) and item == original_channels:
                    if i < 5:
                        config[i] = pruned_channels
                        print(f"  Updated index {i}: {original_channels} -> {pruned_channels}")



class RealStructurePrunedRTDETR:

    def __init__(self, model_path: str, cfg_path: str, device: Optional[str] = None):
        self.model_path = model_path
        self.cfg_path = cfg_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # load RTDETR wrapper (may raise if not available)
        self.model = RTDETR(model_path)
        # 将 ultralytics 模型放到 device
        try:
            self.model.model.to(self.device)
        except Exception:
            pass

        self.structural_pruner = StructuralPruner(cfg_path)

        # 收集所有可剪枝的BN层
        self.bn_layers = []
        self._collect_bn_layers()

        # 层敏感性字典（layer_name -> score，score 越大越敏感）
        self.layer_sensitivity = {}

    def _collect_bn_layers(self):
        """收集所有BN层信息"""
        print("[Prune-Init] Collecting BN layers...")

        for name, module in self.model.model.named_modules():
            if isinstance(module, nn.BatchNorm2d) and module.num_features >= 8:
                self.bn_layers.append({
                    'name': name,
                    'module': module,
                    'original_channels': module.num_features,
                    'pruned_channels': module.num_features
                })
                print(f"  Found BN: {name}, channels: {module.num_features}")

        print(f"[Prune-Init] Total BN layers: {len(self.bn_layers)}")

    def compute_layer_sensitivities(self, calib_loader=None, n_batches: int = 8, use_taylor: bool = False):
        """
        计算每个候选 BN 层的敏感性分数
        """
        print("[Sensitivity] Computing layer sensitivities...")

        sens = {}
        
        for layer in self.bn_layers:
            bn = layer['module']
            with torch.no_grad():
                gamma = bn.weight.data.abs().cpu()
                rv = (bn.running_var + 1e-6).sqrt().cpu()
                proxy = (gamma * rv).mean().item()
                sens[layer['name']] = proxy

        
        if use_taylor and calib_loader is not None:
            print("[Sensitivity] Using data-driven first-order Taylor (requires grad).")
            try:
               
                self.model.model.zero_grad()
                self.model.model.eval()
                
                grad_acc = {layer['name']: torch.zeros_like(layer['module'].weight.data.cpu()) for layer in self.bn_layers}

                batches = 0
                for imgs, targets in calib_loader:
                    imgs = imgs.to(self.device)
                    
                    try:
                        outputs = self.model.model(imgs)  

                        if isinstance(outputs, dict) and 'loss' in outputs:
                            loss = outputs['loss']
                        else:
                           
                            if isinstance(outputs, (tuple, list)):
                                loss = sum([o.sum() for o in outputs if torch.is_tensor(o)])
                            elif torch.is_tensor(outputs):
                                loss = outputs.sum()
                            else:
                                
                                raise RuntimeError("Model outputs not tensor-like for Taylor sensitivity.")
                        loss = loss / max(1, len(calib_loader))  
                        loss.backward()

                        # 收集每个 BN 的 weight.grad（first-order sensitivity）
                        for layer in self.bn_layers:
                            bn = layer['module']
                            if bn.weight.grad is not None:
                                grad_acc[layer['name']] += bn.weight.grad.detach().cpu().abs()
                        self.model.model.zero_grad()
                        batches += 1
                        if batches >= n_batches:
                            break
                    except Exception as e:
                       
                        print(f"[Sensitivity] Data-driven Taylor step error: {e}")
                        break

                # 计算 grad-based sensitivity（均值）
                for name in grad_acc:
                    val = grad_acc[name].mean().item()
                    
                    sens[name] = 0.5 * sens.get(name, 0.0) + 0.5 * val

                print(f"[Sensitivity] Data-driven Taylor computed over {batches} batches.")
            except Exception as e:
                print(f"[Sensitivity] Failed to compute Taylor sensitivities, using proxy. Err: {e}")

        # 归一化（0..1）
        vals = np.array(list(sens.values()), dtype=np.float32)
        if vals.size == 0:
            print("[Sensitivity] No BN layers found; skipping sensitivity calculation.")
            return {}
        if vals.max() - vals.min() > 0:
            norm = (vals - vals.min()) / (vals.max() - vals.min())
        else:
            norm = np.zeros_like(vals)
        for i, k in enumerate(list(sens.keys())):
            self.layer_sensitivity[k] = float(norm[i])

        print("[Sensitivity] Done. Example sensitivities (layer:score):")
        for k, v in list(self.layer_sensitivity.items())[:8]:
            print(f"  {k}: {v:.3f}")

        return self.layer_sensitivity

    
    def apply_structural_pruning(self, prune_rates: list = None, global_prune_rate: float = 0.3):
        
        if prune_rates is None:
            prune_rates = [global_prune_rate] * len(self.bn_layers)

        print(f"\n[Structural-Prune] Starting structural pruning with {len(prune_rates)} layer rates")

        total_params_before = sum(p.numel() for p in self.model.model.parameters())
        print(f"[Structural-Prune] Params before: {total_params_before:,}")

        success_count = 0
        total_reduction = 0

        for i, (layer_info, prune_rate) in enumerate(zip(self.bn_layers, prune_rates)):
            try:
                original_channels = layer_info['original_channels']
                prune_rate = max(0.00, min(1.0, prune_rate))  # safe 安全边界
                pruned_channels = max(8, int(original_channels * (1 - prune_rate)))

                if pruned_channels < original_channels:
                    # 1. 剪枝模型权重
                    if self._prune_single_layer(layer_info, pruned_channels):
                        # 2. 更新配置文件
                        self.structural_pruner.find_and_update_channels(
                            layer_info['name'], original_channels, pruned_channels
                        )

                        layer_info['pruned_channels'] = pruned_channels
                        reduction = original_channels - pruned_channels
                        total_reduction += reduction
                        success_count += 1

                        print(
                            f"Layer {i}: {original_channels} → {pruned_channels} channels (-{reduction}, rate={prune_rate:.1%})")
                    else:
                        print(f"Layer {i} pruning failed")
                else:
                    print(f"Layer {i}: skipped (would increase channels)")

            except Exception as e:
                print(f"Layer {i} error: {e}")
                continue

        total_params_after = sum(p.numel() for p in self.model.model.parameters())
        actual_reduction = (total_params_before - total_params_after) / total_params_before if total_params_before>0 else 0.0

        print(f"\n[Structural-Prune] Results:")
        print(f"  Successfully pruned: {success_count}/{len(self.bn_layers)} layers")
        print(f"  Total channels reduced: {total_reduction}")
        print(f"  Params after: {total_params_after:,}")
        print(f"  Actual reduction: {actual_reduction:.1%}")

        return success_count > 0, actual_reduction

    def _prune_single_layer(self, layer_info: dict, pruned_channels: int) -> bool:
        """剪枝单个BN层及其相邻的卷积层"""
        try:
            bn_module = layer_info['module']
            original_channels = layer_info['original_channels']

            if pruned_channels >= original_channels:
                return False

            with torch.no_grad():
                importance = torch.abs(bn_module.weight.data)
                _, keep_indices = torch.topk(importance, k=pruned_channels)
                keep_indices = keep_indices.sort().values


                bn_module.weight.data = bn_module.weight.data[keep_indices].clone()
                bn_module.bias.data = bn_module.bias.data[keep_indices].clone()
                bn_module.running_mean = bn_module.running_mean[keep_indices].clone()
                bn_module.running_var = bn_module.running_var[keep_indices].clone()
                # 更新 num_features
                try:
                    bn_module.num_features = pruned_channels
                except Exception:
                    
                    pass

                # 剪枝相邻 conv
                self._prune_adjacent_convs(layer_info['name'], keep_indices, original_channels, pruned_channels)

            return True

        except Exception as e:
            print(f"Prune layer error: {e}")
            return False

    def _prune_adjacent_convs(self, bn_name: str, keep_indices: torch.Tensor,
                              original_channels: int, pruned_channels: int):
        """剪枝相邻的卷积层"""
        bn_path = bn_name.split('.')

        for name, module in self.model.model.named_modules():
            if isinstance(module, nn.Conv2d):
                module_path = name.split('.')

                if self._are_layers_adjacent(bn_path, module_path):
                    # 修正 out_channels/in_channels 与权重张量维度
                    if hasattr(module, 'out_channels') and module.out_channels == original_channels:
                        try:
                            module.out_channels = pruned_channels
                        except Exception:
                            pass
                        if module.weight.size(0) == original_channels:
                            module.weight.data = module.weight.data[keep_indices, :, :, :].clone()
                        if hasattr(module, 'bias') and module.bias is not None and module.bias.data.numel() == original_channels:
                            module.bias.data = module.bias.data[keep_indices].clone()
                        print(f"    Pruned prev conv: {name}")

                    if hasattr(module, 'in_channels') and module.in_channels == original_channels:
                        try:
                            module.in_channels = pruned_channels
                        except Exception:
                            pass
                        if module.weight.size(1) == original_channels:
                            module.weight.data = module.weight.data[:, keep_indices, :, :].clone()
                        print(f"    Pruned next conv: {name}")

    def _are_layers_adjacent(self, path1: list, path2: list) -> bool:
        """判断两个层是否相邻"""
        if len(path1) != len(path2):
            return False

        if path1[:-1] == path2[:-1]:
            try:
                idx1 = int(path1[-1]) if path1[-1].isdigit() else -1
                idx2 = int(path2[-1]) if path2[-1].isdigit() else -1
                return abs(idx1 - idx2) <= 2
            except Exception:
                return False

        return False

    def save_pruned_model(self, save_path: str, cfg_save_path: str = None, fine_tune_with_kd: bool = False,
                          teacher_model_path: Optional[str] = None, kd_train_loader=None, kd_epochs: int = 3):
        """保存剪枝后的模型和配置文件"""
        if cfg_save_path is None:
            cfg_save_path = save_path.replace('.pt', '.yaml')

        print(f"\n[Saving] Saving pruned model to: {save_path}")
        print(f"[Saving] Saving pruned config to: {cfg_save_path}")

        # 1. 保存剪枝后的YAML配置文件
        self.structural_pruner.save_yaml_config(cfg_save_path)

        # 2. 构建完整的checkpoint
        checkpoint = {
            'model': self.model.model.state_dict(),
            'prune_info': {
                'original_params': sum(p.numel() for p in self.model.model.parameters()),
                'prune_time': datetime.datetime.now().isoformat(),
                'pruned_layers': [
                    {
                        'name': layer['name'],
                        'original_channels': layer['original_channels'],
                        'pruned_channels': layer['pruned_channels']
                    } for layer in self.bn_layers if layer['pruned_channels'] != layer['original_channels']
                ]
            },
            'model_yaml': cfg_save_path,
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        # 3. 保存模型
        torch.save(checkpoint, save_path)

        
        if fine_tune_with_kd and teacher_model_path is not None and kd_train_loader is not None:
            print("[KD] Starting post-prune knowledge distillation fine-tuning...")
            try:
                teacher = RTDETR(teacher_model_path)
                
                try:
                    teacher.model.to(self.device)
                except Exception:
                    pass

               
                self._post_prune_kd_finetune(student_model=self.model.model,
                                             teacher_model=teacher.model,
                                             train_loader=kd_train_loader,
                                             epochs=kd_epochs)
            except Exception as e:
                print(f"[KD] KD fine-tune failed to start: {e}")

        # 4. 验证保存结果
        self._verify_saved_model(save_path, cfg_save_path)

        print("Pruned model and config saved successfully!")

    def _post_prune_kd_finetune(self, student_model, teacher_model, train_loader, epochs=3, lr=1e-4):

        student_model.train()
        teacher_model.eval()

        optimizer = optim.Adam(student_model.parameters(), lr=lr)
        for epoch in range(epochs):
            t0 = time.time()
            for i, (imgs, targets) in enumerate(train_loader):
                imgs = imgs.to(self.device)
               
                with torch.no_grad():
                    try:
                        t_out = teacher_model(imgs)
                    except Exception:
                        t_out = None

                s_out = student_model(imgs)
              
                loss = None
                try:
                    if t_out is not None and torch.is_tensor(t_out) and torch.is_tensor(s_out):
                        
                        loss = nn.MSELoss()(s_out, t_out.detach())
                    else:
                        
                        if isinstance(s_out, dict) and 'loss' in s_out:
                            loss = s_out['loss']
                        else:
                            
                            if torch.is_tensor(s_out):
                                loss = s_out.sum() * 0.0  
                            else:
                                loss = torch.tensor(0.0, requires_grad=True).to(self.device)
                except Exception:
                    loss = torch.tensor(0.0, requires_grad=True).to(self.device)

                optimizer.zero_grad()
                if loss is not None:
                    loss.backward()
                    optimizer.step()

            print(f"[KD] Epoch {epoch+1}/{epochs} done. Time: {time.time()-t0:.1f}s")

    def _verify_saved_model(self, model_path: str, cfg_path: str):
       
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            print(f"Model file verified: {len(checkpoint.get('model', {}))} parameter groups")

            with open(cfg_path, 'r') as f:
                cfg_content = f.read()
            print(f"Config file verified: {len(cfg_content)} characters")

            for layer in self.bn_layers:
                if layer['pruned_channels'] != layer['original_channels']:
                    if str(layer['pruned_channels']) in cfg_content:
                        print(f"Config updated for {layer['name']}: {layer['pruned_channels']}")
                    else:
                        print(f"Config may not be updated for {layer['name']}")
        except Exception as e:
            print(f"Verification failed: {e}")

    def get_model_copy(self):
        """获取模型的深拷贝，用于蜂群评估"""

        try:
            model_copy = RTDETR(self.model_path)
            try:
                model_copy.model.load_state_dict(copy.deepcopy(self.model.model.state_dict()))
            except Exception:
                
                src_state = self.model.model.state_dict()
                tgt_state = model_copy.model.state_dict()
                for k in tgt_state.keys():
                    if k in src_state and src_state[k].shape == tgt_state[k].shape:
                        tgt_state[k] = src_state[k].clone()
                model_copy.model.load_state_dict(tgt_state)
            try:
                model_copy.model.to(self.device)
            except Exception:
                pass
            return model_copy
        except Exception as e:
            print(f"[get_model_copy] Failed to create model copy: {e}")
            
            return self.model

    def evaluate_pruned_model(self, prune_rates: list) -> float:
        """评估剪枝配置的适应度"""
        try:
            
            try:
                
                temp_pruner = RealStructurePrunedRTDETR(self.model_path, self.cfg_path, self.device)
                
                try:
                    temp_pruner.model.model.load_state_dict(copy.deepcopy(self.model.model.state_dict()))
                except Exception:
                    
                    src_state = self.model.model.state_dict()
                    tgt_state = temp_pruner.model.model.state_dict()
                    for k in list(tgt_state.keys()):
                        if k in src_state and src_state[k].shape == tgt_state[k].shape:
                            tgt_state[k] = src_state[k].clone()
                    temp_pruner.model.model.load_state_dict(tgt_state)
                
                temp_pruner.bn_layers = []
                name_to_module = {n: m for n, m in temp_pruner.model.model.named_modules()}
                for layer in self.bn_layers:
                    name = layer['name']
                    if name in name_to_module:
                        m = name_to_module[name]
                        temp_pruner.bn_layers.append({
                            'name': name,
                            'module': m,
                            'original_channels': layer['original_channels'],
                            'pruned_channels': layer['original_channels']
                        })
                
                temp_pruner.layer_sensitivity = copy.deepcopy(self.layer_sensitivity)
            except Exception as e:
                
                try:
                    temp_pruner = copy.deepcopy(self)
                except Exception:
                    print(f"[evaluate_pruned_model] Warning: failed to make temp pruner cleanly: {e}")
                    temp_pruner = self  

            
            success, param_reduction = temp_pruner.apply_structural_pruning(prune_rates=prune_rates)
            if not success:
                return 0.0


            param_score = min(param_reduction * 60, 60)

            if len(prune_rates) > 1:
                rate_std = np.std(prune_rates)
                uniformity_score = max(25 - rate_std * 50, 0)
            else:
                uniformity_score = 25

            # 敏感性惩罚
            sens_penalty = 0.0
            for layer, rate in zip(self.bn_layers, prune_rates):
                lname = layer['name']
                sens = self.layer_sensitivity.get(lname, 0.5)  # 默认为中等敏感
                # 如果对敏感层剪枝率较高，则根据敏感度乘以惩罚
                if rate > 0.4:
                    sens_penalty += sens * (rate - 0.4) * 20.0  # 权重调节因子
            sens_penalty = min(sens_penalty, 10.0)

            # 过度剪枝惩罚（层级）
            over_prune_penalty = 0.0
            for rate in prune_rates:
                if rate > 0.6:
                    over_prune_penalty += (rate - 0.6) * 20
            over_prune_penalty = min(over_prune_penalty, 5.0)

            fitness = param_score + uniformity_score - sens_penalty - over_prune_penalty

            # 添加少量随机噪声
            fitness += random.uniform(-1, 1)
            return max(0.1, min(99.9, fitness))

        except Exception as e:
            print(f"Evaluation error: {e}")
            return 0.1


class BeeGroup:
    def __init__(self):
        self.code = []
        self.fitness = 0.0
        self.rfitness = 0.0
        self.trail = 0

class FullBeeOptimizer:


    def __init__(self, pruner: RealStructurePrunedRTDETR, args=None):
        self.pruner = pruner
        self.num_layers = len(pruner.bn_layers)

        if args is None:
            self.args = self._default_args()
        else:
            self.args = args

        self.best_honey = BeeGroup()
        self.NectraSource = []
        self.EmployedBee = []
        self.OnLooker = []
        self.best_honey_fitness = 0.0
        self.best_honey_code = []

        # 每层的采样上下限（初始化为空，initilize 时计算）
        self.layer_bounds = [(self.args.min_prune_rate, self.args.max_prune_rate)] * self.num_layers

    def _default_args(self):
        class Args:
            def __init__(self):
                self.food_number = 5
                self.food_limit = 3
                self.honeychange_num = 2
                self.max_cycle = 10
                self.min_prune_rate = 0.1
                self.max_prune_rate = 0.6
        return Args()

    def _compute_layer_bounds_from_sensitivity(self):

        bounds = []
        # 读取 sensitivity（0..1）
        sens_map = self.pruner.layer_sensitivity
        # 如果没有 sensitivity 信息，使用统一 bounds
        if not sens_map:
            self.layer_bounds = [(self.args.min_prune_rate, self.args.max_prune_rate)] * self.num_layers
            return self.layer_bounds

        for i, layer in enumerate(self.pruner.bn_layers):
            s = sens_map.get(layer['name'], 0.5)  # 0..1
            # map sensitivity to max_prune: high sensitivity -> lower max prune
            # max_prune = base_max * (1 - alpha*s) where alpha 控制调整幅度
            alpha = 0.96 #best为0.55 #0.7
            base_max = self.args.max_prune_rate
            layer_max = max(self.args.min_prune_rate + 0.01, base_max * (1 - alpha * s))
            # 允许在敏感层下放宽最小剪枝率一点（避免 0）
            layer_min = self.args.min_prune_rate
            bounds.append((layer_min, layer_max))
        self.layer_bounds = bounds
        print("[Bee-Opt] Layer-specific prune bounds computed from sensitivity (first 8):")
        for i, b in enumerate(self.layer_bounds[:8]):
            print(f"  Layer {i}: min={b[0]:.3f}, max={b[1]:.3f}")
        return self.layer_bounds

    def initilize(self):
        """初始化蜜源；"""
        print(f"[Bee-Opt] Initializing bee colony with {self.args.food_number} food sources...")
        # 先确保 pruner 已经计算了 layer_sensitivity（若没有，则调用轻量方法）
        if not self.pruner.layer_sensitivity:
            self.pruner.compute_layer_sensitivities(use_taylor=False)

        self._compute_layer_bounds_from_sensitivity()

        # 初始化蜜源
        for i in range(self.args.food_number):
            nectar = BeeGroup()
            # 每层在对应 bounds 内随机产生剪枝率
            code = []
            for (lo, hi) in self.layer_bounds:
                val = random.uniform(lo, hi)
                code.append(val)
            nectar.code = code
            nectar.fitness = self.pruner.evaluate_pruned_model(nectar.code)
            nectar.rfitness = 0.0
            nectar.trail = 0

            self.NectraSource.append(nectar)
            self.EmployedBee.append(copy.deepcopy(nectar))
            self.OnLooker.append(copy.deepcopy(nectar))

            if nectar.fitness > self.best_honey.fitness:
                self.best_honey = copy.deepcopy(nectar)
                self.best_honey_fitness = nectar.fitness
                self.best_honey_code = copy.deepcopy(nectar.code)

        print(f"[Bee-Opt] Initialization completed. Best initial fitness: {self.best_honey_fitness:.2f}")

    def calculateProbabilities(self):
        maxfit = max([nectar.fitness for nectar in self.NectraSource]) if self.NectraSource else 1.0
        for nectar in self.NectraSource:
            nectar.rfitness = (0.9 * (nectar.fitness / maxfit)) + 0.1

    def sendEmployedBees(self):
        for i in range(self.args.food_number):
            if not self.NectraSource:
                break
            while True:
                k = random.randint(0, self.args.food_number - 1)
                if k != i:
                    break

            new_code = copy.deepcopy(self.NectraSource[i].code)

            # 随机选择维度并用同层 bounds 做变动（保持在 bounds 内）
            param2change = np.random.randint(0, self.num_layers, self.args.honeychange_num)
            R = np.random.uniform(-0.2, 0.2, self.args.honeychange_num)

            for j, idx in enumerate(param2change):
                new_code[idx] = self.NectraSource[i].code[idx] + R[j] * (
                        self.NectraSource[i].code[idx] - self.NectraSource[k].code[idx]
                )
                lo, hi = self.layer_bounds[idx]
                new_code[idx] = max(lo, min(hi, new_code[idx]))

            new_fitness = self.pruner.evaluate_pruned_model(new_code)

            if new_fitness > self.NectraSource[i].fitness:
                self.NectraSource[i].code = new_code
                self.NectraSource[i].fitness = new_fitness
                self.NectraSource[i].trail = 0
                self.EmployedBee[i] = copy.deepcopy(self.NectraSource[i])

                if new_fitness > self.best_honey.fitness:
                    self.best_honey = copy.deepcopy(self.NectraSource[i])
                    self.best_honey_fitness = new_fitness
                    self.best_honey_code = copy.deepcopy(new_code)
                    print(f"[Bee-Opt] Employed bee found better honey: fitness={new_fitness:.2f}")
            else:
                self.NectraSource[i].trail += 1

    def sendOnlookerBees(self):
        i = 0
        t = 0
        while t < self.args.food_number and self.NectraSource:
            R_choosed = random.uniform(0, 1)
            if R_choosed <= self.NectraSource[i].rfitness:
                t += 1
                while True:
                    k = random.randint(0, self.args.food_number - 1)
                    if k != i:
                        break

                new_code = copy.deepcopy(self.NectraSource[i].code)
                param2change = np.random.randint(0, self.num_layers, self.args.honeychange_num)
                R = np.random.uniform(-0.2, 0.2, self.args.honeychange_num)

                for j, idx in enumerate(param2change):
                    new_code[idx] = self.NectraSource[i].code[idx] + R[j] * (
                            self.NectraSource[i].code[idx] - self.NectraSource[k].code[idx]
                    )
                    lo, hi = self.layer_bounds[idx]
                    new_code[idx] = max(lo, min(hi, new_code[idx]))

                new_fitness = self.pruner.evaluate_pruned_model(new_code)
                if new_fitness > self.NectraSource[i].fitness:
                    self.NectraSource[i].code = new_code
                    self.NectraSource[i].fitness = new_fitness
                    self.NectraSource[i].trail = 0
                    self.OnLooker[i] = copy.deepcopy(self.NectraSource[i])

                    if new_fitness > self.best_honey.fitness:
                        self.best_honey = copy.deepcopy(self.NectraSource[i])
                        self.best_honey_fitness = new_fitness
                        self.best_honey_code = copy.deepcopy(new_code)
                        print(f"[Bee-Opt] Onlooker bee found better honey: fitness={new_fitness:.2f}")
                else:
                    self.NectraSource[i].trail += 1

            i += 1
            if i == self.args.food_number:
                i = 0

    def sendScoutBees(self):
        for i in range(self.args.food_number):
            if self.NectraSource[i].trail >= self.args.food_limit:
                # 侦察蜂根据 per-layer bounds 重新初始化
                new_code = [random.uniform(lo, hi) for (lo, hi) in self.layer_bounds]
                new_fitness = self.pruner.evaluate_pruned_model(new_code)

                self.NectraSource[i].code = new_code
                self.NectraSource[i].fitness = new_fitness
                self.NectraSource[i].trail = 0
                self.EmployedBee[i] = copy.deepcopy(self.NectraSource[i])
                self.OnLooker[i] = copy.deepcopy(self.NectraSource[i])

                print(f"[Bee-Opt] Scout bee reinitialized food source {i}: fitness={new_fitness:.2f}")

                if new_fitness > self.best_honey.fitness:
                    self.best_honey = copy.deepcopy(self.NectraSource[i])
                    self.best_honey_fitness = new_fitness
                    self.best_honey_code = copy.deepcopy(new_code)

    def memorizeBestSource(self):
        for nectar in self.NectraSource:
            if nectar.fitness > self.best_honey.fitness:
                self.best_honey = copy.deepcopy(nectar)
                self.best_honey_fitness = nectar.fitness
                self.best_honey_code = copy.deepcopy(nectar.code)

    def optimize(self):
        """执行完整的蜂群优化过程"""
        print("\n" + "=" * 60)
        print("🐝 FULL BEE COLONY OPTIMIZATION STARTED (SENSITIVITY-AWARE)")
        print("=" * 60)
        print(f"[Bee-Opt] Configuration:")
        print(f"  - Number of layers (food dimension): {self.num_layers}")
        print(f"  - Food sources (employed bees): {self.args.food_number}")
        print(f"  - Max cycles: {self.args.max_cycle}")
        print(f"  - Food limit: {self.args.food_limit}")
        print(f"  - Prune rate range: [{self.args.min_prune_rate:.1%}, {self.args.max_prune_rate:.1%}]")
        print("=" * 60)

        start_time = time.time()

        # 初始化
        self.initilize()

        # 迭代搜索
        for cycle in range(self.args.max_cycle):
            cycle_start = time.time()

            self.sendEmployedBees()
            self.calculateProbabilities()
            self.sendOnlookerBees()
            self.memorizeBestSource()
            self.sendScoutBees()

            cycle_time = time.time() - cycle_start
            avg_fitness = np.mean([nectar.fitness for nectar in self.NectraSource]) if self.NectraSource else 0.0
            print(f"\n[Bee-Opt] Cycle {cycle + 1}/{self.args.max_cycle}")
            print(f"  - Average fitness: {avg_fitness:.2f}")
            print(f"  - Best fitness: {self.best_honey_fitness:.2f}")
            print(f"  - Cycle time: {cycle_time:.2f}s")
            if self.best_honey_code:
                print(f"  - Best average prune rate: {np.mean(self.best_honey_code):.1%}")

        total_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("BEE COLONY OPTIMIZATION COMPLETED")
        print("=" * 60)
        print(f"[Bee-Opt] Total optimization time: {total_time:.2f}s")
        print(f"[Bee-Opt] Best fitness score: {self.best_honey_fitness:.2f}")
        if self.best_honey_code:
            print(f"[Bee-Opt] Best prune rates statistics:")
            print(f"  - Average: {np.mean(self.best_honey_code):.1%}")
            print(f"  - Std: {np.std(self.best_honey_code):.1%}")
            print(f"  - Min: {np.min(self.best_honey_code):.1%}")
            print(f"  - Max: {np.max(self.best_honey_code):.1%}")
        print("=" * 60)

        return self.best_honey_code


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))
    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'



def main():
    print("=" * 60)
    print("IMPROVED Structural Pruning for RT-DETR (Bee + Sensitivity )")
    print("=" * 60)

    model_path = "best.pt"
    cfg_path = "OUR-DETR.yaml"
    output_dir = "PrunFile"

    os.makedirs(output_dir, exist_ok=True)

    try:
        print("Loading model and configuration...")
        if not os.path.exists(model_path):
            print("Model file not found, using pretrained RT-DETR")
            model_path = "rtdetr-l.pt"

        pruner = RealStructurePrunedRTDETR(model_path, cfg_path)


        pruner.compute_layer_sensitivities(use_taylor=False)

        # 设置蜂群参数
        class BeeArgs:
            def __init__(self):
                self.food_number = 6
                self.food_limit = 4
                self.honeychange_num = 2
                self.max_cycle = 8
                self.min_prune_rate = 0.00 #0.05
                self.max_prune_rate = 1.0 #0.5

        bee_args = BeeArgs()
        bee_optimizer = FullBeeOptimizer(pruner, bee_args)
        best_prune_rates = bee_optimizer.optimize()

        # 应用结构剪枝（使用蜂群优化得到的每层剪枝率）
        print(f"Applying structural pruning with optimized rates...")
        success, actual_reduction = pruner.apply_structural_pruning(prune_rates=best_prune_rates)

        if not success:
            print("Pruning failed, using conservative pruning with 20% global rate")
            pruner.apply_structural_pruning(global_prune_rate=0.2)

     
        print("Saving pruned model and configuration...")
        model_save_path = os.path.join(output_dir, "pruned_rtdetr_alpha009.pt")
        cfg_save_path = os.path.join(output_dir, "pruned_rtdetr_alpha009.yaml")


        pruner.save_pruned_model(model_save_path, cfg_save_path)

        # 验证结果
        print("Verifying results...")
        verify_pruning_results(pruner, model_save_path, cfg_save_path)

        print("\n" + "=" * 60)
        print("STRUCTURAL PRUNING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Model: {model_save_path}")
        print(f" onfig: {cfg_save_path}")
        print(f"Pruning Summary:")
        print(f"  - Total pruned layers: {sum(1 for rate in best_prune_rates if rate > 0.05)}")
        print(f"  - Average prune rate: {np.mean(best_prune_rates):.1%}")
        print(f"  - Parameter reduction: {actual_reduction:.1%}")
        print("Usage for fine-tuning:")
        print(f"from ultralytics import RTDETR")
        print(f"model = RTDETR('{model_save_path}')")
        print(f"model.train(data='your_dataset.yaml', epochs=100)")

    except Exception as e:
        print(f"PRUNING FAILED: {e}")
        import traceback
        traceback.print_exc()


def verify_pruning_results(pruner, model_path: str, cfg_path: str):
    print("\n" + "=" * 40)
    print("VERIFICATION RESULTS")
    print("=" * 40)

    checkpoint = torch.load(model_path, map_location='cpu')
    prune_info = checkpoint.get('prune_info', {})

    print("Pruning Statistics:")
    pruned_layers = prune_info.get('pruned_layers', [])
    if pruned_layers:
        for layer_info in pruned_layers[:10]:
            reduction = layer_info['original_channels'] - layer_info['pruned_channels']
            reduction_pct = reduction / layer_info['original_channels'] * 100
            print(f"  {layer_info['name']}: {layer_info['original_channels']} → {layer_info['pruned_channels']} "
                  f"(-{reduction}, -{reduction_pct:.1f}%)")
        if len(pruned_layers) > 10:
            print(f"  ... and {len(pruned_layers) - 10} more layers")
    else:
        print("  No layers were pruned")

    with open(cfg_path, 'r') as f:
        cfg_content = f.read()

    original_cfg = pruner.structural_pruner.original_cfg
    pruned_cfg = pruner.structural_pruner.pruned_cfg

    print("Configuration Changes:")
    changes_found = find_config_changes(original_cfg, pruned_cfg)

    if changes_found:
        print("Configuration file has been properly updated!")
    else:
        print("No changes detected in configuration file")

    total_params = sum(p.numel() for p in pruner.model.model.parameters())
    print(f"Final parameter count: {total_params:,}")


def find_config_changes(original: dict, pruned: dict, path: str = "") -> bool:
    changes_found = False

    if isinstance(original, dict) and isinstance(pruned, dict):
        for key in set(original.keys()) | set(pruned.keys()):
            current_path = f"{path}.{key}" if path else key

            if key in original and key in pruned:
                if original[key] != pruned[key]:
                    if isinstance(original[key], (int, float)) and isinstance(pruned[key], (int, float)):
                        if abs(original[key] - pruned[key]) > 1:
                            print(f"{current_path}: {original[key]} → {pruned[key]}")
                            changes_found = True

                if find_config_changes(original[key], pruned[key], current_path):
                    changes_found = True

    elif isinstance(original, list) and isinstance(pruned, list):
        for i, (orig_item, pruned_item) in enumerate(zip(original, pruned)):
            if find_config_changes(orig_item, pruned_item, f"{path}[{i}]"):
                changes_found = True

    return changes_found


if __name__ == "__main__":
    main()