import torch
import torch.nn.functional as F
import sys
import os
import numpy as np
import types

# ======================================================================
# [0] 终极重构器: 解决 HuggingFace 定长占位符报错，同时严格维持 K 个信息瓶颈
# ======================================================================
def reconstruct_to_original_length(original_patches, compressed_patches):
    sim = F.cosine_similarity(original_patches.unsqueeze(2), compressed_patches.unsqueeze(1), dim=-1)
    nearest_idx = torch.argmax(sim, dim=-1)
    expanded_idx = nearest_idx.unsqueeze(-1).expand(-1, -1, compressed_patches.shape[-1])
    return torch.gather(compressed_patches, 1, expanded_idx)

# ======================================================================
# [1] FlowCut 纯血核心逻辑 (Information Flow Scoring)
# ======================================================================
outputs_dict_fc = {}
def hook_fc_v(module, input, output): outputs_dict_fc['value'] = output

def patch_flowcut_official(vision_model):
    if hasattr(vision_model, "is_flowcut_patched"): return
    original_forward = vision_model.forward
    
    def flowcut_forward(self, pixel_values, output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
        K = getattr(self, 'current_K', 576)
        if K >= 576 or K <= 0:
            return original_forward(pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, **kwargs)

        # 挂载 Hook 以获取原作者需要的 Value 矩阵
        layer_idx = -2
        hook_handle_v = self.encoder.layers[layer_idx].self_attn.v_proj.register_forward_hook(hook_fc_v)
        out = original_forward(pixel_values, output_attentions=True, output_hidden_states=True, return_dict=True, **kwargs)
        hook_handle_v.remove()

        image_features = out.hidden_states[-2]
        cls_token = image_features[:, 0:1]
        original_patch_features = image_features[:, 1:]
        B, N, C = original_patch_features.shape

        # 1. Relation Attention (cls_attn)
        attn_weights = out.attentions[-2]
        cls_attn = attn_weights[:, :, 0, 1:].mean(dim=1) # [B, 576]

        # 2. 还原原作者的 Semantic Attention 与 Value Metric
        value_output = outputs_dict_fc['value'] # [B, 577, embed_dim]
        num_heads = self.encoder.layers[layer_idx].self_attn.num_heads
        head_dim = C // num_heads

        value_states = value_output.view(B, N + 1, num_heads, head_dim).transpose(1, 2) # [B, num_heads, 577, head_dim]
        cls_value = value_states[:, :, 0:1, :] # [B, num_heads, 1, head_dim]
        
        semantic_weight = torch.matmul(cls_value, value_states.transpose(-1, -2)) # [B, num_heads, 1, 577]
        semantic_attn = F.softmax(semantic_weight, dim=-1).squeeze(2).mean(dim=1)[:, 1:] # [B, 576]

        value_metric = value_states.mean(dim=1)[:, 1:, :] # [B, 576, head_dim]

        # 3. FlowCut 核心打分公式 (完全一字不差复现官方公式)
        relation_score = cls_attn / (cls_attn.sum(dim=-1, keepdim=True) + 1e-8)
        semantic_score = semantic_attn / (semantic_attn.sum(dim=-1, keepdim=True) + 1e-8)
        final_score = (relation_score + semantic_score) * torch.norm(value_metric, p=1, dim=-1)

        # 4. 根据分数提取 Top-K
        _, keep_indices = torch.topk(final_score, int(K), dim=1)
        keep_indices = keep_indices.sort(dim=1).values

        pruned_features = []
        for b in range(B):
            pruned_features.append(original_patch_features[b, keep_indices[b]])
        compressed_patch_features = torch.stack(pruned_features, dim=0)

        # 5. 完美重构至 576 长度，绕过 HF 报错
        reconstructed_features = reconstruct_to_original_length(original_patch_features, compressed_patch_features)

        new_hidden_states = list(out.hidden_states)
        new_hidden_states[-2] = torch.cat([cls_token, reconstructed_features], dim=1)
        out['hidden_states'] = tuple(new_hidden_states)
        return out

    vision_model.forward = types.MethodType(flowcut_forward, vision_model)
    vision_model.is_flowcut_patched = True

# ======================================================================
# [2] PruMerge 官方源码
# ======================================================================
def complement_idx(idx, dim):
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim): a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl

outputs_dict = {}
def hook_k(module, input, output): outputs_dict['desired_k'] = output
def hook_q(module, input, output): outputs_dict['desired_q'] = output

def patch_prumerge_official(vision_model):
    if hasattr(vision_model, "is_prumerge_patched"): return
    original_forward = vision_model.forward
    
    def prumerge_forward(self, pixel_values, output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
        K = getattr(self, 'current_K', 576)
        if K >= 576 or K <= 0:
            return original_forward(pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, **kwargs)
        
        hook_handle_k = self.encoder.layers[23].self_attn.k_proj.register_forward_hook(hook_k)
        hook_handle_q = self.encoder.layers[23].self_attn.q_proj.register_forward_hook(hook_q)
        out = original_forward(pixel_values, output_attentions=output_attentions, output_hidden_states=True, return_dict=True, **kwargs)
        hook_handle_k.remove()
        hook_handle_q.remove()
        
        image_features = out.hidden_states[-2] 
        cls_token_last_layer = image_features[:, 0:1]
        original_patch_features = image_features[:, 1:] 
        B, N, C = original_patch_features.shape

        desired_layer_k = outputs_dict["desired_k"]
        desired_layer_q = outputs_dict["desired_q"]

        attn = (desired_layer_q @ desired_layer_k.transpose(-2, -1)) * C ** -0.5
        attn = F.softmax(attn, dim=-1)
        cls_attn = attn[:, 0, 1:]  

        target_k = int(K)
        if target_k >= N: target_k = N // 2
        top_k_num = target_k // 2
        spatial_num = target_k - top_k_num
        
        _, topk_idx = torch.topk(cls_attn, top_k_num, dim=1, largest=True)  
        
        idx_list = []
        for b in range(B):
            cur_topk = topk_idx[b]
            all_idx = torch.arange(N, device=next(self.parameters()).device)
            mask = ~torch.isin(all_idx, cur_topk)
            remain_idx = all_idx[mask]
            if len(remain_idx) >= spatial_num:
                sample_pos = torch.linspace(0, len(remain_idx)-1, steps=spatial_num).long()
                sampled = remain_idx[sample_pos]
            else:
                sampled = remain_idx
            final_idx = torch.cat((cur_topk, sampled), dim=0)
            idx_list.append(final_idx)
            
        idx = torch.stack(idx_list, dim=0)
        index = idx.unsqueeze(-1).expand(-1, -1, C)  
        Key_wo_cls = desired_layer_k[:, 1:]  
        x_others = torch.gather(original_patch_features, dim=1, index=index)  
        x_others_attn = torch.gather(cls_attn, dim=1, index=idx)  
        Key_others = torch.gather(Key_wo_cls, dim=1, index=index)  
        compl = complement_idx(idx, N)  
        non_topk = torch.gather(original_patch_features, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  
        non_topk_Key = torch.gather(Key_wo_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))
        non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  

        Key_others_norm = F.normalize(Key_others, p=2, dim=-1)
        non_topk_Key_norm = F.normalize(non_topk_Key, p=2, dim=-1)

        B, left_tokens, C = x_others.size()
        updated_x_others = torch.zeros_like(x_others)

        for b in range(B):
            for i in range(left_tokens):
                key_others_norm = Key_others_norm[b,i,:].unsqueeze(0).unsqueeze(0)
                before_i_Key = Key_others_norm[b, :i, :].unsqueeze(0)  
                after_i_Key = Key_others_norm[b, i+1:, :].unsqueeze(0) 
                before_i_x_others = x_others[b, :i, :].unsqueeze(0)  
                after_i_x_others = x_others[b, i+1:, :].unsqueeze(0)   
                rest_x_others = torch.cat([before_i_x_others, after_i_x_others, non_topk[b,:,:].unsqueeze(0)], dim=1)   
                before_i_x_others_attn = x_others_attn[b, :i].unsqueeze(0)  
                after_i_x_others_attn = x_others_attn[b, i+1:].unsqueeze(0)  
                rest_x_others_attn = torch.cat([before_i_x_others_attn, after_i_x_others_attn, non_topk_attn[b,:].unsqueeze(0)], dim=1)  

                rest_Keys = torch.cat([before_i_Key, after_i_Key, non_topk_Key_norm[b,:,:].unsqueeze(0)], dim=1)
                cos_sim_matrix = torch.bmm(key_others_norm, rest_Keys.transpose(1, 2))
                _, cluster_indices = torch.topk(cos_sim_matrix, k=int(32), dim=2, largest=True)

                cluster_tokens = rest_x_others[:,cluster_indices.squeeze(),:]
                weights = rest_x_others_attn[:,cluster_indices.squeeze()].unsqueeze(-1)
                weighted_avg = torch.sum(cluster_tokens * weights, dim=1)
                updated_center = x_others[b, i, :]  + weighted_avg 
                updated_x_others[b, i, :] = updated_center 
        
        # ======================================================================
        # 【神级修复核心】：分离匹配与赋值，杜绝特征漂移导致的空间马赛克
        # ======================================================================
        # 1. 使用未被污染的原始保留特征 (x_others) 进行精准的余弦寻址
        sim = F.cosine_similarity(original_patch_features.unsqueeze(2), x_others.unsqueeze(1), dim=-1)
        nearest_idx = torch.argmax(sim, dim=-1)
        expanded_idx = nearest_idx.unsqueeze(-1).expand(-1, -1, C)
        
        # 2. 找到位置后，把融合加强后的特征 (updated_x_others) 填进去
        reconstructed_features = torch.gather(updated_x_others, 1, expanded_idx)

        new_hidden_states = list(out.hidden_states)
        new_hidden_states[-2] = torch.cat([cls_token_last_layer, reconstructed_features], dim=1)
        out['hidden_states'] = tuple(new_hidden_states)
        return out
    
    vision_model.forward = types.MethodType(prumerge_forward, vision_model)
    vision_model.is_prumerge_patched = True


# ======================================================================
# [3] VisionZIP 官方源码
# ======================================================================
def patch_visionzip_official(vision_model):
    if hasattr(vision_model, "is_visionzip_patched"): return
    original_forward = vision_model.forward
    
    def visionzip_forward(self, pixel_values, output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
        K = getattr(self, 'current_K', 576)
        if K >= 576 or K <= 0:
            return original_forward(pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, **kwargs)
        
        out = original_forward(pixel_values, output_attentions=True, output_hidden_states=True, return_dict=True, **kwargs)
        attn_weights  = out.attentions[-2]
        hidden_states = out.hidden_states[-2]
        metric = hidden_states
        original_patch_features = hidden_states[:, 1:]
        
        dominant_num = int(K * 0.75) - 1 
        if dominant_num <= 0: dominant_num = 1
        contextual_num = K - dominant_num - 1

        cls_idx = 0
        cls_attention = attn_weights[:, :, cls_idx, cls_idx+1:]  
        cls_attention_sum = cls_attention.sum(dim=1)  
        topk_indices = cls_attention_sum.topk(dominant_num, dim=1).indices + 1
        all_indices = torch.cat([torch.zeros((hidden_states.shape[0], 1), dtype=topk_indices.dtype, device=topk_indices.device), topk_indices], dim=1)
        
        mask = torch.ones_like(hidden_states[:, :, 0], dtype=torch.bool, device=metric.device).scatter_(1, all_indices, False)
        dominant_tokens = hidden_states.masked_select(~mask.unsqueeze(-1)).view(hidden_states.shape[0], dominant_num + 1, hidden_states.shape[2])
        
        metric_filtered = metric[mask].view(hidden_states.shape[0], hidden_states.shape[1] - (dominant_num + 1), metric.shape[2])
        hidden_states_filtered = hidden_states.masked_select(mask.unsqueeze(-1)).view(hidden_states.shape[0], hidden_states.shape[1] - (dominant_num +1), hidden_states.shape[2])  
        metric_normalized = metric_filtered / metric_filtered.norm(dim=-1, keepdim=True) 

        if contextual_num > 0:
            step = max(1, metric_normalized.shape[1] // contextual_num)
            target_indices = torch.arange(0, metric_normalized.shape[1], step, device=metric_normalized.device)[:contextual_num]
            target_tokens = metric_normalized[:, target_indices, :]

            tokens_to_merge = metric_normalized[:, ~torch.isin(torch.arange(metric_normalized.shape[1], device=metric_normalized.device), target_indices), :]
            similarity = torch.bmm(tokens_to_merge, target_tokens.transpose(1, 2))
            assign_one_hot = torch.zeros(tokens_to_merge.shape[0], tokens_to_merge.shape[1], contextual_num, dtype=hidden_states_filtered.dtype, device=metric_normalized.device)
            assign_one_hot.scatter_(2, similarity.argmax(dim=2).unsqueeze(-1), 1)
            counts = assign_one_hot.sum(dim=1).clamp(min=1).unsqueeze(-1)
            hidden_to_merge = hidden_states_filtered[:, ~torch.isin(torch.arange(hidden_states_filtered.shape[1], device=hidden_states_filtered.device), target_indices), :]
            aggregated_hidden = torch.bmm(assign_one_hot.transpose(1, 2), hidden_to_merge) / counts
            target_hidden = hidden_states_filtered[:, target_indices, :]  
            
            contextual_tokens = target_hidden + aggregated_hidden
            compressed_patch_features = torch.cat([dominant_tokens[:, 1:], contextual_tokens], dim=1)
        else:
            compressed_patch_features = dominant_tokens[:, 1:]
            
        reconstructed_features = reconstruct_to_original_length(original_patch_features, compressed_patch_features)
        
        new_hidden_states = list(out.hidden_states)
        new_hidden_states[-2] = torch.cat([hidden_states[:, 0:1], reconstructed_features], dim=1)
        out['hidden_states'] = tuple(new_hidden_states)
        return out

    vision_model.forward = types.MethodType(visionzip_forward, vision_model)
    vision_model.is_visionzip_patched = True


# ======================================================================
# [4] VisPruner 官方源码
# ======================================================================
def patch_vispruner_official(vision_model):
    if hasattr(vision_model, "is_vispruner_patched"): return
    original_forward = vision_model.forward
    
    def vispruner_forward(self, pixel_values, output_attentions=None, output_hidden_states=None, return_dict=None, **kwargs):
        K = getattr(self, 'current_K', 576)
        if K >= 576 or K <= 0:
            return original_forward(pixel_values, output_attentions=output_attentions, output_hidden_states=output_hidden_states, return_dict=return_dict, **kwargs)
            
        out = original_forward(pixel_values, output_attentions=True, output_hidden_states=True, return_dict=True, **kwargs)
        
        image_features = out.hidden_states[-2]
        cls_token = image_features[:, 0:1]
        original_patch_features = image_features[:, 1:] 
        
        image_attentions = out.attentions[-2][:, :, 0, 1:].mean(dim=1) 
        features_normalized = original_patch_features / original_patch_features.norm(dim=-1, keepdim=True)
        B, N = original_patch_features.shape[:2]
        
        important_token_num = int(K * 0.5) 
        if important_token_num <= 0: important_token_num = 1
        diverse_token_num = K - important_token_num 

        token_indices = image_attentions.argsort(dim=-1, descending=True)
        important_indices = token_indices[:, :important_token_num]
        residual_indices = token_indices[:, important_token_num:]

        while True:
            residual_tokens = features_normalized[torch.arange(B), residual_indices]
            r = min(8, residual_tokens.shape[1] - diverse_token_num)
            if r <= 0: break

            a, b = residual_tokens[..., ::2, :], residual_tokens[..., 1::2, :]
            scores = a @ b.transpose(-1, -2)
            scores = scores.max(dim=-1).values

            distinct_indices = scores.argsort(dim=-1, descending=True)[:, r:]
            residual_indices = torch.cat([residual_indices[..., ::2][torch.arange(B), distinct_indices], residual_indices[..., 1::2]], dim=-1)

        token_indices = torch.cat([important_indices, residual_indices], dim=-1)
        token_indices = torch.sort(token_indices).values
        
        pruned_features = []
        for b in range(B): pruned_features.append(original_patch_features[b, token_indices[b]])
        compressed_patch_features = torch.stack(pruned_features, dim=0)

        reconstructed_features = reconstruct_to_original_length(original_patch_features, compressed_patch_features)

        new_hidden_states = list(out.hidden_states)
        new_hidden_states[-2] = torch.cat([cls_token, reconstructed_features], dim=1)
        out['hidden_states'] = tuple(new_hidden_states)
        return out
        
    vision_model.forward = types.MethodType(vispruner_forward, vision_model)
    vision_model.is_vispruner_patched = True

# ======================================================================
# 统一路由总控 (Router)
# ======================================================================
def apply_compression_patch(vision_model, method_name):
    paper_methods = {"VisionZIP", "VisPruner", "PruMerge", "FlowCut"}
    if method_name not in paper_methods:
        raise ValueError(
            f"{method_name!r} is not in the final four-compressor paper protocol; "
            f"choose one of {sorted(paper_methods)}"
        )
    if method_name == "PruMerge":
        patch_prumerge_official(vision_model)
    elif method_name == "VisionZIP":
        patch_visionzip_official(vision_model)
    elif method_name == "VisPruner":
        patch_vispruner_official(vision_model)
    elif method_name == "FlowCut":
        patch_flowcut_official(vision_model)
