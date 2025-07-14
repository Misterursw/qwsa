#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import torch
import torch.nn as nn
from transformers import Qwen2_5_VLForConditionalGeneration, AutoConfig

class QwenMetaModel(nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        
        # self.vision_tower = None  
        # self.mm_projector = None
        
        if hasattr(config, "use_vision") and config.use_vision:
            self.vision_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                config.model_name,
                torch_dtype="auto",
                device_map="auto"
            ).vision_model

    def get_vision_tower(self):
        """Qwen的视觉编码器"""
        return self.vision_model if hasattr(self, 'vision_model') else None

    def encode_multimodal(self, text_ids, images, attention_mask):
        """使用 Qwen 原生方法融合多模态输入"""
        return self.vision_model.encode_multimodal(
            text_ids=text_ids,
            images=images,
            attention_mask=attention_mask
        )

class QwenMetaForCausalLM(nn.Module):
    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen2_5_VLForConditionalGeneration(config)
        
    @classmethod
    def from_config(cls, config, **kwargs):
        return cls(config)  # 必须显式定义

    def get_model(self):
        return self.model
    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    def encode_images(self, images):
        return self.model.vision_model(images).last_hidden_state

    def prepare_inputs_for_multimodal(
        self,
        input_ids,
        attention_mask,
        past_key_values,
        labels,
        images
    ):
      
        # Qwen多模态输入
        if images is None:
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "labels": labels
            }

        # 生成视觉特征
        image_features = self.encode_images(images)
        
        # 创建多模态输入
        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        
        # 拼接
        combined_embeds = torch.cat([inputs_embeds, image_features], dim=1)
        
        # 调整 attention mask
        new_attention_mask = torch.cat([
            attention_mask,
            torch.ones(image_features.shape[:2], device=attention_mask.device)
        ], dim=1)

        return {
            "inputs_embeds": combined_embeds,
            "attention_mask": new_attention_mask,
            "past_key_values": past_key_values,
            "labels": labels
        }

    def forward(self, **kwargs):
        
        images = kwargs.pop('images', None)
        
        if images is not None:
            # 多模态
            inputs = self.prepare_inputs_for_multimodal(**kwargs, images=images)
            return self.model(**inputs)
        else:
            # 纯文本
            return self.model(**kwargs)

    def generate(self, **kwargs):
        
        images = kwargs.pop('images', None)
        
        if images is not None:
           
            inputs = self.prepare_inputs_for_multimodal(**kwargs, images=images)
            return self.model.generate(**inputs)
        else:
            return self.model.generate(**kwargs)

    def initialize_vision_tokenizer(self, model_args, tokenizer):

        # Qwen的token
        new_tokens = ["<image>", "</image>"] 
        tokenizer.add_tokens(new_tokens, special_tokens=True)
        self.model.resize_token_embeddings(len(tokenizer))
        
        # 初始化embedding
        with torch.no_grad():
            input_embeddings = self.model.get_input_embeddings().weight.data
            output_embeddings = self.model.get_output_embeddings().weight.data
            
            # 使用平均初始化
            avg_embed = input_embeddings[:-len(new_tokens)].mean(dim=0)
            for token in new_tokens:
                input_embeddings[-len(new_tokens)] = avg_embed
                output_embeddings[-len(new_tokens)] = avg_embed