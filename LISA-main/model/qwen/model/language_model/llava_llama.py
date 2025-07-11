
from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast


from transformers import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration
from ..qwen_arch import QwenMetaForCausalLM, QwenMetaModel

class QwenConfig(Qwen2_5_VLConfig):  
    model_type = "qwen_vl"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)



class QwenModel(QwenMetaModel, Qwen2_5_VLForConditionalGeneration):  # 替换基类
    config_class = QwenConfig
    
    def __init__(self, config: QwenConfig):
        super().__init__(config)

        self.vision_model = self.get_vision_tower()  

class QwenForCausalLM(QwenMetaForCausalLM, AutoModelForCausalLM):  # 重构继承关系
    config_class = QwenConfig
    
    def __init__(self, config):
        super(AutoModelForCausalLM, self).__init__(config)
        self.model = QwenModel(config)
        
        # 保持语言模型头
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 初始化权重
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,  # 直接接收图像张量
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        # 调用 Qwen 原生多模态处理
        if images is not None:
            # 多模态输入
            inputs_embeds = self.model.encode_multimodal(
                text_ids=input_ids,
                images=images,
                attention_mask=attention_mask
            )
            input_ids = None  

        # 调用父类前向传播
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        images=None,
        **kwargs
    ):
        # 简化生成准备逻辑
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        
        # 添加图像输入
        if images is not None:
            model_inputs["images"] = images
            
        return model_inputs

# 注册新的模型类型
AutoModelForCausalLM.register(QwenConfig, QwenForCausalLM)
AutoModelForCausalLM.register(QwenConfig, QwenModel)