#! -*- coding: utf-8 -*-
# GAU-α 模型实现
from bert4keras.models import *


class GAU_alpha(RoFormerV2):
    """GAU-α
    改动：基本模块换成GAU
    链接：https://kexue.fm/archives/9052
    """
    def initializer(self, shape, dtype=None, order=3, gain=1.0):
        return super(GAU_alpha, self).initializer(shape, dtype, order, gain)

    def apply_main_layers(self, inputs, index):
        """GAU-α 的主体是基于Gated Attention Unit的模块
        顺序：GAU  --> Add --> LN
        """
        x = inputs

        attention_name = 'Transformer-%d-GatedAttentionUnit' % index
        attention_mask = self.compute_attention_bias(index)
        position_bias = self.compute_position_bias(x)

        # Self Attention
        xi = x
        x = [x, position_bias]
        arguments = {'a_bias': None, 'p_bias': 'rotary'}
        if attention_mask is not None:
            arguments['a_bias'] = True
            x.insert(1, attention_mask)
        x = self.apply(
            inputs=x,
            layer=GatedAttentionUnit,
            arguments=arguments,
            units=self.intermediate_size,
            key_size=self.attention_key_size,
            activation=self.hidden_act,
            use_bias=False,
            normalization='softmax_plus',
            attention_dropout=self.attention_dropout_rate,
            kernel_initializer=self.initializer,
            name=attention_name
        )
        x = self.apply(
            inputs=x,
            layer=Dropout,
            rate=self.dropout_rate,
            name='%s-Dropout' % attention_name
        )
        x = self.apply(
            inputs=[xi, x], layer=Add, name='%s-Add' % attention_name
        )
        x = self.apply(
            inputs=x,
            layer=LayerNormalization,
            zero_mean=False,
            scale=False,
            offset=False,
            name='%s-Norm' % attention_name
        )

        return x

    def variable_mapping(self):
        """重新定义权重映射
        """
        mapping = {
            'Embedding-Token': ['bert/embeddings/word_embeddings'],
            'Embedding-Segment': ['bert/embeddings/token_type_embeddings'],
        }

        for i in range(self.num_hidden_layers):
            prefix = 'GAU_alpha/encoder/layer_%d/' % i
            mapping['Transformer-%d-GatedAttentionUnit' % i] = [
                prefix + 'gau/i_dense/kernel',
                # prefix + 'gau/i_dense/bias',
                prefix + 'gau/o_dense/kernel',
                # prefix + 'gau/o_dense/bias',
                # prefix + 'gau/q_scaleoffset/beta',
                prefix + 'gau/q_scaleoffset/gamma',
                # prefix + 'gau/k_scaleoffset/beta',
                prefix + 'gau/k_scaleoffset/gamma',
            ]

        return mapping
