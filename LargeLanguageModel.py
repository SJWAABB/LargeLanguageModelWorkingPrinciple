from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F


#-----------------------------------------第一步把用户输入语句转换为Token ID
# 本地分词器路径
LOCAL_MODEL_PATH = "./bert-base-chinese-tokenizer/"
#加载预训练分词器（核心：获取模型绑定的词表和拆分规则）
tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
#用户输入语句
input_sentence = "帮我写一句新年祝福"
#执行Token拆分
token_split_result = tokenizer(
    input_sentence,
    return_tensors="pt",  # 返回PyTorch张量格式,因为普通数组不支持操作，所以要转换为张量
    add_special_tokens=True # 是否添加特殊Token（[CLS]和[SEP]）CLS是句子开始，SEP是句子结束，['[CLS]', '帮', '我', '写', '一', '句', '新', '年', '祝', '福', '[SEP]']
)
#-----------------------------------------第二步：将Token ID转换为PyTorch张量
#转换设备为cpu
DEVICE = torch.device("cpu") #如果有GPU，这里可以改为"cuda"
#token_ids是转移到cpu上的Token ID张量，shape: [1, 11]
token_ids = token_split_result["input_ids"].to(DEVICE)  
#attention_mask是转移到cpu上的注意力张量，shape: [1, 11]
attention_mask = token_split_result["attention_mask"].to(DEVICE) 
#得到迁移到cpu上的预训练模型，这里去拿了本地文件夹中的预训练pytorch_model.bin文件
pretrained_model = AutoModel.from_pretrained(
    LOCAL_MODEL_PATH,
    trust_remote_code=True #信任指定路径下的配置文件和模型代码，不进行联网验证或安全性校验
).to(DEVICE)

#-----------------------------------------第三步：提取预训练Embedding层,然后把Token ID转换为高维向量
# 提取预训练Embedding层（包含Token Embedding和位置Embedding，与真实LLM一致）
token_embedding_layer = pretrained_model.embeddings

# 冻结权重（仅用于推理，不进行训练，节省算力）
for param in token_embedding_layer.parameters():
    param.requires_grad = False # 冻结权重，不进行训练,只进行推理

#with在python中创建一个临时的运行环境，在这个环境内执行的代码会遵循该环境的特殊规则，离开环境后会自动恢复原有规则，无需手动清理
#torch.no_grad()是上下文管理器，用于在代码块执行前和执行后关闭梯度计算，避免不必要的算力消耗
with torch.no_grad():
    embedding_vectors = token_embedding_layer(token_ids) #核心步骤：将Token ID转换为高维向量

print(f"高维向量形状：{embedding_vectors.shape}")
#-----------------------------------------第四步：高维向量 → 加权求和后的新向量（注意力加权求和核心流程）得到的已经是融合的其他向量信息的上下文向量（注意：并非只有一个向量，只是每个向量都融合了其他向量的信息）
# 提取预训练Transformer Encoder层
bert_encoder = pretrained_model.encoder

#冻结encoder层权重（仅用于推理，不训练，节省算力+保护预训练权重）encoder层是Transformer的核心，负责将高维向量转换为上下文感知向量
for param in bert_encoder.parameters():
    param.requires_grad = False

#无梯度环境下执行注意力计算（完成加权求和，避免不必要的资源消耗）
with torch.no_grad():
    # 核心调用：encoder层接收高维向量和注意力掩码，内部自动完成加权求和，在bert_encoder自己就完成了注意力权重计算然后加权求和
    model_outputs = pretrained_model(
        inputs_embeds=embedding_vectors,  # 传入第一步生成的高维向量（[1, 11, 768]）
        attention_mask=attention_mask     # 传入注意力掩码，标记有效Token，避免关注填充Token
    )
# last_hidden_state：获取最后一层Transformer的输出，即完成加权求和后的上下文向量
context_aware_vectors = model_outputs.last_hidden_state

#--------------------------------------------------第五步：上下文向量 → 持续语句生成（多步迭代，序列级任务核心）
# 步骤1：定义LM Head（语言模型头）：将768维上下文向量转为词表大小的概率分布
vocab_size = tokenizer.vocab_size  # 获取bert-base-chinese词表大小（约21128）
lm_head = torch.nn.Linear(768, vocab_size).to(DEVICE)  # 输入768维，输出词表大小维
lm_head.requires_grad_(False)  # 冻结LM Head，仅用于推理，不训练

# 步骤2：设置持续生成的关键参数（控制生成过程，必须保留）
max_generate_length = 20  # 最大生成长度（兜底，防止无限循环）
current_token_ids = token_ids  # 初始化当前Token ID序列（后续不断更新），形状[1, 11]
current_attention_mask = attention_mask  # 初始化当前注意力掩码，形状[1, 11]
generated_tokens = []  # 存储所有生成的新Token

# 步骤3：循环迭代，实现持续生成（核心：每轮重新编码+预测+更新序列）
for _ in range(max_generate_length):
    with torch.no_grad():
        # 3.1 基于当前序列，重新生成Embedding向量和上下文感知向量（复用前四步的工具）
        current_embedding_vectors = token_embedding_layer(current_token_ids)
        current_model_outputs = pretrained_model(
            inputs_embeds=current_embedding_vectors,
            attention_mask=current_attention_mask
        )  #####这里已经全部融合成上下文向量了，每个向量都融合了其他向量的信息
        current_context_aware_vectors = current_model_outputs.last_hidden_state #这里相当于从结果集合里提取出当前序列的上下文感知向量（全文向量）
        
        # 3.2 提取当前序列最后一个有效Token的上下文向量（跳过[SEP]，聚焦正文末尾）
        last_effective_token_vector = current_context_aware_vectors[:, -2, :]
        
        # 3.3 LM Head预测：768维向量 → 词表中所有Token的得分
        prediction_scores = lm_head(last_effective_token_vector)
        
        # 3.4 Softmax转换为0~1的概率分布
        prediction_probs = F.softmax(prediction_scores, dim=-1)
        
        # 3.5 选概率最大的Token ID（贪心搜索），调整维度为[1,1]（和current_token_ids匹配）
        predicted_token_id = torch.argmax(prediction_probs, dim=-1)  # 形状[1]
        predicted_token_id = predicted_token_id.unsqueeze(1)  #增加一维
        predicted_token = tokenizer.convert_ids_to_tokens(predicted_token_id.item()) #拿到概率最大的Tokenid对应的Token
        
        # 3.6 强化停止条件（遇到结束标记就终止，句子更自然）
        stop_tokens = ["[SEP]", "。", "！", "？"]
        if predicted_token in stop_tokens:
            generated_tokens.append(predicted_token)
            break
        
        # 3.7 更新序列数据（核心：维度匹配，可正常拼接）
        # 更新Token ID序列：[1,n] + [1,1] → [1,n+1]
        current_token_ids = torch.cat([current_token_ids, predicted_token_id], dim=1)
        # 更新注意力掩码：[1,n] + [1,1] → [1,n+1]
        new_mask = torch.ones((1, 1), dtype=torch.long).to(DEVICE)
        current_attention_mask = torch.cat([current_attention_mask, new_mask], dim=1)
        # 存储生成的Token
        generated_tokens.append(predicted_token)

# 步骤4：拼接所有Token，生成完整新句子
# 4.1 提取原始Token列表和所有生成的Token列表
original_token_ids = token_ids[0].tolist()
original_tokens = tokenizer.convert_ids_to_tokens(original_token_ids)
all_tokens = original_tokens + generated_tokens

clean_gen_tokens = [t.replace("##", "") for t in generated_tokens]
# 第二步：直接拼接成纯新生成的字符串（全程不碰原始句子）
generated_sentence = "".join(clean_gen_tokens)

# 4.2 转换为自然语句（去除特殊Token和多余空格）
new_sentence = tokenizer.convert_tokens_to_string(all_tokens)
new_sentence = new_sentence.replace(" ", "").replace("[CLS]", "").replace("[SEP]", "")

# 步骤5：打印生成结果
print("=" * 60)
print(f"原始句子：{input_sentence}")
print(f"生成的新Token列表：{generated_tokens}")
print(f"生成：{generated_sentence}")
print("=" * 60)

