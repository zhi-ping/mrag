#include "generationengine.h"

#include <format>
#include <print>
#include <iostream>
#include <utility>

GenerationEngine::GenerationEngine(const AppConfig& config) :
    LlamaModelBase(),
    n_ctx_(config.gen_n_ctx), n_batch_(config.gen_n_batch),
    n_ubatch_(config.gen_n_ubatch), max_output_tokens_(config.max_output_tokens),
    temperature_(config.temperature), top_k_sampler_(config.top_k_sampler), top_p_(config.top_p)
    { }

void GenerationEngine::generateStream(const std::string& query, const std::vector<Chunk>& chunks) {
    
    llama_memory_clear(llama_get_memory(context_), false);
    
    std::string context;
    for(auto& chunk : chunks) {
        context += std::format(
            "[来源: {}] {}", 
            chunk.metadata, chunk.text
        );
    }

    std::string prompt = R"(<|im_start|>system
你是一个严谨的小说阅读助手。请你【必须并且只能】根据下面提供的[参考上下文]来回答用户的问题。
如果上下文中有答案，请详细回答并引用出处；如果上下文中没有提到相关信息，请直接回答"根据当前片段无法得知"，绝不允许自行编造。<|im_end|>
<|im_start|>user
[参考上下文]：
)" + context + R"(
用户问题：)" + query + R"(<|im_end|>
<|im_start|>assistant
)";

    // Tokenize
    auto tokens = tokenize(prompt, true, true); 
    
    // Prompt Prefill
    llama_batch batch = llama_batch_init(n_batch_, 0, 1);
    for (size_t i = 0; i < tokens.size(); ++i) {
        int idx = batch.n_tokens; // 当前要填装的数组索引

        batch.token[idx] = tokens[i];
        batch.pos[idx] = i;             // 词在句子中的绝对位置
        batch.n_seq_id[idx] = 1;        // 单用户系统，序列数量始终为 1
        batch.seq_id[idx][0] = 0;       // 序列 ID 始终分配给 0 号

        //设置logits
        if (i == tokens.size() - 1) {
            batch.logits[idx] = 1; //最后一个token,设置logits为1
        } else {
            batch.logits[idx] = 0;
        }

        ++batch.n_tokens;
    }
    
    if (llama_decode(context_, batch) != 0) {
        llama_batch_free(batch); // 异常安全
        throw std::runtime_error("GenerationEngine: Prefill 阶段 llama_decode 失败，可能是上下文超出了 n_ctx 上限。");
    }

    // 创建采样链指针
    llama_sampler* sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    
    // 依次挂载过滤器
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(top_k_sampler_));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(top_p_, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(temperature_));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(std::time(nullptr))); // std::time(nullptr)产生随机种子

    // 自回归生成循环
    int n_past = tokens.size(); // 记录当前对话在时间轴上的绝对位置
    
    for (int i = 0; i < max_output_tokens_; ++i) {
        // 采样
        //基于 batch 中最后一个 token 的特征来预测
        llama_token id = llama_sampler_sample(sampler, context_, -1);
        
        // 检测 EOG

        if (llama_vocab_is_eog(llama_model_get_vocab(model_), id)) {
            break; 
        }

        // 输出
        // 将数字 ID 转成人类可读的字符串，并立即刷新缓冲区，实现打字机效果
        std::string piece = tokenToString(id);
        std::cout << piece << std::flush;

        // 清空 batch & 加入新 token
        // 不需要重新分配内存，只需把游标清零，重用刚才的 batch
        batch.n_tokens = 0; 
        
        batch.token[0] = id;
        batch.pos[0] = n_past;        // 词的绝对位置一直在递增
        batch.n_seq_id[0] = 1;
        batch.seq_id[0][0] = 0;
        batch.logits[0] = 1;          // 为下一次循环预测打开开关
        
        batch.n_tokens = 1;           // 增量推断，每次只喂 1 个 Token

        // decode
        // 将新 token 喂给模型，更新内部的 KV Cache
        if (llama_decode(context_, batch) != 0) {
            std::println(stderr,"\n[错误] 自回归生成中断：llama_decode 失败");
            break;
        }
        
        n_past++; // 时间轴向前推移一步
    }

    std::cout << std::endl; // 结束回答后换行
    // ==========================================
    // 3. 释放资源
    // ==========================================
    llama_sampler_free(sampler);
    llama_batch_free(batch);
}

std::string GenerationEngine::tokenToString(const llama_token id) {
    auto vocab =llama_model_get_vocab(model_);
    // 先尝试获取所需长度
    // 传入空 buffer 和 0 长度，llama_token_to_piece 会返回需要的缓冲区大小的负值
    int32_t n = llama_token_to_piece(vocab, id, nullptr, 0, 0, false);

    if(n < 0) n = -n;

    // 分配缓冲区
    std::vector<char> buf(n);
    
    // 执行转换
    // lstrip=0 表示保留所有前导空格，special=false 表示不渲染 <s> 或 <unk> 等特殊字符
    llama_token_to_piece(vocab, id, buf.data(), n, 0, false);

    return std::string(buf.data(), n);
}

void GenerationEngine::reset(const AppConfig& config) {
    try{
        GenerationEngine new_gen;
        new_gen.loadGeneration(config);

        new_gen.n_batch_ = config.gen_n_batch;
        new_gen.n_ctx_ = config.gen_n_ctx;
        new_gen.n_ubatch_ = config.gen_n_ubatch;
        new_gen.max_output_tokens_ = config.max_output_tokens;
        new_gen.temperature_ = config.temperature;
        new_gen.top_k_sampler_ = config.top_k_sampler;
        new_gen.top_p_ = config.top_p;

        *this = std::move(new_gen);   
        std::println(stderr, "已切换生成模型为: {}", config.gen_model_path);
    } catch(std::exception& exception) {
        std::println(stderr,"生成模型切换失败");
    }
}