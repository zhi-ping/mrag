#include "embeddingengine.h"
#include <print>

EmbeddingEngine::EmbeddingEngine(const AppConfig& config) 
    : LlamaModelBase(),
        n_ctx_(config.emb_n_ctx), n_batch_(config.emb_n_batch), n_ubatch_(config.emb_n_ubatch)
    { }

void sanitizeUtf8(std::string& text) {
    if (text.empty()) {
        return;
    }

    size_t i = 0;
    size_t valid_len = 0; // 记录最后一次确认安全的 UTF-8 边界

    while (i < text.length()) {
        unsigned char c = text[i];
        size_t char_len = 0;

        // 判断当前字符的“预期长度”
        if ((c & 0x80) == 0x00) {
            char_len = 1; // 0xxxxxxx: 单字节 (ASCII)
        } else if ((c & 0xE0) == 0xC0) {
            char_len = 2; // 110xxxxx: 2 字节字符
        } else if ((c & 0xF0) == 0xE0) {
            char_len = 3; // 1110xxxx: 3 字节字符
        } else if ((c & 0xF8) == 0xF0) {
            char_len = 4; // 11110xxx: 4 字节字符
        } else {
            // 遇到非法的首字节（比如直接遇到 10xxxxxx 的孤儿尾巴），停止解析
            break; 
        }

        // 检查长度是否越界（残缺序列的核心检测点）
        if (i + char_len > text.length()) {
            break; // 剩余的字节数不够拼出这个字了，触发截断
        }

        // 严格校验后续字节（Continuation bytes）必须是 10xxxxxx
        bool valid_continuation = true;
        for (size_t j = 1; j < char_len; ++j) {
            if ((text[i + j] & 0xC0) != 0x80) {
                valid_continuation = false;
                break;
            }
        }

        if (!valid_continuation) {
            break; // 后续字节损坏，停止解析
        }

        // 更新安全游标，跳到下一个字符
        i += char_len;
        valid_len = i;
    }

    // 如果安全长度小于总长度，说明遇到了残缺或乱码，进行截断
    if (valid_len < text.length()) {
        text.resize(valid_len);
    }
}

std::vector<float> EmbeddingEngine::generateEmbeddings(std::string& text) {
    sanitizeUtf8(text);
    auto tokens = tokenize(text,true,false);
    //截断至 min(n_ctx, n_batch) 上限
    tokens.resize(std::min({tokens.size(),size_t(n_ctx_),size_t(n_batch_)}));
    llama_memory_clear(llama_get_memory(context_), false);
    
    /// @test
    // std::println("tokens.size: {}",tokens.size());

    //构造 batch
    llama_batch batch = llama_batch_init(n_batch_, 0, 1);
    batch.n_tokens = 0; 
    for (size_t i = 0; i < tokens.size(); i++) {
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

    // decode

    if (llama_encode(context_, batch) != 0) {  //记得改回 !=
        llama_batch_free(batch); // 异常安全
        throw std::runtime_error("EmbeddingEngine: llama decode failed");
    }

    // 优先尝试高级的序列级池化指针
    float* emb_ptr = llama_get_embeddings_seq(context_, 0);

    // 如果返回空指针，说明当前模型或配置没有激活序列池化，回退到词级原始指针
    if (emb_ptr == nullptr) {
        emb_ptr = llama_get_embeddings(context_);
    }
    
    // 获取当前模型定义的物理嵌入维度（例如 BGE-Small 通常是 512）
    int n_embd = llama_model_n_embd(model_);
    
    //释放 batch
    llama_batch_free(batch);


    /// @test
    // std::println("embedding: {},{},{},{},{} ……",emb_ptr[0],emb_ptr[1],emb_ptr[2],emb_ptr[3],emb_ptr[4]);


    // 利用 std::vector 的迭代器构造函数，提取嵌入向量
    return std::vector<float>(emb_ptr, emb_ptr + n_embd);
}

void EmbeddingEngine::reset(const AppConfig& config) {
    try{
        
        EmbeddingEngine new_emb;
        new_emb.loadEmbedding(config);

        llama_free(this->context_);
        llama_model_free(this->model_);

        *this = std::move(new_emb);   
        std::println(stderr, "已切换向量模型为: {}", config.emb_model_path);
    } catch(...) {
        std::println(stderr,"向量模型切换失败");
    }
}
