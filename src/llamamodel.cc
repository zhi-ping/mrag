#include "llamamodel.h"
#include <print>

LlamaModelBase::LlamaModelBase(ModelType model_type, const AppConfig& config) {
    if (model_type == ModelType::Embedding) {
        this->loadEmbedding(config);
    } else {
        this->loadGeneration(config);
    }
}

LlamaModelBase::~LlamaModelBase() {
    if (context_) {
        llama_free(context_);
        context_ = nullptr;
    }
    if (model_) {
        llama_model_free(model_);
        model_ = nullptr;
    }

}

LlamaModelBase::LlamaModelBase(LlamaModelBase&& other) noexcept 
        : model_(other.model_), context_(other.context_) {
        other.model_ = nullptr;
        other.context_ = nullptr;
}

LlamaModelBase& LlamaModelBase::operator=(LlamaModelBase&& other) noexcept {
        if (this != &other) { // 防止自我赋值

            if (context_) llama_free(context_);
            if (model_) llama_model_free(model_);

            model_ = other.model_;
            context_ = other.context_;

            other.model_ = nullptr;
            other.context_ = nullptr;
        }
        return *this;
}

void LlamaModelBase::loadEmbedding(const AppConfig& config) {
    llama_model_params model_params = llama_model_default_params();
    llama_context_params ctx_params = llama_context_default_params();

    model_params.n_gpu_layers = config.n_gpu_layers;

    std::println(stderr,"正在加载向量模型...");
    ctx_params.embeddings = true;
    ctx_params.pooling_type = LLAMA_POOLING_TYPE_MEAN;
    ctx_params.n_ctx        = config.emb_n_ctx;
    ctx_params.n_batch      = config.emb_n_batch;
    ctx_params.n_ubatch     = config.emb_n_ubatch;
    model_ = llama_model_load_from_file(config.emb_model_path.c_str(),model_params);
    if (!model_) {
        throw std::runtime_error("无法加载模型文件: " + config.emb_model_path);
    }
    context_ = llama_init_from_model(model_, ctx_params);
}

void LlamaModelBase::loadGeneration(const AppConfig& config) {
    llama_model_params model_params = llama_model_default_params();
    llama_context_params ctx_params = llama_context_default_params();

    model_params.n_gpu_layers = config.n_gpu_layers;

    std::println(stderr,"正在加载生成模型...");

    ctx_params.embeddings = false;
    ctx_params.n_ctx        = config.gen_n_ctx;
    ctx_params.n_batch      = config.gen_n_batch;
    ctx_params.n_ubatch     = config.gen_n_ubatch;
    model_ = llama_model_load_from_file(config.gen_model_path.c_str(),model_params);
    if (!model_) {
        throw std::runtime_error("无法加载模型文件: " + config.gen_model_path);
    }
    context_ = llama_init_from_model(model_, ctx_params);
}

std::vector<llama_token> LlamaModelBase::tokenize(const std::string& text, bool add_special, bool parse_special) const {
    if (text.empty()) {
        return std::vector<llama_token>();
    }

    // 第一次调用：探测所需的空间大小
    // llama.cpp 会在底层计算好所需数量，并以负数的形式返回
    int n_tokens_needed = llama_tokenize(
        llama_model_get_vocab(model_), 
        text.c_str(), 
        text.length(), 
        nullptr, // 暂不接收数据
        0,       // 最大容量给 0
        add_special, 
        parse_special
    );

    // 获取绝对值，得出真正需要的数组长度
    if (n_tokens_needed < 0) {
        n_tokens_needed = -n_tokens_needed;
    }

    // 分配内存
    std::vector<llama_token> tokens(n_tokens_needed);

    // 第二次调用：真正将 Token ID 写入分配好的 vector 中
    int final_tokens_count = llama_tokenize(
        llama_model_get_vocab(model_), 
        text.c_str(), 
        text.length(), 
        tokens.data(),
        tokens.size(), // 传入容量上限
        add_special, 
        parse_special
    );

    tokens.resize(final_tokens_count);

    return tokens;
}