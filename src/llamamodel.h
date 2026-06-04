#pragma once

#include "config.h"
#include <llama.h>

enum class ModelType {
    Embedding,
    Generation
};

class LlamaModelBase {
protected:
    llama_model* model_ = nullptr;
    llama_context* context_ = nullptr;

    std::vector<llama_token> tokenize(const std::string& text, bool add_special, bool parse_special) const;
public:
    
    void loadEmbedding(const AppConfig& config);    //加载向量模型
    void loadGeneration(const AppConfig& config);   //加载生成模型
    
    explicit LlamaModelBase() = default;
    explicit LlamaModelBase(ModelType model_type, const AppConfig& config);
    virtual ~LlamaModelBase();

    //单例模式，禁止复制
    LlamaModelBase(const LlamaModelBase&) = delete;
    LlamaModelBase& operator=(const LlamaModelBase&) = delete;

    LlamaModelBase(LlamaModelBase&& other) noexcept;
    LlamaModelBase& operator=(LlamaModelBase&& other) noexcept;
};