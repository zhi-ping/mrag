#pragma once

#include "llamamodel.h"

class EmbeddingEngine : public LlamaModelBase {
private:
    int n_ctx_;
    int n_batch_;
    int n_ubatch_;
public:

    explicit EmbeddingEngine() = default;

    explicit EmbeddingEngine(const AppConfig& config);
    ~EmbeddingEngine() override = default;

    EmbeddingEngine(EmbeddingEngine&&) noexcept = default;
    EmbeddingEngine& operator=(EmbeddingEngine&&) noexcept = default;

    //清理输入文本中残缺的 UTF-8 序列
    friend void sanitizeUtf8(std::string& text);


    //根据输入文本生成嵌入向量
    std::vector<float> generateEmbeddings(std::string& text);
    
    // 切换模型
    void reset(const AppConfig& config);
};