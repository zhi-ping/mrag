#pragma once

#include "llamamodel.h"
#include "chunk.h"

class GenerationEngine : public LlamaModelBase {
private:
    int n_ctx_;
    int n_batch_;
    int n_ubatch_;
    int max_output_tokens_;
    float temperature_;
    int top_k_sampler_;
    float top_p_;
public:
    explicit GenerationEngine() = default;
    explicit GenerationEngine(const AppConfig& config);
    ~GenerationEngine() override = default;

    GenerationEngine(GenerationEngine&&) noexcept = default;
    GenerationEngine& operator=(GenerationEngine&&) noexcept = default;

    // 进行RAG推理，并实时输出结果
    void generateStream(const std::string& query, const std::vector<Chunk>& chunks);

    // 将token id 转换为字符串
    std::string tokenToString(const llama_token id);

    // 切换模型
    void reset(const AppConfig& config);
};