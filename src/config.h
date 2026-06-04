#pragma once

#include <nlohmann/json.hpp>
#include <string>

struct AppConfig {
    // 模型路径与通用配置
    std::string emb_model_path = "./models/bge-small-zh-v1.5-f16.gguf"; 
    std::string gen_model_path = "./models/qwen2.5-1.5b-instruct-q4_k_m.gguf";
    int n_gpu_layers = 99; // 默认全量加载

    // 文档切分参数
    size_t chunk_size = 500;
    size_t overlap_size = 50;

    // 检索参数
    int top_k = 3;

    // 生成模型上下文与采样参数
    int gen_n_ctx = 4096;
    int gen_n_batch = 4096;
    int gen_n_ubatch = 512;
    int max_output_tokens = 512;
    float temperature = 0.7f;
    int top_k_sampler = 40;
    float top_p = 0.9f;

    // 嵌入模型上下文参数
    int emb_n_ctx = 512;
    int emb_n_batch = 512;
    int emb_n_ubatch = 512;

    //还是算了，最终要去掉
    explicit AppConfig() = default;

    AppConfig(AppConfig&&) = default;
    AppConfig& operator=(AppConfig&&) = default;
    
    /**
     * @brief 辅助函数：从JSON对象创建配置对象
     * 支持部分字段覆盖，未提供的字段将使用默认值
     */
    friend AppConfig fromJson(const nlohmann::json& config_json);

    /**
     * @brief 读取配置文件构造配置对象
     */
    explicit AppConfig(const std::string& file_path);

    // 比较函数，只比较--chat模式相关参数
    bool operator==(const AppConfig& config) const;

    // 比较emb相关参数是否相同
    bool embChange(const AppConfig& config) const;

    // 比较gen相关参数是否相同
    bool genChange(const AppConfig& config) const;

    // 比较top_k参数是否相同
    bool topkChange(const AppConfig& config) const;
};
