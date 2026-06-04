#include "config.h"
#include <fstream>
#include <print>

AppConfig fromJson(const nlohmann::json& config_json) {
    AppConfig config;

    if (config_json.contains("models") && config_json["models"].is_object()) {
        const auto& m = config_json["models"];
        config.emb_model_path = m.value("embedding", config.emb_model_path);
        config.gen_model_path = m.value("generation", config.gen_model_path);
        config.n_gpu_layers   = m.value("n_gpu_layers", config.n_gpu_layers);
    }

    if(config_json.contains("document") && config_json["document"].is_object()) {
        const auto& d = config_json["document"];
        config.chunk_size = d.value("chunk_size", config.chunk_size);
        config.overlap_size = d.value("overlap_size", config.overlap_size);
    }

    if (config_json.contains("retrieval") && config_json["retrieval"].is_object()) {
        config.top_k = config_json["retrieval"].value("top_k", config.top_k);
    }

    if (config_json.contains("generation") && config_json["generation"].is_object()) {
        const auto& g = config_json["generation"];
        config.gen_n_ctx    = g.value("n_ctx", config.gen_n_ctx);
        config.gen_n_batch  = g.value("n_batch", config.gen_n_batch);
        config.gen_n_ubatch = g.value("n_ubatch", config.gen_n_ubatch);
        config.max_output_tokens = g.value("max_output_tokens", config.max_output_tokens);
        config.temperature  = g.value("temperature", config.temperature);
        config.top_k_sampler = g.value("top_k", config.top_k_sampler);
        config.top_p        = g.value("top_p", config.top_p);
    }
    
    if (config_json.contains("embedding") && config_json["embedding"].is_object()) {
        const auto& e = config_json["embedding"];
        config.emb_n_ctx    = e.value("n_ctx", config.emb_n_ctx);
        config.emb_n_batch  = e.value("n_batch", config.emb_n_batch);
        config.emb_n_ubatch = e.value("n_ubatch", config.emb_n_ubatch);
    }
    return config;
}

AppConfig::AppConfig(const std::string& file_path) {
    std::println(stderr,"正在读取配置文件...");
    std::ifstream config_file(file_path);
    if (!config_file.is_open()) {
        std::println(stderr,"读取配置失败，将采用默认配置！");
        return;
    }
    try {
        nlohmann::json config_json;
        config_file >> config_json;
        *this = fromJson(config_json);
        return;
    } catch (...) {
        std::println(stderr,"解析{}失败，将采用默认配置！", file_path);
        return;
    }
}

bool AppConfig::operator==(const AppConfig& config) const {
    return this->emb_model_path == config.emb_model_path 
        && this->emb_n_batch == config.emb_n_batch
        && this->emb_n_ctx == config.emb_n_ctx
        && this->emb_n_ubatch == config.emb_n_ubatch
        && this->n_gpu_layers == config.n_gpu_layers
        && this->gen_model_path == config.gen_model_path
        && this->gen_n_batch == config.gen_n_batch
        && this->gen_n_ctx == config.gen_n_ctx
        && this->gen_n_ubatch == config.gen_n_ubatch
        && this->max_output_tokens == config.max_output_tokens
        && this->temperature == config.temperature
        && this->top_k_sampler == config.top_k_sampler
        && this->top_k == config.top_k
        && this->top_p == config.top_p;
}

bool AppConfig::embChange(const AppConfig& config) const {
    return this->emb_model_path != config.emb_model_path 
        || this->emb_n_batch != config.emb_n_batch
        || this->emb_n_ctx != config.emb_n_ctx
        || this->emb_n_ubatch != config.emb_n_ubatch
        || this->n_gpu_layers != config.n_gpu_layers;
}

bool AppConfig::genChange(const AppConfig& config) const {
    return this->n_gpu_layers != config.n_gpu_layers
        || this->gen_model_path != config.gen_model_path
        || this->gen_n_batch != config.gen_n_batch
        || this->gen_n_ctx != config.gen_n_ctx
        || this->gen_n_ubatch != config.gen_n_ubatch
        || this->max_output_tokens != config.max_output_tokens
        || this->temperature != config.temperature
        || this->top_k_sampler != config.top_k_sampler
        || this->top_p != config.top_p;
}

bool AppConfig::topkChange(const AppConfig& config) const {
    return this->top_k != config.top_k;
}
