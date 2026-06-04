#include "mragapp.h"
#include <print>
#include <iostream>
#include <filesystem>

namespace fs = std::filesystem;

void my_log_callback(ggml_log_level level, const char* text, void* user_data) {
    // 空回调
};

BackendGuard::BackendGuard() {
    //屏蔽 llama.cpp 的底层日志
    llama_log_set(my_log_callback,nullptr);
    std::println(stderr,"正在初始化计算引擎...");
    llama_backend_init();
}

BackendGuard::~BackendGuard() noexcept {
    llama_backend_free();
}

MragApp::MragApp(const std::string& config_file_path) : config_(config_file_path),
    emb_(config_), gen_(config_), db_(config_), docu_processor_(config_) { }

void MragApp::loadEmbedding() {
    emb_.loadEmbedding(config_);
}

void MragApp::loadGeneration() {
    gen_.loadGeneration(config_);
}

void printUsage(const char* prog_name) {
    std::println(stderr, "Usage: {} [COMMAND] [ARGS]\n", prog_name);
    std::println(stderr,"命令:");
    std::println(stderr,"  --ingest <txt_path> [txt_path2 ...] <db_path>   离线建库");
    std::println(stderr,"  --chat   <db_path>                              在线问答");
    std::println(stderr,"  --help                                          显示此帮助信息");
}

void MragApp::loadDatabase(const std::string& db_path) {
    if(!db_.loadFromDisk(db_path)) {
        throw std::runtime_error("数据库加载失败");
    }
}

void MragApp::loadText(const std::vector<std::string>& text_paths) {
    for(size_t i = 0;i < text_paths.size();++i) {
        std::print(stderr,"\r正在导入训练文本[{}/{}]    ", i+1, text_paths.size());
        docu_processor_.addNovel(Document(text_paths[i]));
    }
    std::println(stderr,"");
}

void MragApp::buildKnowledgeBase(const std::string& db_output_path) {
    auto chunks = docu_processor_.processNovel();
    for(auto& chunk : chunks) {
        chunk.embedding = emb_.generateEmbeddings(chunk.text);
    }
    db_.insert(chunks);
    if(!db_.saveToDisk(db_output_path)) {
        throw std::runtime_error("构建数据库失败");
    }
}

void MragApp::chatLoop() {
    while(true) {
        std::print(">> ");
        std::string query;
        std::getline(std::cin, query);
        if(query.empty()) continue; // 防止用户直接输入回车，导致 UB
        if (query[0] == '/') {   // 处理内置命令
            std::stringstream ss(query);
            std::string cmd;
            ss >> cmd; // 提取第一个单词作为命令

            if (cmd == "/model") {          // /model 切换模型
                std::string flag;
                std::string new_model_path;
                
                ss >> flag; // 提取第二个参数（模型类别标志）

                // 分支 1：切换生成模型
                if (flag == "--generation" || flag == "-g") {
                    ss >> new_model_path; // 提取第三个参数（路径）
                    if (new_model_path.empty()) {
                        std::println(stderr, "错误: 缺少模型路径。用法: /model --generation <路径>");
                    } else {
                        std::println("正在切换生成模型 (Generation) 至: {} ...", new_model_path);
                        this->resetGeneration(new_model_path);
                    }
                } 
                // 分支 2：切换向量模型
                else if (flag == "--embedding" || flag == "-e") {
                    ss >> new_model_path; 
                    if (new_model_path.empty()) {
                        std::println(stderr, "错误: 缺少模型路径。用法: /model --embedding <路径>");
                    } else {
                        std::println("正在切换向量模型 (Embedding) 至: {}...", new_model_path);
                        /// @todo: 释放旧向量模型，加载新向量模型
                        this->resetEmbedding(new_model_path);
                    }
                } 
                // 错误处理：标志输入错误或为空
                else {
                    std::println(stderr, "错误: 参数无法识别。");
                    std::println(stderr, "正确用法: /model --generation <路径> 或 /model --embedding <路径>");
                }

            } else if (cmd == "/config") {  // /config <路径> 热切换配置文件
                std::string new_config_path;
                ss >> new_config_path;

                if (new_config_path.empty()) {
                    std::println(stderr, "错误: 缺少配置文件路径。用法: /config <路径>");
                } else {
                    try {
                        std::println("正在切换配置文件为:{}", new_config_path);     
                        this->switchConfig(new_config_path);
                    } catch (const std::exception& e) {
                        std::println(stderr, "错误: 配置文件路径解析失败: {}", e.what());
                    }
                } 
            } else if (cmd == "/exit" || cmd == "/quit") {   // /exit 退出会话
                std::println("\nBye from mrag~~ ");
                break; // 退出 while(true) 循环
                
            } else if (cmd == "/help") {    // /help 显示帮助信息
                std::println("\n====================== Mrag 内部指令 =======================");
                std::println("  /model --generation <路径>    热切换大语言模型 (可简写为 -g)");
                std::println("  /model --embedding  <路径>    热切换向量模型   (可简写为 -e)");
                std::println("  /config <配置文件路径>        热重载配置文件   (JSON 格式)");
                std::println("  /help                         显示此帮助信息");
                std::println("  /exit, /quit                  退出问答会话");
                std::println("============================================================");
            } else {
                std::println(stderr, "未知指令: '{}'。输入 /help 查看支持的指令。", cmd);
            } 
        } else {
            // std::print("回答：");
            std::vector<float> query_emb = emb_.generateEmbeddings(query);
            std::vector<Chunk> topk_chunks = db_.search(query_emb);
            
            ///@test
            // for(auto& chunk :topk_chunks) {
            //     std::println("\nchunk id: {}",chunk.id);
            //     std::println("章节： {}", chunk.metadata);
            //     std::println("chunk内容: {}",chunk.text);
            //     std::println("相似度: {}",cosine(chunk.embedding,query_emb));
            // }
            ///@test

            gen_.generateStream(query, topk_chunks);
        }
    }
}

void MragApp::resetEmbedding(const std::string& file_path) {
    /// @test
    // auto old_abso = fs::canonical(config_.emb_model_path);
    // auto new_abso = fs::canonical(file_path);
    /// @test
    try {
        if(fs::canonical(config_.emb_model_path) != fs::canonical(file_path)) {
        config_.emb_model_path = file_path;
        emb_.reset(config_);
        } else {
            std::println(stderr, "切换模型失败，当前模型已经是: {}", file_path);
        }
    } catch (const fs::filesystem_error& e) {
        std::println(stderr,"文件不存在或路径错误");
    }
}

void MragApp::resetGeneration(const std::string& file_path) {
    try {
        if(fs::canonical(config_.gen_model_path) != fs::canonical(file_path)) {
        config_.gen_model_path = file_path;
        gen_.reset(config_);
        } else {
            std::println(stderr, "切换模型失败，当前模型已经是{}", file_path);
        }
    } catch (const fs::filesystem_error& e) {
        std::println(stderr,"模型文件不存在或路径错误");
    }
}

void MragApp::switchConfig(const std::string& file_path) {
    // 备份
    auto old_config = std::move(config_);
    try{
        config_ = AppConfig(file_path);
        if(config_ != old_config) {
            if(config_.embChange(old_config)) emb_.reset(config_);
            if(config_.genChange(old_config)) gen_.reset(config_);
            if(config_.topkChange(old_config)) db_.setTopk(config_);
            std::println(stderr, "已切换配置为: {}", file_path);
        } else {
            std::println(stderr,"当前chat模式配置已与{}相同", file_path);
        }
    } catch(...) {
        std::println(stderr,"配置文件切换失败，继续使用原来配置");
        config_ = std::move(old_config);
    }
}

std::vector<std::string> parseCli(int argc, char* argv[]) {
    
    //paths.size()=0表示命令错误，或输出帮助信息
    //paths.size()=1表示chat模式
    //paths.size()>1表示ingest模式
    std::vector<std::string> paths;

    // 检查是否传入了基础命令
    if (argc < 2) {
        std::println(stderr, "Usage: {} [COMMAND] [ARGS]\n", argv[0]);
        std::println(stderr,"Try --help for more help.");
        return paths;
    }

    std::string cmd = argv[1];

    // 解析 --help
    if (cmd == "--help") {
        printUsage(argv[0]);
        return paths;
    } 
    // 解析 --ingest 模式
    else if (cmd == "--ingest") {
        if (argc < 4) {
            std::println(stderr, "Usage: {} [COMMAND] [ARGS]\n", argv[0]);
            std::println(stderr,"Try --help for more help.");
            return paths;
        }
        for (size_t i = 2;i < argc;++i) {
            paths.push_back(argv[i]);
        } //最后一个为数据库路径
        return paths;
    } 
    // 解析 --chat 模式
    else if (cmd == "--chat") {
        if (argc != 3) {
            std::println(stderr, "Usage: {} [COMMAND] [ARGS]\n", argv[0]);
            std::println(stderr,"Try --help for more help.");
            return paths;
        }
        paths.push_back(argv[2]);
        return paths;
    } 
    else {
        std::println(stderr, "Usage: {} [COMMAND] [ARGS]\n", argv[0]);
        std::println(stderr,"Try --help for more help.");
        printUsage(argv[0]);
        return paths;
    }
}
