#pragma once

#include "generationengine.h"
#include "embeddingengine.h"
#include "vectordatabase.h"
#include "document.h"

enum class RunMode {
    Chat,
    Ingest,
    Err
};

struct BackendGuard {
    BackendGuard();
    ~BackendGuard();
};

class MragApp {
private:
    BackendGuard backendguard_;
    AppConfig config_;
    EmbeddingEngine emb_;
    GenerationEngine gen_;
    VectorDatabase db_;
    DocumentProcessor docu_processor_;
public:

    explicit MragApp() = default;
    explicit MragApp(const std::string& config_file_path);
  
    ~MragApp() = default;


    // 禁止复制
    MragApp(const MragApp&) = delete;
    MragApp& operator=(const MragApp&) = delete;

    void loadEmbedding();   // 加载向量模型
    void loadGeneration();  // 加载生成模型

    // 加载数据库
    void loadDatabase(const std::string& db_path);

    // 设置数据库保存路径
    void setDatabasebSavePath(const std::string& db_path);

    // 批量导入文本
    void loadText(const std::vector<std::string>& text_paths);

    // 建立本地知识库
    void buildKnowledgeBase(const std::string& db_output_path);

    // 切换配置文件
    void switchConfig(const std::string& file_path);

    // 切换向量模型
    void resetEmbedding(const std::string& file_path);

    // 切换生成模型
    void resetGeneration(const std::string& file_path);

    void chatLoop();

};

// 输出帮助信息
void printUsage();

// 解析命令行参数 
std::vector<std::string> parseCli(int argc, char* argv[]);
