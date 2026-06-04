#pragma once

#include "chunk.h"
#include "config.h"

class VectorDatabase {
private:
    std::vector<Chunk> chunks_;
    int top_k_{};
    inline static const std::string VERSION = "v1.0.0";  //版本号
    inline static const int MAGIC_NUMBER = 0x4D524147;   //魔数
public:

    explicit VectorDatabase(const AppConfig& config); //保证对象存在，则一定读取了配置信息

    void insert(const Chunk& chunk); //单个Chunk插入
    void insert(const std::vector<Chunk>& chunks); //批量插入Chunk
    void setTopk(const AppConfig& config);  // 设置top_k参数

    std::vector<Chunk> search(const std::vector<float>& query_emb) const;

    bool saveToDisk(const std::string& out_path) const;
    bool loadFromDisk(const std::string& in_path);

};

float cosine(const std::vector<float>& v1,const std::vector<float>& v2);

/// @brief 写完向量引擎后再测试