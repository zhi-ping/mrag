#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <fstream>

struct Chunk {
    uint64_t id{};                  // 全局唯一递增ID
    std::string text;               // 文本内容
    std::string metadata;           // 所属章节名
    std::vector<float> embedding;   // 嵌入向量
    
    Chunk() = default;
    explicit Chunk(const std::string& t, const std::string& meta);
    
    void serialize(std::ofstream& out) const;       //序列化
    static Chunk deserialize(std::ifstream& in);    //反序列化
};