#pragma once

#include "chunk.h"
#include "config.h"
#include <fstream>
#include <string>
#include <regex>
#include <string_view>




struct Document {

    std::string full_text_;  
    std::regex chapter_regex_;  //匹配章节标题的正则表达式
    std::vector<std::pair<std::string, size_t>> chapter_table_;

    Document() = default;
    Document(const std::string& file_path);
    void loadNovel(const std::string& file_path);
    /**
     * @brief 允许用户自定义正则表达式
     */
    void setRegex(const std::string& pattern);

    /**
     * @brief 建立章节索引表chapter_table_
     */
    void matchChapterTitle();

    /**
     * @brief 根据索引定位章节
     */
    const std::string& locateChapter(const size_t pos) const;
};

class DocumentProcessor {
private:
    std::vector<Document> documents_; // 支持多文件处理

    size_t chunk_size_{};
    size_t overlap_size_{};
    
public:
    explicit DocumentProcessor(const AppConfig& config);
    std::vector<Chunk> processNovel();

    /// @test 测试用
    explicit DocumentProcessor(const std::string& file_path,const AppConfig& config);

    // 添加文本接口
    void addNovel(const Document& novel);
    void addNovel(const std::vector<Document>& novels);
};

/**
 * @brief 寻找下一个合法的 UTF-8 字符边界
 */
size_t nextUtf8Boundary(const std::string& text, size_t pos);

/**
 * @brief 将字符串切片窗口对齐到按句末标点、换行符
 */
std::string semanticJust(const std::string& text, size_t& start, size_t& end);