#include "document.h"
#include <algorithm>
#include <print>

std::vector<Chunk> DocumentProcessor::processNovel() {
    std::vector<Chunk> chunks;
    chunks.reserve(2000); //预分配内存
    std::println(stderr, "正在处理文本...");
    for(auto& document : documents_) {
        document.matchChapterTitle();
        size_t pos_1 = 0, pos_2 = 0;
        while(pos_1 <= pos_2 && pos_2 < document.full_text_.size()) {
            pos_2 = pos_1+chunk_size_;
            pos_2 = nextUtf8Boundary(document.full_text_, pos_2);
            chunks.emplace_back(
                semanticJust(document.full_text_, pos_1, pos_2),
                document.locateChapter(pos_1)
            );

            /// @test
            // Chunk chunk = chunks.back();
            // if(chunk.id >= 609) {
            //     int a = 1;
            // }
            // std::println("chunk {}",chunk.id);
            // std::println("chunk size: {}", chunk.text.size());
            // std::println("metadata: {}", chunk.metadata);
            // std::println("text: {}", chunk.text);
            /// @test

            pos_1 = nextUtf8Boundary(document.full_text_, pos_2 - overlap_size_);
        }
    }
    return chunks;
}

void Document::loadNovel(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + file_path);
    }

    // 预处理：去掉每行的\r回车符
    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        full_text_ += line + "\n";
    }

}

DocumentProcessor::DocumentProcessor(const AppConfig& config)
    : chunk_size_(config.chunk_size), overlap_size_(config.overlap_size)
        { }

/// @test
DocumentProcessor::DocumentProcessor(const std::string& file_path, const AppConfig& config)
    : chunk_size_(config.chunk_size), overlap_size_(config.overlap_size) {
        this->addNovel(Document(file_path));
    }

void DocumentProcessor::addNovel(const Document& novel) {
    documents_.push_back(novel);
}

void DocumentProcessor::addNovel(const std::vector<Document>& novels) {
    documents_.append_range(novels);
}

Document::Document(const std::string& file_path) :
     chapter_regex_(R"(((^第[零一二三四五六七八九十百\d]+[章节回卷][　\s：]|Chapter\s+\d+).*?)\n)",std::regex::ECMAScript | std::regex::multiline) {
    this->loadNovel(file_path);
}

void Document::setRegex(const std::string& pattern) {
    chapter_regex_ = std::regex(pattern);
}

void Document::matchChapterTitle() {
    auto words_begin = std::sregex_iterator(full_text_.begin(), full_text_.end(), chapter_regex_);
    auto words_end = std::sregex_iterator();
    chapter_table_.emplace_back(std::string("序言"), 0);
    for (auto i = words_begin; i != words_end; ++i) {
        const std::smatch& match = *i;
        chapter_table_.emplace_back(match.str(1), match.position(0));
    }
}

const std::string& Document::locateChapter(const size_t pos) const {
    //采用二分查找法
    // 二分查找：找到第一个起始位置 > pos 的章节
    auto it = std::upper_bound(chapter_table_.begin(), chapter_table_.end(), pos,
        [](size_t val, const std::pair<std::string, size_t>& pair) {
            return val < pair.second;
        });
    return (--it)->first;   
}

size_t nextUtf8Boundary(const std::string& text, size_t pos) {
    // 1. 如果位置已经超出文本长度，直接返回末尾
    if (pos >= text.size()) return text.size();
    // 2. 检查当前字节是否为 UTF-8 后续字节 (Continuation Byte)
    // 后续字节的特征是二进制开头为 10，即范围在 0x80 到 0xBF 之间
    // 位运算逻辑：(c & 0xC0) == 0x80
    while (pos < text.size() && 
           (static_cast<unsigned char>(text[pos]) & 0xC0) == 0x80) {
        ++pos;
    }

    return pos;
}

std::string semanticJust(const std::string& text, size_t& start, size_t& end) {
    std::string_view slice(text.data()+start,end-start);
    std::vector<std::string> signs = {"\n","\"","？","！","。","”",".","!","?"};
    const size_t bias_forward = 40;
    const size_t bias_backword = 100;
    if(end != text.size()) {
        size_t best_pos = std::string_view::npos;
    
        for (const auto& s : signs) {
            size_t p = slice.rfind(s);
            if (p != std::string_view::npos) {
                if (best_pos == std::string_view::npos || p > best_pos) {
                    best_pos = p;
                }
            }
        }

        if (best_pos != std::string_view::npos) {
            // 根据首字节正确判断 UTF-8 字符的字节长度
            unsigned char c = slice[best_pos];
            size_t char_len = 1; // fallback
            if ((c & 0x80) == 0x00) {
                char_len = 1;
            } else if ((c & 0xE0) == 0xC0) {
                char_len = 2;
            } else if ((c & 0xF0) == 0xE0) {
                char_len = 3;
            } else if ((c & 0xF8) == 0xF0) {
                char_len = 4;
            }
            if(best_pos >= slice.size() - bias_backword) //如果标点在切片前bias个字节内，才进行调整，否则可能切掉太多内容且导致死循环
                end = start + best_pos + char_len;
        }
    }

    if(start != 0) {
        size_t best_pos = std::string_view::npos;

        for (const auto& s : signs) {
            size_t p = slice.find(s);
            if (p != std::string_view::npos) {
                if (best_pos == std::string_view::npos || p < best_pos) {
                    best_pos = p;
                }
            }
        }

        if (best_pos != std::string_view::npos) {
            // 根据首字节正确判断 UTF-8 字符的字节长度
            unsigned char c = slice[best_pos];
            size_t char_len = 1; // fallback
            if ((c & 0x80) == 0x00) {
                char_len = 1;
            } else if ((c & 0xE0) == 0xC0) {
                char_len = 2;
            } else if ((c & 0xF0) == 0xE0) {
                char_len = 3;
            } else if ((c & 0xF8) == 0xF0) {
                char_len = 4;
            }
            if(best_pos <= bias_forward)  //同上
                start = start + best_pos + char_len;
        }
    }
    return text.substr(start,end-start);
}

