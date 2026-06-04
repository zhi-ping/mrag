#include "vectordatabase.h"
#include <queue>
#include <print>

VectorDatabase:: VectorDatabase(const AppConfig& config) : top_k_(config.top_k) { }

bool VectorDatabase::saveToDisk(const std::string& out_path) const {
    std::println(stderr,"正在构建数据库...");
    //确保目录存在
    std::filesystem::create_directories(std::filesystem::path(out_path).parent_path());
    //再创建或打开文件
    std::ofstream out(out_path);
    if (!out) {
        std::println(stderr, "错误: 无法打开或创建{}", out_path);
        return false;
    }

    //写入魔数
    out.write(reinterpret_cast<const char*>(&MAGIC_NUMBER),sizeof(MAGIC_NUMBER));
    
    //写入版本号
    uint64_t v_size = VERSION.size();
    out.write(reinterpret_cast<const char*>(&v_size), sizeof(v_size));
    out.write(VERSION.data(), v_size);

    //写入数据总量
    uint64_t chunk_count = chunks_.size();
    out.write(reinterpret_cast<const char*>(&chunk_count),sizeof(chunk_count));

    //逐个序列化chunk
    for(const auto& chunk : chunks_) {
        chunk.serialize(out);
    }

    std::println(stderr,"数据库已保存到 {}", out_path);
    
    return true;
}

//to be tested
bool VectorDatabase::loadFromDisk(const std::string& in_path) {
    //检测数据库文件是否成功打开
    std::println(stderr,"正在加载数据库...");
    std::ifstream in(in_path);
    if(!in.is_open()) {
        std::println(stderr,"Error: Failed to open {}", in_path);
        return false;
    }
    //读取文件头
    int magic_number{};
    in.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));

    uint64_t v_size{};
    in.read(reinterpret_cast<char*>(&v_size), sizeof(v_size));
    std::string version; version.resize(v_size);
    in.read(version.data(),v_size);

    //校验文件头
    if(magic_number != MAGIC_NUMBER || version != VERSION) {
        std::println(stderr,"错误：数据库魔数或版本号不匹配");
        return false;
    }

    //读取数据总量
    uint64_t chunk_count{};
    in.read(reinterpret_cast<char*>(&chunk_count),sizeof(chunk_count));

    for (uint64_t i = 0; i < chunk_count; ++i) {
            Chunk tmp = Chunk::deserialize(in);
            this->insert(tmp);
        }
    return true;
}

/**
 *  @brief lamada表达式版本，会频繁计算余弦相似度，效率较低
std::vector<Chunk> VectorDatabase::search(std::vector<float>& query_emb) const {
    std::vector<Chunk> topk_chunks;
    topk_chunks.reserve(top_k_);

    auto cmp = [&query_emb] (const Chunk& left, const Chunk& right) {
        return cosine(query_emb,left.embedding) > cosine(query_emb,right.embedding);
    };

    std::priority_queue<Chunk,std::vector<Chunk>,decltype(cmp)> chunk_pq(cmp);

    //利用优先队列筛选出 topk chunk
    for(const auto& chunk : chunks_) {
        if(chunk_pq.size() < top_k_) {
            chunk_pq.push(chunk);
            continue;
        }
        
        if (cosine(query_emb,chunk.embedding) > cosine(query_emb,chunk_pq.top().embedding)) {
            chunk_pq.pop();
            chunk_pq.push(chunk);
        }
    }

    //取出优先队列中的 topk chunk
    while(!chunk_pq.empty()) {
        topk_chunks.push_back(chunk_pq.top());
        chunk_pq.pop();
    }

    //反转，保证chunk按相似度从大到小排列
    std::reverse(topk_chunks.begin(),topk_chunks.end());

    return topk_chunks;
}
*/

/**
 * @brief 每个chunk只计算一次相似值，效率较高
 */
std::vector<Chunk> VectorDatabase::search(const std::vector<float>& query_emb) const {
    struct Node {
        float score;
        size_t idx; // 存下标，避免重复存储chunk对象
        bool operator>(const Node& other) const { return score > other.score; }
    };

    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;

    for (size_t i = 0; i < chunks_.size(); ++i) {
        // 每个chunk只计算一次相似度，提高效率
        float cur_sim = cosine(query_emb, chunks_[i].embedding);
        
        /// @test
        // std::println("相似度： {}",cur_sim);
        
        if (pq.size() < top_k_) {
            pq.emplace(cur_sim, i);
        } else if (cur_sim > pq.top().score) {
            pq.pop();
            pq.emplace(cur_sim, i);
        }
    }

    std::vector<Chunk> results;
    results.reserve(pq.size());
    while (!pq.empty()) {
        results.push_back(chunks_[pq.top().idx]); // 根据下标拷贝最终结果
        pq.pop();
    }

    //调整顺序，按相似度从大到小排列
    std::reverse(results.begin(), results.end());
    return results;
}

float cosine(const std::vector<float>& v1, const std::vector<float>& v2) {
    
    if (v1.size() != v2.size())
        throw std::invalid_argument("Error: Vector dimensions must match to compute cosine similarity.");
    if (v1.empty() || v2.empty())
        throw std::invalid_argument("Error: Vector should not be empty");

    float dot_product = 0.0f;
    float norm_v1 = 0.0f;
    float norm_v2 = 0.0f;

    for (size_t i = 0; i < v1.size(); ++i) {
        dot_product += v1[i] * v2[i];
        norm_v1 += v1[i] * v1[i];
        norm_v2 += v2[i] * v2[i];
    }

    if (norm_v1 == 0.0f || norm_v2 == 0.0f) {
        return 0.0f;
    }

    return dot_product / (std::sqrt(norm_v1 * norm_v2));
}

void VectorDatabase::insert(const Chunk& chunk) {
    chunks_.push_back(chunk);
}

void VectorDatabase::insert(const std::vector<Chunk>& chunks) {
    chunks_.append_range(chunks);
}

void VectorDatabase::setTopk(const AppConfig& config) {
    this->top_k_ = config.top_k;
}