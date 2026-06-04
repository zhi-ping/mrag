#include "chunk.h"

Chunk::Chunk(const std::string& t, const std::string& meta) 
    :text(t), metadata(meta) {
        static uint64_t global_chunk_counter = 0; //全局chunk计数器
        id = global_chunk_counter++;
    }

void Chunk::serialize(std::ofstream &out) const {
    // 1. 序列化 ID (8字节)
    out.write(reinterpret_cast<const char*>(&id), sizeof(id));

    // 2. 序列化 text (长度 + 内容)
    uint64_t text_len = text.size();
    out.write(reinterpret_cast<const char*>(&text_len), sizeof(text_len));
    out.write(text.data(), text_len); // string 底层就是 char 数组，直接写入

    // 3. 序列化 metadata (长度 + 内容)
    uint64_t meta_len = metadata.size();
    out.write(reinterpret_cast<const char*>(&meta_len), sizeof(meta_len));
    out.write(metadata.data(), meta_len);

    // 4. 序列化 embedding (维度 + 数组内容)
    uint64_t emb_len = embedding.size();
    out.write(reinterpret_cast<const char*>(&emb_len), sizeof(emb_len));
    out.write(reinterpret_cast<const char*>(embedding.data()), emb_len * sizeof(float));
}

Chunk Chunk::deserialize(std::ifstream& in) {
    Chunk chunk; 
    // 1. 反序列化 ID
    in.read(reinterpret_cast<char*>(&chunk.id), sizeof(chunk.id));

    // 2. 反序列化 text
    uint64_t text_len = 0;
    in.read(reinterpret_cast<char*>(&text_len), sizeof(text_len));
    chunk.text.resize(text_len);
    in.read(chunk.text.data(), text_len); // 再向这块内存灌入数据

    // 3. 反序列化 metadata
    uint64_t meta_len = 0;
    in.read(reinterpret_cast<char*>(&meta_len), sizeof(meta_len));
    chunk.metadata.resize(meta_len);
    in.read(chunk.metadata.data(), meta_len);

    // 4. 反序列化 embedding
    uint64_t emb_len = 0;
    in.read(reinterpret_cast<char*>(&emb_len), sizeof(emb_len));
    chunk.embedding.resize(emb_len); 
    in.read(reinterpret_cast<char*>(chunk.embedding.data()), emb_len * sizeof(float));

    return chunk;
}