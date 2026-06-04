#include "mragapp.h"
#include <print>

int main(int argc, char* argv[]) {
    try {
        std::vector<std::string> paths = parseCli(argc, argv);
        if(paths.empty()) {
            return 0;   // 直接退出，并输出错误信息
        }

        MragApp app("config.json");
        if(paths.size() == 1) {     // chat 模式 同时加载生成模型和向量模型
            app.loadEmbedding();
            app.loadGeneration();
            app.loadDatabase(paths[0]);
            app.chatLoop();
        } else if(paths.size() > 1) {   // ingest 模式 只需加载向量模型
            std::string db_output_path = paths.back();
            app.loadEmbedding();   
            paths.pop_back();
            app.loadText(paths);
            app.buildKnowledgeBase(db_output_path);
        }
    } catch(const std::exception& exception) {
        std::println(stderr, "\n[致命错误] 程序抛出异常终止：{}", exception.what());
        return 1;
    } catch(...) {
        std::println(stderr, "\n[致命错误] 发生了未知的异常！");
        return 1;
    }
    return 0;
}