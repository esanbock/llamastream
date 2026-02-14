#pragma once
#include <string>
#include "llama.h"

class llamastream {
private:
    llama_model* model;
    llama_context* ctx;
    const llama_vocab* vocab;
    llama_sampler* sampler;
    std::string response_buffer;

public:
    llamastream(const std::string& model_path, int context_params);
    ~llamastream();
    
    llamastream& operator<<(const std::string& prompt);
    llamastream& operator>>(std::string& response);
    void restart();

};
