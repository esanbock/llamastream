#pragma once

#include <string>

class ollamastream {
private:
    std::string response_buffer;
    std::string model_name;
    int max_tokens;

public:
    // model_name is optional; max_tokens defaults to 512
    ollamastream(const std::string& model_name );
    ~ollamastream();

    ollamastream& operator<<(const std::string& prompt);
    ollamastream& operator>>(std::string& response);
    void restart();
};
