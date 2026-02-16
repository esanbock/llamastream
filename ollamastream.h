#pragma once

#include <string>

class ollamastream {
private:
    std::string response_buffer;
    std::string model_name;
    std::string server_host;
    int max_tokens;

public:
    ollamastream(const std::string& model_name, const std::string& server_host = "localhost");
    ~ollamastream();

    ollamastream& operator<<(const std::string& prompt);
    ollamastream& operator>>(std::string& response);
    void restart();
};
