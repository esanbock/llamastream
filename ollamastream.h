#pragma once

#include <string>
#include "llmstream.h"

class ollamastream : public llmstream
{
private:
    std::string response_buffer;
    std::string model_name;
    std::string server_host;
    int max_tokens;

public:
    ollamastream(const std::string& model_name, const std::string& server_host = "localhost");
    ~ollamastream();

    ollamastream& operator<<(const std::string& prompt) override;
    ollamastream& operator>>(std::string& response) override;
    void restart();
};
