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
    double temperature;
    double top_p;

public:
    ollamastream(const std::string& model_name, const std::string& server_host = "localhost",
                 double temperature = 0.3, double top_p = 0.9);
    ~ollamastream();

    ollamastream& operator<<(const std::string& prompt) override;
    ollamastream& operator>>(std::string& response) override;
    void restart();
};
