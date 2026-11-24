#pragma once

#include <string>
#include "llmstream.h"

class claudestream : public llmstream
{
private:
    std::string response_buffer;
    std::string model_id;
    std::string api_key;
    std::string effort;
    int max_tokens;

public:
    claudestream(const std::string& model_id, const std::string& api_key,
                 const std::string& effort = "low", int max_tokens = 4096);
    ~claudestream();

    claudestream& operator<<(const std::string& prompt) override;
    claudestream& operator>>(std::string& response) override;

    void restart();
};
