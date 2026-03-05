#pragma once

#include <string>
#include "llmstream.h"

class bedrockstream : public llmstream
{
private:
    std::string response_buffer;
    std::string model_id;
    std::string region;
    double temperature;
    double top_p;
    int max_tokens;

public:
    bedrockstream(const std::string& model_id, const std::string& region = "us-east-1");
    ~bedrockstream();

    bedrockstream& operator<<(const std::string& prompt) override;
    bedrockstream& operator>>(std::string& response) override;
    
    void restart();
    void set_temperature(double temp) { temperature = temp; }
    void set_max_tokens(int tokens) { max_tokens = tokens; }
    void set_top_p(double p) { top_p = p; }
};
