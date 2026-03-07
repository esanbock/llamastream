#pragma once

#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <string>
#include <memory>
#include "llmstream.h"

namespace Aws { namespace BedrockRuntime { class BedrockRuntimeClient; namespace Model { class InferenceConfiguration; } } }

class bedrockstream : public llmstream
{
private:
    std::string response_buffer;
    std::string model_id;
    std::unique_ptr<Aws::BedrockRuntime::BedrockRuntimeClient> client;
    std::unique_ptr<Aws::BedrockRuntime::Model::InferenceConfiguration> inference_config;

public:
    bedrockstream(const std::string& model_id, const std::string& region = "us-east-1",
                  double temperature = 0.3, int max_tokens = 2048, double top_p = 0.9);
    ~bedrockstream();

    bedrockstream& operator<<(const std::string& prompt) override;
    bedrockstream& operator>>(std::string& response) override;
    
    void restart();
};
