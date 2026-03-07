#ifndef NOMINMAX
#define NOMINMAX
#endif

#include "llmstream.h"
#include "bedrockstream.h"
#include <aws/core/Aws.h>
#include <aws/bedrock-runtime/BedrockRuntimeClient.h>
#include <aws/bedrock-runtime/model/ConverseRequest.h>
#include <aws/bedrock-runtime/model/ConverseResult.h>
#include <aws/bedrock-runtime/model/Message.h>
#include <aws/bedrock-runtime/model/ContentBlock.h>
#include <aws/bedrock-runtime/model/ConversationRole.h>
#include <aws/bedrock-runtime/model/InferenceConfiguration.h>
#include <iostream>

using namespace Aws::BedrockRuntime;
using namespace Aws::BedrockRuntime::Model;

bedrockstream::bedrockstream(const std::string& model_id, const std::string& region,
    double temperature, int max_tokens, double top_p)
    : response_buffer(), model_id(model_id)
{
    Aws::Client::ClientConfiguration config;
    config.region = region;
    client = std::make_unique<BedrockRuntimeClient>(config);

    inference_config = std::make_unique<InferenceConfiguration>();
    inference_config->SetMaxTokens(max_tokens);
    inference_config->SetTemperature(temperature);
    inference_config->SetTopP(top_p);
}

bedrockstream::~bedrockstream()
{
}

bedrockstream& bedrockstream::operator<<(const std::string& prompt)
{
    try
    {
        ContentBlock content;
        content.SetText(prompt);

        Message message;
        message.SetRole(ConversationRole::user);
        message.AddContent(std::move(content));

        ConverseRequest request;
        request.SetModelId(model_id);
        request.AddMessages(std::move(message));
        request.SetInferenceConfig(*inference_config);

        auto outcome = client->Converse(request);

        if (outcome.IsSuccess())
        {
            auto& result = outcome.GetResult();
            auto& output_message = result.GetOutput().GetMessage();
            auto& content_blocks = output_message.GetContent();

            response_buffer.clear();
            for (auto& block : content_blocks)
            {
                if (block.TextHasBeenSet())
                {
                    response_buffer += block.GetText();
                }
            }
        }
        else
        {
            std::cerr << "Bedrock error: " << outcome.GetError().GetMessage() << std::endl;
            response_buffer.clear();
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "Exception in bedrockstream: " << e.what() << std::endl;
        response_buffer.clear();
    }

    return *this;
}

bedrockstream& bedrockstream::operator>>(std::string& response)
{
    response = response_buffer;
    response_buffer.clear();
    return *this;
}

void bedrockstream::restart()
{
    response_buffer.clear();
}
