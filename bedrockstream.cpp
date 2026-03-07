#include "llmstream.h"
#include "bedrockstream.h"
#include <aws/core/Aws.h>
#include <aws/bedrock-runtime/BedrockRuntimeClient.h>
#include <aws/bedrock-runtime/model/InvokeModelRequest.h>
#include <aws/core/utils/json/JsonSerializer.h>
#include <aws/core/utils/memory/stl/AWSStringStream.h>
#include <iostream>


using namespace Aws::BedrockRuntime;
using namespace Aws::BedrockRuntime::Model;

bedrockstream::bedrockstream(const std::string& model_id, const std::string& region)
	: response_buffer(), model_id(model_id), region(region), temperature(0.3), top_p(0.9), max_tokens(2048)
{
}

bedrockstream::~bedrockstream()
{
}

bedrockstream& bedrockstream::operator<<(const std::string& prompt)
{
	try
	{
		Aws::Client::ClientConfiguration config;
		config.region = region;

		BedrockRuntimeClient client(config);

		// Build request payload based on model family
		Aws::Utils::Json::JsonValue payload;

		if (model_id.find("anthropic") != std::string::npos)
		{
			// Anthropic Claude format
			payload.WithString("prompt", "\n\nHuman: " + prompt + "\n\nAssistant:");
			payload.WithDouble("temperature", temperature);
			payload.WithDouble("top_p", top_p);
			payload.WithInteger("max_tokens_to_sample", max_tokens);
		}
		else if (model_id.find("amazon.nova") != std::string::npos)
		{
			// Amazon Nova format - uses messages API
			Aws::Utils::Json::JsonValue content_item;
			content_item.WithString("text", prompt);

			Aws::Utils::Array<Aws::Utils::Json::JsonValue> content_array(1);
			content_array[0] = std::move(content_item);

			Aws::Utils::Json::JsonValue message;
			message.WithString("role", "user");
			message.WithArray("content", std::move(content_array));

			Aws::Utils::Array<Aws::Utils::Json::JsonValue> messages_array(1);
			messages_array[0] = std::move(message);

			payload.WithArray("messages", std::move(messages_array));

			Aws::Utils::Json::JsonValue inference_config;
			inference_config.WithDouble("temperature", temperature);
			inference_config.WithDouble("topP", top_p);
			inference_config.WithInteger("maxTokens", max_tokens);
			payload.WithObject("inferenceConfig", std::move(inference_config));
		}
		else
		{
			// Generic format
			payload.WithString("prompt", prompt);
			payload.WithDouble("temperature", temperature);
			payload.WithDouble("top_p", top_p);
			payload.WithInteger("max_tokens", max_tokens);
		}

		std::string payload_str = payload.View().WriteCompact();

		InvokeModelRequest request;
		request.SetModelId(model_id);
		request.SetContentType("application/json");
		request.SetAccept("application/json");

		auto payload_stream = Aws::MakeShared<Aws::StringStream>("BedrockPayload");
		*payload_stream << payload_str;
		request.SetBody(payload_stream);

		auto outcome = client.InvokeModel(request);

		if (outcome.IsSuccess())
		{
			auto& result = outcome.GetResult();
			auto& body = result.GetBody();

			std::string response_str((std::istreambuf_iterator<char>(body)), std::istreambuf_iterator<char>());

			Aws::Utils::Json::JsonValue response_json(response_str);

			// Parse response based on model family
			if (model_id.find("anthropic") != std::string::npos)
			{
				response_buffer = response_json.View().GetString("completion");
			}
			else if (model_id.find("amazon.nova") != std::string::npos)
			{
				auto output = response_json.View().GetObject("output");
				auto message = output.GetObject("message");
				auto content = message.GetArray("content");
				if (content.GetLength() > 0)
				{
					response_buffer = content[0].AsObject().GetString("text");
				}
			}
			else
			{
				// Try common fields
				if (response_json.View().ValueExists("completion"))
				{
					response_buffer = response_json.View().GetString("completion");
				}
				else if (response_json.View().ValueExists("text"))
				{
					response_buffer = response_json.View().GetString("text");
				}
				else
				{
					response_buffer = response_str;
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
