// llamastream.cpp : Defines the functions for the static library.
//

#include "llamastream.h"
#include <vector>
#include <stdexcept>
#include <iostream>
#include <algorithm>
using namespace std;

llamastream::llamastream(const std::string& model_path, int context_params)
{
	// load backend
	ggml_backend_load_all();
	// initialize model
	llama_model_params model_params = llama_model_default_params();
	model_params.n_gpu_layers = 99; // <--- enables GPU offload
	model = llama_model_load_from_file(model_path.c_str(), model_params);
	if (!model)
		throw std::runtime_error("Failed to load model");
	
	vocab = llama_model_get_vocab(model);
	
	llama_context_params ctx_params = llama_context_default_params();
	ctx_params.n_ctx = context_params;
	ctx = llama_init_from_model(model, ctx_params);
	if (!ctx) throw std::runtime_error("Failed to create context");
}

llamastream::~llamastream()
{
	llama_sampler_free(sampler);
	llama_free(ctx);
	llama_model_free(model);
}

llamastream& llamastream::operator<<(const std::string& prompt)
{
	sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
	llama_sampler_chain_add(sampler, llama_sampler_init_greedy());


	std::string formatted = "<|start_header_id|>user<|end_header_id|>\n\n" + prompt + "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";

	std::vector<llama_token> tokens(formatted.size() + 1);
	int n_tokens = llama_tokenize(vocab, formatted.c_str(), formatted.size(), tokens.data(), tokens.size(), true, true);

	int max_ctx = llama_n_ctx(ctx);
	int reserved = 100; // Reserve space for response
	if (n_tokens > max_ctx - reserved)
	{
		// Truncate oldest tokens to fit context window
		int start = n_tokens - (max_ctx - reserved);
		if (start < n_tokens)
		{
			std::vector<llama_token> truncated_tokens(tokens.begin() + start, tokens.begin() + n_tokens);
			tokens = std::move(truncated_tokens);
			n_tokens = static_cast<int>(tokens.size());
			cerr << "Input truncated (oldest tokens removed) to fit context window" << endl;
		}
	}

	llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
	if (llama_decode(ctx, batch) != 0)
	{
		cerr << "Decode failed, restarting context" << endl;
		restart();
		// Retry with fresh context
		batch = llama_batch_get_one(tokens.data(), min(n_tokens, 512));
		llama_decode(ctx, batch);
	}

	for (int i = 0; i < 500; i++)
	{
		llama_token new_token = llama_sampler_sample(sampler, ctx, -1);
		if (llama_vocab_is_eog(vocab, new_token)) break;

		char buf[256];
		int n = llama_token_to_piece(vocab, new_token, buf, sizeof(buf), 0, true);
		response_buffer += std::string(buf, n);

		batch = llama_batch_get_one(&new_token, 1);
		if (llama_decode(ctx, batch) != 0) 
			break; // Stop on decode error
	}
	return *this;
}

llamastream& llamastream::operator>>(std::string& response)
{
	response = response_buffer;
	response_buffer.clear();
	return *this;
}

void llamastream::restart()
{
	llama_sampler_free(sampler);
	llama_free(ctx);

	llama_context_params ctx_params = llama_context_default_params();
	ctx_params.n_ctx = 1024;  // Reduced from 2048 to prevent memory issues
	ctx = llama_init_from_model(model, ctx_params);
	if (!ctx) throw std::runtime_error("Failed to recreate context");

	sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
	llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
}
