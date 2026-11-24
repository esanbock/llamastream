#include "claudestream.h"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <iostream>

using json = nlohmann::json;

static size_t claude_write_cb(void* contents, size_t size, size_t nmemb, void* userp)
{
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

claudestream::claudestream(const std::string& model_id, const std::string& api_key,
                           const std::string& effort, int max_tokens)
    : response_buffer(), model_id(model_id), api_key(api_key),
      effort(effort), max_tokens(max_tokens)
{
}

claudestream::~claudestream()
{
}

claudestream& claudestream::operator<<(const std::string& prompt)
{
    CURL* curl = curl_easy_init();
    if (!curl)
    {
        response_buffer.clear();
        return *this;
    }

    curl_easy_setopt(curl, CURLOPT_URL, "https://api.anthropic.com/v1/messages");
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300L);

    json payload;
    payload["model"] = model_id;
    payload["max_tokens"] = max_tokens;
    payload["messages"] = json::array({
        {{"role", "user"}, {"content", prompt}}
    });

    payload["output_config"] = {{"effort", effort}};

    std::string payload_str = payload.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload_str.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE_LARGE, (curl_off_t)payload_str.size());

    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");

    std::string auth_header = "x-api-key: " + api_key;
    headers = curl_slist_append(headers, auth_header.c_str());
    headers = curl_slist_append(headers, "anthropic-version: 2023-06-01");

    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    std::string response_string;
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, claude_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);

    CURLcode res = curl_easy_perform(curl);

    if (res != CURLE_OK)
    {
        std::cerr << "Claude API error: " << curl_easy_strerror(res) << std::endl;
        response_buffer.clear();
    }
    else
    {
        try
        {
            auto parsed = json::parse(response_string);

            if (parsed.contains("error"))
            {
                std::string err_msg = parsed["error"].value("message", "unknown error");
                std::cerr << "Claude API error: " << err_msg << std::endl;
                response_buffer.clear();
            }
            else if (parsed.contains("content") && parsed["content"].is_array())
            {
                response_buffer.clear();
                for (auto& block : parsed["content"])
                {
                    if (block.value("type", "") == "text")
                    {
                        response_buffer += block.value("text", "");
                    }
                }
            }
            else
            {
                std::cerr << "Claude API: unexpected response format" << std::endl;
                response_buffer.clear();
            }
        }
        catch (const std::exception& e)
        {
            std::cerr << "Claude API JSON parse error: " << e.what() << std::endl;
            response_buffer.clear();
        }
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    return *this;
}

claudestream& claudestream::operator>>(std::string& response)
{
    response = response_buffer;
    response_buffer.clear();
    return *this;
}

void claudestream::restart()
{
    response_buffer.clear();
}
