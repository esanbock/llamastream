#include "llmstream.h"
#include "ollamastream.h"
#include <curl/curl.h>
#include <nlohmann/json.hpp>
#include <iostream>

using json = nlohmann::json;

static size_t curl_write_cb(void* contents, size_t size, size_t nmemb, void* userp)
{
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

ollamastream::ollamastream(const std::string& model_name, const std::string& server_host,
                           double temperature, double top_p)
    : response_buffer(), model_name(model_name), server_host(server_host),
      temperature(temperature), top_p(top_p)
{
}

ollamastream::~ollamastream()
{
}

ollamastream& ollamastream::operator<<(const std::string& prompt)
{
    CURL* curl = curl_easy_init();
    if (!curl)
    {
        response_buffer.clear();
        return *this;
    }

    std::string url = "http://" + server_host + ":11434/api/generate";
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 30L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 300L);

    json payload = json::object();
    if (!model_name.empty()) payload["model"] = model_name;
    payload["prompt"] = prompt;
    payload["stream"] = false;

    json options = json::object();
    options["temperature"] = temperature;
    options["top_p"] = top_p;
    payload["options"] = options;

    std::string payload_str = payload.dump(-1, ' ', false, nlohmann::json::error_handler_t::replace);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload_str.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE_LARGE, (curl_off_t)payload_str.size());

    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    std::string response_string;
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);

    CURLcode res = curl_easy_perform(curl);

    if (res != CURLE_OK)
    {
        std::cerr << "Ollama API error: " << curl_easy_strerror(res) << std::endl;
        response_buffer.clear();
    }
    else
    {
        try
        {
            auto parsed = json::parse(response_string);

            if (parsed.contains("response"))
            {
                response_buffer = parsed["response"].get<std::string>();
            }
            else if (parsed.contains("text"))
            {
                response_buffer = parsed["text"].get<std::string>();
            }
            else if (parsed.contains("results") && parsed["results"].is_array() && !parsed["results"].empty())
            {
                auto &first = parsed["results"][0];
                if (first.contains("text"))
                    response_buffer = first["text"].get<std::string>();
                else
                    response_buffer = response_string;
            }
            else if (parsed.contains("output") && parsed["output"].is_string())
            {
                response_buffer = parsed["output"].get<std::string>();
            }
            else
            {
                std::cerr << "Ollama API: unexpected response format" << std::endl;
                response_buffer = response_string;
            }
        }
        catch (const std::exception& e)
        {
            std::cerr << "Ollama API JSON parse error: " << e.what() << std::endl;
            response_buffer = response_string;
        }
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    return *this;
}

ollamastream& ollamastream::operator>>(std::string& response)
{
    response = response_buffer;
    response_buffer.clear();
    return *this;
}

void ollamastream::restart()
{
    response_buffer.clear();
}
