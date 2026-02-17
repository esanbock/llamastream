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

ollamastream::ollamastream(const std::string& model_name, const std::string& server_host)
    : response_buffer(), model_name(model_name), server_host(server_host)
{
}

ollamastream::~ollamastream()
{
}

ollamastream& ollamastream::operator<<(const std::string& prompt)
{
    CURL* curl = curl_easy_init();
    if (!curl) {
        response_buffer.clear();
        return *this;
    }

    std::string url = "http://" + server_host + ":11434/api/generate";
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);

    // Construct payload according to Ollama's API: model (optional), prompt, stream, options
    json payload = json::object();
    if (!model_name.empty()) payload["model"] = model_name;
    payload["prompt"] = prompt;

    // Explicitly request non-streaming output so we receive a single JSON response
    payload["stream"] = false;

    // Add options to control model behavior
    json options = json::object();
    options["temperature"] = 0.3;  // Lower temperature for more focused responses
    options["top_p"] = 0.9;
    payload["options"] = options;

    std::string payload_str = payload.dump();
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, payload_str.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE_LARGE, (curl_off_t)payload_str.size());

    struct curl_slist* headers = NULL;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);

    std::string response_string;
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, curl_write_cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_string);

    CURLcode res = curl_easy_perform(curl);
    
    std::cerr << "DEBUG: CURL result: " << res << std::endl;
    std::cerr << "DEBUG: Raw response: " << response_string.substr(0, 500) << std::endl;
    
    if (res != CURLE_OK) {
        std::cerr << "DEBUG: CURL error: " << curl_easy_strerror(res) << std::endl;
        response_buffer.clear();
    } else {
        try {
            auto parsed = json::parse(response_string);
            std::cerr << "DEBUG: Parsed JSON successfully" << std::endl;
            
            // Try several common fields that might contain the generated text
            if (parsed.contains("response")) {
                response_buffer = parsed["response"].get<std::string>();
                std::cerr << "DEBUG: Found 'response' field" << std::endl;
            } else if (parsed.contains("text")) {
                response_buffer = parsed["text"].get<std::string>();
                std::cerr << "DEBUG: Found 'text' field" << std::endl;
            } else if (parsed.contains("results") && parsed["results"].is_array() && !parsed["results"].empty()) {
                auto &first = parsed["results"][0];
                if (first.contains("text")) response_buffer = first["text"].get<std::string>();
                else response_buffer = response_string;
                std::cerr << "DEBUG: Found 'results' array" << std::endl;
            } else if (parsed.contains("output") && parsed["output"].is_string()) {
                response_buffer = parsed["output"].get<std::string>();
                std::cerr << "DEBUG: Found 'output' field" << std::endl;
            } else {
                std::cerr << "DEBUG: No known field found, using raw response" << std::endl;
                response_buffer = response_string;
            }
            
            std::cerr << "DEBUG: Response buffer length: " << response_buffer.length() << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "DEBUG: JSON parse error: " << e.what() << std::endl;
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
