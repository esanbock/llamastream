#include "ollamastream.h"
#include <curl/curl.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

static size_t curl_write_cb(void* contents, size_t size, size_t nmemb, void* userp)
{
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
}

// model_name optional; max_tokens controls request
ollamastream::ollamastream(const std::string& model_name)
    : response_buffer(), model_name(model_name)
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

    std::string url = "http://localhost:11434/api/generate";
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);

    // Construct payload according to Ollama's API: model (optional), prompt, stream, options
    json payload = json::object();
    if (!model_name.empty()) payload["model"] = model_name;
    payload["prompt"] = prompt;

    // Explicitly request non-streaming output so we receive a single JSON response
    payload["stream"] = false;

    // Disable separate 'thinking' output so the response field contains the model text
    payload["think"] = false;


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
    if (res != CURLE_OK) {
        response_buffer.clear();
    } else {
        try {
            auto parsed = json::parse(response_string);
            // Try several common fields that might contain the generated text
            if (parsed.contains("text")) {
                response_buffer = parsed["text"].get<std::string>();
            } else if (parsed.contains("response")) {
                response_buffer = parsed["response"].get<std::string>();
            } else if (parsed.contains("results") && parsed["results"].is_array() && !parsed["results"].empty()) {
                // Some Ollama responses include a results array with outputs
                auto &first = parsed["results"][0];
                if (first.contains("text")) response_buffer = first["text"].get<std::string>();
                else response_buffer = response_string;
            } else if (parsed.contains("output") && parsed["output"].is_string()) {
                response_buffer = parsed["output"].get<std::string>();
            } else {
                // fallback to raw
                response_buffer = response_string;
            }
        } catch (...) {
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
