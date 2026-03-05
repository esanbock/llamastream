#pragma once
#include <string>

class llmstream
{
public:
    virtual ~llmstream() = default;
    virtual llmstream& operator<<(const std::string& prompt)=0;
    virtual llmstream& operator>>(std::string& response)=0;
};
