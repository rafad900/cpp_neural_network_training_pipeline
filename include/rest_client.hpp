#pragma once

#include <string>

class CustomClient {
private:
  static size_t WriteCallback(void *contents, size_t size, size_t nmemb, void *userp) {
    ((std::string*)userp)->append((char*)contents, size * nmemb);
    return size * nmemb;
  }

public:

  CustomClient() {};
  ~CustomClient() {};
  int create_client(const std::string& url, const std::string& http_method = "", const std::string& json = "");
};