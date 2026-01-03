#pragma once

#include <mongocxx/client.hpp>
#include <mongocxx/uri.hpp>
#include <bsoncxx/builder/stream/document.hpp>

class MongoManager {
public:
    MongoManager(const std::string& uri_str);
    void saveData(const std::string& url, const std::string& html_title, int status_code);
private:
    mongocxx::uri uri;
    mongocxx::client client;
    mongocxx::collection collection;
};