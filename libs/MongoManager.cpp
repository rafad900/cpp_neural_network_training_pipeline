#include "MongoManager.hpp"
#include <bsoncxx/json.hpp>
#include <

namespace custom_bson = bsoncxx::builder::stream;

MongoManager::MongoManager(const std::string& uri_str) : uri(uri_str), client(uri) { 
  collection = client["scraper_db"]["results"];
}

void MongoManager::saveData(const std::string& url, const std::string& html_title, int status_code) {
    try {
        auto builder = custom_bson::document{};
        bsoncxx::document::value doc_value = builder
        << "url" << url << "title" << html_title << "status" << status_code
        << "time" << bsoncxx::types::b_date{std::chrono::system_clock::now()}
        << custom_bson::finalize;

        collection.insert_one(doc_value.view());
    } catch (const std::exception& e) {

    }
}
