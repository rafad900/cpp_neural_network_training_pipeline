#include <iostream>
#include <string>
#include <curl/curl.h>
#include "rest_client.hpp"
// Callback function to handle the data received from the server

/*
    I'm using CURL and all it does it just get the chuncks from the server.
    It does not know how to parse and process the datagram chunks it receives.
    I have to put them all into a string that I can later do some processing on.
*/ 

int CustomClient::create_client(const std::string& url, const std::string& http_method, const std::string& json) {
    CURL * curl;
    CURLcode res;
    std::string readbuffer;
    curl = curl_easy_init();
    struct curl_slist *headers = NULL;
    
    if (!curl) {
        std::cout << "DID NOT CREATE CURL\n";
        curl_easy_cleanup(curl);
        return 1;
    }

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, CustomClient::WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &readbuffer);

    if (!http_method.empty()) {
        
        headers = curl_slist_append(headers, "Content-Type: application/type");
        curl_easy_setopt(curl, CURLOPT_CUSTOMREQUEST, http_method.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    }

    if (!json.empty()) {
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json.c_str());
    }

    res = curl_easy_perform(curl);

    if (res != CURLE_OK) {
        std::cerr << "CURL FAILED: " << curl_easy_strerror(res) << std::endl;
        curl_easy_cleanup(curl);
        curl_slist_free_all(headers);
        return 1;
    }
    
    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
    std::cout << "Response code: " << http_code << std::endl;

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    return 0;
}

    


