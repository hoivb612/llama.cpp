Offspring for hoivb612
https://github.com/hoivb612/llama.cpp

#include <sys/stat.h>
#include <sys/types.h>
#include <string>
#include <hash>

static std::hash<std::string> hasher;
static const char* dir = "./llama_cache";

// create the cache dir if it does not exist yet
struct stat info;
if (stat(dir, &info) != 0) {
    mkdir(dir, 0777);
}

// default generated file name
std::string pfx_path(dir);
std::string full_file_path = pfx_path + "/" + std::to_string(hasher(pfx));
