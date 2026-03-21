#include <catch_amalgamated.hpp>

#include "turbo_ocr/common/serialization.h"

using turbo_ocr::OCRResultItem;
using turbo_ocr::Box;
using turbo_ocr::results_to_json;

TEST_CASE("results_to_json empty results", "[serialization]") {
  std::vector<OCRResultItem> results;
  auto json = results_to_json(results);
  CHECK(json == R"({"results":[]})");
}

TEST_CASE("results_to_json single result", "[serialization]") {
  Box box{{{{{10, 20}}, {{30, 20}}, {{30, 40}}, {{10, 40}}}}};
  std::vector<OCRResultItem> results = {
      {.text = "Hello", .confidence = 0.95f, .box = box}};
  auto json = results_to_json(results);

  // Check structure -- contains the text and bounding box
  CHECK(json.find("\"text\":\"Hello\"") != std::string::npos);
  CHECK(json.find("\"bounding_box\":[[10,20],[30,20],[30,40],[10,40]]") !=
        std::string::npos);
  // Starts and ends correctly
  CHECK(json.substr(0, 13) == "{\"results\":[{");
  CHECK(json.back() == '}');
}

TEST_CASE("results_to_json escapes special characters", "[serialization]") {
  Box box{};
  std::vector<OCRResultItem> results = {
      {.text = "He said \"hello\" \\ world", .confidence = 0.9f, .box = box}};
  auto json = results_to_json(results);

  CHECK(json.find(R"(He said \"hello\" \\ world)") != std::string::npos);
}

TEST_CASE("results_to_json escapes control characters", "[serialization]") {
  Box box{};
  std::string text_with_controls = "line1\nline2\ttab\rreturn";
  std::vector<OCRResultItem> results = {
      {.text = text_with_controls, .confidence = 0.8f, .box = box}};
  auto json = results_to_json(results);

  CHECK(json.find(R"(line1\nline2\ttab\rreturn)") != std::string::npos);
}

TEST_CASE("results_to_json escapes low control chars as unicode", "[serialization]") {
  Box box{};
  // \x01 should be escaped as \u0001
  std::string text = "a";
  text += '\x01';
  text += "b";
  std::vector<OCRResultItem> results = {
      {.text = text, .confidence = 0.8f, .box = box}};
  auto json = results_to_json(results);

  CHECK(json.find(R"(a\u0001b)") != std::string::npos);
}

TEST_CASE("results_to_json multiple results separated by commas", "[serialization]") {
  Box box{};
  std::vector<OCRResultItem> results = {
      {.text = "A", .confidence = 0.9f, .box = box},
      {.text = "B", .confidence = 0.8f, .box = box},
      {.text = "C", .confidence = 0.7f, .box = box},
  };
  auto json = results_to_json(results);

  // Count commas between result objects (should be 2)
  int comma_count = 0;
  bool in_results = false;
  for (size_t i = 0; i < json.size(); ++i) {
    if (json[i] == '[' && i > 0 && json[i - 1] == ':')
      in_results = true;
    if (in_results && json[i] == '}' && i + 1 < json.size() && json[i + 1] == ',')
      comma_count++;
  }
  CHECK(comma_count == 2);
}
