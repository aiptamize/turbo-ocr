#include "turbo_ocr/recognition/ctc_decode.h"

#include <format>
#include <fstream>
#include <iostream>
#include <string>

namespace turbo_ocr::recognition {

std::pair<std::string, float>
ctc_greedy_decode(const int *indices, const float *scores, int seq_len,
                  const std::vector<std::string> &label_list) {
  std::string text;
  text.reserve(seq_len);
  float score = 0.0f;
  int count = 0;
  int last_index = -1;

  for (int i = 0; i < seq_len; i++) {
    int index = indices[i];
    if (index != last_index) {
      if (index != 0 && index < static_cast<int>(label_list.size())) {
        text += label_list[index];
        score += scores[i];
        count++;
      }
    }
    last_index = index;
  }
  if (count > 0)
    score /= count;
  return {text, score};
}

std::pair<std::string, float>
ctc_greedy_decode_raw(const float *logits, int seq_len, int num_classes,
                      const std::vector<std::string> &label_list) {
  std::string text;
  text.reserve(seq_len);
  float score = 0.0f;
  int count = 0;
  int last_index = -1;

  for (int i = 0; i < seq_len; i++) {
    const float *row = logits + i * num_classes;
    int index = 0;
    float max_val = row[0];
    for (int j = 1; j < num_classes; j++) {
      if (row[j] > max_val) {
        max_val = row[j];
        index = j;
      }
    }

    if (index != last_index) {
      if (index != 0 && index < static_cast<int>(label_list.size())) {
        text += label_list[index];
        score += max_val;
        count++;
      }
    }
    last_index = index;
  }
  if (count > 0)
    score /= count;
  return {text, score};
}

bool load_label_dict(const std::string &dict_path,
                     std::vector<std::string> &label_list) {
  std::ifstream in(dict_path);
  if (!in) [[unlikely]] {
    std::cerr << std::format("[Rec] Failed to open dictionary: {}",
                             dict_path)
              << '\n';
    return false;
  }
  std::string line;
  while (getline(in, line)) {
    if (!line.empty() && line.back() == '\n')
      line.pop_back();
    if (!line.empty() && line.back() == '\r')
      line.pop_back();
    label_list.push_back(std::move(line));
  }
  label_list.push_back(" ");
  return true;
}

} // namespace turbo_ocr::recognition
