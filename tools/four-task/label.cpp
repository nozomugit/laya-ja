// Read JSONL prompts, reuse their common prefix, output the logits of answer
// tokens.
#include "ggml-backend.h"
#include "llama.h"
#include "nlohmann/json.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using json = nlohmann::json;
using tokens = std::vector<llama_token>;
tokens encode(const llama_vocab *v, const std::string &s, bool special) {
  tokens result(s.size() + 16);
  int n = llama_tokenize(v, s.data(), s.size(), result.data(), result.size(),
                         false, special);
  if (n < 0)
    throw std::runtime_error("token count");
  result.resize(n);
  return result;
}
void add(llama_batch &b, int token, int pos, int seq, bool logits) {
  int i = b.n_tokens++;
  b.token[i] = token;
  b.pos[i] = pos;
  b.n_seq_id[i] = 1;
  b.seq_id[i][0] = seq;
  b.logits[i] = logits;
}
int main(int argc, char **argv) {
  if (argc != 6 && argc != 7)
    return 2;
  ggml_backend_load_all();
  llama_backend_init();
  auto mp = llama_model_default_params();
  mp.n_gpu_layers = 0;
  auto m = llama_model_load_from_file(argv[1], mp);
  if (!m)
    return 3;
  auto vocab = llama_model_get_vocab(m);
  int group = std::stoi(argv[4]), threads = std::stoi(argv[5]);
  const int budget = argc == 7 ? std::stoi(argv[6]) : 0;
  if (budget < 0 || budget > 256)
    return 2;
  auto cp = llama_context_default_params();
  cp.n_seq_max = group;
  cp.n_ctx = 768 * group;
  cp.n_batch = 1024;
  cp.n_ubatch = 256;
  cp.n_threads = threads;
  cp.n_threads_batch = threads;
  cp.kv_unified = false;
  auto c = llama_init_from_model(m, cp);
  if (!c)
    return 4;
  std::ifstream input(argv[2]);
  std::ofstream out(argv[3], std::ios::app);
  if (!input || !out)
    return 5;
  std::vector<tokens> data;
  std::vector<json> ids;
  std::string line;
  while (std::getline(input, line)) {
    auto r = json::parse(line);
    data.push_back(encode(vocab, r.at("prompt"), true));
    ids.push_back(r.at("id"));
  }
  if (data.empty())
    return 0;
  size_t common = data[0].size() - 1;
  for (auto &v : data) {
    size_t n = 0;
    while (n < common && n + 1 < v.size() && v[n] == data[0][n])
      ++n;
    common = n;
  }
  tokens letters;
  for (auto s : {"A", "B", "C"}) {
    auto a = encode(vocab, s, false);
    if (a.size() != 1)
      throw std::runtime_error("label");
    letters.push_back(a[0]);
  }
  auto b = llama_batch_init(cp.n_batch, 0, 1);
  b.n_tokens = 0;
  for (size_t j = 0; j < common; j++)
    add(b, data[0][j], j, 0, j + 1 == common);
  if (common == 0 || llama_decode(c, b))
    return 6;
  std::vector<uint8_t> state(llama_state_seq_get_size(c, 0));
  if (llama_state_seq_get_data(c, state.data(), state.size(), 0) !=
      state.size())
    return 7;
  std::cerr << "prefix=" << common << " state=" << state.size() << std::endl;
  for (size_t start = 0; start < data.size(); start += group) {
    int count = std::min((size_t)group, data.size() - start);
    llama_memory_clear(llama_get_memory(c), false);
    b.n_tokens = 0;
    std::vector<int> last(count);
    for (int s = 0; s < count; s++) {
      if (llama_state_seq_set_data(c, state.data(), state.size(), s) !=
          state.size())
        return 8;
      auto &v = data[start + s];
      if (b.n_tokens + v.size() - common > cp.n_batch ||
          v.size() + budget + 12 > 768)
        throw std::runtime_error("input too large");
      for (size_t j = common; j < v.size(); j++)
        add(b, v[j], j, s, j + 1 == v.size());
      last[s] = b.n_tokens - 1;
    }
    if (llama_decode(c, b))
      return 9;
    if (budget) {
      const auto close = encode(vocab, "</think>", true);
      const auto suffix = encode(vocab, "\n</think>\n\n", true);
      if (close.size() != 1)
        throw std::runtime_error("Unrecognized reasoning delimiter");
      std::vector<int> pos(count);
      for (int s = 0; s < count; ++s)
        pos[s] = data[start + s].size();
      const int vocab_size = llama_vocab_n_tokens(vocab);
      for (int step = 0; step < budget; ++step) {
        b.n_tokens = 0;
        for (int s = 0; s < count; ++s) {
          if (last[s] < 0)
            continue;
          const auto logits = llama_get_logits_ith(c, last[s]);
          if (!logits)
            return 10;
          const int next =
              std::max_element(logits, logits + vocab_size) - logits;
          if (next == close[0] || llama_vocab_is_eog(vocab, next)) {
            last[s] = -1;
          } else {
            add(b, next, pos[s]++, s, true);
            last[s] = b.n_tokens - 1;
          }
        }
        if (!b.n_tokens)
          break;
        if (llama_decode(c, b))
          return 9;
      }
      b.n_tokens = 0;
      for (int s = 0; s < count; ++s) {
        for (size_t j = 0; j < suffix.size(); ++j)
          add(b, suffix[j], pos[s]++, s, j + 1 == suffix.size());
        last[s] = b.n_tokens - 1;
      }
      if (llama_decode(c, b))
        return 9;
    }
    for (int s = 0; s < count; s++) {
      auto logits = llama_get_logits_ith(c, last[s]);
      if (!logits)
        return 10;
      std::vector<double> scores;
      for (int l : letters)
        scores.push_back(logits[l]);
      out << json{{"id", ids[start + s]}, {"logits", scores}}.dump()
          << std::endl;
    }
  }
  llama_batch_free(b);
  llama_free(c);
  llama_model_free(m);
  llama_backend_free();
}
