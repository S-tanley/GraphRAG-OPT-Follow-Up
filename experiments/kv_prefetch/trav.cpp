// R0 (KV prefetch v2): path enumerators on a CSR graph, called from traversal_profile.py via ctypes.
//
// The CSR keeps networkx's neighbour order (G.adj[u] insertion order), so the BFS here visits states
// in exactly the order of graphrag-main/graph_utils.get_paths_bfs.
//
//   bfs_gold  RoG BFS to the answer entities. prune=1 adds target-distance pruning (PathEnum-style
//             index: multi-source BFS from the targets, drop a child that cannot reach a target in the
//             hops left). Pruning only removes states with no emitting descendant and keeps FIFO order,
//             so the emitted list (and its cap cut) is identical to the unpruned BFS.
//   expand    answer entity unknown: paths from the topic entities to any node.
//             mode 0  BFS order, first N paths (cap)
//             mode 1  all simple paths up to `hop` (DFS), keep the top N by mean edge score
//             mode 2  beam: per depth keep the top W extensions by mean edge score, then the top N
//
// Build: clang++/g++ -O3 -std=c++17 -shared -fPIC trav.cpp -o libtrav.{dylib,so}
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <queue>
#include <vector>

using clk = std::chrono::steady_clock;

static inline double us_since(clk::time_point t0) {
  return std::chrono::duration<double, std::micro>(clk::now() - t0).count();
}

struct States {
  std::vector<int> node, parent, depth;
  std::vector<double> ssum;
  void clear() { node.clear(); parent.clear(); depth.clear(); ssum.clear(); }
  int push(int v, int p, int d, double s) {
    node.push_back(v); parent.push_back(p); depth.push_back(d); ssum.push_back(s);
    return (int)node.size() - 1;
  }
  bool on_path(int st, int v) const {
    for (; st >= 0; st = parent[st])
      if (node[st] == v) return true;
    return false;
  }
  int write_path(int st, int* out) const {
    int d = depth[st];
    for (int i = d; i >= 0; --i) { out[i] = node[st]; st = parent[st]; }
    return d + 1;
  }
};

// cnt: [0] states dequeued, [1] edges scanned, [2] states enqueued, [3] edges scanned by the distance BFS
extern "C" int bfs_gold(int n, const int* indptr, const int* indices,
                        const int* starts, int n_starts, const int* targets, int n_targets,
                        int hop, int cap, int prune,
                        int* out_nodes, int* out_len, double* out_t_us,
                        long long* cnt, double* total_us) {
  auto t0 = clk::now();
  std::vector<uint8_t> is_t(n, 0);
  for (int i = 0; i < n_targets; ++i) is_t[targets[i]] = 1;
  std::vector<int> dist;
  if (prune) {  // distances up to hop-1 are all a child (depth >= 1) can use
    dist.assign(n, hop + 1);
    std::vector<int> q;
    q.reserve(n);
    for (int i = 0; i < n_targets; ++i)
      if (dist[targets[i]] != 0) { dist[targets[i]] = 0; q.push_back(targets[i]); }
    for (size_t h = 0; h < q.size(); ++h) {
      int u = q[h];
      if (dist[u] + 1 > hop - 1) continue;
      for (int e = indptr[u]; e < indptr[u + 1]; ++e) {
        int v = indices[e];
        cnt[3]++;
        if (dist[v] > dist[u] + 1) { dist[v] = dist[u] + 1; q.push_back(v); }
      }
    }
  }
  int emitted = 0;
  States S;
  for (int si = 0; si < n_starts && emitted < cap; ++si) {
    S.clear();
    S.push(starts[si], -1, 0, 0.0);
    for (size_t h = 0; h < S.node.size(); ++h) {
      int st = (int)h, cur = S.node[st], d = S.depth[st];
      cnt[0]++;
      if (is_t[cur] && d >= 1) {
        out_len[emitted] = S.write_path(st, out_nodes + (size_t)emitted * (hop + 1));
        out_t_us[emitted] = us_since(t0);
        if (++emitted >= cap) break;
        continue;
      }
      if (d >= hop) continue;
      for (int e = indptr[cur]; e < indptr[cur + 1]; ++e) {
        int v = indices[e];
        cnt[1]++;
        if (prune && dist[v] > hop - d - 1) continue;
        if (S.on_path(st, v)) continue;
        S.push(v, st, d + 1, 0.0);
        cnt[2]++;
      }
    }
  }
  *total_us = us_since(t0);
  return emitted;
}

struct Item {
  double score;
  long long order;
  int len;
  int nodes[8];
};
// "a is better than b": higher score, then earlier enumeration order
static inline bool better(const Item& a, const Item& b) {
  return a.score > b.score || (a.score == b.score && a.order < b.order);
}
struct WorstOnTop {
  bool operator()(const Item& a, const Item& b) const { return better(a, b); }
};

// cnt: [0] paths (states) generated, [1] edges scanned, [4] 1 if mode 1 hit `limit` and stopped early
extern "C" int expand(int n, const int* indptr, const int* indices, const float* escore,
                      const int* starts, int n_starts, int hop, int mode, int N, int W, long long limit,
                      int* out_nodes, int* out_len, double* out_score, double* out_t_us,
                      long long* cnt, double* total_us) {
  auto t0 = clk::now();
  int emitted = 0;
  if (hop > 7) return -1;
  if (mode == 0) {
    States S;
    for (int si = 0; si < n_starts && emitted < N; ++si) {
      S.clear();
      S.push(starts[si], -1, 0, 0.0);
      for (size_t h = 0; h < S.node.size(); ++h) {
        int st = (int)h, cur = S.node[st], d = S.depth[st];
        if (d >= 1) {
          out_len[emitted] = S.write_path(st, out_nodes + (size_t)emitted * (hop + 1));
          out_score[emitted] = 0.0;
          out_t_us[emitted] = us_since(t0);
          if (++emitted >= N) break;
        }
        if (d >= hop) continue;
        for (int e = indptr[cur]; e < indptr[cur + 1]; ++e) {
          int v = indices[e];
          cnt[1]++;
          if (S.on_path(st, v)) continue;
          S.push(v, st, d + 1, 0.0);
          cnt[0]++;
        }
      }
    }
  } else if (mode == 1) {
    std::priority_queue<Item, std::vector<Item>, WorstOnTop> heap;
    int path[8], pos[8];
    double ssum[8];
    bool stop = false;
    for (int si = 0; si < n_starts && !stop; ++si) {
      int d = 0;
      path[0] = starts[si]; pos[0] = indptr[path[0]]; ssum[0] = 0.0;
      while (d >= 0) {
        if (pos[d] >= indptr[path[d] + 1]) { --d; continue; }
        int e = pos[d]++, v = indices[e];
        cnt[1]++;
        bool seen = false;
        for (int i = 0; i <= d; ++i) if (path[i] == v) { seen = true; break; }
        if (seen) continue;
        int d1 = d + 1;
        path[d1] = v;
        ssum[d1] = ssum[d] + (double)escore[e];
        Item it;
        it.score = ssum[d1] / d1;
        it.order = cnt[0]++;
        if ((int)heap.size() < N || better(it, heap.top())) {
          it.len = d1 + 1;
          for (int i = 0; i <= d1; ++i) it.nodes[i] = path[i];
          heap.push(it);
          if ((int)heap.size() > N) heap.pop();
        }
        if (cnt[0] >= limit) { cnt[4] = 1; stop = true; break; }
        if (d1 < hop) { pos[d1] = indptr[v]; d = d1; }
      }
    }
    std::vector<Item> items;
    while (!heap.empty()) { items.push_back(heap.top()); heap.pop(); }
    std::sort(items.begin(), items.end(), better);
    for (auto& it : items) {
      out_len[emitted] = it.len;
      for (int i = 0; i < it.len; ++i) out_nodes[(size_t)emitted * (hop + 1) + i] = it.nodes[i];
      out_score[emitted] = it.score;
      out_t_us[emitted] = 0.0;
      ++emitted;
    }
  } else {
    States S;
    std::vector<int> frontier, children, cand;
    for (int si = 0; si < n_starts; ++si) frontier.push_back(S.push(starts[si], -1, 0, 0.0));
    auto key_better = [&](int a, int b) {
      double sa = S.ssum[a] / S.depth[a], sb = S.ssum[b] / S.depth[b];
      return sa > sb || (sa == sb && a < b);
    };
    for (int d = 0; d < hop && !frontier.empty(); ++d) {
      children.clear();
      for (int st : frontier) {
        int cur = S.node[st];
        for (int e = indptr[cur]; e < indptr[cur + 1]; ++e) {
          int v = indices[e];
          cnt[1]++;
          if (S.on_path(st, v)) continue;
          children.push_back(S.push(v, st, d + 1, S.ssum[st] + (double)escore[e]));
          cnt[0]++;
        }
      }
      // always sort: the next depth's generation order (the tie-break) follows the frontier order
      if ((int)children.size() > W) {
        std::partial_sort(children.begin(), children.begin() + W, children.end(), key_better);
        children.resize(W);
      } else {
        std::sort(children.begin(), children.end(), key_better);
      }
      frontier = children;
      cand.insert(cand.end(), children.begin(), children.end());
    }
    std::sort(cand.begin(), cand.end(), key_better);
    for (int st : cand) {
      if (emitted >= N) break;
      out_len[emitted] = S.write_path(st, out_nodes + (size_t)emitted * (hop + 1));
      out_score[emitted] = S.ssum[st] / S.depth[st];
      out_t_us[emitted] = 0.0;
      ++emitted;
    }
  }
  double t = us_since(t0);
  if (mode != 0)
    for (int i = 0; i < emitted; ++i) out_t_us[i] = t;
  *total_us = t;
  return emitted;
}
