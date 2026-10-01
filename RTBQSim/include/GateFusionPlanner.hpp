#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "GatePrimitive.hpp"
#include "operations/OpType.hpp"

namespace bqsim_rt {

struct GateFusionPlan {
  std::vector<qc::GatePrimitive> ordered_primitives{};
  std::vector<std::size_t> block_sizes{};
};

inline bool plannerIsZeroMatrixEntry(const bqsim_rt::MatrixElem& value) {
  return value.x == 0.0 && value.y == 0.0;
}

inline int plannerGateRowNNZUpperBound(const qc::GatePrimitive& gate) { // used for debug information
  if (gate.target_count <= 0 || gate.matrix_dim <= 0) {
    return 1;
  }
  int max_row = 0;
  for (int r = 0; r < gate.matrix_dim; ++r) {
    int nnz = 0;
    for (int c = 0; c < gate.matrix_dim; ++c) {
      if (!plannerIsZeroMatrixEntry(gate.matrix[r * gate.matrix_dim + c])) {
        ++nnz;
      }
    }
    max_row = std::max(max_row, nnz);
  }
  return std::max(1, max_row);
}

inline bool plannerIsDiagonalGate(const qc::GatePrimitive& gate) { // detect diagonal gate 
  const int dim = gate.matrix_dim;
  if (dim <= 0 || dim > 4) {
    return false;
  }
  for (int r = 0; r < dim; ++r) {
    for (int c = 0; c < dim; ++c) {
      if (r != c && !plannerIsZeroMatrixEntry(gate.matrix[r * dim + c])) {
        return false;
      }
    }
  }
  return true;
}

inline bool plannerGateHasOneRayPerRow(const qc::GatePrimitive& gate) { // for row nnz = 1
  if (gate.target_count != 1 || gate.matrix_dim != 2) {
    return false;
  }
  for (int r = 0; r < 2; ++r) {
    int row_nnz = 0;
    for (int c = 0; c < 2; ++c) {
      if (!plannerIsZeroMatrixEntry(gate.matrix[r * 2 + c])) {
        ++row_nnz;
      }
    }
    if (row_nnz != 1) {
      return false;
    }
  }
  return true;
}

inline bool plannerGateIsWidthPreserving(const qc::GatePrimitive& gate) { // for row nnz = 1 (including diagonal gate)
  return plannerIsDiagonalGate(gate) || plannerGateHasOneRayPerRow(gate);
}

inline bool plannerGateIsCNOT(const qc::GatePrimitive& gate) {
  return gate.control_count == 1 &&
         gate.target_count == 1 &&
         static_cast<qc::OpType>(gate.gate_type) == qc::X &&
         plannerGateHasOneRayPerRow(gate);
}

inline std::vector<int> plannerTouchedQubits(const qc::GatePrimitive& gate) {
  std::vector<int> qubits;
  qubits.reserve(static_cast<std::size_t>(gate.control_count + gate.target_count));
  for (int i = 0; i < gate.control_count; ++i) {
    qubits.push_back(gate.controls[i]);
  }
  for (int i = 0; i < gate.target_count; ++i) {
    qubits.push_back(gate.targets[i]);
  }
  std::sort(qubits.begin(), qubits.end());
  qubits.erase(std::unique(qubits.begin(), qubits.end()), qubits.end());
  return qubits;
}

inline std::vector<int> plannerUnionQubits(const std::vector<int>& lhs,
                                           const std::vector<int>& rhs) {
  std::vector<int> merged;
  merged.reserve(lhs.size() + rhs.size());
  std::set_union(lhs.begin(), lhs.end(),
                 rhs.begin(), rhs.end(),
                 std::back_inserter(merged));
  return merged;
}

struct PlannerBlockState {
  std::vector<std::vector<int>> qubit_sets{};
};

inline bool plannerHasSingleton(const PlannerBlockState& state, int qubit) {
  return std::any_of(state.qubit_sets.begin(),
                     state.qubit_sets.end(),
                     [qubit](const std::vector<int>& qubit_set) {
                       return qubit_set.size() == 1 && qubit_set.front() == qubit;
                     });
}

inline bool plannerQubitSetsIntersect(const std::vector<int>& lhs,
                                      const std::vector<int>& rhs) {
  std::size_t i = 0;
  std::size_t j = 0;
  while (i < lhs.size() && j < rhs.size()) {
    if (lhs[i] == rhs[j]) {
      return true;
    }
    if (lhs[i] < rhs[j]) {
      ++i;
    } else {
      ++j;
    }
  }
  return false;
}

inline void plannerRemoveFullyCoveredCompositeSets(PlannerBlockState& state) {
  std::vector<int> singleton_qubits;
  singleton_qubits.reserve(state.qubit_sets.size());
  for (const auto& qubit_set : state.qubit_sets) {
    if (qubit_set.size() == 1) {
      singleton_qubits.push_back(qubit_set.front());
    }
  }
  std::sort(singleton_qubits.begin(), singleton_qubits.end());
  singleton_qubits.erase(std::unique(singleton_qubits.begin(), singleton_qubits.end()),
                         singleton_qubits.end());

  state.qubit_sets.erase(
      std::remove_if(state.qubit_sets.begin(),
                     state.qubit_sets.end(),
                     [&singleton_qubits](const std::vector<int>& qubit_set) {
                       return qubit_set.size() > 1 &&
                              std::includes(singleton_qubits.begin(),
                                            singleton_qubits.end(),
                                            qubit_set.begin(),
                                            qubit_set.end());
                     }),
      state.qubit_sets.end());
}

inline bool plannerComponentHasMoreDimensionsThanQubits(const PlannerBlockState& state,
                                                        std::size_t seed) {
  std::vector<char> in_component(state.qubit_sets.size(), 0);
  in_component[seed] = 1;

  bool expanded = true;
  while (expanded) {
    expanded = false;
    for (std::size_t i = 0; i < state.qubit_sets.size(); ++i) {
      if (in_component[i]) {
        continue;
      }
      for (std::size_t j = 0; j < state.qubit_sets.size(); ++j) {
        if (in_component[j] &&
            plannerQubitSetsIntersect(state.qubit_sets[i], state.qubit_sets[j])) {
          in_component[i] = 1;
          expanded = true;
          break;
        }
      }
    }
  }

  std::vector<int> physical_qubits;
  std::size_t dimensions = 0;
  for (std::size_t i = 0; i < state.qubit_sets.size(); ++i) {
    if (!in_component[i]) {
      continue;
    }
    physical_qubits = plannerUnionQubits(physical_qubits, state.qubit_sets[i]);
    ++dimensions;
  }
  return dimensions > physical_qubits.size();
}

inline void plannerRemovePhysicallyRedundantDuplicateCompositeSets(
    PlannerBlockState& state) {
  while (true) {
    bool removed = false;
    for (std::size_t i = 0; i < state.qubit_sets.size(); ++i) {
      if (state.qubit_sets[i].size() <= 1) {
        continue;
      }
      const auto duplicate = std::find(state.qubit_sets.begin() + i + 1,
                                       state.qubit_sets.end(),
                                       state.qubit_sets[i]);
      if (duplicate == state.qubit_sets.end() ||
          !plannerComponentHasMoreDimensionsThanQubits(state, i)) {
        continue;
      }
      state.qubit_sets.erase(state.qubit_sets.begin() + i);
      removed = true;
      break;
    }
    if (!removed) {
      return;
    }
  }
}

struct PlannerSearchState {
  PlannerBlockState block_state{};
  std::vector<std::size_t> chosen{};
  std::vector<char> in_block{};
  std::size_t first_gate = std::numeric_limits<std::size_t>::max();
  std::size_t last_gate = 0;
};

inline std::size_t plannerBlockSpan(const PlannerSearchState& state) {
  if (state.chosen.empty()) {
    return 0;
  }
  return state.last_gate - state.first_gate + 1;
}

inline bool plannerSearchStateBetter(const PlannerSearchState& lhs,
                                     const PlannerSearchState& rhs) {
  if (lhs.chosen.size() != rhs.chosen.size()) {
    return lhs.chosen.size() > rhs.chosen.size();
  }
  if (lhs.block_state.qubit_sets.size() != rhs.block_state.qubit_sets.size()) {
    return lhs.block_state.qubit_sets.size() < rhs.block_state.qubit_sets.size();
  }
  const std::size_t lhs_span = plannerBlockSpan(lhs);
  const std::size_t rhs_span = plannerBlockSpan(rhs);
  if (lhs_span != rhs_span) {
    return lhs_span < rhs_span;
  }
  if (lhs.last_gate != rhs.last_gate) {
    return lhs.last_gate < rhs.last_gate;
  }
  return lhs.chosen < rhs.chosen;
}

inline bool plannerIsReadyForBlock(std::size_t idx,
                                   const std::vector<std::vector<std::size_t>>& predecessors,
                                   const std::vector<char>& globally_scheduled,
                                   const std::vector<char>& in_block) {
  if (globally_scheduled[idx] || in_block[idx]) {
    return false;
  }
  for (std::size_t pred : predecessors[idx]) {
    if (!globally_scheduled[pred] && !in_block[pred]) {
      return false;
    }
  }
  return true;
}

inline void plannerCollectReadyCandidates(
    const std::vector<std::vector<std::size_t>>& predecessors,
    const std::vector<char>& globally_scheduled,
    const std::vector<char>& in_block,
    std::vector<std::size_t>& out_candidates) {
  out_candidates.clear();
  for (std::size_t idx = 0; idx < predecessors.size(); ++idx) {
    if (!plannerIsReadyForBlock(idx, predecessors, globally_scheduled, in_block)) {
      continue;
    }
    out_candidates.push_back(idx);
  }
}

inline bool plannerApplyGateToBlockState(const qc::GatePrimitive& gate,
                                         const std::vector<int>& gate_qubits,
                                         int max_group_qubits,
                                         PlannerBlockState& state) {
  if (plannerIsDiagonalGate(gate)) {
    return true;
  }

  if (plannerGateIsCNOT(gate)) {
    const int control = gate.controls[0];
    const int target = gate.targets[0];
    if (plannerHasSingleton(state, control) && plannerHasSingleton(state, target)) {
      return true;
    }

    bool updated = false;
    for (auto& qubit_set : state.qubit_sets) {
      if (qubit_set.size() == 1 && qubit_set.front() == control) {
        qubit_set = plannerUnionQubits(qubit_set, gate_qubits);
        updated = true;
        break;
      }
    }

    if (!updated) {
      for (auto it = state.qubit_sets.rbegin(); it != state.qubit_sets.rend(); ++it) {
        if (std::binary_search(it->begin(), it->end(), control)) {
          *it = plannerUnionQubits(*it, gate_qubits);
          updated = true;
          break;
        }
      }
    }

    if (!updated) {
      return true;
    }

    while (true) {
      bool split = false;
      for (const auto& candidate : state.qubit_sets) {
        if (candidate.size() <= 1) {
          continue;
        }
        const auto copies = static_cast<std::size_t>(std::count(
            state.qubit_sets.begin(), state.qubit_sets.end(), candidate));
        if (copies != candidate.size()) {
          continue;
        }

        const std::vector<int> copied_set = candidate;
        state.qubit_sets.erase(
            std::remove(state.qubit_sets.begin(), state.qubit_sets.end(), copied_set),
            state.qubit_sets.end());
        for (int qubit : copied_set) {
          if (!plannerHasSingleton(state, qubit)) {
            state.qubit_sets.push_back({qubit});
          }
        }
        split = true;
        break;
      }
      if (!split) {
        break;
      }
    }

    plannerRemoveFullyCoveredCompositeSets(state);
    plannerRemovePhysicallyRedundantDuplicateCompositeSets(state);
    return true;
  }

  for (int qubit : gate_qubits) {
    if (!plannerHasSingleton(state, qubit)) {
      state.qubit_sets.push_back({qubit});
    }
  }
  plannerRemoveFullyCoveredCompositeSets(state);
  plannerRemovePhysicallyRedundantDuplicateCompositeSets(state);

  if (static_cast<int>(state.qubit_sets.size()) > max_group_qubits) {
    return false;
  }
  return true;
}

inline bool plannerAppendGateRangeToState(const std::vector<qc::GatePrimitive>& gates,
                                          std::size_t begin,
                                          std::size_t count,
                                          int max_group_qubits,
                                          PlannerBlockState& state) {
  const std::size_t end = begin + count;
  for (std::size_t idx = begin; idx < end; ++idx) {
    const auto gate_qubits = plannerTouchedQubits(gates[idx]);
    if (!plannerApplyGateToBlockState(gates[idx],
                                      gate_qubits,
                                      max_group_qubits,
                                      state)) {
      return false;
    }
  }
  return true;
}

struct PlannerSearchKey {
  std::vector<char> in_block{};
  std::vector<std::vector<int>> qubit_sets{};

  bool operator==(const PlannerSearchKey& other) const {
    return in_block == other.in_block && qubit_sets == other.qubit_sets;
  }
};

struct PlannerSearchKeyHash {
  std::size_t operator()(const PlannerSearchKey& key) const {
    std::size_t hash = key.in_block.size();
    const auto combine = [&hash](std::size_t value) {
      hash ^= value + 0x9e3779b9U + (hash << 6U) + (hash >> 2U);
    };

    for (char selected : key.in_block) {
      combine(static_cast<std::size_t>(selected));
    }
    combine(key.qubit_sets.size());
    for (const auto& qubit_set : key.qubit_sets) {
      combine(qubit_set.size());
      for (int qubit : qubit_set) {
        combine(std::hash<int>{}(qubit));
      }
    }
    return hash;
  }
};

inline void plannerSearchMaximumBlock(
    const std::vector<qc::GatePrimitive>& primitives,
    const std::vector<std::vector<int>>& gate_qubits,
    const std::vector<std::vector<std::size_t>>& predecessors,
    const std::vector<char>& globally_scheduled,
    int max_group_qubits,
    const PlannerSearchState& state,
    std::unordered_set<PlannerSearchKey, PlannerSearchKeyHash>& visited,
    PlannerSearchState& best_state,
    bool& have_best_state) {
  PlannerSearchKey key{state.in_block, state.block_state.qubit_sets};
  if (!visited.insert(std::move(key)).second) {
    return;
  }

  if (!state.chosen.empty() &&
      (!have_best_state || plannerSearchStateBetter(state, best_state))) {
    best_state = state;
    have_best_state = true;
  }

  std::vector<std::size_t> ready_candidates;
  plannerCollectReadyCandidates(predecessors,
                                globally_scheduled,
                                state.in_block,
                                ready_candidates);

  for (std::size_t idx : ready_candidates) {
    PlannerSearchState next_state = state;
    if (!plannerApplyGateToBlockState(primitives[idx],
                                      gate_qubits[idx],
                                      max_group_qubits,
                                      next_state.block_state)) {
      continue;
    }

    next_state.in_block[idx] = 1;
    next_state.chosen.push_back(idx);
    next_state.first_gate = std::min(next_state.first_gate, idx);
    next_state.last_gate = std::max(next_state.last_gate, idx);
    plannerSearchMaximumBlock(primitives,
                              gate_qubits,
                              predecessors,
                              globally_scheduled,
                              max_group_qubits,
                              next_state,
                              visited,
                              best_state,
                              have_best_state);
  }
}

inline bool plannerChooseNextBlock(const std::vector<qc::GatePrimitive>& primitives,
                                   const std::vector<std::vector<int>>& gate_qubits,
                                   const std::vector<std::vector<std::size_t>>& predecessors,
                                   const std::vector<char>& globally_scheduled,
                                   int max_group_qubits,
                                   std::vector<std::size_t>& chosen_block) {
  chosen_block.clear();

  PlannerSearchState initial;
  initial.in_block.assign(primitives.size(), 0);

  PlannerSearchState best_state;
  bool have_best_state = false;
  std::unordered_set<PlannerSearchKey, PlannerSearchKeyHash> visited;
  plannerSearchMaximumBlock(primitives,
                            gate_qubits,
                            predecessors,
                            globally_scheduled,
                            max_group_qubits,
                            initial,
                            visited,
                            best_state,
                            have_best_state);

  if (!have_best_state) {
    return false;
  }
  chosen_block = std::move(best_state.chosen);
  return !chosen_block.empty();
}

inline int plannerMaxGroupQubitsFromRowNNZLimit(int row_nnz_limit) {
  if (row_nnz_limit <= 0) {
    return std::numeric_limits<int>::max();
  }
  int max_qubits = 0;
  int capacity = 1;
  while (capacity < row_nnz_limit) {
    capacity <<= 1;
    ++max_qubits;
  }
  if (capacity > row_nnz_limit && max_qubits > 0) {
    --max_qubits;
  }
  return std::max(0, max_qubits);
}

inline bool buildGateFusionPlan(const std::vector<qc::GatePrimitive>& primitives,
                                int row_nnz_limit,
                                GateFusionPlan& out) {
  out.ordered_primitives.clear();
  out.block_sizes.clear();
  if (primitives.empty()) {
    return true;
  }
  const int max_group_qubits = plannerMaxGroupQubitsFromRowNNZLimit(row_nnz_limit);

  const std::size_t n = primitives.size();
  std::vector<std::vector<int>> gate_qubits(n);
  std::vector<std::vector<std::size_t>> predecessors(n);
  std::unordered_map<int, std::size_t> last_touch;
  last_touch.reserve(n * 2);

  for (std::size_t idx = 0; idx < n; ++idx) {
    gate_qubits[idx] = plannerTouchedQubits(primitives[idx]);
    const auto& qubits = gate_qubits[idx];
    std::vector<std::size_t> preds;
    preds.reserve(qubits.size());
    for (int q : qubits) {
      const auto it = last_touch.find(q);
      if (it != last_touch.end()) {
        const std::size_t pred = it->second;
        if (std::find(preds.begin(), preds.end(), pred) == preds.end()) {
          preds.push_back(pred);
        }
      }
    }
    predecessors[idx] = std::move(preds);
    for (int q : qubits) {
      last_touch[q] = idx;
    }
  }

  std::vector<char> globally_scheduled(n, 0);
  std::size_t scheduled_count = 0;

  out.ordered_primitives.reserve(n);
  out.block_sizes.reserve(n);

  while (scheduled_count < n) {
    std::vector<std::size_t> chosen_block;
    if (!plannerChooseNextBlock(primitives,
                                gate_qubits,
                                predecessors,
                                globally_scheduled,
                                max_group_qubits,
                                chosen_block)) {
      return false;
    }

    for (std::size_t idx : chosen_block) {
      if (globally_scheduled[idx]) {
        return false;
      }
      globally_scheduled[idx] = 1;
      ++scheduled_count;
      out.ordered_primitives.push_back(primitives[idx]);
    }
    out.block_sizes.push_back(chosen_block.size());
  }

  if (out.block_sizes.size() > 1) {
    std::vector<std::size_t> merged_block_sizes;
    merged_block_sizes.reserve(out.block_sizes.size());

    std::size_t gate_cursor = 0;
    std::size_t current_block_size = out.block_sizes[0];
    PlannerBlockState current_state;
    if (!plannerAppendGateRangeToState(out.ordered_primitives,
                                       gate_cursor,
                                       current_block_size,
                                       max_group_qubits,
                                       current_state)) {
      return false;
    }

    gate_cursor += current_block_size;
    for (std::size_t block_idx = 1; block_idx < out.block_sizes.size(); ++block_idx) {
      const std::size_t next_block_size = out.block_sizes[block_idx];
      PlannerBlockState trial_state = current_state;
      if (plannerAppendGateRangeToState(out.ordered_primitives,
                                        gate_cursor,
                                        next_block_size,
                                        max_group_qubits,
                                        trial_state)) {
        current_block_size += next_block_size;
        current_state = std::move(trial_state);
      } else {
        merged_block_sizes.push_back(current_block_size);
        current_block_size = next_block_size;
        current_state = PlannerBlockState{};
        if (!plannerAppendGateRangeToState(out.ordered_primitives,
                                           gate_cursor,
                                           current_block_size,
                                           max_group_qubits,
                                           current_state)) {
          return false;
        }
      }
      gate_cursor += next_block_size;
    }
    merged_block_sizes.push_back(current_block_size);
    out.block_sizes = std::move(merged_block_sizes);
  }

  return out.ordered_primitives.size() == primitives.size();
}

inline bool buildSequentialGateFusionPlan(const std::vector<qc::GatePrimitive>& primitives,
                                          int row_nnz_limit,
                                          GateFusionPlan& out) {
  out.ordered_primitives.clear();
  out.block_sizes.clear();
  if (primitives.empty()) {
    return true;
  }

  const int max_group_qubits = plannerMaxGroupQubitsFromRowNNZLimit(row_nnz_limit);
  out.ordered_primitives.reserve(primitives.size());
  out.block_sizes.reserve(primitives.size());

  PlannerBlockState current_state;
  std::size_t current_block_size = 0;

  for (std::size_t idx = 0; idx < primitives.size(); ++idx) {
    const auto gate_qubits = plannerTouchedQubits(primitives[idx]);
    PlannerBlockState trial_state = current_state;
    if (plannerApplyGateToBlockState(primitives[idx],
                                     gate_qubits,
                                     max_group_qubits,
                                     trial_state)) {
      current_state = std::move(trial_state);
      ++current_block_size;
      out.ordered_primitives.push_back(primitives[idx]);
      continue;
    }

    if (current_block_size == 0) {
      return false;
    }
    out.block_sizes.push_back(current_block_size);

    current_state = PlannerBlockState{};
    current_block_size = 0;
    if (!plannerApplyGateToBlockState(primitives[idx],
                                      gate_qubits,
                                      max_group_qubits,
                                      current_state)) {
      return false;
    }
    current_block_size = 1;
    out.ordered_primitives.push_back(primitives[idx]);
  }

  if (current_block_size > 0) {
    out.block_sizes.push_back(current_block_size);
  }
  return out.ordered_primitives.size() == primitives.size();
}

}  // namespace bqsim_rt
