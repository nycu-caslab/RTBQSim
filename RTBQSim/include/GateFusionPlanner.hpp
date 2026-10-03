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

struct PlannerGateInfo {
  std::vector<int> qubits{};
  bool is_diagonal = false;
  bool is_cnot = false;
  int control = -1;
  int target = -1;
};

inline PlannerGateInfo plannerBuildGateInfo(const qc::GatePrimitive& gate) {
  PlannerGateInfo info;
  info.qubits = plannerTouchedQubits(gate);
  info.is_diagonal = plannerIsDiagonalGate(gate);
  info.is_cnot = gate.control_count == 1 &&
                 gate.target_count == 1 &&
                 static_cast<qc::OpType>(gate.gate_type) == qc::X &&
                 plannerGateHasOneRayPerRow(gate);
  if (info.is_cnot) {
    info.control = gate.controls[0];
    info.target = gate.targets[0];
  }
  return info;
}

inline bool plannerHasSingleton(const PlannerBlockState& state, int qubit) {
  return std::any_of(state.qubit_sets.begin(),
                     state.qubit_sets.end(),
                     [qubit](const std::vector<int>& qubit_set) {
                       return qubit_set.size() == 1 && qubit_set.front() == qubit;
                     });
}

inline bool plannerFindSaturatedSubsetOfSize(
    const PlannerBlockState& state,
    std::size_t target_size,
    std::size_t next_index,
    std::vector<std::size_t>& selected_indices,
    const std::vector<int>& selected_qubits,
    bool has_composite,
    std::vector<std::size_t>& result_indices,
    std::vector<int>& result_qubits) {
  if (selected_indices.size() == target_size) {
    if (has_composite && selected_qubits.size() <= target_size) {
      result_indices = selected_indices;
      result_qubits = selected_qubits;
      return true;
    }
    return false;
  }

  const std::size_t needed = target_size - selected_indices.size();
  const std::size_t set_count = state.qubit_sets.size();
  for (std::size_t index = next_index; index + needed <= set_count; ++index) {
    const auto merged_qubits = plannerUnionQubits(selected_qubits,
                                                  state.qubit_sets[index]);
    if (merged_qubits.size() > target_size) {
      continue;
    }
    selected_indices.push_back(index);
    if (plannerFindSaturatedSubsetOfSize(state,
                                         target_size,
                                         index + 1,
                                         selected_indices,
                                         merged_qubits,
                                         has_composite || state.qubit_sets[index].size() > 1,
                                         result_indices,
                                         result_qubits)) {
      return true;
    }
    selected_indices.pop_back();
  }
  return false;
}

inline bool plannerFindSaturatedSubset(const PlannerBlockState& state,
                                       std::vector<std::size_t>& result_indices,
                                       std::vector<int>& result_qubits) {
  for (std::size_t target_size = 2;
       target_size <= state.qubit_sets.size();
       ++target_size) {
    std::vector<std::size_t> selected_indices;
    selected_indices.reserve(target_size);
    if (plannerFindSaturatedSubsetOfSize(state,
                                         target_size,
                                         0,
                                         selected_indices,
                                         {},
                                         false,
                                         result_indices,
                                         result_qubits)) {
      return true;
    }
  }
  return false;
}

inline void plannerNormalizeSaturatedSets(PlannerBlockState& state) {
  std::vector<std::size_t> saturated_indices;
  std::vector<int> saturated_qubits;
  while (plannerFindSaturatedSubset(state, saturated_indices, saturated_qubits)) {
    std::vector<char> remove_set(state.qubit_sets.size(), 0);
    for (std::size_t index : saturated_indices) {
      remove_set[index] = 1;
    }

    std::vector<std::vector<int>> normalized_sets;
    normalized_sets.reserve(state.qubit_sets.size() - saturated_indices.size() +
                            saturated_qubits.size());
    for (std::size_t index = 0; index < state.qubit_sets.size(); ++index) {
      if (!remove_set[index]) {
        normalized_sets.push_back(state.qubit_sets[index]);
      }
    }
    state.qubit_sets = std::move(normalized_sets);
    for (int qubit : saturated_qubits) {
      if (!plannerHasSingleton(state, qubit)) {
        state.qubit_sets.push_back({qubit});
      }
    }
  }
}

struct PlannerSearchState {
  PlannerBlockState block_state{};
  std::vector<std::size_t> chosen{};
  std::vector<std::uint64_t> selected{};
  std::size_t first_gate = std::numeric_limits<std::size_t>::max();
  std::size_t last_gate = 0;
};

inline bool plannerGateSelected(const std::vector<std::uint64_t>& selected,
                                std::size_t gate) {
  return (selected[gate >> 6U] & (1ULL << (gate & 63U))) != 0;
}

inline void plannerSetGateSelected(std::vector<std::uint64_t>& selected,
                                   std::size_t gate) {
  selected[gate >> 6U] |= 1ULL << (gate & 63U);
}

inline void plannerClearGateSelected(std::vector<std::uint64_t>& selected,
                                     std::size_t gate) {
  selected[gate >> 6U] &= ~(1ULL << (gate & 63U));
}

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

inline void plannerInitializeReadyCandidates(
    const std::vector<std::vector<std::size_t>>& predecessors,
    const std::vector<char>& globally_scheduled,
    std::vector<std::size_t>& pending_predecessors,
    std::vector<std::size_t>& out_candidates) {
  pending_predecessors.assign(predecessors.size(), 0);
  out_candidates.clear();
  for (std::size_t idx = 0; idx < predecessors.size(); ++idx) {
    if (globally_scheduled[idx]) {
      continue;
    }
    for (std::size_t pred : predecessors[idx]) {
      if (!globally_scheduled[pred]) {
        ++pending_predecessors[idx];
      }
    }
    if (pending_predecessors[idx] == 0) {
      out_candidates.push_back(idx);
    }
  }
}

inline bool plannerApplyGateToBlockState(const PlannerGateInfo& gate_info,
                                         int max_group_qubits,
                                         PlannerBlockState& state) {
  if (gate_info.is_diagonal) {
    return true;
  }

  if (gate_info.is_cnot) {
    const int control = gate_info.control;
    const int target = gate_info.target;
    if (plannerHasSingleton(state, control) && plannerHasSingleton(state, target)) {
      return true;
    }

    bool updated = false;
    for (auto& qubit_set : state.qubit_sets) {
      if (qubit_set.size() == 1 && qubit_set.front() == control) {
        qubit_set = plannerUnionQubits(qubit_set, gate_info.qubits);
        updated = true;
        break;
      }
    }

    if (!updated) {
      for (auto it = state.qubit_sets.rbegin(); it != state.qubit_sets.rend(); ++it) {
        if (std::binary_search(it->begin(), it->end(), control)) {
          *it = plannerUnionQubits(*it, gate_info.qubits);
          updated = true;
          break;
        }
      }
    }

    if (!updated) {
      return true;
    }

    plannerNormalizeSaturatedSets(state);
    return true;
  }

  for (int qubit : gate_info.qubits) {
    if (!plannerHasSingleton(state, qubit)) {
      state.qubit_sets.push_back({qubit});
    }
  }
  plannerNormalizeSaturatedSets(state);

  if (static_cast<int>(state.qubit_sets.size()) > max_group_qubits) {
    return false;
  }
  return true;
}

inline bool plannerApplyGateToBlockState(const qc::GatePrimitive& gate,
                                         const std::vector<int>& gate_qubits,
                                         int max_group_qubits,
                                         PlannerBlockState& state) {
  PlannerGateInfo gate_info = plannerBuildGateInfo(gate);
  gate_info.qubits = gate_qubits;
  return plannerApplyGateToBlockState(gate_info, max_group_qubits, state);
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
  std::vector<std::uint64_t> selected{};
  std::vector<std::vector<int>> qubit_sets{};

  bool operator==(const PlannerSearchKey& other) const {
    return selected == other.selected && qubit_sets == other.qubit_sets;
  }
};

struct PlannerSearchKeyHash {
  std::size_t operator()(const PlannerSearchKey& key) const {
    std::size_t hash = key.selected.size();
    const auto combine = [&hash](std::size_t value) {
      hash ^= value + 0x9e3779b9U + (hash << 6U) + (hash >> 2U);
    };

    for (std::uint64_t word : key.selected) {
      combine(std::hash<std::uint64_t>{}(word));
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
    const std::vector<PlannerGateInfo>& gate_infos,
    const std::vector<std::vector<std::size_t>>& successors,
    const std::vector<char>& globally_scheduled,
    int max_group_qubits,
    PlannerSearchState& state,
    std::vector<std::size_t>& pending_predecessors,
    std::vector<std::size_t>& ready_candidates,
    std::unordered_set<PlannerSearchKey, PlannerSearchKeyHash>& visited,
    PlannerSearchState& best_state,
    bool& have_best_state) {
  PlannerSearchKey key{state.selected, state.block_state.qubit_sets};
  if (!visited.insert(std::move(key)).second) {
    return;
  }

  if (!state.chosen.empty() &&
      (!have_best_state || plannerSearchStateBetter(state, best_state))) {
    best_state = state;
    have_best_state = true;
  }

  for (std::size_t candidate_pos = 0;
       candidate_pos < ready_candidates.size();
       ++candidate_pos) {
    const std::size_t idx = ready_candidates[candidate_pos];
    PlannerBlockState next_block_state = state.block_state;
    if (!plannerApplyGateToBlockState(gate_infos[idx],
                                      max_group_qubits,
                                      next_block_state)) {
      continue;
    }

    PlannerBlockState previous_block_state = std::move(state.block_state);
    state.block_state = std::move(next_block_state);
    plannerSetGateSelected(state.selected, idx);
    state.chosen.push_back(idx);
    const std::size_t previous_first_gate = state.first_gate;
    const std::size_t previous_last_gate = state.last_gate;
    state.first_gate = std::min(state.first_gate, idx);
    state.last_gate = std::max(state.last_gate, idx);

    ready_candidates.erase(ready_candidates.begin() + candidate_pos);
    std::vector<std::size_t> newly_ready;
    newly_ready.reserve(successors[idx].size());
    for (std::size_t successor : successors[idx]) {
      if (globally_scheduled[successor] || plannerGateSelected(state.selected, successor)) {
        continue;
      }
      --pending_predecessors[successor];
      if (pending_predecessors[successor] == 0) {
        const auto position = std::lower_bound(ready_candidates.begin(),
                                               ready_candidates.end(),
                                               successor);
        ready_candidates.insert(position, successor);
        newly_ready.push_back(successor);
      }
    }

    plannerSearchMaximumBlock(gate_infos,
                              successors,
                              globally_scheduled,
                              max_group_qubits,
                              state,
                              pending_predecessors,
                              ready_candidates,
                              visited,
                              best_state,
                              have_best_state);

    for (std::size_t successor : successors[idx]) {
      if (globally_scheduled[successor] || plannerGateSelected(state.selected, successor)) {
        continue;
      }
      ++pending_predecessors[successor];
    }
    for (std::size_t successor : newly_ready) {
      const auto position = std::lower_bound(ready_candidates.begin(),
                                             ready_candidates.end(),
                                             successor);
      if (position != ready_candidates.end() && *position == successor) {
        ready_candidates.erase(position);
      }
    }
    ready_candidates.insert(ready_candidates.begin() + candidate_pos, idx);

    state.last_gate = previous_last_gate;
    state.first_gate = previous_first_gate;
    state.chosen.pop_back();
    plannerClearGateSelected(state.selected, idx);
    state.block_state = std::move(previous_block_state);
  }
}

inline bool plannerChooseNextBlock(const std::vector<PlannerGateInfo>& gate_infos,
                                   const std::vector<std::vector<std::size_t>>& predecessors,
                                   const std::vector<std::vector<std::size_t>>& successors,
                                   const std::vector<char>& globally_scheduled,
                                   int max_group_qubits,
                                   std::vector<std::size_t>& chosen_block) {
  chosen_block.clear();

  PlannerSearchState initial;
  initial.selected.assign((gate_infos.size() + 63U) / 64U, 0);

  std::vector<std::size_t> pending_predecessors;
  std::vector<std::size_t> ready_candidates;
  plannerInitializeReadyCandidates(predecessors,
                                   globally_scheduled,
                                   pending_predecessors,
                                   ready_candidates);

  PlannerSearchState best_state;
  bool have_best_state = false;
  std::unordered_set<PlannerSearchKey, PlannerSearchKeyHash> visited;
  plannerSearchMaximumBlock(gate_infos,
                            successors,
                            globally_scheduled,
                            max_group_qubits,
                            initial,
                            pending_predecessors,
                            ready_candidates,
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
  std::vector<PlannerGateInfo> gate_infos(n);
  std::vector<std::vector<std::size_t>> predecessors(n);
  std::vector<std::vector<std::size_t>> successors(n);
  std::unordered_map<int, std::size_t> last_touch;
  last_touch.reserve(n * 2);

  for (std::size_t idx = 0; idx < n; ++idx) {
    gate_infos[idx] = plannerBuildGateInfo(primitives[idx]);
    const auto& qubits = gate_infos[idx].qubits;
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
  for (std::size_t idx = 0; idx < n; ++idx) {
    for (std::size_t predecessor : predecessors[idx]) {
      successors[predecessor].push_back(idx);
    }
  }

  std::vector<char> globally_scheduled(n, 0);
  std::size_t scheduled_count = 0;

  out.ordered_primitives.reserve(n);
  out.block_sizes.reserve(n);

  while (scheduled_count < n) {
    std::vector<std::size_t> chosen_block;
    if (!plannerChooseNextBlock(gate_infos,
                                predecessors,
                                successors,
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
